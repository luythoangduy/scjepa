
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
import importlib   
import yaml, numpy as np, torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from src.utils.tensors import trunc_normal_
from src.datasets.dataset import build_dataloader
import src.dinov2.models.vision_transformer as vit
from transformers import AutoModel, AutoProcessor, SiglipVisionModel, CLIPVisionModel


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_setting(cfg: Optional[Dict[str, Any]], name: str, env_name: str, default=None):
    value = (cfg or {}).get(name)
    if value not in (None, ""):
        return value
    return os.environ.get(env_name, default)


def _configure_torch_hub(cfg: Optional[Dict[str, Any]] = None) -> None:
    hub_dir = _get_setting(cfg, "torchhub_dir", "SCJEPA_TORCH_HUB_DIR")
    if hub_dir:
        hub_dir = Path(hub_dir).expanduser()
    else:
        torch_home = Path(
            _get_setting(cfg, "torch_home", "SCJEPA_TORCH_HOME", _repo_root() / ".cache" / "torch")
        ).expanduser()
        os.environ["TORCH_HOME"] = str(torch_home)
        hub_dir = torch_home / "hub"

    hub_dir.mkdir(parents=True, exist_ok=True)
    (hub_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))


def _hub_load(repo_or_dir: str, model: str, cfg: Optional[Dict[str, Any]] = None, source: Optional[str] = None, **kwargs):
    _configure_torch_hub(cfg)
    repo_path = Path(str(repo_or_dir)).expanduser()
    source = source or ("local" if repo_path.exists() else "github")
    if source == "local":
        repo_or_dir = str(repo_path)
    return torch.hub.load(repo_or_dir, model, source=source, **kwargs)


def _dinov3_hf_id(torch_hub_name: str) -> str:
    mapping = {
        "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "dinov3_vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    }
    return mapping.get(torch_hub_name, "facebook/dinov3-vitb16-pretrain-lvd1689m")


class HuggingFaceDinoV3Backbone(nn.Module):
    def __init__(self, model_id_or_path: str, image_size: int = 512):
        super().__init__()
        model_path = Path(str(model_id_or_path)).expanduser()
        if model_path.exists() and model_path.is_dir():
            required = ["config.json", "model.safetensors"]
            missing = [name for name in required if not (model_path / name).exists()]
            if missing:
                raise FileNotFoundError(
                    f"Hugging Face DINOv3 folder is missing {missing}: {model_path}. "
                    "Download the full model snapshot, not only one file."
                )
        kwargs = {"local_files_only": model_path.exists(), "trust_remote_code": True} if model_path.exists() else {"trust_remote_code": True}
        self.model = AutoModel.from_pretrained(str(model_path) if model_path.exists() else model_id_or_path, **kwargs)
        model_cls = type(self.model).__name__.lower()
        if "dino" not in model_cls:
            raise RuntimeError(
                f"Expected a DINOv3 Hugging Face model, got {type(self.model).__name__}. "
                "Please upgrade transformers to >=4.56.0 and re-download the DINOv3 snapshot."
            )
        self.config = self.model.config
        self.embed_dim = int(getattr(self.config, "hidden_size"))
        self.patch_size = int(getattr(self.config, "patch_size", 16))
        self.num_patches = (int(image_size) // self.patch_size) ** 2

    def get_intermediate_layers(self, imgs: torch.Tensor, n: int = 3, return_class_token: bool = False):
        out = self.model(pixel_values=imgs, output_hidden_states=True)
        hidden_states = out.hidden_states[1:]
        n = max(1, min(int(n), len(hidden_states)))
        layers = hidden_states[-n:]
        outputs = []
        expected_patches = (imgs.shape[-2] // self.patch_size) * (imgs.shape[-1] // self.patch_size)
        for h in layers:
            patch_tokens = h[:, -expected_patches:, :]
            if return_class_token:
                outputs.append((patch_tokens, h[:, 0, :]))
            else:
                outputs.append(patch_tokens)
        return outputs



class LinearProjector(torch.nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = torch.nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class VisionModule(nn.Module):
    def __init__(
        self,
        model_name: str,
        pred_depth: int,
        pred_emb_dim: int,
        use_cuda: bool = True,
        if_pe: bool = True,
        feat_normed: bool = False,
        encoder_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.encoder_cfg = encoder_cfg or {}
        (self.encoder, self.num_patches, self.embed_dim, self.processor, self.projector) = self._build_encoder(model_name)
        self.model_name = model_name

        self.predictor = vit.__dict__["vit_predictor"](num_patches=self.num_patches, embed_dim=self.embed_dim,
                                                         predictor_embed_dim=pred_emb_dim, depth=pred_depth, if_pe=if_pe, feat_normed=feat_normed)
        self._init_predictor(self.predictor)
        self.dropout = nn.Dropout(0.2)
        if use_cuda and torch.cuda.is_available():
            self.cuda()
        self.feat_normed = self.predictor.feat_normed # it depends on the predictor
        print(f"Normed features: {self.feat_normed}")

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)
    
    def target_features(self, images, paths, n_layer=3):
        with torch.no_grad():
            return self._extract(images, paths, n_layer=n_layer)

    def context_features(self, images, paths, n_layer=3):
        z = self._extract(images, paths, n_layer=n_layer)
        p = self.predictor(self.dropout(z))
        return z, p

    def _build_encoder(self, model: str):

        projector = processor = None
        if model == "dinov2":
            enc = _hub_load("facebookresearch/dinov2", "dinov2_vitb14", cfg=self.encoder_cfg).eval(); num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dinov3":
            dinov3_repo = _get_setting(self.encoder_cfg, "dinov3_repo", "DINOV3_REPO", "facebookresearch/dinov3")
            dinov3_model = _get_setting(self.encoder_cfg, "dinov3_model", "DINOV3_MODEL", "dinov3_vitb16")
            dinov3_source = _get_setting(self.encoder_cfg, "dinov3_source", "DINOV3_SOURCE")
            dinov3_weights = _get_setting(self.encoder_cfg, "dinov3_weights", "DINOV3_WEIGHTS")
            dinov3_hf_model = _get_setting(self.encoder_cfg, "dinov3_hf_model", "DINOV3_HF_MODEL")
            image_size = int(self.encoder_cfg.get("crop_size", 512))
            hub_kwargs = {}
            weights_path = Path(str(dinov3_weights)).expanduser() if dinov3_weights else None
            if dinov3_source == "hf" or dinov3_hf_model:
                enc = HuggingFaceDinoV3Backbone(dinov3_hf_model or _dinov3_hf_id(dinov3_model), image_size=image_size).eval()
                num_patches, embed_dim = enc.num_patches, enc.embed_dim
            elif weights_path and (weights_path.is_dir() or weights_path.suffix == ".safetensors"):
                hf_path = weights_path if weights_path.is_dir() else weights_path.parent
                enc = HuggingFaceDinoV3Backbone(str(hf_path), image_size=image_size).eval()
                num_patches, embed_dim = enc.num_patches, enc.embed_dim
            else:
                weights_path = Path(str(dinov3_weights)).expanduser()
                if dinov3_weights and not weights_path.exists():
                    raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
                if dinov3_weights:
                    hub_kwargs["weights"] = str(weights_path)
                enc = _hub_load(dinov3_repo, dinov3_model, cfg=self.encoder_cfg, source=dinov3_source, **hub_kwargs).eval()
                num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dino":
            enc = _hub_load("facebookresearch/dino:main", "dino_vitb16", cfg=self.encoder_cfg).eval(); num_patches, embed_dim = 1024, enc.embed_dim
        elif model == "siglip":
            enc = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-512").eval(); processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512"); num_patches, embed_dim = 1024, 768
        elif model == "clip":
            enc = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").eval(); processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16"); num_patches, embed_dim = 196, 768
        elif model == "dinosiglip":
            from src.vision_backbone.scripts.vit_inference import init_vit_backbone, Config      
            
            config = Config()
            enc = init_vit_backbone(config)

            projector = LinearProjector(2176, 2176).cuda()
            num_patches, embed_dim = 729, 2176
        else:
            raise ValueError(f"Unknown model: {model}")
        if model != 'dinosiglip':
            for p in enc.parameters(): 
                p.requires_grad = False
        return enc, num_patches, embed_dim, processor, projector

    def _extract(self, imgs: torch.Tensor, paths: List[str], n_layer: int = 3):
        if self.model_name == "dinov2":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer, return_class_token=False)[0] # the thrid last block
        elif self.model_name == "dinov3":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer, return_class_token=False)[0] 
        elif self.model_name == "dino":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer)[0][:,1:,:]
        elif self.model_name == "siglip":
            pil_list = [Image.open(p).convert("RGB") for p in paths]
            proc = self.processor(images=pil_list, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(imgs.device)

            with torch.no_grad():
                out = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
                hs = out.hidden_states  # tuple: [embeddings, block1, ..., blockL]; len = L+1

            L = len(hs) - 1  # number of transformer blocks
            n = max(1, min(n_layer, L))
            h = hs[-n][:, :, :]   # [B, 1024, 768] for 512/16 patches
            # print(h.shape)
        elif self.model_name == "clip":
            hs = self.encoder(pixel_values=imgs, output_hidden_states=True).hidden_states
            L = len(hs) - 1  # number of transformer blocks
            n = max(1, min(n_layer, L))
            h = hs[-n][:, 1:, :]   # [B, 1024, 768] for 512/16 patches
            # print(h.shape)
        elif self.model_name == "dinosiglip":
            feats = [self.encoder.generate(Image.open(p).convert("RGB"))[0] for p in paths]
            h = torch.cat(feats).view(imgs.size(0), 2176, -1).permute(0,2,1)
            h = self.projector(h) if self.projector else h
        else:
            raise NotImplementedError(self.model_name)

        if self.feat_normed:
            h = F.normalize(h, dim=-1)

        return h

    @staticmethod
    def _init_predictor(module):
        for m in module.modules():
            if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=0.02); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0)
