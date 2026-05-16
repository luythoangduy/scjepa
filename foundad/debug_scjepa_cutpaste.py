from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib import cm
from PIL import Image
from torchvision import transforms

from src.AD_support import _build_model, _support_for_class
from src.datasets.dataset import IMAGENET_MEAN, IMAGENET_STD, build_base_transform
from src.train_support import SupportBank
from src.utils.synthesis import CutPasteUnion


def _default_cfg() -> Dict[str, Any]:
    return {
        "meta": {
            "model": "dinov3",
            "crop_size": 512,
            "pred_depth": 4,
            "pred_emb_dim": 384,
            "if_pred_pe": False,
            "feat_normed": False,
            "n_layer": 3,
            "support_shots": 1,
            "top_m": 8,
            "n_heads": 8,
            "support_aggregation": "mean",
            "matcher_mode": "window",
            "window_size": 4,
            "window_shift": 0,
            "use_rel_pos": True,
            "rel_pos_weight": 0.1,
            "dropout": 0.0,
        },
        "data": {
            "train_root": "",
        },
    }


def _merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _load_cfg(path: str | None) -> Dict[str, Any]:
    cfg = _default_cfg()
    if path:
        with open(path, "r") as f:
            cfg = _merge_dict(cfg, yaml.safe_load(f) or {})
    return cfg


def _first_image(train_root: Path, classname: str | None):
    root = train_root / "train"
    if classname is None:
        class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not class_dirs:
            raise FileNotFoundError(f"No class folders under {root}")
        classname = class_dirs[0].name
    class_dir = root / classname
    paths = sorted([p for p in class_dir.iterdir() if p.is_file()])
    if not paths:
        raise FileNotFoundError(f"No images under {class_dir}")
    return classname, paths[0]


def _to_image(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(3, 1, 1)
    x = (t.detach() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _norm_map(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu().numpy()
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def _upsample_tokens(tokens: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    side = int(round(tokens.shape[-1] ** 0.5))
    if side * side != tokens.shape[-1]:
        raise ValueError(f"Token count is not square: {tokens.shape[-1]}")
    return F.interpolate(tokens.view(1, 1, side, side), size=size, mode="bilinear", align_corners=False)[0, 0]


def _heatmap(x: np.ndarray) -> np.ndarray:
    return (cm.jet(x)[..., :3] * 255).astype(np.uint8)


def _overlay(img: np.ndarray, heat: np.ndarray) -> np.ndarray:
    heat_rgb = _heatmap(heat)
    alpha = np.clip((heat - 0.1) / 0.9, 0, 1) * 0.65
    return (img.astype(np.float32) * (1 - alpha[..., None]) + heat_rgb.astype(np.float32) * alpha[..., None]).astype(np.uint8)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Debug SC-JEPA CutPaste matcher/predictor maps")
    parser.add_argument("--params", default="", help="Saved params.yaml from a SC-JEPA run")
    parser.add_argument("--ckpt", default="", help="Optional checkpoint to load")
    parser.add_argument("--train-root", default="", help="Few-shot root containing train/<class>")
    parser.add_argument("--classname", default="", help="Class to visualize")
    parser.add_argument("--image", default="", help="Optional clean image path")
    parser.add_argument("--dinov3-weights", default="", help="Override DINOv3 HF folder or .pth")
    parser.add_argument("--support-shots", type=int, default=0)
    parser.add_argument("--output", default="logs/debug_scjepa_cutpaste")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = _load_cfg(args.params or None)
    meta = cfg["meta"]
    if args.dinov3_weights:
        meta["dinov3_weights"] = args.dinov3_weights
    if args.support_shots > 0:
        meta["support_shots"] = args.support_shots

    train_root_value = args.train_root or cfg["data"].get("train_root", "")
    if not train_root_value:
        raise ValueError("Provide --train-root or --params with data.train_root")
    train_root = Path(train_root_value)

    classname = args.classname or None
    image_path = Path(args.image) if args.image else None
    if image_path is None:
        classname, image_path = _first_image(train_root, classname)
    elif classname is None:
        classname = image_path.parent.name

    device = torch.device(args.device)
    model = _build_model(meta).to(device).eval()
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.predictor.load_state_dict(state["predictor"])
        if model.projector is not None and state.get("projector") is not None:
            model.projector.load_state_dict(state["projector"])

    crop = int(meta.get("crop_size", 512))
    n_layer = int(meta.get("n_layer", 3))
    support_shots = int(meta.get("support_shots", 1))
    transform = transforms.Compose(build_base_transform(crop))
    cutpaste = CutPasteUnion(colorJitter=0.5)

    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    _, img_abn = cutpaste(img, [classname])

    support_bank = SupportBank(str(train_root), crop, support_shots)
    support = _support_for_class(model, support_bank, classname, device, n_layer)

    clean_feat = model.target_features(img, [str(image_path)], n_layer=n_layer)
    query_feat = model.target_features(img_abn, [str(image_path)], n_layer=n_layer)
    out = model.predict(query_feat, support)

    h, w = img.shape[-2:]
    clean_np = _to_image(img[0])
    query_np = _to_image(img_abn[0])
    diff = (img_abn - img).abs().mean(dim=1)
    diff_map = _norm_map(diff[0])

    sim_map = _norm_map(_upsample_tokens(out["top_sim"].max(dim=-1).values[0], (h, w)))
    weight_map = _norm_map(_upsample_tokens(out["top_weights"].max(dim=-1).values[0], (h, w)))
    gate_map = _norm_map(_upsample_tokens(out["gate"][0], (h, w)))
    residual_map = _norm_map(_upsample_tokens(out["residual"][0], (h, w)))
    score_map = _norm_map(_upsample_tokens(out["score"][0], (h, w)))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{classname}_{image_path.stem}_cutpaste_debug.png"
    out_path = out_dir / safe_name

    panels = [
        ("Clean", clean_np),
        ("CutPaste query", query_np),
        ("CutPaste diff", _heatmap(diff_map)),
        ("Max similarity", _heatmap(sim_map)),
        ("Max match weight", _heatmap(weight_map)),
        ("Gate", _heatmap(gate_map)),
        ("Residual", _heatmap(residual_map)),
        ("Score overlay", _overlay(query_np, score_map)),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (title, panel) in zip(axs.ravel(), panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Saved debug visualization: {out_path}")
    print(f"class={classname}")
    print(f"image={image_path}")
    print(f"matcher_mode={meta.get('matcher_mode', 'window')} window_size={meta.get('window_size', 4)}")


if __name__ == "__main__":
    main()
