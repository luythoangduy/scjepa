from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib import cm

from src.AD_support import _build_model, _support_for_class
from src.datasets.dataset import IMAGENET_MEAN, IMAGENET_STD, TestDataset
from src.train_support import SupportBank


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
            "dataset": "mvtec",
            "train_root": "",
            "test_root": "",
            "mvtec_classnames": [
                "bottle",
                "cable",
                "capsule",
                "carpet",
                "grid",
                "hazelnut",
                "leather",
                "metal_nut",
                "pill",
                "screw",
                "tile",
                "toothbrush",
                "transistor",
                "wood",
                "zipper",
            ],
            "visa_classnames": [
                "candle",
                "capsules",
                "cashew",
                "chewinggum",
                "fryum",
                "macaroni1",
                "macaroni2",
                "pcb1",
                "pcb2",
                "pcb3",
                "pcb4",
                "pipe_fryum",
            ],
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


def _parse_classes(value: str, cfg: Dict[str, Any]) -> List[str]:
    if value:
        return [x.strip() for x in value.split(",") if x.strip()]
    dataset_name = cfg["data"].get("dataset", "mvtec")
    key = "mvtec_classnames" if dataset_name == "mvtec" else "visa_classnames"
    return list(cfg["data"][key])


def _select_test_index(dataset: TestDataset, prefer: str) -> int:
    normal_names = {"good", "ok"}
    if prefer == "first":
        return 0
    for idx, (_, anomaly, _, _) in enumerate(dataset.data_to_iterate):
        is_anomaly = anomaly not in normal_names
        if prefer == "anomaly" and is_anomaly:
            return idx
        if prefer == "normal" and not is_anomaly:
            return idx
    return 0


def _save_panels(path: Path, panels: Iterable[tuple[str, np.ndarray]]) -> None:
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (title, panel) in zip(axs.ravel(), panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Debug SC-JEPA test matcher/predictor maps")
    parser.add_argument("--params", default="", help="Saved params.yaml from a SC-JEPA run")
    parser.add_argument("--ckpt", default="", help="Optional checkpoint to load")
    parser.add_argument("--train-root", default="", help="Few-shot root containing train/<class>")
    parser.add_argument("--test-root", default="", help="Dataset root containing <class>/test")
    parser.add_argument("--classname", default="", help="Class or comma-separated classes. Empty means all classes.")
    parser.add_argument("--prefer", choices=["anomaly", "normal", "first"], default="anomaly")
    parser.add_argument("--dinov3-weights", default="", help="Override DINOv3 HF folder or .pth")
    parser.add_argument("--support-shots", type=int, default=0)
    parser.add_argument("--output", default="logs/debug_scjepa_test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = _load_cfg(args.params or None)
    meta = cfg["meta"]
    data = cfg["data"]
    if args.dinov3_weights:
        meta["dinov3_weights"] = args.dinov3_weights
    if args.support_shots > 0:
        meta["support_shots"] = args.support_shots

    train_root_value = args.train_root or data.get("train_root", "")
    test_root_value = args.test_root or data.get("test_root", "")
    if not train_root_value:
        raise ValueError("Provide --train-root or --params with data.train_root")
    if not test_root_value:
        raise ValueError("Provide --test-root or --params with data.test_root")

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
    dataset_name = data.get("dataset", "mvtec")
    support_bank = SupportBank(str(train_root_value), crop, support_shots)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for classname in _parse_classes(args.classname, cfg):
        dataset = TestDataset(
            source=str(test_root_value),
            classname=classname,
            resize=crop,
            datasetname=dataset_name,
        )
        sample = dataset[_select_test_index(dataset, args.prefer)]
        img = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].squeeze(0).float().cpu().numpy()
        paths = [sample["image_path"]]

        support = _support_for_class(model, support_bank, classname, device, n_layer)
        enc = model.target_features(img, paths, n_layer=n_layer)
        out = model.predict(enc, support)

        h, w = img.shape[-2:]
        img_np = _to_image(img[0])
        gt_np = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        sim_map = _norm_map(_upsample_tokens(out["top_sim"].max(dim=-1).values[0], (h, w)))
        weight_map = _norm_map(_upsample_tokens(out["top_weights"].max(dim=-1).values[0], (h, w)))
        gate_map = _norm_map(_upsample_tokens(out["gate"][0], (h, w)))
        residual_map = _norm_map(_upsample_tokens(out["residual"][0], (h, w)))
        score_map = _norm_map(_upsample_tokens(out["score"][0], (h, w)))

        anomaly = str(sample["anomaly"])
        stem = Path(sample["image_path"]).stem
        out_path = out_dir / f"{classname}_{anomaly}_{stem}_test_debug.png"
        panels = [
            ("Original", img_np),
            ("GT mask", gt_np),
            ("Score heatmap", _heatmap(score_map)),
            ("Max similarity", _heatmap(sim_map)),
            ("Max match weight", _heatmap(weight_map)),
            ("Gate", _heatmap(gate_map)),
            ("Residual", _heatmap(residual_map)),
            ("Score overlay", _overlay(img_np, score_map)),
        ]
        _save_panels(out_path, panels)
        print(f"Saved {out_path}")
        print(f"  class={classname} anomaly={anomaly} image={sample['image_path']}")


if __name__ == "__main__":
    main()
