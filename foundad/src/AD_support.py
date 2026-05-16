import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.AD import _demo as _foundad_demo
from src.datasets.dataset import build_base_transform, build_dataloader
from src.support_jepa import SupportConditionedVisionModule
from src.train_support import SupportBank
from src.utils.logging import CSVLogger
from src.utils.metrics import calculate_pro, compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics
from src.helper import save_segmentation_grid

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("support_evaluator")


def _build_model(meta: Dict[str, Any]) -> SupportConditionedVisionModule:
    return SupportConditionedVisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
        top_m=meta.get("top_m", 8),
        n_heads=meta.get("n_heads", 8),
        aggregation=meta.get("support_aggregation", "mean"),
        use_rel_pos=meta.get("use_rel_pos", True),
        rel_pos_weight=meta.get("rel_pos_weight", 0.1),
        dropout=meta.get("dropout", 0.0),
        encoder_cfg=meta,
    )


@torch.inference_mode()
def _support_for_class(model, support_bank: SupportBank, classname: str, device, n_layer: int):
    chosen = support_bank.paths[classname][: support_bank.shots]
    if not chosen:
        raise FileNotFoundError(f"No support image found for class {classname}")
    imgs = [support_bank.transform(Image.open(p).convert("RGB")) for p in chosen]
    support_imgs = torch.stack([torch.stack(imgs, dim=0)], dim=0)
    support_paths = [[str(p) for p in chosen]]
    support_imgs = support_imgs.to(device)
    return model.support_features(support_imgs, support_paths, n_layer=n_layer)


@torch.inference_mode()
def _evaluate_single_ckpt(ckpt: Path, cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None and state.get("projector") is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    support_shots = int(cfg["meta"].get("support_shots", cfg["data"].get("support_shots", 4)))
    support_bank = SupportBank(cfg["data"]["train_root"], crop, support_shots)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    if dataset_name == "mvtec":
        classnames = cfg["data"]["mvtec_classnames"]
        K = cfg["testing"]["K_top_mvtec"]
    elif dataset_name == "visa":
        classnames = cfg["data"]["visa_classnames"]
        K = cfg["testing"]["K_top_visa"]
    else:
        raise NotImplementedError(dataset_name)

    os.makedirs(Path(cfg["logging"]["folder"]), exist_ok=True)
    csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
        ("%.8f", "pix_auroc"), ("%.8f", "pro_auc"),
    )

    inst_auc, inst_aupr, pix_auc, pro_auc = [], [], [], []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )
        support = _support_for_class(model, support_bank, cls, device, n_layer)
        patch_scores, labels = [], []
        pix_buf, img_buf, mask_buf, name_buf = [], [], [], []

        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]
            labels.extend(batch["is_anomaly"])
            name_buf.extend(batch["image_name"])

            enc = model.target_features(img, paths, n_layer=n_layer)
            out = model.predict(enc, support.expand(enc.shape[0], -1, -1, -1))
            l = out["score"]

            topk = torch.topk(l, min(K, l.shape[1]), dim=1).values.mean(dim=1)
            patch_scores.extend(topk.cpu())
            h = w = int(math.sqrt(l.size(1)))
            pix = F.interpolate(l.view(-1, 1, h, w), size=img.shape[2:], mode="bilinear", align_corners=False)
            pix_buf.append(pix.squeeze(1).cpu())
            img_buf.append(img.cpu())
            mask_buf.append(mask.cpu())

        p_np = torch.tensor(patch_scores).numpy()
        p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8)
        pix_all = torch.cat(pix_buf)
        gmin, gmax = pix_all.min(), pix_all.max()
        pix_norm = ((pix_all - gmin) / (gmax - gmin + 1e-8)).numpy()
        mask_np = torch.cat(mask_buf).squeeze(1).numpy()

        inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))
        pix = compute_pixelwise_retrieval_metrics(pix_norm, mask_np)
        pro = calculate_pro(mask_np, pix_norm, max_steps=cfg["testing"]["max_steps"], expect_fpr=cfg["testing"]["expect_fpr"])

        logger.info("%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f", cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)
        csv_logger.log(ckpt.name, cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)
        inst_auc.append(inst["auroc"])
        inst_aupr.append(inst["aupr"])
        pix_auc.append(pix["auroc"])
        pro_auc.append(pro)

        if cfg["testing"].get("segmentation_vis", False):
            imgs_un = (torch.cat(img_buf) * std.cpu() + mean.cpu()).permute(0, 2, 3, 1).numpy()
            out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)

    logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f", np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))
    csv_logger.log(ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))


def main(args: Dict[str, Any]) -> None:
    ckpt = Path(args["ckpt_path"])
    logger.info("loading %s...", ckpt)
    _evaluate_single_ckpt(ckpt, args)
