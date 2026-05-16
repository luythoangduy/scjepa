import numpy as np
import torch
from skimage.measure import label, regionprops
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
    binary_precision_recall_curve,
    binary_roc,
)


def _as_flat_tensor(values, dtype):
    if isinstance(values, list):
        values = np.stack(values, axis=0)
    if isinstance(values, torch.Tensor):
        return values.detach().flatten().cpu().to(dtype)
    return torch.as_tensor(np.asarray(values), dtype=dtype).flatten().cpu()


def _to_numpy(values):
    return values.detach().cpu().numpy()


def _compute_binary_retrieval_metrics(scores, labels):
    scores = _as_flat_tensor(scores, torch.float32)
    labels = _as_flat_tensor(labels, torch.int32)

    fpr, tpr, roc_thresholds = binary_roc(scores, labels)
    auroc = binary_auroc(scores, labels)

    precision, recall, pr_thresholds = binary_precision_recall_curve(scores, labels)
    aupr = binary_average_precision(scores, labels)

    f1_scores = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-8)
    best_idx = torch.argmax(f1_scores)
    if best_idx < len(pr_thresholds):
        f1_max_threshold = pr_thresholds[best_idx]
    else:
        f1_max_threshold = torch.tensor(1.0, dtype=scores.dtype)

    return {
        "auroc": float(auroc.item()),
        "fpr": _to_numpy(fpr),
        "tpr": _to_numpy(tpr),
        "roc_thresholds": _to_numpy(roc_thresholds),
        "aupr": float(aupr.item()),
        "precision": _to_numpy(precision),
        "recall": _to_numpy(recall),
        "pr_thresholds": _to_numpy(pr_thresholds),
        "f1_max": float(f1_scores[best_idx].item()),
        "f1_max_threshold": float(f1_max_threshold.item())
    }


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights,
    anomaly_ground_truth_labels
):
    return _compute_binary_retrieval_metrics(
        anomaly_prediction_weights,
        anomaly_ground_truth_labels,
    )


def compute_pixelwise_retrieval_metrics(
    anomaly_segmentations,
    ground_truth_masks
):
    return _compute_binary_retrieval_metrics(
        anomaly_segmentations,
        ground_truth_masks,
    )


def calculate_pro(masks, scores, max_steps=200, expect_fpr=0.3):
    if isinstance(masks, list):
        masks = np.stack(masks, axis=0)
    if isinstance(scores, list):
        scores = np.stack(scores, axis=0)

    masks_np = np.asarray(masks).astype(np.uint8)
    scores_t = torch.as_tensor(np.asarray(scores), dtype=torch.float32).cpu()
    masks_t = torch.as_tensor(masks_np, dtype=torch.bool).cpu()
    thresholds = torch.linspace(scores_t.min(), scores_t.max(), max_steps)
    pros = []
    fprs = []

    for threshold in thresholds:
        binary_scores_t = scores_t > threshold
        binary_scores_np = binary_scores_t.numpy().astype(np.uint8)

        # Calculate Pro
        pro_values = []
        for binary_score, mask in zip(binary_scores_np, masks_np):
            regions = regionprops(label(mask))
            for region in regions:
                tp_pixels = binary_score[region.coords[:, 0], region.coords[:, 1]].sum()
                pro_values.append(tp_pixels / region.area)
        pros.append(float(np.mean(pro_values)) if pro_values else 0.0)

        # Calculate FPR
        inverse_masks = ~masks_t
        fp_pixels = torch.logical_and(inverse_masks, binary_scores_t).sum()
        fpr = fp_pixels.float() / torch.clamp(inverse_masks.sum().float(), min=1.0)
        fprs.append(fpr)

    pros = torch.as_tensor(pros, dtype=torch.float32)
    fprs = torch.stack(fprs).float()

    # Filter FPRs below the expected threshold
    valid_idxs = fprs <= expect_fpr
    fprs = fprs[valid_idxs]
    pros = pros[valid_idxs]

    # Normalize FPRs for AUC computation
    if len(fprs) > 1:
        fpr_range = fprs.max() - fprs.min()
        if fpr_range > 0:
            fprs = (fprs - fprs.min()) / fpr_range
            order = torch.argsort(fprs)
            pro_auc = torch.trapezoid(pros[order], fprs[order]).item()
        else:
            pro_auc = 0.0
    else:
        pro_auc = 0.0

    return pro_auc
