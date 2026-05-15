"""
Segmentation Metrics Module

This module computes pixel-level and boundary-level segmentation metrics
for binary nanoparticle masks.

Implemented metrics
-------------------
- IoU
- Dice
- Precision
- Recall
- Specificity
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- Boundary F1 Score
- Hausdorff Distance (HD95)
- Average Surface Distance (ASD)

The implementation combines torchmetrics and surface-distance utilities
for reproducible segmentation evaluation.
"""

import numpy as np
import surface_distance
import torch
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryAccuracy,
    BinaryMatthewsCorrCoef
)


def boundary_f1_score(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 2):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    true_boundary = find_boundaries(y_true, mode="thick")
    pred_boundary = find_boundaries(y_pred, mode="thick")

    dt_true = distance_transform_edt(~true_boundary)
    dt_pred = distance_transform_edt(~pred_boundary)

    pred_match = pred_boundary & (dt_true <= tolerance)
    true_match = true_boundary & (dt_pred <= tolerance)

    precision = pred_match.sum() / (pred_boundary.sum() + 1e-8)
    recall = true_match.sum() / (true_boundary.sum() + 1e-8)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(pred_mask: np.ndarray,
                    gt_mask: np.ndarray,
                    device: torch.device):

    # Convert masks to binary format
    y_true_bin = (gt_mask > 127).astype(np.int64)
    y_pred_bin = (pred_mask > 127).astype(np.int64)

    # Convert to tensor format [N, 1, H, W]
    y_true_t = torch.from_numpy(y_true_bin).unsqueeze(0).unsqueeze(0).to(device)
    y_pred_t = torch.from_numpy(y_pred_bin).unsqueeze(0).unsqueeze(0).to(device)

    # TorchMetrics evaluation
    iou_metric = BinaryJaccardIndex().to(device)
    dice_metric = BinaryF1Score().to(device)
    prec_metric = BinaryPrecision().to(device)
    rec_metric = BinaryRecall().to(device)
    spec_metric = BinarySpecificity().to(device)
    balacc_metric = BinaryAccuracy().to(device)
    mcc_metric = BinaryMatthewsCorrCoef().to(device)

    iou = iou_metric(y_pred_t, y_true_t).item()
    dice = dice_metric(y_pred_t, y_true_t).item()
    prec = prec_metric(y_pred_t, y_true_t).item()
    rec = rec_metric(y_pred_t, y_true_t).item()
    spec = spec_metric(y_pred_t, y_true_t).item()
    balacc = balacc_metric(y_pred_t, y_true_t).item()
    mcc = mcc_metric(y_pred_t, y_true_t).item()

    # Surface-distance metrics
    surface_distances = surface_distance.compute_surface_distances(
        y_true_bin.astype(bool), y_pred_bin.astype(bool), spacing_mm=(1.0, 1.0)
    )
    bf1 = boundary_f1_score(y_true_bin, y_pred_bin, tolerance=2)
    hd95 = surface_distance.compute_robust_hausdorff(surface_distances, percent=95)
    asd = np.mean(surface_distance.compute_average_surface_distance(surface_distances))

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "BalancedAcc": balacc,
        "MCC": mcc,
        "BF1_tau2": bf1,
        "HD95": hd95,
        "ASD": asd
    }
