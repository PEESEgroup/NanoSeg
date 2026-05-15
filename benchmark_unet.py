"""
One-Shot U-Net Segmentation Benchmark

This script trains a U-Net segmentation model in a one-shot setting,
where each run uses a single annotated image for training and all
remaining images for evaluation.

Main features
-------------
- One-shot supervised training
- Leave-one-out evaluation
- Detailed segmentation metrics
- Boundary-aware evaluation
- Automatic checkpoint selection
- Cross-run log summarization

The framework serves as a one-shot supervised baseline for
NanoSeg segmentation benchmarking.
"""

import os
import random
import shutil
import warnings
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial.distance import cdist
from segmentation_models_pytorch import create_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ----------------------------------------
# Config
# ----------------------------------------
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "path/to/dataset/images"
MASK_DIR = "path/to/dataset/masks"

# one-shot result root
SAVE_ROOT = "path/to/output/unet_one_shot"

# keep final copied global-best checkpoint
BEST_MODEL_COPY_PATH = "path/to/checkpoints/unet_best.pt"

# requested extra outputs
METRICS_CSV_PATH = "path/to/output/metrics/unet.csv"
LOG_SUMMARY_CSV_PATH = "path/to/output/metrics/unet_log_summary.csv"
LOG_SUMMARY_PNG_PATH = "path/to/output/metrics/unet_log_summary.png"

EPOCHS = 35
BATCH_SIZE = 256
LR = 1e-3
TARGET_SIZE = 512
NUM_CLASSES = 2
NUM_WORKERS = 4

# Boundary-aware metric configuration
BF1_TAU = 2.0


# ----------------------------------------
# Reproducibility
# ----------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_image_mask_pairs(image_paths, mask_paths):
    print("[INFO] Checking image-mask size consistency...")
    mismatch_count = 0

    for img_path, mask_path in zip(image_paths, mask_paths):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"[WARN] Cannot read image: {img_path}")
            mismatch_count += 1
            continue
        if mask is None:
            print(f"[WARN] Cannot read mask: {mask_path}")
            mismatch_count += 1
            continue

        hi, wi = image.shape[:2]
        hm, wm = mask.shape[:2]

        if hi != hm or wi != wm:
            print(
                f"[WARN] Size mismatch: {os.path.basename(img_path)} | "
                f"image=({hi},{wi}) mask=({hm},{wm})"
            )
            mismatch_count += 1

    print(f"[INFO] Check done. Mismatch pairs: {mismatch_count}")


# ----------------------------------------
# Dataset
# ----------------------------------------
class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure image-mask size consistency
        h_img, w_img = image.shape[:2]
        h_msk, w_msk = mask.shape[:2]
        if (h_img != h_msk) or (w_img != w_msk):
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.uint8)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"].long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask, os.path.basename(img_path)


# ----------------------------------------
# Transform
# ----------------------------------------
train_transform = A.Compose([
    A.Resize(TARGET_SIZE, TARGET_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(TARGET_SIZE, TARGET_SIZE),
    A.Normalize(),
    ToTensorV2(),
])


# ----------------------------------------
# Model
# ----------------------------------------
def build_model():
    model = create_model(
        arch="Unet",
        encoder_name="resnet18",
        classes=NUM_CLASSES,
    )
    return model


# ----------------------------------------
# Train / Eval epoch metrics
# ----------------------------------------
def run_one_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_tp = total_tn = total_fp = total_fn = 0

    pbar = tqdm(loader, leave=False)
    for images, masks, _ in pbar:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)

        tp = ((preds == 1) & (masks == 1)).sum().item()
        tn = ((preds == 0) & (masks == 0)).sum().item()
        fp = ((preds == 1) & (masks == 0)).sum().item()
        fn = ((preds == 0) & (masks == 1)).sum().item()

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        pbar.set_description(
            f"{'Train' if is_train else 'Test '} Loss:{loss.item():.4f}"
        )

    n = len(loader.dataset)
    avg_loss = total_loss / max(n, 1)

    eps = 1e-7
    pa = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)

    return {
        "loss": avg_loss,
        "PA": pa,
        "Dice": dice,
        "IoU": iou,
        "Recall": recall,
        "Precision": precision,
    }


# ----------------------------------------
# Detailed evaluation metrics
# ----------------------------------------
def mask_to_boundary(mask_bool):
    if mask_bool.sum() == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    eroded = binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = mask_bool ^ eroded
    return boundary


def boundary_f1_score(pred_bool, gt_bool, tau=2.0):
    pred_b = mask_to_boundary(pred_bool)
    gt_b = mask_to_boundary(gt_bool)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0

    gt_dist = distance_transform_edt(~gt_b)
    pred_dist = distance_transform_edt(~pred_b)

    pred_match = pred_b & (gt_dist <= tau)
    gt_match = gt_b & (pred_dist <= tau)

    precision = pred_match.sum() / max(pred_b.sum(), 1)
    recall = gt_match.sum() / max(gt_b.sum(), 1)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def surface_distances(pred_bool, gt_bool):
    pred_b = mask_to_boundary(pred_bool)
    gt_b = mask_to_boundary(gt_bool)

    pred_pts = np.column_stack(np.where(pred_b))
    gt_pts = np.column_stack(np.where(gt_b))

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return np.array([0.0]), np.array([0.0])
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return None, None

    dist_mat = cdist(pred_pts, gt_pts, metric="euclidean")
    d_pred_to_gt = dist_mat.min(axis=1)
    d_gt_to_pred = dist_mat.min(axis=0)
    return d_pred_to_gt, d_gt_to_pred


def hd95_and_asd(pred_bool, gt_bool):
    d1, d2 = surface_distances(pred_bool, gt_bool)
    if d1 is None or d2 is None:
        return np.nan, np.nan

    all_d = np.concatenate([d1, d2], axis=0)
    hd95 = np.percentile(all_d, 95) if len(all_d) > 0 else 0.0
    asd = all_d.mean() if len(all_d) > 0 else 0.0
    return float(hd95), float(asd)


def compute_metrics_from_confusion(tp, tn, fp, fn):
    eps = 1e-7

    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    balanced_acc = (recall + specificity) / 2.0

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    mcc = ((tp * tn) - (fp * fn)) / denom

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "BalancedAcc": float(balanced_acc),
        "MCC": float(mcc),
    }


@torch.no_grad()
def evaluate_model_detailed(model, loader):
    model.eval()

    total_tp = total_tn = total_fp = total_fn = 0
    bf1_list = []
    hd95_list = []
    asd_list = []

    for images, masks, _ in tqdm(loader, leave=False, desc="DetailedEval"):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        preds_np = preds.detach().cpu().numpy().astype(np.uint8)
        masks_np = masks.detach().cpu().numpy().astype(np.uint8)

        for pred_u8, gt_u8 in zip(preds_np, masks_np):
            pred_bool = pred_u8 > 0
            gt_bool = gt_u8 > 0

            tp = np.logical_and(pred_bool, gt_bool).sum()
            tn = np.logical_and(~pred_bool, ~gt_bool).sum()
            fp = np.logical_and(pred_bool, ~gt_bool).sum()
            fn = np.logical_and(~pred_bool, gt_bool).sum()

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

            bf1 = boundary_f1_score(pred_bool, gt_bool, tau=BF1_TAU)
            hd95, asd = hd95_and_asd(pred_bool, gt_bool)

            bf1_list.append(float(bf1))
            if not np.isnan(hd95):
                hd95_list.append(float(hd95))
            if not np.isnan(asd):
                asd_list.append(float(asd))

    out = compute_metrics_from_confusion(total_tp, total_tn, total_fp, total_fn)
    out["BF1_tau2"] = float(np.mean(bf1_list)) if len(bf1_list) > 0 else np.nan
    out["HD95"] = float(np.mean(hd95_list)) if len(hd95_list) > 0 else np.nan
    out["ASD"] = float(np.mean(asd_list)) if len(asd_list) > 0 else np.nan
    return out


# ----------------------------------------
# Visualization
# ----------------------------------------
PLOT_METRICS = ["loss", "PA", "Dice", "IoU", "Recall", "Precision"]


def plot_log(df, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.flatten()

    df_plot = df[df["epoch"] != "best"].copy()
    df_plot["epoch"] = df_plot["epoch"].astype(int)

    for ax, metric in zip(axes, PLOT_METRICS):
        train_col = f"train_{metric}"
        test_col = f"test_{metric}"

        train_y = df_plot[train_col].to_numpy(dtype=float)
        test_y = df_plot[test_col].to_numpy(dtype=float)
        epoch = df_plot["epoch"].to_numpy(dtype=int)

        ax.plot(epoch, train_y, label="train")
        ax.plot(epoch, test_y, label="test")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def summarize_logs_across_runs(log_dfs):
    if len(log_dfs) == 0:
        raise RuntimeError("No log dataframes found for summary.")

    common_epochs = set(log_dfs[0]["epoch"].tolist())
    for df in log_dfs[1:]:
        common_epochs &= set(df["epoch"].tolist())
    common_epochs = sorted(common_epochs)

    if len(common_epochs) == 0:
        raise RuntimeError("No common epochs across one-shot runs.")

    rows = []
    for ep in common_epochs:
        row = {"epoch": ep, "n_runs": len(log_dfs)}
        cur_rows = [df[df["epoch"] == ep].iloc[0] for df in log_dfs]

        for metric in PLOT_METRICS:
            for prefix in ["train", "test"]:
                col = f"{prefix}_{metric}"
                vals = np.array([float(x[col]) for x in cur_rows], dtype=np.float64)
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def plot_summary_log(summary_df, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.flatten()
    epoch = summary_df["epoch"].to_numpy()

    for ax, metric in zip(axes, PLOT_METRICS):
        train_mean = summary_df[f"train_{metric}_mean"].to_numpy(dtype=float)
        train_std = summary_df[f"train_{metric}_std"].to_numpy(dtype=float)
        test_mean = summary_df[f"test_{metric}_mean"].to_numpy(dtype=float)
        test_std = summary_df[f"test_{metric}_std"].to_numpy(dtype=float)

        ax.plot(epoch, train_mean, label="train")
        ax.fill_between(epoch, train_mean - train_std, train_mean + train_std, alpha=0.25)

        ax.plot(epoch, test_mean, label="test")
        ax.fill_between(epoch, test_mean - test_std, test_mean + test_std, alpha=0.25)

        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ----------------------------------------
# File helpers
# ----------------------------------------
def numeric_stem_sort_key(path_or_name):
    stem = Path(path_or_name).stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.lower())


def get_all_pairs(image_dir, mask_dir):
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)],
        key=numeric_stem_sort_key,
    )

    image_paths = []
    mask_paths = []

    for f in image_files:
        img_path = os.path.join(image_dir, f)
        msk_path = os.path.join(mask_dir, f)

        if not os.path.exists(msk_path):
            print(f"[WARN] Missing GT for {f}, skip.")
            continue

        image_paths.append(img_path)
        mask_paths.append(msk_path)

    if len(image_paths) == 0:
        raise RuntimeError("No valid image-mask pairs found.")

    return image_paths, mask_paths


# ----------------------------------------
# Main training
# ----------------------------------------
def main():
    seed_everything(SEED)

    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(BEST_MODEL_COPY_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_CSV_PATH), exist_ok=True)

    image_paths, mask_paths = get_all_pairs(IMAGE_DIR, MASK_DIR)
    check_image_mask_pairs(image_paths, mask_paths)
    n_samples = len(image_paths)

    print(f"[INFO] DEVICE: {DEVICE}")
    print(f"[INFO] Total samples: {n_samples}")
    print(f"[INFO] One-shot runs: {n_samples}")

    summary_rows = []
    metrics_rows = []
    all_log_dfs = []

    global_best_dice = -1.0
    global_best_model_path = None
    global_best_image = None

    for shot_idx in range(n_samples):
        train_image_path = image_paths[shot_idx]
        train_mask_path = mask_paths[shot_idx]
        image_name = os.path.basename(train_image_path)
        image_stem = Path(image_name).stem

        print(f"\n{'=' * 90}")
        print(f"[INFO] One-shot run {shot_idx + 1}/{n_samples} | train image: {image_name}")
        print(f"{'=' * 90}")

        run_save_dir = os.path.join(SAVE_ROOT, image_stem)
        os.makedirs(run_save_dir, exist_ok=True)

        log_csv_path = os.path.join(run_save_dir, "log.csv")
        log_png_path = os.path.join(run_save_dir, "log.png")
        model_path = os.path.join(run_save_dir, "model.pt")

        train_images = [train_image_path]
        train_masks = [train_mask_path]
        test_images = [image_paths[i] for i in range(n_samples) if i != shot_idx]
        test_masks = [mask_paths[i] for i in range(n_samples) if i != shot_idx]

        print(f"[INFO] Train size: {len(train_images)} | Test size: {len(test_images)}")

        train_dataset = SegDataset(train_images, train_masks, transform=train_transform)
        test_dataset = SegDataset(test_images, test_masks, transform=test_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(BATCH_SIZE, len(train_dataset)),
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        model = build_model().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        logs = []
        best_record = None
        best_dice = -1.0
        best_epoch = -1

        for epoch in range(1, EPOCHS + 1):
            print(f"\n[INFO] Train image {image_name} | Epoch {epoch}/{EPOCHS}")

            train_metrics = run_one_epoch(model, train_loader, criterion, optimizer=optimizer)
            test_metrics = run_one_epoch(model, test_loader, criterion, optimizer=None)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_PA": train_metrics["PA"],
                "train_Dice": train_metrics["Dice"],
                "train_IoU": train_metrics["IoU"],
                "train_Recall": train_metrics["Recall"],
                "train_Precision": train_metrics["Precision"],
                "test_loss": test_metrics["loss"],
                "test_PA": test_metrics["PA"],
                "test_Dice": test_metrics["Dice"],
                "test_IoU": test_metrics["IoU"],
                "test_Recall": test_metrics["Recall"],
                "test_Precision": test_metrics["Precision"],
            }
            logs.append(row)

            print(
                f"[EPOCH {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.6f}, "
                f"train_dice={train_metrics['Dice']:.6f}, "
                f"test_loss={test_metrics['loss']:.6f}, "
                f"test_dice={test_metrics['Dice']:.6f}"
            )

            if test_metrics["Dice"] > best_dice:
                best_dice = test_metrics["Dice"]
                best_epoch = epoch

                best_record = {
                    "epoch": "best",
                    "train_loss": train_metrics["loss"],
                    "train_PA": train_metrics["PA"],
                    "train_Dice": train_metrics["Dice"],
                    "train_IoU": train_metrics["IoU"],
                    "train_Recall": train_metrics["Recall"],
                    "train_Precision": train_metrics["Precision"],
                    "test_loss": test_metrics["loss"],
                    "test_PA": test_metrics["PA"],
                    "test_Dice": test_metrics["Dice"],
                    "test_IoU": test_metrics["IoU"],
                    "test_Recall": test_metrics["Recall"],
                    "test_Precision": test_metrics["Precision"],
                    "best_epoch": best_epoch,
                }

                torch.save(model.state_dict(), model_path)
                print(f"[INFO] Best model updated at epoch {epoch}, Dice={best_dice:.6f}")

        # Save training log CSV
        df_log = pd.DataFrame(logs)

        best_row_for_csv = {
            "epoch": "best",
            "train_loss": best_record["train_loss"],
            "train_PA": best_record["train_PA"],
            "train_Dice": best_record["train_Dice"],
            "train_IoU": best_record["train_IoU"],
            "train_Recall": best_record["train_Recall"],
            "train_Precision": best_record["train_Precision"],
            "test_loss": best_record["test_loss"],
            "test_PA": best_record["test_PA"],
            "test_Dice": best_record["test_Dice"],
            "test_IoU": best_record["test_IoU"],
            "test_Recall": best_record["test_Recall"],
            "test_Precision": best_record["test_Precision"],
            "best_epoch": best_record["best_epoch"],
        }

        if "best_epoch" not in df_log.columns:
            df_log["best_epoch"] = ""

        df_log_with_best = pd.concat(
            [df_log, pd.DataFrame([best_row_for_csv])],
            ignore_index=True
        )
        df_log_with_best.to_csv(log_csv_path, index=False)

        # Save training-curve visualization
        plot_log(df_log_with_best, log_png_path)

        print(f"[SAVED] {log_csv_path}")
        print(f"[SAVED] {log_png_path}")
        print(f"[SAVED] {model_path}")

        # detailed evaluation using the saved best checkpoint
        best_model = build_model().to(DEVICE)
        best_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        detailed_metrics = evaluate_model_detailed(best_model, test_loader)

        metrics_rows.append({
            "image": image_name,
            "IoU": detailed_metrics["IoU"],
            "Dice": detailed_metrics["Dice"],
            "Precision": detailed_metrics["Precision"],
            "Recall": detailed_metrics["Recall"],
            "Specificity": detailed_metrics["Specificity"],
            "BalancedAcc": detailed_metrics["BalancedAcc"],
            "MCC": detailed_metrics["MCC"],
            "BF1_tau2": detailed_metrics["BF1_tau2"],
            "HD95": detailed_metrics["HD95"],
            "ASD": detailed_metrics["ASD"],
        })

        summary_row = {
            "image": image_name,
            "best_epoch": best_record["best_epoch"],
            "train_loss": best_record["train_loss"],
            "train_PA": best_record["train_PA"],
            "train_Dice": best_record["train_Dice"],
            "train_IoU": best_record["train_IoU"],
            "train_Recall": best_record["train_Recall"],
            "train_Precision": best_record["train_Precision"],
            "test_loss": best_record["test_loss"],
            "test_PA": best_record["test_PA"],
            "test_Dice": best_record["test_Dice"],
            "test_IoU": best_record["test_IoU"],
            "test_Recall": best_record["test_Recall"],
            "test_Precision": best_record["test_Precision"],
            "model_path": model_path,
        }
        summary_rows.append(summary_row)

        # Collect epoch logs for summary statistics
        all_log_dfs.append(df_log.copy())

        if best_record["test_Dice"] > global_best_dice:
            global_best_dice = best_record["test_Dice"]
            global_best_model_path = model_path
            global_best_image = image_name

    # ----------------------------------------
    # summary.csv under SAVE_ROOT
    # ----------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    numeric_cols = [
        "best_epoch",
        "train_loss", "train_PA", "train_Dice", "train_IoU", "train_Recall", "train_Precision",
        "test_loss", "test_PA", "test_Dice", "test_IoU", "test_Recall", "test_Precision",
    ]

    mean_row = {"image": "mean", "model_path": ""}
    std_row = {"image": "std", "model_path": ""}

    for col in numeric_cols:
        mean_row[col] = summary_df[col].mean()
        std_row[col] = summary_df[col].std(ddof=0)

    summary_df = pd.concat(
        [summary_df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )

    summary_csv_path = os.path.join(SAVE_ROOT, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[SAVED] {summary_csv_path}")

    # ----------------------------------------
    # Export final metrics CSV
    # path/to/output/metrics/unet.csv
    # ----------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[
        ["image", "IoU", "Dice", "Precision", "Recall", "Specificity",
         "BalancedAcc", "MCC", "BF1_tau2", "HD95", "ASD"]
    ]

    mean_row = {"image": "mean"}
    std_row = {"image": "std"}
    for col in metrics_df.columns[1:]:
        mean_row[col] = metrics_df[col].mean()
        std_row[col] = metrics_df[col].std(ddof=0)

    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"[SAVED] {METRICS_CSV_PATH}")

    # ----------------------------------------
    # Export summarized training curves
    # ----------------------------------------
    log_summary_df = summarize_logs_across_runs(all_log_dfs)
    log_summary_df.to_csv(LOG_SUMMARY_CSV_PATH, index=False)
    plot_summary_log(log_summary_df, LOG_SUMMARY_PNG_PATH)
    print(f"[SAVED] {LOG_SUMMARY_CSV_PATH}")
    print(f"[SAVED] {LOG_SUMMARY_PNG_PATH}")

    # ----------------------------------------
    # Export best checkpoint
    # ----------------------------------------
    if global_best_model_path is not None:
        shutil.copy2(global_best_model_path, BEST_MODEL_COPY_PATH)
        print(f"[SAVED] Best model copied to: {BEST_MODEL_COPY_PATH}")
        print(f"[INFO] Global best Dice: {global_best_dice:.6f}")
        print(f"[INFO] Global best training image: {global_best_image}")
    else:
        print("[WARN] No best model found.")

    print("\nDone.")


if __name__ == "__main__":
    main()