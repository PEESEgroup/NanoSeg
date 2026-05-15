"""
HAM Supervised Segmentation Training Pipeline

This script trains a fully supervised semantic segmentation model
for nanoparticle segmentation using U-Net with a ResNet18 encoder.

Main features
-------------
- K-fold cross-validation training
- Binary segmentation
- Albumentations preprocessing
- Automatic metric logging
- Best-model checkpoint saving
- Training-curve visualization

The framework serves as a supervised baseline benchmark for
NanoSeg segmentation experiments.
"""

import gc
import os
import random
import shutil
import warnings

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
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

SAVE_ROOT = "path/to/output/training_results"
BEST_MODEL_COPY_PATH = "path/to/checkpoints/ham.pt"

NUM_FOLDS = 10
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
TARGET_SIZE = 512
NUM_CLASSES = 2
NUM_WORKERS = 4


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
            print(f"[WARN] Size mismatch: {os.path.basename(img_path)} | image=({hi},{wi}) mask=({hm},{wm})")
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

        # Convert mask to binary format
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
        classes=NUM_CLASSES
    )
    return model


# ----------------------------------------
# Metrics
# ----------------------------------------
def compute_batch_metrics_from_logits(logits, targets, eps=1e-7):
    """
    logits: [B, C, H, W]
    targets: [B, H, W], values in {0,1}
    """
    preds = torch.argmax(logits, dim=1)

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    pa = (tp + tn) / (tp + tn + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)

    return {
        "PA": pa,
        "Dice": dice,
        "IoU": iou,
        "Recall": recall,
        "Precision": precision,
    }


# ----------------------------------------
# Train / Eval one epoch
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
            f"{'Train' if is_train else 'Test '} "
            f"Loss:{loss.item():.4f}"
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
# Visualization
# ----------------------------------------
def plot_log(df, save_path):
    metrics = ["loss", "PA", "Dice", "IoU", "Recall", "Precision"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.flatten()

    df_plot = df[df["epoch"] != "best"].copy()
    df_plot["epoch"] = df_plot["epoch"].astype(int)

    for ax, metric in zip(axes, metrics):
        ax.plot(df_plot["epoch"], df_plot[f"train_{metric}"], label="train")
        ax.plot(df_plot["epoch"], df_plot[f"test_{metric}"], label="test")
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
def get_all_pairs(image_dir, mask_dir):
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(valid_exts)
    ])

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


def make_folds(n, num_folds=10):
    indices = list(range(n))
    fold_sizes = [n // num_folds] * num_folds
    for i in range(n % num_folds):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        folds.append(indices[start:end])
        start = end
    return folds


# ----------------------------------------
# Main training
# ----------------------------------------
def main():
    seed_everything(SEED)

    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(BEST_MODEL_COPY_PATH), exist_ok=True)

    image_paths, mask_paths = get_all_pairs(IMAGE_DIR, MASK_DIR)
    check_image_mask_pairs(image_paths, mask_paths)
    n_samples = len(image_paths)

    print(f"[INFO] DEVICE: {DEVICE}")
    print(f"[INFO] Total samples: {n_samples}")

    folds = make_folds(n_samples, NUM_FOLDS)

    summary_rows = []
    global_best_dice = -1.0
    global_best_model_path = None

    for fold_idx in range(NUM_FOLDS):
        print(f"\n{'=' * 80}")
        print(f"[INFO] Fold {fold_idx + 1}/{NUM_FOLDS}")
        print(f"{'=' * 80}")

        fold_save_dir = os.path.join(SAVE_ROOT, str(fold_idx + 1))
        os.makedirs(fold_save_dir, exist_ok=True)

        log_csv_path = os.path.join(fold_save_dir, "log.csv")
        log_png_path = os.path.join(fold_save_dir, "log.png")
        model_path = os.path.join(fold_save_dir, "model.pt")

        test_idx = folds[fold_idx]
        train_idx = [i for i in range(n_samples) if i not in test_idx]

        train_images = [image_paths[i] for i in train_idx]
        train_masks = [mask_paths[i] for i in train_idx]
        test_images = [image_paths[i] for i in test_idx]
        test_masks = [mask_paths[i] for i in test_idx]

        print(f"[INFO] Train size: {len(train_images)} | Test size: {len(test_images)}")

        train_dataset = SegDataset(train_images, train_masks, transform=train_transform)
        test_dataset = SegDataset(test_images, test_masks, transform=test_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
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
            print(f"\n[INFO] Fold {fold_idx + 1} | Epoch {epoch}/{EPOCHS}")

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

            torch.cuda.empty_cache()

        # Save training log CSV
        df_log = pd.DataFrame(logs)

        # Append best-epoch summary row
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

        df_log = pd.concat([df_log, pd.DataFrame([best_row_for_csv])], ignore_index=True)
        df_log.to_csv(log_csv_path, index=False)

        # Save training curve visualization
        plot_log(df_log, log_png_path)

        print(f"[SAVED] {log_csv_path}")
        print(f"[SAVED] {log_png_path}")
        print(f"[SAVED] {model_path}")

        summary_row = {
            "fold": fold_idx + 1,
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

        if best_record["test_Dice"] > global_best_dice:
            global_best_dice = best_record["test_Dice"]
            global_best_model_path = model_path

        # Release GPU memory after each fold
        del model
        del optimizer
        del criterion
        del train_loader
        del test_loader
        del train_dataset
        del test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # ----------------------------------------
    # Save cross-validation summary
    # ----------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    numeric_cols = [
        "best_epoch",
        "train_loss", "train_PA", "train_Dice", "train_IoU", "train_Recall", "train_Precision",
        "test_loss", "test_PA", "test_Dice", "test_IoU", "test_Recall", "test_Precision"
    ]

    mean_row = {"fold": "mean", "model_path": ""}
    std_row = {"fold": "std", "model_path": ""}

    for col in numeric_cols:
        mean_row[col] = summary_df[col].mean()
        std_row[col] = summary_df[col].std()

    summary_df = pd.concat(
        [summary_df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )

    summary_csv_path = os.path.join(SAVE_ROOT, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[SAVED] {summary_csv_path}")

    # ----------------------------------------
    # Export best model checkpoint
    # ----------------------------------------
    if global_best_model_path is not None:
        shutil.copy2(global_best_model_path, BEST_MODEL_COPY_PATH)
        print(f"[SAVED] Best model copied to: {BEST_MODEL_COPY_PATH}")
        print(f"[INFO] Global best Dice: {global_best_dice:.6f}")
    else:
        print("[WARN] No best model found.")

    print("\nDone.")


if __name__ == "__main__":
    main()