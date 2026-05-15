"""
U-Net Segmentation Training Pipeline

This script trains a U-Net segmentation model using data loaders provided
by the project dataset module.

Main features
-------------
- Fixed-subset or user-specified training image selection
- U-Net with ResNet18 encoder
- Focal + Jaccard loss
- Train/validation metric logging
- Periodic checkpoint saving
- Best-model checkpoint saving
- Training-curve visualization

This script integrates the original training entry point and utility
functions into a single reproducible file.
"""

import argparse
import os
import random
import time
from glob import glob
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from segmentation_models_pytorch import create_model, losses
from sklearn.metrics import confusion_matrix

from datasets import get_data


VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------
# Reproducibility
# ----------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------
# Metrics
# ----------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Compute mPA, mDice, mIoU, recall, and precision."""
    cm = confusion_matrix(
        y_true.flatten(),
        y_pred.flatten(),
        labels=[0, 1],
    )

    mpa = np.nanmean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1))

    iou = np.diag(cm) / (
        cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-7
    )
    miou = np.nanmean(iou)

    dice = 2 * np.diag(cm) / (
        cm.sum(axis=1) + cm.sum(axis=0) + 1e-7
    )
    mdice = np.nanmean(dice)

    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]

    recall = tp / (tp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)

    return float(mpa), float(mdice), float(miou), float(recall), float(precision)


def init_metric_history() -> Dict[str, List[float]]:
    """Initialize metric history container."""
    return {
        "train_loss": [],
        "val_loss": [],
        "train_mPA": [],
        "val_mPA": [],
        "train_mDice": [],
        "val_mDice": [],
        "train_mIoU": [],
        "val_mIoU": [],
        "train_recall": [],
        "val_recall": [],
        "train_precision": [],
        "val_precision": [],
    }


# ----------------------------------------
# Visualization
# ----------------------------------------
def plot_metrics(metrics: Dict[str, List[float]], save_path: str) -> None:
    """Save training and validation metric curves."""
    num_epochs = len(metrics["train_loss"])
    if num_epochs == 0:
        raise ValueError("No metrics available for plotting.")

    epochs = range(1, num_epochs + 1)

    curves = [
        ("Loss", "train_loss", "val_loss"),
        ("Mean Pixel Accuracy", "train_mPA", "val_mPA"),
        ("Mean Dice", "train_mDice", "val_mDice"),
        ("Mean IoU", "train_mIoU", "val_mIoU"),
        ("Recall", "train_recall", "val_recall"),
        ("Precision", "train_precision", "val_precision"),
    ]

    plt.figure(figsize=(12, 15))

    for idx, (title, train_key, val_key) in enumerate(curves, start=1):
        plt.subplot(3, 2, idx)
        plt.plot(epochs, metrics[train_key], label="train", linewidth=2)
        plt.plot(epochs, metrics[val_key], label="validation", linewidth=2)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ----------------------------------------
# Training
# ----------------------------------------
def build_model(
    architecture: str = "Unet",
    encoder_name: str = "resnet18",
    num_classes: int = 2,
):
    """Build segmentation model."""
    return create_model(
        architecture,
        encoder_name=encoder_name,
        classes=num_classes,
    )


def run_one_epoch(
    model,
    loader,
    device: torch.device,
    loss_fn,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    """Run one training or validation epoch."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    progress = tqdm.tqdm(
        loader,
        desc="Train" if is_train else "Validation",
        leave=False,
    )

    for images, masks in progress:
        images = images.to(device).float()
        masks = masks.to(device).long()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = loss_fn(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)

        all_true.append(masks.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    mpa, mdice, miou, recall, precision = compute_metrics(y_true, y_pred)

    return {
        "loss": total_loss / max(len(loader), 1),
        "mPA": mpa,
        "mDice": mdice,
        "mIoU": miou,
        "recall": recall,
        "precision": precision,
    }


def append_metrics(
    history: Dict[str, List[float]],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    """Append train and validation metrics to history."""
    key_map = {
        "loss": "loss",
        "mPA": "mPA",
        "mDice": "mDice",
        "mIoU": "mIoU",
        "recall": "recall",
        "precision": "precision",
    }

    for metric_key, history_key in key_map.items():
        history[f"train_{history_key}"].append(train_metrics[metric_key])
        history[f"val_{history_key}"].append(val_metrics[metric_key])


def save_log_csv(
    history: Dict[str, List[float]],
    save_path: str,
    best_metrics: Dict[str, float],
    total_time_seconds: float,
) -> None:
    """Save training log and best validation metrics."""
    num_epochs = len(history["train_loss"])

    header = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_mPA",
        "val_mPA",
        "train_mDice",
        "val_mDice",
        "train_mIoU",
        "val_mIoU",
        "train_recall",
        "val_recall",
        "train_precision",
        "val_precision",
    ]

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(",".join(header) + "\n")

        for idx in range(num_epochs):
            row = [
                idx + 1,
                history["train_loss"][idx],
                history["val_loss"][idx],
                history["train_mPA"][idx],
                history["val_mPA"][idx],
                history["train_mDice"][idx],
                history["val_mDice"][idx],
                history["train_mIoU"][idx],
                history["val_mIoU"][idx],
                history["train_recall"][idx],
                history["val_recall"][idx],
                history["train_precision"][idx],
                history["val_precision"][idx],
            ]
            file.write(",".join(map(str, row)) + "\n")

        file.write("\n")
        file.write(f"Best mPA,{best_metrics['mPA']}\n")
        file.write(f"Best mDice,{best_metrics['mDice']}\n")
        file.write(f"Best mIoU,{best_metrics['mIoU']}\n")
        file.write(f"Best recall,{best_metrics['recall']}\n")
        file.write(f"Best precision,{best_metrics['precision']}\n")
        file.write(f"Total training time seconds,{total_time_seconds:.2f}\n")


def train_model(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    results_dir: str,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    save_every: int = 5,
) -> Dict[str, str]:
    """Train model and save logs, curves, and checkpoints."""
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "log.csv")
    best_model_path = os.path.join(results_dir, "best_model.pt")
    plot_path = os.path.join(results_dir, "metrics.png")

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    focal_loss = losses.FocalLoss(mode="multiclass", alpha=0.25)
    jaccard_loss = losses.JaccardLoss(mode="multiclass")

    def loss_fn(logits, targets):
        return focal_loss(logits, targets) + jaccard_loss(logits, targets)

    history = init_metric_history()

    best_metrics = {
        "mPA": 0.0,
        "mDice": 0.0,
        "mIoU": 0.0,
        "recall": 0.0,
        "precision": 0.0,
    }
    best_miou = -1.0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{epochs}")

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
        )

        append_metrics(history, train_metrics, val_metrics)

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_mDice={val_metrics['mDice']:.4f}, "
            f"val_mIoU={val_metrics['mIoU']:.4f}"
        )

        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            best_metrics = {
                "mPA": val_metrics["mPA"],
                "mDice": val_metrics["mDice"],
                "mIoU": val_metrics["mIoU"],
                "recall": val_metrics["recall"],
                "precision": val_metrics["precision"],
            }
            torch.save(model.state_dict(), best_model_path)
            print(f"[SAVED] Best model: {best_model_path}")

        if save_every > 0 and epoch % save_every == 0:
            epoch_model_path = os.path.join(results_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), epoch_model_path)
            print(f"[SAVED] Checkpoint: {epoch_model_path}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - start_time

    plot_metrics(history, plot_path)
    save_log_csv(
        history=history,
        save_path=log_path,
        best_metrics=best_metrics,
        total_time_seconds=total_time,
    )

    print(f"[SAVED] Log: {log_path}")
    print(f"[SAVED] Metrics plot: {plot_path}")

    return {
        "log_path": log_path,
        "best_model_path": best_model_path,
        "plot_path": plot_path,
    }


# ----------------------------------------
# Data split helpers
# ----------------------------------------
def list_dataset_images(data_root: str) -> List[str]:
    """List image names under data_root/image."""
    image_dir = os.path.join(data_root, "image")
    image_paths = sorted(glob(os.path.join(image_dir, "*")))

    image_names = [
        os.path.basename(path)
        for path in image_paths
        if path.lower().endswith(VALID_EXTS)
    ]

    if len(image_names) == 0:
        raise FileNotFoundError(f"No valid images found in: {image_dir}")

    return image_names


def select_train_images(
    image_names: List[str],
    train_images: Optional[List[str]],
    num_train_images: int,
) -> List[str]:
    """Select training image names."""
    if train_images is not None and len(train_images) > 0:
        return train_images

    return image_names[:num_train_images]


# ----------------------------------------
# Command-line interface
# ----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a U-Net segmentation model with fixed or user-defined training images."
    )

    parser.add_argument("--data_root", default="path/to/data_root")
    parser.add_argument("--results_dir", default="path/to/output")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_train_images", type=int, default=12)
    parser.add_argument("--train_images", nargs="*", default=None)

    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--architecture", default="Unet")
    parser.add_argument("--encoder_name", default="resnet18")
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--save_every", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)

    image_names = list_dataset_images(args.data_root)
    train_img_names = select_train_images(
        image_names=image_names,
        train_images=args.train_images,
        num_train_images=args.num_train_images,
    )

    print(f"[INFO] Found {len(image_names)} images.")
    print(f"[INFO] Training images: {train_img_names}")

    train_loader, val_loader = get_data(
        args.data_root,
        args.batch_size,
        train_img_names=train_img_names,
    )

    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        num_classes=args.num_classes,
    )
    model = model.to(device)

    print(f"[INFO] Training on device: {device}")
    print(f"[INFO] Results directory: {args.results_dir}")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        results_dir=args.results_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_every=args.save_every,
    )

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
