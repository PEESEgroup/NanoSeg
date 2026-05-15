"""Select representative reference images using DINOv2 embeddings.

This script scans datasets under a root directory, extracts DINOv2 image
features, computes each image's mean distance to all others, and selects the
medoid image as the most representative reference.

Expected dataset layout:
    data_root/
        dataset_001/
            images/
                image_001.png
                image_002.png
        dataset_002/
            images/
                ...

Example:
    python ref_pick_clean.py \
        --data-root /path/to/data \
        --checkpoint-dir /path/to/checkpoints \
        --output-root /path/to/results/ref_pick
"""

from __future__ import annotations

import argparse
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
DEFAULT_REPO_ID = "facebook/dinov2-base"
OUTPUT_CSV_NAME = "medoid_scores.csv"
OUTPUT_PLOT_NAME = "dino_embedding_scatter.png"
CBAR_LABEL = r"$\theta_i$ (mean distance to all images)"


@dataclass(frozen=True)
class Config:
    data_root: Path
    checkpoint_dir: Path
    output_root: Path
    repo_id: str = DEFAULT_REPO_ID
    target_size: int = 512
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    skip_existing: bool = True

    @property
    def model_dir(self) -> Path:
        return self.checkpoint_dir / self.repo_id.replace("/", "_")


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    image_dir: Path


class DINOv2Encoder:
    """Lightweight wrapper for Hugging Face DINOv2 feature extraction."""

    def __init__(self, repo_id: str, model_dir: Path, device: torch.device):
        self.repo_id = repo_id
        self.model_dir = model_dir
        self.device = device
        self.processor, self.model = self._load_model()

    @staticmethod
    def _import_transformers():
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required. Install it with: "
                "pip install transformers"
            ) from exc
        return AutoImageProcessor, AutoModel

    @staticmethod
    def _has_local_model(model_dir: Path) -> bool:
        if not model_dir.is_dir():
            return False
        required_files = {
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "pytorch_model.bin",
            "model.safetensors",
        }
        return any((model_dir / filename).exists() for filename in required_files)

    def _load_model(self):
        AutoImageProcessor, AutoModel = self._import_transformers()
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if not self._has_local_model(self.model_dir):
            print(f"[INFO] Downloading {self.repo_id} to {self.model_dir}")
            processor = AutoImageProcessor.from_pretrained(self.repo_id)
            model = AutoModel.from_pretrained(self.repo_id)
            processor.save_pretrained(self.model_dir)
            model.save_pretrained(self.model_dir)

        processor = AutoImageProcessor.from_pretrained(self.model_dir)
        model = AutoModel.from_pretrained(self.model_dir)
        model.to(self.device)
        model.eval()
        print(f"[INFO] Loaded DINOv2 model from {self.model_dir}")
        return processor, model

    @torch.no_grad()
    def encode(self, image_rgb: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)

        if getattr(outputs, "pooler_output", None) is not None:
            vector = outputs.pooler_output
        elif getattr(outputs, "last_hidden_state", None) is not None:
            vector = outputs.last_hidden_state[:, 0, :]
        else:
            raise RuntimeError("DINOv2 output does not contain usable features.")

        return vector.squeeze(0).detach().cpu().numpy().astype(np.float64)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Select representative reference images using DINOv2 medoid scoring."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory containing dataset folders.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory for cached Hugging Face models.")
    parser.add_argument("--output-root", type=Path, default=Path("results/ref_pick"), help="Directory for output CSV files and plots.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face model ID.")
    parser.add_argument("--target-size", type=int, default=512, help="Square image size used before encoding.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute datasets even when outputs already exist.")
    args = parser.parse_args()

    return Config(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        output_root=args.output_root,
        repo_id=args.repo_id,
        target_size=args.target_size,
        device=args.device,
        skip_existing=not args.overwrite,
    )


def natural_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def find_datasets(data_root: Path) -> list[DatasetInfo]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    datasets: list[DatasetInfo] = []
    for dataset_dir in sorted(data_root.iterdir(), key=lambda p: natural_key(p.name)):
        image_dir = dataset_dir / "images"
        if dataset_dir.is_dir() and image_dir.is_dir():
            datasets.append(DatasetInfo(name=dataset_dir.name, image_dir=image_dir))
    return datasets


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        [path for path in image_dir.iterdir() if path.suffix.lower() in VALID_EXTS],
        key=lambda p: natural_key(p.name),
    )


def read_rgb_image(image_path: Path, target_size: int) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    height, width = image_bgr.shape[:2]
    if height != target_size or width != target_size:
        image_bgr = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def outputs_exist(output_dir: Path) -> bool:
    return (output_dir / OUTPUT_CSV_NAME).exists() and (output_dir / OUTPUT_PLOT_NAME).exists()


def extract_features(encoder: DINOv2Encoder, image_paths: list[Path], target_size: int, dataset_name: str) -> np.ndarray:
    features = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{dataset_name}] Encoding {index}/{len(image_paths)}: {image_path.name}")
        image_rgb = read_rgb_image(image_path, target_size)
        features.append(encoder.encode(image_rgb))
    return np.stack(features, axis=0)


def compute_medoid_scores(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = features[:, None, :] - features[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    theta = distances.mean(axis=1)

    order = np.argsort(theta, kind="mergesort")
    rank = np.empty_like(order, dtype=np.int64)
    rank[order] = np.arange(1, len(theta) + 1)
    return theta, rank


def project_features(features: np.ndarray) -> np.ndarray:
    if len(features) < 2:
        return np.zeros((len(features), 2), dtype=np.float64)
    return PCA(n_components=2, random_state=0).fit_transform(features)


def save_scatter_plot(summary: pd.DataFrame, dataset_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    non_medoid = summary["is_medoid"] == 0
    scatter = ax.scatter(
        summary.loc[non_medoid, "pc1"],
        summary.loc[non_medoid, "pc2"],
        c=summary.loc[non_medoid, "theta"],
        cmap="viridis_r",
        s=70,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.95,
        zorder=2,
    )

    medoid = summary["is_medoid"] == 1
    ax.scatter(
        summary.loc[medoid, "pc1"],
        summary.loc[medoid, "pc2"],
        c=summary.loc[medoid, "theta"],
        cmap="viridis_r",
        s=220,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=4,
    )

    x_span = summary["pc1"].max() - summary["pc1"].min() + 1e-12
    y_span = summary["pc2"].max() - summary["pc2"].min() + 1e-12
    for _, row in summary.iterrows():
        ax.text(
            row["pc1"] + 0.02 * x_span,
            row["pc2"] + 0.02 * y_span,
            str(int(row["index"])),
            fontsize=9,
            ha="left",
            va="bottom",
            zorder=5,
        )

    medoid_row = summary.loc[medoid].iloc[0]
    ax.text(
        medoid_row["pc1"],
        medoid_row["pc2"],
        "  Medoid",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=6,
    )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(CBAR_LABEL, fontsize=11)
    ax.set_title(f"{dataset_name} | DINOv2 embedding space", fontsize=14)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_existing_result(dataset_name: str, output_dir: Path) -> tuple[str, str] | None:
    csv_path = output_dir / OUTPUT_CSV_NAME
    try:
        summary = pd.read_csv(csv_path).sort_values("dino_rank", ascending=True)
        return dataset_name, str(summary.iloc[0]["image_name"])
    except Exception as exc:
        print(f"[WARN] Failed to read existing result for {dataset_name}: {exc}")
        return None


def run_dataset(encoder: DINOv2Encoder, dataset: DatasetInfo, output_dir: Path, target_size: int) -> tuple[str, str] | None:
    image_paths = list_images(dataset.image_dir)
    if not image_paths:
        print(f"[WARN] No valid images found in {dataset.image_dir}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[RUN] {dataset.name}: {len(image_paths)} images")

    features = extract_features(encoder, image_paths, target_size, dataset.name)
    theta, rank = compute_medoid_scores(features)
    coords = project_features(features)

    summary = pd.DataFrame(
        {
            "index": np.arange(1, len(image_paths) + 1),
            "image_name": [path.name for path in image_paths],
            "dino_medoid": theta,
            "dino_rank": rank,
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "theta": theta,
            "is_medoid": (rank == 1).astype(int),
        }
    )

    summary[["index", "image_name", "dino_medoid", "dino_rank"]].to_csv(
        output_dir / OUTPUT_CSV_NAME, index=False, float_format="%.10f"
    )
    save_scatter_plot(summary, dataset.name, output_dir / OUTPUT_PLOT_NAME)

    medoid_row = summary.sort_values("dino_rank", ascending=True).iloc[0]
    best_image = str(medoid_row["image_name"])
    best_score = float(medoid_row["dino_medoid"])
    print(f"[RESULT] {dataset.name}: {best_image} (rank=1, theta={best_score:.10f})")

    del features, theta, rank, coords, summary
    release_memory()
    return dataset.name, best_image


def main() -> None:
    config = parse_args()
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.output_root.mkdir(parents=True, exist_ok=True)

    datasets = find_datasets(config.data_root)
    if not datasets:
        raise FileNotFoundError(f"No dataset folders with an 'images' subdirectory were found under {config.data_root}")

    print(f"[INFO] Device: {config.device}")
    print(f"[INFO] Data root: {config.data_root}")
    print(f"[INFO] Output root: {config.output_root}")
    print(f"[INFO] Found datasets: {len(datasets)}")

    encoder = DINOv2Encoder(config.repo_id, config.model_dir, torch.device(config.device))
    recommendations: list[tuple[str, str]] = []

    for dataset in datasets:
        output_dir = config.output_root / dataset.name
        if config.skip_existing and outputs_exist(output_dir):
            print(f"[SKIP] {dataset.name}: existing outputs found")
            result = summarize_existing_result(dataset.name, output_dir)
        else:
            try:
                result = run_dataset(encoder, dataset, output_dir, config.target_size)
            except Exception as exc:
                print(f"[ERROR] {dataset.name}: {exc}")
                release_memory()
                result = None

        if result is not None:
            recommendations.append(result)

    print("\n[SUMMARY] Recommended reference images")
    if not recommendations:
        print("No valid recommendation was generated.")
    else:
        for dataset_name, image_name in recommendations:
            print(f"{dataset_name}: {image_name}")


if __name__ == "__main__":
    main()
