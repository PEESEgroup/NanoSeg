"""
Multi-Encoder Medoid Selection Framework

This script identifies the most representative reference image
(medoid) using feature embeddings from multiple vision foundation models.

Supported encoders
------------------
- SAM
- MicroSAM
- MedSAM
- DINOv2
- CLIP
- MAE

Main features
-------------
- Image embedding extraction
- Pairwise distance computation
- Exact medoid ranking
- PCA visualization
- Embedding-space scatter plots

The framework is designed for reference-image selection in
one-shot nanoparticle segmentation workflows.
"""

import gc
import os
import re
from typing import Dict, List, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from segment_anything import sam_model_registry, SamPredictor
from sklearn.decomposition import PCA

# ----------------------------------------
# Config
# ----------------------------------------
IMAGE_DIR = "path/to/dataset/images"
CHECKPOINT_DIR = "path/to/checkpoints"
OUTPUT_DIR = "path/to/output/medoid_selection"

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "medoid_scores.csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
PLOT_DATA_DIR = os.path.join(OUTPUT_DIR, "plot_data")

SAM_MODEL_TYPE = "vit_b"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Unified image encoding size.
TARGET_SIZE = 512

# Exact medoid score definition.
DISTANCE_METRIC = "euclidean"

# SAM-family checkpoints.
SAM_MODEL_SPECS = {
    "sam": os.path.join(CHECKPOINT_DIR, "sam_vit_b.pth"),
    "microsam": os.path.join(CHECKPOINT_DIR, "microsam_vit_b.pth"),
    "medsam": os.path.join(CHECKPOINT_DIR, "medsam_vit_b.pth"),
}

# Hugging Face encoder configurations.
HF_MODEL_SPECS = {
    "dinov2": {
        "family": "dinov2",
        "repo_id": "facebook/dinov2-base",
        "local_dir": os.path.join(CHECKPOINT_DIR, "facebook_dinov2-base"),
    },
    "clip": {
        "family": "clip",
        "repo_id": "openai/clip-vit-base-patch32",
        "local_dir": os.path.join(CHECKPOINT_DIR, "openai_clip-vit-base-patch32"),
    },
    "mae": {
        "family": "mae",
        "repo_id": "facebook/vit-mae-base",
        "local_dir": os.path.join(CHECKPOINT_DIR, "facebook_vit-mae-base"),
    },
}

# Visualization config
FIG_DPI = 300
POINT_SIZE = 70
MEDOID_POINT_SIZE = 220
TEXT_FONT_SIZE = 9
TITLE_FONT_SIZE = 14
CBAR_LABEL = r"$\theta_i$ (mean distance to all images)"


# ----------------------------------------
# Utilities
# ----------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def natural_key(s: str):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


def list_images(image_dir: str) -> List[str]:
    files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(VALID_EXTS)
    ]
    files.sort(key=natural_key)
    return files


def resize_image(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_size and w == target_size:
        return image
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def read_image_rgb_resized(image_path: str, target_size: int = 512) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_bgr = resize_image(image_bgr, target_size=target_size)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------
# SAM loading and features
# ----------------------------------------
def robust_load_checkpoint(model, checkpoint_path: str):
    """
    Robust checkpoint loader.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        cleaned_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    return missing, unexpected


def build_sam_predictor(checkpoint_path: str) -> SamPredictor:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=None)
    missing, unexpected = robust_load_checkpoint(model, checkpoint_path)

    model.to(DEVICE)
    model.eval()

    print(f"[INFO] Loaded SAM checkpoint: {checkpoint_path}")
    if len(missing) > 0:
        print(f"[WARN] Missing keys: {len(missing)}")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    predictor = SamPredictor(model)
    return predictor


@torch.no_grad()
def extract_image_vector_sam(predictor: SamPredictor, image_rgb: np.ndarray) -> np.ndarray:
    """
    SAM image encoder output is [1, C, H, W].
    We apply global average pooling to obtain one vector per image.
    """
    predictor.set_image(image_rgb)
    emb = predictor.get_image_embedding()          # [1, C, H, W]
    vec = emb.mean(dim=(2, 3)).squeeze(0)          # [C]
    vec = vec.detach().cpu().numpy().astype(np.float64)
    return vec


# ----------------------------------------
# Hugging Face models
# ----------------------------------------
def import_hf_modules():
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            CLIPProcessor,
            CLIPModel,
            ViTMAEModel,
        )
    except Exception as e:
        raise ImportError(
            "transformers is required for DINOv2 / CLIP / MAE.\n"
            "Please install it first, e.g.:\n"
            "pip install transformers"
        ) from e

    return AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel, ViTMAEModel


def is_valid_hf_local_dir(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    required_any = [
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    return any(os.path.exists(os.path.join(local_dir, x)) for x in required_any)


def ensure_hf_model_local(spec: Dict[str, str]):
    """
    Ensure the model exists in spec['local_dir'].
    If absent, download from HF repo_id and save_pretrained to local_dir.
    """
    local_dir = spec["local_dir"]
    repo_id = spec["repo_id"]
    family = spec["family"]

    ensure_dir(local_dir)

    if is_valid_hf_local_dir(local_dir):
        print(f"[INFO] Found local HF model: {local_dir}")
        return

    print(f"[INFO] Local model not found. Downloading {repo_id} to {local_dir} ...")

    AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel, ViTMAEModel = import_hf_modules()

    if family == "dinov2":
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = AutoModel.from_pretrained(repo_id)
        processor.save_pretrained(local_dir)
        model.save_pretrained(local_dir)

    elif family == "clip":
        processor = CLIPProcessor.from_pretrained(repo_id)
        model = CLIPModel.from_pretrained(repo_id)
        processor.save_pretrained(local_dir)
        model.save_pretrained(local_dir)

    elif family == "mae":
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = ViTMAEModel.from_pretrained(repo_id)
        processor.save_pretrained(local_dir)
        model.save_pretrained(local_dir)

    else:
        raise ValueError(f"Unsupported HF family: {family}")

    print(f"[INFO] Downloaded and saved {repo_id} to {local_dir}")


def build_hf_encoder(spec: Dict[str, str]) -> Dict[str, Any]:
    """
    Initialize Hugging Face encoder.
    """
    family = spec["family"]
    local_dir = spec["local_dir"]

    ensure_hf_model_local(spec)

    AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel, ViTMAEModel = import_hf_modules()

    if family == "dinov2":
        processor = AutoImageProcessor.from_pretrained(local_dir)
        model = AutoModel.from_pretrained(local_dir)
        model.to(DEVICE)
        model.eval()
        print(f"[INFO] Loaded DINOv2 from: {local_dir}")
        return {
            "family": family,
            "processor": processor,
            "model": model,
        }

    elif family == "clip":
        processor = CLIPProcessor.from_pretrained(local_dir)
        model = CLIPModel.from_pretrained(local_dir)
        model.to(DEVICE)
        model.eval()
        print(f"[INFO] Loaded CLIP from: {local_dir}")
        return {
            "family": family,
            "processor": processor,
            "model": model,
        }

    elif family == "mae":
        processor = AutoImageProcessor.from_pretrained(local_dir)
        model = ViTMAEModel.from_pretrained(local_dir)
        model.to(DEVICE)
        model.eval()
        print(f"[INFO] Loaded MAE from: {local_dir}")
        return {
            "family": family,
            "processor": processor,
            "model": model,
        }

    else:
        raise ValueError(f"Unsupported HF family: {family}")


@torch.no_grad()
def extract_image_vector_hf(encoder: Dict[str, Any], image_rgb: np.ndarray) -> np.ndarray:
    """
    Extract image-level feature embedding.

    DINOv2:
        prefer pooler_output; otherwise use CLS token.
    CLIP:
        use get_image_features().
    MAE:
        use CLS token from last_hidden_state.
    """
    family = encoder["family"]
    processor = encoder["processor"]
    model = encoder["model"]

    if family == "dinov2":
        inputs = processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            vec = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            vec = outputs.last_hidden_state[:, 0, :]
        else:
            raise RuntimeError("DINOv2 output does not contain usable pooled features.")

        vec = vec.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return vec

    elif family == "clip":
        inputs = processor(images=image_rgb, return_tensors="pt")
        if "pixel_values" not in inputs:
            raise RuntimeError("CLIPProcessor did not return pixel_values.")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        vec = model.get_image_features(pixel_values=pixel_values)
        vec = vec.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return vec

    elif family == "mae":
        inputs = processor(images=image_rgb, return_tensors="pt")
        if "pixel_values" not in inputs:
            raise RuntimeError("MAE processor did not return pixel_values.")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        outputs = model(pixel_values=pixel_values)
        if not hasattr(outputs, "last_hidden_state") or outputs.last_hidden_state is None:
            raise RuntimeError("MAE output does not contain last_hidden_state.")

        # Use CLS token as image-level representation
        vec = outputs.last_hidden_state[:, 0, :]
        vec = vec.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return vec

    else:
        raise ValueError(f"Unsupported HF family: {family}")


# ----------------------------------------
# Distance / rank / plot
# ----------------------------------------
def compute_pairwise_distances(features: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if metric != "euclidean":
        raise ValueError(f"Unsupported metric: {metric}")

    diff = features[:, None, :] - features[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist


def compute_theta_exact(features: np.ndarray) -> np.ndarray:
    """
    Exact medoid score definition.
    Self-distance is included and equals zero.
    """
    dist_mat = compute_pairwise_distances(features, metric=DISTANCE_METRIC)
    theta = dist_mat.mean(axis=1)
    return theta


def compute_rank_from_theta(theta: np.ndarray) -> np.ndarray:
    """
    Rank 1 = smallest theta = most representative image.
    """
    order = np.argsort(theta, kind="mergesort")
    rank = np.empty_like(order, dtype=np.int64)
    rank[order] = np.arange(1, len(theta) + 1)
    return rank


def project_to_2d(features: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(features)
    return coords


def make_plot_dataframe(
    image_files: List[str],
    theta: np.ndarray,
    rank: np.ndarray,
    coords_2d: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "image_index": np.arange(1, len(image_files) + 1),
        "image_name": image_files,
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
        "theta": theta,
        "rank": rank,
        "is_medoid": (rank == 1).astype(int),
    })
    return df


def save_plot_dataframe(df_plot: pd.DataFrame, model_name: str):
    out_csv = os.path.join(PLOT_DATA_DIR, f"{model_name}_scatter_data.csv")
    df_plot.to_csv(out_csv, index=False, float_format="%.10f")
    print(f"[SAVED] Plot data: {out_csv}")


def save_scatter_plot(df_plot: pd.DataFrame, model_name: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    non_medoid = df_plot["is_medoid"] == 0
    sc = ax.scatter(
        df_plot.loc[non_medoid, "x"],
        df_plot.loc[non_medoid, "y"],
        c=df_plot.loc[non_medoid, "theta"],
        cmap="viridis_r",
        s=POINT_SIZE,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.95,
        zorder=2,
    )

    medoid = df_plot["is_medoid"] == 1
    ax.scatter(
        df_plot.loc[medoid, "x"],
        df_plot.loc[medoid, "y"],
        c=df_plot.loc[medoid, "theta"],
        cmap="viridis_r",
        s=MEDOID_POINT_SIZE,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        alpha=1.0,
        zorder=4,
    )

    x_span = df_plot["x"].max() - df_plot["x"].min() + 1e-12
    y_span = df_plot["y"].max() - df_plot["y"].min() + 1e-12

    for _, row in df_plot.iterrows():
        ax.text(
            row["x"] + 0.02 * x_span,
            row["y"] + 0.02 * y_span,
            str(int(row["image_index"])),
            fontsize=TEXT_FONT_SIZE,
            color="black",
            ha="left",
            va="bottom",
            zorder=5,
        )

    medoid_row = df_plot[df_plot["is_medoid"] == 1].iloc[0]
    ax.text(
        medoid_row["x"],
        medoid_row["y"],
        "  Medoid",
        fontsize=TEXT_FONT_SIZE + 1,
        color="black",
        ha="left",
        va="center",
        fontweight="bold",
        zorder=6,
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(CBAR_LABEL, fontsize=11)

    ax.set_title(f"{model_name.upper()} embedding space", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out_png = os.path.join(PLOT_DIR, f"{model_name}_embedding_scatter.png")
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] Scatter plot: {out_png}")


# ----------------------------------------
# Feature extraction wrappers
# ----------------------------------------
def extract_features_for_sam_model(
    model_name: str,
    checkpoint_path: str,
    image_dir: str,
    image_files: List[str],
) -> np.ndarray:
    predictor = build_sam_predictor(checkpoint_path)

    all_vecs = []
    for idx, image_name in enumerate(image_files, start=1):
        image_path = os.path.join(image_dir, image_name)
        print(f"[{model_name}] Encoding {idx}/{len(image_files)}: {image_name}")
        image_rgb = read_image_rgb_resized(image_path, target_size=TARGET_SIZE)
        vec = extract_image_vector_sam(predictor, image_rgb)
        all_vecs.append(vec)

    features = np.stack(all_vecs, axis=0)
    print(f"[{model_name}] Feature matrix shape: {features.shape}")
    return features


def extract_features_for_hf_model(
    model_name: str,
    spec: Dict[str, str],
    image_dir: str,
    image_files: List[str],
) -> np.ndarray:
    encoder = build_hf_encoder(spec)

    all_vecs = []
    for idx, image_name in enumerate(image_files, start=1):
        image_path = os.path.join(image_dir, image_name)
        print(f"[{model_name}] Encoding {idx}/{len(image_files)}: {image_name}")
        image_rgb = read_image_rgb_resized(image_path, target_size=TARGET_SIZE)
        vec = extract_image_vector_hf(encoder, image_rgb)
        all_vecs.append(vec)

    features = np.stack(all_vecs, axis=0)
    print(f"[{model_name}] Feature matrix shape: {features.shape}")
    return features


# ----------------------------------------
# Main
# ----------------------------------------
def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOT_DIR)
    ensure_dir(PLOT_DATA_DIR)
    ensure_dir(CHECKPOINT_DIR)

    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"IMAGE_DIR not found: {IMAGE_DIR}")

    image_files = list_images(IMAGE_DIR)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No valid images found in {IMAGE_DIR}")

    print("=" * 100)
    print(f"[INFO] DEVICE              : {DEVICE}")
    print(f"[INFO] IMAGE_DIR           : {IMAGE_DIR}")
    print(f"[INFO] CHECKPOINT_DIR      : {CHECKPOINT_DIR}")
    print(f"[INFO] OUTPUT_DIR          : {OUTPUT_DIR}")
    print(f"[INFO] OUTPUT_CSV          : {OUTPUT_CSV}")
    print(f"[INFO] PLOT_DIR            : {PLOT_DIR}")
    print(f"[INFO] PLOT_DATA_DIR       : {PLOT_DATA_DIR}")
    print(f"[INFO] SAM_MODEL_TYPE      : {SAM_MODEL_TYPE}")
    print(f"[INFO] TARGET_SIZE         : {TARGET_SIZE}")
    print(f"[INFO] DISTANCE_METRIC     : {DISTANCE_METRIC}")
    print(f"[INFO] Total images        : {len(image_files)}")
    print("=" * 100)

    # Check SAM checkpoints
    for model_name, ckpt_path in SAM_MODEL_SPECS.items():
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[{model_name}] checkpoint not found: {ckpt_path}")

    # Initialize summary table.
    df_summary = pd.DataFrame({
        "index": np.arange(1, len(image_files) + 1),
        "image_name": image_files,
    })

    # Evaluate SAM-family encoders.
    for model_name, ckpt_path in SAM_MODEL_SPECS.items():
        print("\n" + "=" * 100)
        print(f"[RUN] Model: {model_name}")
        print("=" * 100)

        features = extract_features_for_sam_model(
            model_name=model_name,
            checkpoint_path=ckpt_path,
            image_dir=IMAGE_DIR,
            image_files=image_files,
        )

        theta = compute_theta_exact(features)
        rank = compute_rank_from_theta(theta)

        df_summary[f"{model_name}_medoid"] = theta
        df_summary[f"{model_name}_rank"] = rank

        coords_2d = project_to_2d(features)

        df_plot = make_plot_dataframe(
            image_files=image_files,
            theta=theta,
            rank=rank,
            coords_2d=coords_2d,
        )
        save_plot_dataframe(df_plot, model_name)
        save_scatter_plot(df_plot, model_name)

        best_idx = int(np.argmin(theta))
        print(f"[RESULT] {model_name} best representative image:")
        print(f"         rank=1 -> index={best_idx + 1}, image={image_files[best_idx]}, theta={theta[best_idx]:.10f}")

        del features, theta, rank, coords_2d, df_plot
        release_memory()

    # Evaluate Hugging Face encoders.
    for model_name, spec in HF_MODEL_SPECS.items():
        print("\n" + "=" * 100)
        print(f"[RUN] Model: {model_name}")
        print("=" * 100)

        features = extract_features_for_hf_model(
            model_name=model_name,
            spec=spec,
            image_dir=IMAGE_DIR,
            image_files=image_files,
        )

        theta = compute_theta_exact(features)
        rank = compute_rank_from_theta(theta)

        df_summary[f"{model_name}_medoid"] = theta
        df_summary[f"{model_name}_rank"] = rank

        coords_2d = project_to_2d(features)

        df_plot = make_plot_dataframe(
            image_files=image_files,
            theta=theta,
            rank=rank,
            coords_2d=coords_2d,
        )
        save_plot_dataframe(df_plot, model_name)
        save_scatter_plot(df_plot, model_name)

        best_idx = int(np.argmin(theta))
        print(f"[RESULT] {model_name} best representative image:")
        print(f"         rank=1 -> index={best_idx + 1}, image={image_files[best_idx]}, theta={theta[best_idx]:.10f}")

        del features, theta, rank, coords_2d, df_plot
        release_memory()

    df_summary.to_csv(OUTPUT_CSV, index=False, float_format="%.10f")

    print("\n" + "=" * 100)
    print("[DONE] Exact medoid theta computation and visualization finished.")
    print(f"[SAVED] Summary CSV: {OUTPUT_CSV}")
    print("=" * 100)


if __name__ == "__main__":
    main()