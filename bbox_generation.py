"""
Automatic Bounding-Box Proposal Generator

This script generates bounding-box proposals for nanoparticle regions
from GradCAM images using Difference-of-Gaussians (DoG)
enhancement and multi-threshold candidate extraction.

Main features
-------------
- Difference-of-Gaussians enhancement
- Multi-threshold proposal extraction
- Bounding-box expansion
- Aspect-ratio outlier filtering
- Intensity-based proposal filtering
- Bounding-box visualization export
- Pixel-coordinate label export

The pipeline is designed for automatic prompt generation in
NanoSeg one-shot segmentation workflows.
"""

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, find_objects, label
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.io import imread

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------
# Image utilities
# ----------------------------------------
def list_images(image_dir: str) -> List[str]:
    """List valid image files."""
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(VALID_EXTS)]
    return sorted(files)


def read_gray_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read image and return:
    - original image
    - normalized grayscale image
    """
    image = imread(image_path)

    if image.ndim == 3:
        gray = rgb2gray(image)
    else:
        gray = image.astype(np.float32) / 255.0

    return image, gray.astype(np.float32)


# ----------------------------------------
# DoG enhancement
# ----------------------------------------
def dog_enhance(
    gray: np.ndarray,
    sigma_small: float = 1.0,
    sigma_large: float = 4.0,
) -> np.ndarray:
    """Apply Difference-of-Gaussians enhancement."""
    blur_small = gaussian_filter(gray, sigma=sigma_small)
    blur_large = gaussian_filter(gray, sigma=sigma_large)

    dog = blur_small - blur_large
    return rescale_intensity(dog, out_range=(0.0, 1.0))


# ----------------------------------------
# Candidate extraction
# ----------------------------------------
def multi_threshold_filter(
    image: np.ndarray,
    thresholds: List[float],
):
    """Extract connected-component candidates using multiple percentile thresholds."""
    total_mask = np.zeros_like(image, dtype=bool)

    for threshold in thresholds:
        binary = image > np.percentile(image, threshold)
        total_mask |= binary

    labeled, _ = label(total_mask)
    objects = find_objects(labeled)

    return objects, labeled


def expand_bbox(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    width: int,
    height: int,
    expand_ratio: float,
) -> Tuple[int, int, int, int]:
    """Expand bounding box around its center."""
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    box_width = (x_max - x_min) * expand_ratio
    box_height = (y_max - y_min) * expand_ratio

    x_min_exp = int(max(center_x - box_width / 2.0, 0))
    x_max_exp = int(min(center_x + box_width / 2.0, width))

    y_min_exp = int(max(center_y - box_height / 2.0, 0))
    y_max_exp = int(min(center_y + box_height / 2.0, height))

    return x_min_exp, y_min_exp, x_max_exp, y_max_exp


def generate_bbox_candidates(
    image: np.ndarray,
    gray: np.ndarray,
    thresholds: List[float],
    min_area: int,
    expand_ratio: float,
):
    """Generate initial bounding-box candidates."""
    enhanced = dog_enhance(gray)
    objects, _ = multi_threshold_filter(enhanced, thresholds)

    height, width = gray.shape
    candidates = []

    for slc in objects:
        if slc is None:
            continue

        y_min, x_min = slc[0].start, slc[1].start
        y_max, x_max = slc[0].stop, slc[1].stop

        area = (x_max - x_min) * (y_max - y_min)
        if area < min_area:
            continue

        x_min_exp, y_min_exp, x_max_exp, y_max_exp = expand_bbox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            width=width,
            height=height,
            expand_ratio=expand_ratio,
        )

        bbox_width = x_max_exp - x_min_exp
        bbox_height = y_max_exp - y_min_exp

        if bbox_width <= 0 or bbox_height <= 0:
            continue

        aspect_ratio = bbox_width / bbox_height

        candidates.append({
            "bbox": (x_min_exp, y_min_exp, x_max_exp, y_max_exp),
            "aspect_ratio": float(aspect_ratio),
        })

    return candidates


# ----------------------------------------
# Proposal filtering
# ----------------------------------------
def compute_iqr_bounds(values: List[float]) -> Tuple[float, float]:
    """Compute IQR outlier bounds."""
    if len(values) == 0:
        return -np.inf, np.inf

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound


def compute_region_mean_intensity(region: np.ndarray) -> float:
    """Compute mean intensity of one image region."""
    if region.ndim == 3:
        region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        return float(np.mean(region_gray))

    return float(np.mean(region))


def filter_bbox_candidates(
    image: np.ndarray,
    candidates,
    intensity_threshold: float,
):
    """Filter proposals using aspect-ratio and intensity rules."""
    ratios = [candidate["aspect_ratio"] for candidate in candidates]
    lower_bound, upper_bound = compute_iqr_bounds(ratios)

    final_bboxes = []

    for candidate in candidates:
        ratio = candidate["aspect_ratio"]

        if ratio < lower_bound or ratio > upper_bound:
            continue

        x_min, y_min, x_max, y_max = candidate["bbox"]

        region = image[y_min:y_max, x_min:x_max]
        mean_intensity = compute_region_mean_intensity(region)

        if mean_intensity < intensity_threshold:
            continue

        final_bboxes.append((x_min, y_min, x_max, y_max))

    return final_bboxes


# ----------------------------------------
# Output utilities
# ----------------------------------------
def draw_bboxes(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    save_path: str,
    line_width: int = 1,
):
    """Draw bounding boxes on image."""
    image_vis = image.copy().astype(np.uint8)

    for (x_min, y_min, x_max, y_max) in bboxes:
        cv2.rectangle(
            image_vis,
            (x_min, y_min),
            (x_max, y_max),
            (255, 0, 0),
            line_width,
        )

    cv2.imwrite(save_path, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))


def save_pixel_labels(
    bboxes: List[Tuple[int, int, int, int]],
    save_path: str,
    class_id: int = 0,
):
    """Save bounding boxes in pixel-coordinate TXT format."""
    with open(save_path, "w") as file:
        for (x_min, y_min, x_max, y_max) in bboxes:
            file.write(f"{class_id} {x_min} {y_min} {x_max} {y_max}\n")


# ----------------------------------------
# Main processing
# ----------------------------------------
def run_bbox_generation(
    image_dir: str,
    output_dir: str,
    thresholds: List[float],
    intensity_threshold: float,
    min_area: int,
    expand_ratio: float,
):
    """Run automatic bounding-box proposal generation."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    label_dir = os.path.join(output_dir, "labels")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    image_files = list_images(image_dir)

    if len(image_files) == 0:
        raise FileNotFoundError(f"No valid images found in: {image_dir}")

    print(f"[INFO] Image directory: {image_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Total images: {len(image_files)}")

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)

        try:
            image, gray = read_gray_image(image_path)

            candidates = generate_bbox_candidates(
                image=image,
                gray=gray,
                thresholds=thresholds,
                min_area=min_area,
                expand_ratio=expand_ratio,
            )

            final_bboxes = filter_bbox_candidates(
                image=image,
                candidates=candidates,
                intensity_threshold=intensity_threshold,
            )

            stem = os.path.splitext(image_name)[0]

            bbox_vis_path = os.path.join(output_dir, f"{stem}_bbox.png")
            label_path = os.path.join(label_dir, f"{stem}.txt")

            draw_bboxes(
                image=image,
                bboxes=final_bboxes,
                save_path=bbox_vis_path,
            )

            save_pixel_labels(
                bboxes=final_bboxes,
                save_path=label_path,
                class_id=0,
            )

            print(f"[OK] {image_name}: {len(final_bboxes)} proposals")

        except Exception as exc:
            print(f"[WARN] Failed on {image_name}: {exc}")
            continue

    print("[DONE] Bounding-box generation finished.")


# ----------------------------------------
# Command-line interface
# ----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate nanoparticle bounding-box proposals from microscopy images."
    )

    parser.add_argument("--image_dir", default="path/to/images")
    parser.add_argument("--output_dir", default="path/to/output")

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[95, 97, 99],
        help="Percentile thresholds used for candidate extraction.",
    )

    parser.add_argument(
        "--intensity_threshold",
        type=float,
        default=50.0,
        help="Minimum mean intensity required for one proposal.",
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=60,
        help="Minimum proposal area in pixels.",
    )

    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=1.2,
        help="Bounding-box expansion ratio.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_bbox_generation(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        intensity_threshold=args.intensity_threshold,
        min_area=args.min_area,
        expand_ratio=args.expand_ratio,
    )


if __name__ == "__main__":
    main()
