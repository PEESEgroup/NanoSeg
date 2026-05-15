"""
Global Point Flow Segmentation Pipeline

This script performs automatic nanoparticle segmentation using the
Segment Anything Model (SAM) with proposal filtering, overlap removal,
brightness-based refinement, and edge filtering.

Main features
-------------
- SAM automatic mask generation
- Duplicate and merged-mask removal
- Dynamic area-based filtering
- Brightness-aware proposal refinement
- Binary mask export for downstream analysis

The script is designed for reproducible large-scale segmentation
experiments and can be integrated into batch-processing workflows.
"""

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage.measure import label, regionprops

# ----------------------------------------
# Default Config
# These defaults are only for standalone testing.
# Batch traversal should call run_global_point_flow_experiment(...)
# with explicit arguments.
# ----------------------------------------
DEFAULT_SAM_CHECKPOINT = "path/to/checkpoints/sam_vit_b.pth"
DEFAULT_MODEL_TYPE = "vit_b"
DEFAULT_MODEL_TAG = "sam_vit_b"

DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_IMAGE_DIR = "path/to/dataset/images"
DEFAULT_GT_DIR = "path/to/dataset/masks"

# Example only: one ref_select subfolder for standalone testing
DEFAULT_REFERENCE_ID = "1"
DEFAULT_OUTPUT_ROOT = "path/to/output"
DEFAULT_AREA_RANGE_TXT = os.path.join(DEFAULT_OUTPUT_ROOT, "area_range.txt")

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Default processing size
TARGET_SIZE = 512

# SAM parameters
POINTS_PER_SIDE = 64
POINTS_PER_BATCH = 64
PRED_IOU_THRESH = 0.01
STABILITY_SCORE_THRESH = 0.9
CROP_N_LAYERS = 1
CROP_N_POINTS_DOWNSCALE_FACTOR = 2
MIN_MASK_REGION_AREA_SAM = 0

# Dynamic area filtering:
# only use upper bound from txt, new upper = txt_max_area * MAX_AREA_SCALE
MAX_AREA_SCALE = 2.0

# Rotated rectangle expansion ratio
RECT_EXPAND_RATIO = 1.2

# Keep object if:
# target_mean - rect_mean >= DIFF_THRESHOLD
DIFF_THRESHOLD = 0.0

# Edge filtering
BOUNDARY_MARGIN = 5

# -----------------------------
# Overlap / merged-mask handling
# Reproduced from original logic
# -----------------------------
MAX_MASK_AREA = 5000

SMALL_IN_BIG_COV = 0.80
SMALL_SMALL_IOU_MAX = 0.30
MIN_CHILDREN = 2

DEDUP_IOU_THRESH = 0.85
DEDUP_CONTAIN_THRESH = 0.90


# ----------------------------------------
# Model
# ----------------------------------------
def build_mask_generator(
    sam_checkpoint: str,
    model_type: str,
    device: torch.device,
    points_per_side: int = POINTS_PER_SIDE,
    points_per_batch: int = POINTS_PER_BATCH,
    pred_iou_thresh: float = PRED_IOU_THRESH,
    stability_score_thresh: float = STABILITY_SCORE_THRESH,
    crop_n_layers: int = CROP_N_LAYERS,
    crop_n_points_downscale_factor: int = CROP_N_POINTS_DOWNSCALE_FACTOR,
    min_mask_region_area: int = MIN_MASK_REGION_AREA_SAM,
) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        output_mode="binary_mask",
    )
    return mask_generator


# ----------------------------------------
# Basic helpers
# ----------------------------------------
def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("Input mask is None.")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask > 0


def save_binary_mask(mask_bool: np.ndarray, save_path: str) -> None:
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(save_path, mask_uint8)


def resize_if_needed(image: np.ndarray, target_size: int, interp: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_size and w == target_size:
        return image
    return cv2.resize(image, (target_size, target_size), interpolation=interp)


def list_valid_images(image_dir: str) -> List[str]:
    image_files = [
        f for f in sorted(os.listdir(image_dir))
        if f.lower().endswith(VALID_EXTS)
    ]
    return image_files


def load_area_range_txt(txt_path: str) -> Dict[str, Tuple[int, int]]:
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Area range txt not found: {txt_path}")

    area_dict: Dict[str, Tuple[int, int]] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if i == 0 and line.startswith("image_name"):
            continue

        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid line in area range txt: {line}")

        image_name, min_area, max_area = parts
        area_dict[image_name] = (int(min_area), int(max_area))

    return area_dict


# ----------------------------------------
# Overlap / duplicate handling
# Reproduced from original point flow
# ----------------------------------------
def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return inter / float(union)


def _coverage_small_in_big(small: np.ndarray, big: np.ndarray) -> float:
    inter = np.logical_and(small, big).sum()
    denom = small.sum()
    return 0.0 if denom == 0 else inter / float(denom)


def _resize_bool_mask(mask_bool: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if mask_bool.shape == (th, tw):
        return mask_bool
    m = (mask_bool.astype(np.uint8) * 255)
    m_rs = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
    return m_rs > 0


def standardize_ann_masks(anns: List[Dict], target_hw: Tuple[int, int]) -> List[Dict]:
    if len(anns) == 0:
        return anns
    out = []
    for ann in anns:
        m = _resize_bool_mask(ann["segmentation"], target_hw)
        ann2 = dict(ann)
        ann2["segmentation"] = m
        ann2["area"] = int(m.sum())
        out.append(ann2)
    return out


def remove_merged_masks(
    anns: List[Dict],
    max_area: int = MAX_MASK_AREA,
    small_in_big_cov: float = SMALL_IN_BIG_COV,
    small_small_iou_max: float = SMALL_SMALL_IOU_MAX,
    min_children: int = MIN_CHILDREN,
) -> List[Dict]:
    anns = [a for a in anns if a["area"] <= max_area]
    if len(anns) <= 2:
        return anns

    anns = sorted(anns, key=lambda x: x["area"])
    masks = [a["segmentation"] for a in anns]
    areas = [a["area"] for a in anns]

    n = len(anns)
    is_merged = [False] * n

    for j in range(n - 1, -1, -1):
        big = masks[j]
        if areas[j] == 0:
            continue

        children = []
        for i in range(0, j):
            cov = _coverage_small_in_big(masks[i], big)
            if cov >= small_in_big_cov:
                children.append(i)

        if len(children) < min_children:
            continue

        distinct = []
        for idx in children:
            ok = True
            for kept in distinct:
                if _iou_bool(masks[idx], masks[kept]) > small_small_iou_max:
                    ok = False
                    break
            if ok:
                distinct.append(idx)

        if len(distinct) >= min_children:
            is_merged[j] = True

    return [a for k, a in enumerate(anns) if not is_merged[k]]


def dedup_same_object(
    anns: List[Dict],
    iou_thresh: float = DEDUP_IOU_THRESH,
    contain_thresh: float = DEDUP_CONTAIN_THRESH,
) -> List[Dict]:
    if len(anns) <= 1:
        return anns

    anns = sorted(anns, key=lambda x: x["area"])
    kept = []
    kept_masks = []
    kept_areas = []

    for ann in anns:
        m = ann["segmentation"]
        a = ann["area"]

        drop = False
        for km, ka in zip(kept_masks, kept_areas):
            inter = np.logical_and(m, km).sum()
            if inter == 0:
                continue

            union = np.logical_or(m, km).sum()
            iou = inter / float(union)
            if iou >= iou_thresh:
                drop = True
                break

            cov_m_in_k = inter / float(a)
            cov_k_in_m = inter / float(ka)
            if max(cov_m_in_k, cov_k_in_m) >= contain_thresh:
                drop = True
                break

        if not drop:
            kept.append(ann)
            kept_masks.append(m)
            kept_areas.append(a)

    return kept


def make_binary_union_mask(anns: List[Dict], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    out = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        m = ann["segmentation"]
        out[m] = 255
    return out


# ----------------------------------------
# Brightness / geometry filtering
# ----------------------------------------
def contour_object_mean_intensity(single_mask: np.ndarray, image_bgr: np.ndarray) -> float:
    masked_region = cv2.bitwise_and(image_bgr, image_bgr, mask=single_mask)
    valid_pixels = masked_region[masked_region > 0]
    return float(np.mean(valid_pixels)) if len(valid_pixels) > 0 else 0.0


def rotated_rect_mean_intensity(
    contour: np.ndarray,
    image_bgr: np.ndarray,
    expand_ratio: float = RECT_EXPAND_RATIO
) -> float:
    """
    Mean intensity inside an expanded rotated minimum-area rectangle.
    """
    h, w = image_bgr.shape[:2]

    if contour is None or len(contour) < 3:
        return 0.0

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect

    rw = max(rw, 1.0)
    rh = max(rh, 1.0)

    expanded_rect = ((cx, cy), (rw * expand_ratio, rh * expand_ratio), angle)
    box = cv2.boxPoints(expanded_rect)
    box = np.round(box).astype(np.int32)

    rect_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(rect_mask, [box], 255)

    valid_pixels = image_bgr[rect_mask > 0]
    if valid_pixels.size == 0:
        return 0.0

    return float(np.mean(valid_pixels))


def process_mask_new(
    mask_uint8: np.ndarray,
    image_bgr: np.ndarray,
    expand_ratio: float = RECT_EXPAND_RATIO,
    diff_threshold: float = DIFF_THRESHOLD
) -> np.ndarray:
    """
    Keep object if:
        target_mean - rect_mean >= diff_threshold
    """
    _, mask_bin = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks_to_keep = np.zeros_like(mask_bin, dtype=np.uint8)

    for contour in contours:
        single_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)

        target_mean = contour_object_mean_intensity(single_mask, image_bgr)
        rect_mean = rotated_rect_mean_intensity(contour, image_bgr, expand_ratio=expand_ratio)

        if (target_mean - rect_mean) >= diff_threshold:
            cv2.drawContours(masks_to_keep, [contour], -1, 255, thickness=cv2.FILLED)

    return masks_to_keep


def edge_filter(
    mask_uint8: np.ndarray,
    target_size: int = TARGET_SIZE,
    boundary_margin: int = BOUNDARY_MARGIN
) -> np.ndarray:
    labeled_mask = label(mask_uint8 > 0)
    filtered_mask = np.zeros_like(mask_uint8, dtype=np.uint8)

    inner_x_min, inner_x_max = boundary_margin, target_size - boundary_margin
    inner_y_min, inner_y_max = boundary_margin, target_size - boundary_margin

    for region in regionprops(labeled_mask):
        min_row, min_col, max_row, max_col = region.bbox
        if (
            min_row >= inner_y_min and max_row <= inner_y_max and
            min_col >= inner_x_min and max_col <= inner_x_max
        ):
            filtered_mask[labeled_mask == region.label] = 255

    return filtered_mask


# ----------------------------------------
# Main point flow
# ----------------------------------------
def generate_processed_raw_binary_mask(
    image_bgr: np.ndarray,
    max_area_limit: int,
    mask_generator: SamAutomaticMaskGenerator,
    target_size: int = TARGET_SIZE,
    max_mask_area: int = MAX_MASK_AREA,
    small_in_big_cov: float = SMALL_IN_BIG_COV,
    small_small_iou_max: float = SMALL_SMALL_IOU_MAX,
    min_children: int = MIN_CHILDREN,
    dedup_iou_thresh: float = DEDUP_IOU_THRESH,
    dedup_contain_thresh: float = DEDUP_CONTAIN_THRESH,
) -> Tuple[np.ndarray, int]:
    """
    1. resize to target_size
    2. run SAM
    3. standardize mask shapes
    4. remove merged masks
    5. dedup same-object masks
    6. apply dynamic upper-area limit
    7. merge kept proposals into one binary mask
    """
    image_bgr_rs = resize_if_needed(image_bgr, target_size, cv2.INTER_LINEAR)
    image_rgb_rs = cv2.cvtColor(image_bgr_rs, cv2.COLOR_BGR2RGB)
    target_hw = image_bgr_rs.shape[:2]

    anns = mask_generator.generate(image_rgb_rs)

    anns = standardize_ann_masks(anns, target_hw)

    anns = remove_merged_masks(
        anns,
        max_area=max_mask_area,
        small_in_big_cov=small_in_big_cov,
        small_small_iou_max=small_small_iou_max,
        min_children=min_children,
    )

    anns = dedup_same_object(
        anns,
        iou_thresh=dedup_iou_thresh,
        contain_thresh=dedup_contain_thresh,
    )

    anns = [ann for ann in anns if ann["area"] <= max_area_limit]

    if len(anns) == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8), 0

    binary_mask = make_binary_union_mask(anns, target_hw)
    return binary_mask, len(anns)


def run_global_point_flow_on_image(
    image_bgr: np.ndarray,
    max_area_limit: int,
    mask_generator: SamAutomaticMaskGenerator,
    target_size: int = TARGET_SIZE,
    rect_expand_ratio: float = RECT_EXPAND_RATIO,
    diff_threshold: float = DIFF_THRESHOLD,
    boundary_margin: int = BOUNDARY_MARGIN,
    max_mask_area: int = MAX_MASK_AREA,
    small_in_big_cov: float = SMALL_IN_BIG_COV,
    small_small_iou_max: float = SMALL_SMALL_IOU_MAX,
    min_children: int = MIN_CHILDREN,
    dedup_iou_thresh: float = DEDUP_IOU_THRESH,
    dedup_contain_thresh: float = DEDUP_CONTAIN_THRESH,
) -> Tuple[np.ndarray, int]:
    image_bgr_rs = resize_if_needed(image_bgr, target_size, cv2.INTER_LINEAR)

    raw_binary_mask, kept_mask_count = generate_processed_raw_binary_mask(
        image_bgr=image_bgr_rs,
        max_area_limit=max_area_limit,
        mask_generator=mask_generator,
        target_size=target_size,
        max_mask_area=max_mask_area,
        small_in_big_cov=small_in_big_cov,
        small_small_iou_max=small_small_iou_max,
        min_children=min_children,
        dedup_iou_thresh=dedup_iou_thresh,
        dedup_contain_thresh=dedup_contain_thresh,
    )

    new_processed = process_mask_new(
        mask_uint8=raw_binary_mask,
        image_bgr=image_bgr_rs,
        expand_ratio=rect_expand_ratio,
        diff_threshold=diff_threshold
    )

    new_final = edge_filter(
        mask_uint8=new_processed,
        target_size=target_size,
        boundary_margin=boundary_margin
    )

    return to_binary_mask(new_final), kept_mask_count


# ----------------------------------------
# Main experiment function
# ----------------------------------------
def run_global_point_flow_experiment(
    image_dir: str,
    gt_dir: str,
    output_root: str,
    reference_id: str,
    area_range_txt: str,
    sam_checkpoint: str = DEFAULT_SAM_CHECKPOINT,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_tag: str = DEFAULT_MODEL_TAG,
    device: torch.device = DEFAULT_DEVICE,
    target_size: int = TARGET_SIZE,
    points_per_side: int = POINTS_PER_SIDE,
    points_per_batch: int = POINTS_PER_BATCH,
    pred_iou_thresh: float = PRED_IOU_THRESH,
    stability_score_thresh: float = STABILITY_SCORE_THRESH,
    crop_n_layers: int = CROP_N_LAYERS,
    crop_n_points_downscale_factor: int = CROP_N_POINTS_DOWNSCALE_FACTOR,
    min_mask_region_area_sam: int = MIN_MASK_REGION_AREA_SAM,
    max_area_scale: float = MAX_AREA_SCALE,
    rect_expand_ratio: float = RECT_EXPAND_RATIO,
    diff_threshold: float = DIFF_THRESHOLD,
    boundary_margin: int = BOUNDARY_MARGIN,
    max_mask_area: int = MAX_MASK_AREA,
    small_in_big_cov: float = SMALL_IN_BIG_COV,
    small_small_iou_max: float = SMALL_SMALL_IOU_MAX,
    min_children: int = MIN_CHILDREN,
    dedup_iou_thresh: float = DEDUP_IOU_THRESH,
    dedup_contain_thresh: float = DEDUP_CONTAIN_THRESH,
) -> Dict:
    """
    Run the full global point flow for one experiment.

    Output
    ----------
    output_root/
    └── mask/point/*.png
    """
    point_mask_dir = os.path.join(output_root, "mask", "point")
    os.makedirs(point_mask_dir, exist_ok=True)

    area_range_dict = load_area_range_txt(area_range_txt)

    mask_generator = build_mask_generator(
        sam_checkpoint=sam_checkpoint,
        model_type=model_type,
        device=device,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area_sam,
    )

    image_files = list_valid_images(image_dir)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No valid images found in {image_dir}")

    print(f"[INFO] DEVICE                  : {device}")
    print(f"[INFO] MODEL_TYPE              : {model_type}")
    print(f"[INFO] MODEL_TAG               : {model_tag}")
    print(f"[INFO] CHECKPOINT              : {sam_checkpoint}")
    print(f"[INFO] IMAGE_DIR               : {image_dir}")
    print(f"[INFO] GT_DIR                  : {gt_dir}")
    print(f"[INFO] OUTPUT_ROOT             : {output_root}")
    print(f"[INFO] POINT_MASK_DIR          : {point_mask_dir}")
    print(f"[INFO] AREA_RANGE_TXT          : {area_range_txt}")
    print(f"[INFO] REFERENCE_ID            : {reference_id}")
    print(f"[INFO] TARGET_SIZE             : {target_size}")
    print(f"[INFO] POINTS_PER_SIDE         : {points_per_side}")
    print(f"[INFO] POINTS_PER_BATCH        : {points_per_batch}")
    print(f"[INFO] PRED_IOU_THRESH         : {pred_iou_thresh}")
    print(f"[INFO] STABILITY_SCORE_THRESH  : {stability_score_thresh}")
    print(f"[INFO] CROP_N_LAYERS           : {crop_n_layers}")
    print(f"[INFO] CROP_N_POINTS_DOWNSCALE : {crop_n_points_downscale_factor}")
    print(f"[INFO] MAX_AREA_SCALE          : {max_area_scale}")
    print(f"[INFO] RECT_EXPAND_RATIO       : {rect_expand_ratio}")
    print(f"[INFO] DIFF_THRESHOLD          : {diff_threshold}")
    print(f"[INFO] BOUNDARY_MARGIN         : {boundary_margin}")
    print(f"[INFO] MAX_MASK_AREA           : {max_mask_area}")
    print(f"[INFO] SMALL_IN_BIG_COV        : {small_in_big_cov}")
    print(f"[INFO] SMALL_SMALL_IOU_MAX     : {small_small_iou_max}")
    print(f"[INFO] MIN_CHILDREN            : {min_children}")
    print(f"[INFO] DEDUP_IOU_THRESH        : {dedup_iou_thresh}")
    print(f"[INFO] DEDUP_CONTAIN_THRESH    : {dedup_contain_thresh}")
    print(f"[INFO] Total images            : {len(image_files)}")
    print(f"[INFO] Loaded txt items        : {len(area_range_dict)}")

    processed = 0
    skipped = 0
    failed = 0

    for image_name in image_files:
        stem = os.path.splitext(image_name)[0]

        # Skip reference image.
        if stem == str(reference_id):
            print("\n" + "=" * 80)
            print(f"[SKIP] Reference image itself: {image_name}")
            skipped += 1
            continue

        img_path = os.path.join(image_dir, image_name)
        gt_path = os.path.join(gt_dir, image_name)

        print("\n" + "=" * 80)
        print(f"[INFO] Processing image : {image_name}")
        print(f"[INFO] IMAGE : {img_path}")
        print(f"[INFO] GT    : {gt_path}")

        if image_name not in area_range_dict:
            print(f"[WARN] No area range found in txt, skip: {image_name}")
            skipped += 1
            continue

        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found, skip: {gt_path}")
            skipped += 1
            continue

        try:
            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if image_bgr is None:
                raise ValueError(f"Failed to read image: {img_path}")
            if gt_mask is None:
                raise ValueError(f"Failed to read GT: {gt_path}")

            # Load GT for consistency checks.
            _ = resize_if_needed(gt_mask, target_size, cv2.INTER_NEAREST)

            raw_min_area, raw_max_area = area_range_dict[image_name]
            max_area_limit = int(raw_max_area * max_area_scale)

            print("[RUN] Global point flow")
            print(f"[INFO] TXT area range         : [{raw_min_area}, {raw_max_area}]")
            print(f"[INFO] Dynamic upper area     : {max_area_limit}")

            pred_mask_bool, kept_mask_count = run_global_point_flow_on_image(
                image_bgr=image_bgr,
                max_area_limit=max_area_limit,
                mask_generator=mask_generator,
                target_size=target_size,
                rect_expand_ratio=rect_expand_ratio,
                diff_threshold=diff_threshold,
                boundary_margin=boundary_margin,
                max_mask_area=max_mask_area,
                small_in_big_cov=small_in_big_cov,
                small_small_iou_max=small_small_iou_max,
                min_children=min_children,
                dedup_iou_thresh=dedup_iou_thresh,
                dedup_contain_thresh=dedup_contain_thresh,
            )

            print(f"[INFO] Final kept proposals   : {kept_mask_count}")

            save_path = os.path.join(point_mask_dir, stem + ".png")
            save_binary_mask(pred_mask_bool, save_path)
            print(f"[SAVED] mask: {save_path}")

            processed += 1

        except Exception as e:
            print(f"[ERROR] Failed on {image_name}: {e}")
            failed += 1
            continue

    print("\n" + "=" * 80)
    print("[DONE] Global point flow finished.")
    print(f"[SUMMARY] processed = {processed}")
    print(f"[SUMMARY] skipped   = {skipped}")
    print(f"[SUMMARY] failed    = {failed}")

    return {
        "reference_id": str(reference_id),
        "model_tag": model_tag,
        "sam_checkpoint": sam_checkpoint,
        "model_type": model_type,
        "output_root": output_root,
        "point_mask_dir": point_mask_dir,
        "area_range_txt": area_range_txt,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }


# ----------------------------------------
# Standalone direct run
# ----------------------------------------
def main():
    """
    Standalone example entry point.
    """
    run_global_point_flow_experiment(
        image_dir=DEFAULT_IMAGE_DIR,
        gt_dir=DEFAULT_GT_DIR,
        output_root=DEFAULT_OUTPUT_ROOT,
        reference_id=DEFAULT_REFERENCE_ID,
        area_range_txt=DEFAULT_AREA_RANGE_TXT,
        sam_checkpoint=DEFAULT_SAM_CHECKPOINT,
        model_type=DEFAULT_MODEL_TYPE,
        model_tag=DEFAULT_MODEL_TAG,
        device=DEFAULT_DEVICE,
        target_size=TARGET_SIZE,
        points_per_side=POINTS_PER_SIDE,
        points_per_batch=POINTS_PER_BATCH,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        crop_n_layers=CROP_N_LAYERS,
        crop_n_points_downscale_factor=CROP_N_POINTS_DOWNSCALE_FACTOR,
        min_mask_region_area_sam=MIN_MASK_REGION_AREA_SAM,
        max_area_scale=MAX_AREA_SCALE,
        rect_expand_ratio=RECT_EXPAND_RATIO,
        diff_threshold=DIFF_THRESHOLD,
        boundary_margin=BOUNDARY_MARGIN,
        max_mask_area=MAX_MASK_AREA,
        small_in_big_cov=SMALL_IN_BIG_COV,
        small_small_iou_max=SMALL_SMALL_IOU_MAX,
        min_children=MIN_CHILDREN,
        dedup_iou_thresh=DEDUP_IOU_THRESH,
        dedup_contain_thresh=DEDUP_CONTAIN_THRESH,
    )


if __name__ == "__main__":
    main()