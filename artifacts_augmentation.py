"""
Artifact Robustness Segmentation Benchmark

This script evaluates NanoSeg segmentation robustness under multiple
synthetic image perturbations and artifact conditions.

Main features
-------------
- Augmented dataset evaluation
- Perturbation-wise robustness benchmarking
- Prompt-flow and point-flow integration
- Quantitative metric aggregation
- Visualization montage generation

The pipeline measures segmentation stability across noise, blur,
scan distortion, background variation, and combined artifacts.
"""

import csv
import os
import shutil
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import label, regionprops

from global_point_flow import run_global_point_flow_experiment
from metrics import compute_metrics
from prompt_flow import (
    build_predictor,
    list_valid_images as prompt_list_valid_images,
    resize_image_and_gt,
    to_binary_mask as prompt_to_binary_mask,
    run_prompt_flow_on_image,
    get_area_range_from_components,
    save_binary_mask as save_binary_mask_prompt,
    write_area_range_txt,
)

# ----------------------------------------
# Config
# ----------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Raw images used for bbox coordinate scaling.
ORIG_IMAGE_DIR = "path/to/raw_images"

# Augmented image dataset.
AUG_IMAGE_DIR = "path/to/augmented_images"

# GT
GT_DIR = "path/to/gt_masks"

# Reference prompt directory.
BBOX_PROMPT_DIR = "path/to/prompts/reference_28"
REFERENCE_ID = "28"

# Output roots
OUTPUT_ROOT = "path/to/augmented_images"
TEST_ROOT = os.path.join(OUTPUT_ROOT, "test")
SUMMARY_CSV = os.path.join(OUTPUT_ROOT, "summary_segmentation.csv")

# Temp working root
TMP_ROOT = os.path.join(OUTPUT_ROOT, "_tmp_seg_eval")

# For test visualization
VIS_IMAGE_ID = "23"

# Model config
SAM_CHECKPOINT = "path/to/checkpoints/sam_vit_b.pth"
MODEL_TYPE = "vit_b"
MODEL_TAG = "sam_vit_b"

TARGET_SIZE = 512
BATCH_SIZE = 2
BOUNDARY_MARGIN = 5
ALPHA = 0.55
RNG_SEED = 0

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Perturbation evaluation order.
PERTURBATION_ORDER = [
    ("raw", "_raw"),
    ("shot", "_shot"),
    ("gaus", "_gaus"),
    ("scan", "_scan"),
    ("sig", "_sig"),
    ("bgfield", "_bgfield"),
    ("defocus", "_defocus"),
    ("combo_shot_gaus", "_combo_shot_gaus"),
    ("combo_shot_gaus_scan", "_combo_shot_gaus_scan"),
    ("combo_shot_gaus_scan_sig", "_combo_shot_gaus_scan_sig"),
    ("combo_shot_gaus_scan_sig_bgfield", "_combo_shot_gaus_scan_sig_bgfield"),
    ("combo_all", "_combo_all"),
]

METRIC_COLUMNS = ["IoU", "Dice", "Precision", "Recall"]


# ----------------------------------------
# Basic helpers
# ----------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sorted_files(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTS)]
    return sorted(files)


def resize_img(img, interp):
    return cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=interp)


def to_bool(mask):
    return mask > 0


def save_mask(mask_bool: np.ndarray, path: str):
    cv2.imwrite(path, (mask_bool.astype(np.uint8) * 255))


def get_stem(fname: str) -> str:
    return os.path.splitext(os.path.basename(fname))[0]


def parse_augmented_filename(fname: str) -> Tuple[str, str]:
    stem = get_stem(fname)

    for key, suffix in sorted(PERTURBATION_ORDER, key=lambda x: len(x[1]), reverse=True):
        if stem.endswith(suffix):
            base_id = stem[: -len(suffix)]
            return base_id, key

    raise ValueError(f"Unrecognized augmented filename pattern: {fname}")


def find_file_by_stem(folder: str, stem: str) -> str:
    for ext in VALID_EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"File not found for stem={stem} in {folder}")


def list_augmented_by_perturbation(image_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    out = {k: [] for k, _ in PERTURBATION_ORDER}
    for fname in sorted_files(image_dir):
        base_id, pkey = parse_augmented_filename(fname)
        full_path = os.path.join(image_dir, fname)
        out[pkey].append((base_id, full_path))
    return out


# ----------------------------------------
# Visualization helpers
# ----------------------------------------
def get_font(size=24):
    font_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_text_with_bg_pil(
    img_bgr,
    text,
    xy,
    font_size=24,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    pad=4,
    anchor=None,
):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size)

    if anchor is None:
        bbox = draw.textbbox(xy, text, font=font)
        x1 = max(0, bbox[0] - pad)
        y1 = max(0, bbox[1] - pad)
        x2 = min(pil_img.width - 1, bbox[2] + pad)
        y2 = min(pil_img.height - 1, bbox[3] + pad)
        draw.rectangle([x1, y1, x2, y2], fill=bg_color)
        draw.text(xy, text, font=font, fill=text_color)
    else:
        bbox = draw.textbbox(xy, text, font=font, anchor=anchor)
        x1 = max(0, bbox[0] - pad)
        y1 = max(0, bbox[1] - pad)
        x2 = min(pil_img.width - 1, bbox[2] + pad)
        y2 = min(pil_img.height - 1, bbox[3] + pad)
        draw.rectangle([x1, y1, x2, y2], fill=bg_color)
        draw.text(xy, text, font=font, fill=text_color, anchor=anchor)

    out_rgb = np.array(pil_img)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def mask_to_anns(mask_bool):
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    anns = []
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        comp_mask = labels == lab
        anns.append({
            "segmentation": comp_mask,
            "area": area,
            "label": lab,
        })
    return anns


def make_color_overlay_bgr(img_bgr: np.ndarray, anns, alpha=0.55, seed=0):
    if len(anns) == 0:
        return img_bgr.copy()

    rng = np.random.default_rng(seed)
    overlay = img_bgr.copy().astype(np.float32)

    anns_sorted = sorted(anns, key=lambda x: x.get("area", 0), reverse=True)
    for ann in anns_sorted:
        m = ann["segmentation"]
        color = rng.integers(low=0, high=256, size=(3,), dtype=np.uint8)
        overlay[m] = (1 - alpha) * overlay[m] + alpha * color.astype(np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_difference_map(pred_mask_bool, gt_mask_bool):
    h, w = gt_mask_bool.shape
    diff_img = np.full((h, w, 3), 255, dtype=np.uint8)

    tp = pred_mask_bool & gt_mask_bool
    fp = pred_mask_bool & (~gt_mask_bool)
    fn = (~pred_mask_bool) & gt_mask_bool

    tp_color_bgr = (220, 220, 220)
    fp_color_bgr = (241, 198, 81)
    fn_color_bgr = (0, 128, 255)

    diff_img[tp] = tp_color_bgr
    diff_img[fp] = fp_color_bgr
    diff_img[fn] = fn_color_bgr

    return diff_img


# ----------------------------------------
# Merge logic
# ----------------------------------------
def edge_filter(mask):
    labeled = label(mask)
    out = np.zeros_like(mask, dtype=np.uint8)

    for r in regionprops(labeled):
        minr, minc, maxr, maxc = r.bbox
        if (
            minr >= BOUNDARY_MARGIN and
            maxr <= TARGET_SIZE - BOUNDARY_MARGIN and
            minc >= BOUNDARY_MARGIN and
            maxc <= TARGET_SIZE - BOUNDARY_MARGIN
        ):
            out[labeled == r.label] = 1

    return out.astype(bool)


def get_components(mask):
    labeled = label(mask)
    comps = []
    for r in regionprops(labeled):
        m = labeled == r.label
        comps.append({
            "mask": m,
            "area": r.area,
        })
    return comps


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return inter / union


def brightness_score(mask, image_bgr):
    if image_bgr.ndim == 3:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        image = image_bgr

    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0

    c = contours[0]
    target_pixels = image[mask]
    target_mean = np.mean(target_pixels) if len(target_pixels) > 0 else 0.0

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    rect_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(rect_mask, [box], 255)

    bg_pixels = image[rect_mask > 0]
    rect_mean = np.mean(bg_pixels) if len(bg_pixels) > 0 else 0.0

    return target_mean - rect_mean


def merge_masks(mask_a, mask_b, image_bgr):
    comps_a = get_components(mask_a)
    comps_b = get_components(mask_b)

    used_b = set()
    final = []

    for ca in comps_a:
        best_j = -1
        best_iou = 0.0

        for j, cb in enumerate(comps_b):
            ov = iou(ca["mask"], cb["mask"])
            if ov > best_iou:
                best_iou = ov
                best_j = j

        if best_iou < 0.1:
            final.append(ca["mask"])
        else:
            cb = comps_b[best_j]
            used_b.add(best_j)

            s1 = brightness_score(ca["mask"], image_bgr)
            s2 = brightness_score(cb["mask"], image_bgr)
            final.append(ca["mask"] if s1 >= s2 else cb["mask"])

    for j, cb in enumerate(comps_b):
        if j not in used_b:
            final.append(cb["mask"])

    out = np.zeros_like(mask_a, dtype=bool)
    for m in final:
        out |= m

    return out


# ----------------------------------------
# Evaluation helpers
# ----------------------------------------
def summarize_metrics(records: List[Dict]) -> Dict:
    df = pd.DataFrame(records)
    out = {}
    for col in METRIC_COLUMNS:
        out[col] = float(df[col].mean()) if len(df) > 0 else np.nan
    return out


def write_summary_csv(rows: List[Dict], out_csv: str):
    fieldnames = ["perturbation"] + METRIC_COLUMNS + ["n_images"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------
# Temp dataset builder
# ----------------------------------------
def build_temp_dataset_for_perturbation(
    perturbation_key: str,
    pairs: List[Tuple[str, str]],
    tmp_root: str,
) -> str:
    work_dir = tempfile.mkdtemp(prefix=f"{perturbation_key}_", dir=tmp_root)
    image_dir = os.path.join(work_dir, "images")
    ensure_dir(image_dir)

    for base_id, aug_path in pairs:
        img = cv2.imread(aug_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read augmented image: {aug_path}")

        img = resize_img(img, cv2.INTER_LINEAR)
        out_path = os.path.join(image_dir, f"{base_id}.png")
        cv2.imwrite(out_path, img)

    return work_dir


# ----------------------------------------
# Prompt flow using original-image coordinate scaling.
# ----------------------------------------
def run_prompt_flow_experiment_fixed(
    image_dir: str,
    gt_dir: str,
    orig_image_dir: str,
    prompt_dir: str,
    output_root: str,
    reference_id: str,
    sam_checkpoint: str,
    model_type: str,
    model_tag: str,
    device: torch.device,
    target_size: Tuple[int, int] = (512, 512),
    batch_size: int = 2,
) -> Dict:
    """
    Same logic as run_prompt_flow_experiment, but bbox scaling uses the ORIGINAL raw image size
    from orig_image_dir, not the augmented image size.
    """
    box_mask_dir = os.path.join(output_root, "mask", "box")
    area_range_txt = os.path.join(output_root, "area_range.txt")
    ensure_dir(box_mask_dir)

    predictor = build_predictor(
        sam_checkpoint=sam_checkpoint,
        model_type=model_type,
        device=device,
    )

    image_files = prompt_list_valid_images(image_dir)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No valid images found in {image_dir}")

    processed = 0
    skipped = 0
    failed = 0
    area_lines = []

    for image_name in image_files:
        stem = os.path.splitext(image_name)[0]

        if stem == str(reference_id):
            skipped += 1
            continue

        aug_img_path = os.path.join(image_dir, image_name)
        gt_path = os.path.join(gt_dir, image_name)
        prompt_path = os.path.join(prompt_dir, stem + ".txt")

        if not os.path.exists(gt_path):
            skipped += 1
            continue
        if not os.path.exists(prompt_path):
            skipped += 1
            continue

        try:
            # augmented image used for actual segmentation
            image_bgr = cv2.imread(aug_img_path, cv2.IMREAD_COLOR)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if image_bgr is None:
                raise ValueError(f"Failed to read augmented image: {aug_img_path}")
            if gt_mask is None:
                raise ValueError(f"Failed to read GT: {gt_path}")

            # original raw image used ONLY to obtain original size for bbox scaling
            orig_img_path = find_file_by_stem(orig_image_dir, stem)
            orig_image_bgr = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
            if orig_image_bgr is None:
                raise ValueError(f"Failed to read original raw image: {orig_img_path}")

            orig_h, orig_w = orig_image_bgr.shape[:2]

            image_bgr_512, gt_mask_512 = resize_image_and_gt(
                image_bgr=image_bgr,
                gt_mask=gt_mask,
                target_size=target_size,
            )

            _ = prompt_to_binary_mask(gt_mask_512)

            box_mask_bool = run_prompt_flow_on_image(
                predictor=predictor,
                image_bgr_512=image_bgr_512,
                prompt_path=prompt_path,
                orig_w=orig_w,
                orig_h=orig_h,
                batch_size=batch_size,
                device=device,
            )

            min_area, max_area, components = get_area_range_from_components(box_mask_bool)
            if min_area is None or max_area is None:
                skipped += 1
                continue

            mask_path = os.path.join(box_mask_dir, stem + ".png")
            save_binary_mask_prompt(box_mask_bool, mask_path)

            area_lines.append(f"{image_name} {int(min_area)} {int(max_area)}\n")
            processed += 1

        except Exception:
            failed += 1
            continue

    write_area_range_txt(area_lines, area_range_txt)

    return {
        "reference_id": str(reference_id),
        "model_tag": model_tag,
        "sam_checkpoint": sam_checkpoint,
        "model_type": model_type,
        "output_root": output_root,
        "box_mask_dir": box_mask_dir,
        "area_range_txt": area_range_txt,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }


# ----------------------------------------
# Main per-perturbation runner
# ----------------------------------------
def run_one_perturbation(
    perturbation_key: str,
    pairs: List[Tuple[str, str]],
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    if len(pairs) == 0:
        raise ValueError(f"No images found for perturbation: {perturbation_key}")

    ensure_dir(TMP_ROOT)
    work_dir = build_temp_dataset_for_perturbation(perturbation_key, pairs, TMP_ROOT)
    image_dir = os.path.join(work_dir, "images")
    output_root = os.path.join(work_dir, "result")

    try:
        # 1) FIXED prompt flow
        prompt_result = run_prompt_flow_experiment_fixed(
            image_dir=image_dir,
            gt_dir=GT_DIR,
            orig_image_dir=ORIG_IMAGE_DIR,
            prompt_dir=BBOX_PROMPT_DIR,
            output_root=output_root,
            reference_id=REFERENCE_ID,
            sam_checkpoint=SAM_CHECKPOINT,
            model_type=MODEL_TYPE,
            model_tag=MODEL_TAG,
            device=DEVICE,
            target_size=(TARGET_SIZE, TARGET_SIZE),
            batch_size=BATCH_SIZE,
        )

        # 2) Global point-flow segmentation.
        run_global_point_flow_experiment(
            image_dir=image_dir,
            gt_dir=GT_DIR,
            output_root=output_root,
            reference_id=REFERENCE_ID,
            area_range_txt=prompt_result["area_range_txt"],
            sam_checkpoint=SAM_CHECKPOINT,
            model_type=MODEL_TYPE,
            model_tag=MODEL_TAG,
            device=DEVICE,
            target_size=TARGET_SIZE,
        )

        # 3) Merge predictions and compute metrics.
        prompt_mask_dir = os.path.join(output_root, "mask", "box")
        point_mask_dir = os.path.join(output_root, "mask", "point")

        files = sorted_files(prompt_mask_dir)
        metric_records = []
        vis_cache = {}

        for f in files:
            base_id = get_stem(f)

            img_path = os.path.join(image_dir, f)
            box_path = os.path.join(prompt_mask_dir, f)
            point_path = os.path.join(point_mask_dir, f)
            gt_path = find_file_by_stem(GT_DIR, base_id)

            if not os.path.exists(img_path):
                continue
            if not os.path.exists(point_path):
                continue
            if not os.path.exists(gt_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            box = cv2.imread(box_path, cv2.IMREAD_GRAYSCALE)
            point = cv2.imread(point_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if img is None or box is None or point is None or gt is None:
                continue

            img = resize_img(img, cv2.INTER_LINEAR)
            box = resize_img(box, cv2.INTER_NEAREST)
            point = resize_img(point, cv2.INTER_NEAREST)
            gt = resize_img(gt, cv2.INTER_NEAREST)

            box = edge_filter(to_bool(box))
            point = edge_filter(to_bool(point))
            gt = edge_filter(to_bool(gt))

            nano = merge_masks(box, point, img)
            nano = edge_filter(nano)

            nano_u8 = (nano.astype(np.uint8) * 255)
            gt_u8 = (gt.astype(np.uint8) * 255)

            m = compute_metrics(nano_u8, gt_u8, DEVICE)
            metric_records.append({
                "image": base_id,
                "IoU": m["IoU"],
                "Dice": m["Dice"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
            })

            if base_id == VIS_IMAGE_ID:
                vis_cache["img"] = img
                vis_cache["mask"] = nano
                vis_cache["gt"] = gt

        summary = summarize_metrics(metric_records)
        summary_row = {
            "perturbation": perturbation_key,
            **summary,
            "n_images": len(metric_records),
        }

        return summary_row, vis_cache

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ----------------------------------------
# Save visualization examples.
# ----------------------------------------
def save_test_visuals(all_vis: Dict[str, Dict[str, np.ndarray]]):
    mask_dir = os.path.join(TEST_ROOT, "mask")
    overlap_dir = os.path.join(TEST_ROOT, "overlap")
    diff_dir = os.path.join(TEST_ROOT, "difference")

    ensure_dir(TEST_ROOT)
    ensure_dir(mask_dir)
    ensure_dir(overlap_dir)
    ensure_dir(diff_dir)

    montage_tiles = []

    for perturbation_key, _suffix in PERTURBATION_ORDER:
        if perturbation_key not in all_vis:
            continue

        cache = all_vis[perturbation_key]
        if not cache:
            continue

        img = cache["img"]
        mask = cache["mask"]
        gt = cache["gt"]

        save_name = f"{VIS_IMAGE_ID}_{perturbation_key}.png"

        save_mask(mask, os.path.join(mask_dir, save_name))

        anns = mask_to_anns(mask)
        overlap = make_color_overlay_bgr(img, anns, alpha=ALPHA, seed=RNG_SEED)
        cv2.imwrite(os.path.join(overlap_dir, save_name), overlap)

        diff_img = make_difference_map(mask, gt)
        cv2.imwrite(os.path.join(diff_dir, save_name), diff_img)

        tile = overlap.copy()
        tile = draw_text_with_bg_pil(
            tile,
            perturbation_key,
            (10, 10),
            font_size=24,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0),
            pad=4,
        )
        montage_tiles.append(tile)

    # 4x3 montage
    n_cols = 4
    n_rows = 3
    gap = 12
    tile_w = TARGET_SIZE
    tile_h = TARGET_SIZE

    canvas_w = n_cols * tile_w + (n_cols - 1) * gap
    canvas_h = n_rows * tile_h + (n_rows - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for idx, tile in enumerate(montage_tiles):
        r = idx // n_cols
        c = idx % n_cols
        if r >= n_rows:
            break
        x0 = c * (tile_w + gap)
        y0 = r * (tile_h + gap)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    cv2.imwrite(os.path.join(TEST_ROOT, "montage.png"), canvas)


# ----------------------------------------
# Main
# ----------------------------------------
def main():
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(TEST_ROOT)
    ensure_dir(TMP_ROOT)

    grouped = list_augmented_by_perturbation(AUG_IMAGE_DIR)

    summary_rows = []
    vis_results = {}

    print("=" * 100)
    print("[INFO] DEVICE:", DEVICE)
    print("[INFO] ORIG_IMAGE_DIR:", ORIG_IMAGE_DIR)
    print("[INFO] AUG_IMAGE_DIR:", AUG_IMAGE_DIR)
    print("[INFO] GT_DIR:", GT_DIR)
    print("[INFO] BBOX_PROMPT_DIR:", BBOX_PROMPT_DIR)
    print("[INFO] REFERENCE_ID:", REFERENCE_ID)
    print("[INFO] VIS_IMAGE_ID:", VIS_IMAGE_ID)
    print("=" * 100)

    for perturbation_key, _suffix in PERTURBATION_ORDER:
        pairs = grouped.get(perturbation_key, [])
        if len(pairs) == 0:
            print(f"[WARN] Skip {perturbation_key}: no images found.")
            continue

        print(f"\n[RUN] perturbation = {perturbation_key} | n_images = {len(pairs)}")
        row, vis_cache = run_one_perturbation(perturbation_key, pairs)
        summary_rows.append(row)

        if vis_cache and ("img" in vis_cache):
            vis_results[perturbation_key] = vis_cache

        print(
            f"[DONE] {perturbation_key} | "
            f"Dice={row['Dice']:.6f}, IoU={row['IoU']:.6f}, "
            f"Precision={row['Precision']:.6f}, Recall={row['Recall']:.6f}, "
            f"n={row['n_images']}"
        )

    write_summary_csv(summary_rows, SUMMARY_CSV)
    print(f"\n[SAVED] summary csv -> {SUMMARY_CSV}")

    save_test_visuals(vis_results)
    print(f"[SAVED] test visuals -> {TEST_ROOT}")

    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()