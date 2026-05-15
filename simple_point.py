"""
Single-Point SAM Segmentation Benchmark

This script evaluates segmentation performance using independent
single-point prompts with the Segment Anything Model.

Main features
-------------
- GT-centroid point generation
- Independent SAM point prompting
- Multi-mask selection
- Segmentation metric evaluation
- Difference-map generation
- Visualization montage export

The framework serves as a simple point-prompt baseline for
NanoSeg segmentation benchmarking.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from segment_anything import sam_model_registry, SamPredictor
from skimage.measure import label, regionprops

from metrics import compute_metrics

# ----------------------------------------
# Config
# ----------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "path/to/dataset/images"
GT_DIR = "path/to/dataset/masks"

POINT_TXT_DIR = "path/to/output/simple_point_prompts"

RESULT_ROOT = "path/to/output"
MASK_DIR = os.path.join(RESULT_ROOT, "mask", "simple_point")
OVERLAP_DIR = os.path.join(RESULT_ROOT, "overlap", "simple_point")
DIFF_DIR = os.path.join(RESULT_ROOT, "difference", "simple_point")
METRICS_CSV = os.path.join(RESULT_ROOT, "metrics", "simple_point.csv")

SAM_CHECKPOINT = "path/to/checkpoints/sam_vit_b.pth"
MODEL_TYPE = "vit_b"
MODEL_TAG = "sam_vit_b"

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
TARGET_SIZE = 512
BOUNDARY_MARGIN = 5

# Single-point prompt configuration.
MULTIMASK_OUTPUT = True

# Point visualization settings.
POINT_RADIUS = 3
POINT_COLOR_BGR = (0, 255, 0)  # green

# Difference-map visualization colors.
#
#
TP_COLOR = (0, 255, 0)
FP_COLOR = (255, 0, 0)
FN_COLOR = (0, 0, 255)
BG_COLOR = (255, 255, 255)

METRIC_COLUMNS = [
    "IoU",
    "Dice",
    "Precision",
    "Recall",
    "Specificity",
    "BalancedAcc",
    "MCC",
    "BF1_tau2",
    "HD95",
    "ASD",
]

# Montage config
MONTAGE_COLS = 4
MONTAGE_ROWS = 5
IMAGES_PER_MONTAGE = 10   # Each image contributes point-overlay and difference tiles.
TOTAL_MONTAGES = 4

CELL_GAP = 16
OUTER_PAD = 24

LABEL_FONT_SIZE = 28
LABEL_MARGIN_X = 12
LABEL_MARGIN_Y = 8

TITLE_FONT_SIZE = 24
TITLE_MARGIN_X = 12
TITLE_MARGIN_Y = 44


# ----------------------------------------
# Helpers
# ----------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_valid_images(image_dir: str) -> List[str]:
    files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(VALID_EXTS)
    ]
    return sorted(files, key=try_parse_int_stem_name)


def try_parse_int_stem_name(name: str):
    stem = Path(name).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem.lower())


def resize_image(img: np.ndarray, interp: int) -> np.ndarray:
    return cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=interp)


def to_bool(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask > 0


def save_binary_mask(mask_bool: np.ndarray, save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(save_path, mask_u8)


def edge_filter(mask_bool: np.ndarray, boundary_margin: int = BOUNDARY_MARGIN) -> np.ndarray:
    labeled = label(mask_bool)
    out = np.zeros_like(mask_bool, dtype=np.uint8)

    for r in regionprops(labeled):
        minr, minc, maxr, maxc = r.bbox
        if (
            minr >= boundary_margin and
            maxr <= TARGET_SIZE - boundary_margin and
            minc >= boundary_margin and
            maxc <= TARGET_SIZE - boundary_margin
        ):
            out[labeled == r.label] = 1

    return out.astype(bool)


def build_predictor(
    sam_checkpoint: str = SAM_CHECKPOINT,
    model_type: str = MODEL_TYPE,
    device: torch.device = DEVICE,
) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    return predictor


# ----------------------------------------
# Point generation from GT
# ----------------------------------------
def extract_centroids_from_gt(gt_mask_512: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract GT centroids from resized masks.
    - find connected components
    - take centroid of each component
    - return integer (x, y) points
    """
    gt_bool = to_bool(gt_mask_512)
    labeled = label(gt_bool)
    props = regionprops(labeled)

    points: List[Tuple[int, int]] = []
    for r in props:
        cy, cx = r.centroid
        x = int(round(cx))
        y = int(round(cy))

        x = max(0, min(TARGET_SIZE - 1, x))
        y = max(0, min(TARGET_SIZE - 1, y))
        points.append((x, y))

    return points


def save_points_txt(points: List[Tuple[int, int]], save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        for x, y in points:
            f.write(f"{x} {y}\n")


def draw_points_on_image(image_bgr_512: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
    vis = image_bgr_512.copy()
    for x, y in points:
        cv2.circle(vis, (x, y), POINT_RADIUS, POINT_COLOR_BGR, thickness=-1, lineType=cv2.LINE_AA)
    return vis


# ----------------------------------------
# SAM point prompting
# ----------------------------------------
def predict_mask_from_points(
    predictor: SamPredictor,
    image_bgr_512: np.ndarray,
    points: List[Tuple[int, int]],
    multimask_output: bool = MULTIMASK_OUTPUT,
) -> np.ndarray:
    """
    Independent point-prompt inference.
    - each point is prompted independently
    - choose the best-scored mask for that point
    - final prediction is the union of all masks
    """
    image_rgb = cv2.cvtColor(image_bgr_512, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    merged = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)

    for x, y in points:
        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
            return_logits=False,
        )

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx] > 0
        merged |= best_mask

    return merged


# ----------------------------------------
# Overlap / Difference
# ----------------------------------------
def generate_component_palette(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate deterministic visualization colors.
    Returned in RGB order.
    """
    if n <= 0:
        return []

    palette = []
    for i in range(n):
        hue = (i * 137) % 180
        hsv = np.uint8([[[hue, 220, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        palette.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return palette


def make_overlap_image(pred_bool: np.ndarray) -> np.ndarray:
    """
    Generate instance-colored overlap visualization.
    each connected component gets a different RGB color
    background is black
    """
    labeled = label(pred_bool)
    props = regionprops(labeled)
    palette = generate_component_palette(len(props))

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)

    for idx, r in enumerate(props):
        color = palette[idx]
        canvas[labeled == r.label] = color

    return canvas


def make_diff(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Generate difference map.
      TP = green
      FP = red
      FN = blue
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    img = np.ones((*gt.shape, 3), dtype=np.uint8) * 255

    tp = pred & gt
    fp = pred & (~gt)
    fn = (~pred) & gt

    img[tp] = (220, 220, 220)
    img[fp] = (241, 198, 81)
    img[fn] = (0, 128, 255)

    return img


def save_rgb_image(rgb_img: np.ndarray, save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    Image.fromarray(rgb_img).save(save_path)


# ----------------------------------------
# Metrics CSV
# ----------------------------------------
def save_metrics_csv_with_mean_std(records: List[Dict], out_path: str) -> pd.DataFrame:
    if len(records) == 0:
        raise ValueError("No records available to save metrics csv.")

    df = pd.DataFrame(records)
    df = df[["image"] + METRIC_COLUMNS]

    mean_row = {"image": "mean"}
    std_row = {"image": "std"}

    for col in METRIC_COLUMNS:
        mean_row[col] = df[col].mean()
        std_row[col] = df[col].std(ddof=0)

    df = pd.concat(
        [df, pd.DataFrame([mean_row]), pd.DataFrame([std_row])],
        ignore_index=True
    )

    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, index=False)
    return df


# ----------------------------------------
# Montage
# ----------------------------------------
def load_font(font_size: int = 28):
    candidate_fonts = [
        "arial.ttf",
        "Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidate_fonts:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font) -> None:
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def save_pair_montages(
    image_names: List[str],
    point_overlay_dir: str,
    diff_dir: str,
    save_dir: str,
) -> None:
    """
    Each source image contributes 2 tiles:
      1) point overlay
      2) difference
    Layout:
      4 cols x 5 rows = 20 tiles = 10 source images per montage
    Total for 33 images => 4 montages
    """
    ensure_dir(save_dir)

    label_font = load_font(LABEL_FONT_SIZE)
    title_font = load_font(TITLE_FONT_SIZE)

    cell_w = TARGET_SIZE
    cell_h = TARGET_SIZE

    tiles_per_montage = MONTAGE_COLS * MONTAGE_ROWS  # 20
    assert tiles_per_montage == 20

    for montage_idx in range(TOTAL_MONTAGES):
        start = montage_idx * IMAGES_PER_MONTAGE
        end = min(start + IMAGES_PER_MONTAGE, len(image_names))
        subset = image_names[start:end]

        grid_w = MONTAGE_COLS * cell_w + (MONTAGE_COLS - 1) * CELL_GAP
        grid_h = MONTAGE_ROWS * cell_h + (MONTAGE_ROWS - 1) * CELL_GAP
        canvas_w = grid_w + 2 * OUTER_PAD
        canvas_h = grid_h + 2 * OUTER_PAD

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        tile_idx = 0
        for image_name in subset:
            stem = Path(image_name).stem

            point_path = os.path.join(point_overlay_dir, image_name)
            diff_path = os.path.join(diff_dir, image_name)

            pair = [
                (point_path, f"{stem} point"),
                (diff_path, f"{stem} diff"),
            ]

            for img_path, title in pair:
                if tile_idx >= tiles_per_montage:
                    break

                r = tile_idx // MONTAGE_COLS
                c = tile_idx % MONTAGE_COLS

                x0 = OUTER_PAD + c * (cell_w + CELL_GAP)
                y0 = OUTER_PAD + r * (cell_h + CELL_GAP)

                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                else:
                    img = Image.new("RGB", (cell_w, cell_h), color=(255, 255, 255))

                canvas.paste(img, (x0, y0))
                draw_label(draw, title, x0 + LABEL_MARGIN_X, y0 + LABEL_MARGIN_Y, title_font)
                tile_idx += 1

        save_path = os.path.join(save_dir, f"montage_{montage_idx + 1}.png")
        canvas.save(save_path)
        print(f"[SAVED] {save_path}")


# ----------------------------------------
# Main
# ----------------------------------------
def run_simple_point_experiment() -> Dict:
    ensure_dir(POINT_TXT_DIR)
    ensure_dir(MASK_DIR)
    ensure_dir(OVERLAP_DIR)
    ensure_dir(DIFF_DIR)

    point_overlay_dir = os.path.join(DIFF_DIR, "_point_overlay_tmp")
    ensure_dir(point_overlay_dir)

    predictor = build_predictor(
        sam_checkpoint=SAM_CHECKPOINT,
        model_type=MODEL_TYPE,
        device=DEVICE,
    )

    image_files = list_valid_images(IMAGE_DIR)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No valid images found in: {IMAGE_DIR}")

    print("=" * 100)
    print("[SIMPLE POINT START]")
    print(f"[INFO] DEVICE         : {DEVICE}")
    print(f"[INFO] CHECKPOINT     : {SAM_CHECKPOINT}")
    print(f"[INFO] MODEL_TYPE     : {MODEL_TYPE}")
    print(f"[INFO] MODEL_TAG      : {MODEL_TAG}")
    print(f"[INFO] IMAGE_DIR      : {IMAGE_DIR}")
    print(f"[INFO] GT_DIR         : {GT_DIR}")
    print(f"[INFO] POINT_TXT_DIR  : {POINT_TXT_DIR}")
    print(f"[INFO] MASK_DIR       : {MASK_DIR}")
    print(f"[INFO] OVERLAP_DIR    : {OVERLAP_DIR}")
    print(f"[INFO] DIFF_DIR       : {DIFF_DIR}")
    print(f"[INFO] METRICS_CSV    : {METRICS_CSV}")
    print(f"[INFO] TARGET_SIZE    : {TARGET_SIZE}")
    print(f"[INFO] MULTIMASK      : {MULTIMASK_OUTPUT}")
    print(f"[INFO] Total images   : {len(image_files)}")
    print("=" * 100)

    records = []
    processed = 0
    failed = 0

    for image_name in image_files:
        stem = Path(image_name).stem
        img_path = os.path.join(IMAGE_DIR, image_name)
        gt_path = os.path.join(GT_DIR, image_name)

        print("\n" + "=" * 80)
        print(f"[INFO] Processing: {image_name}")
        print(f"[INFO] IMAGE: {img_path}")
        print(f"[INFO] GT   : {gt_path}")

        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found, skip: {gt_path}")
            failed += 1
            continue

        try:
            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if image_bgr is None:
                raise ValueError(f"Failed to read image: {img_path}")
            if gt_mask is None:
                raise ValueError(f"Failed to read GT: {gt_path}")

            image_bgr_512 = resize_image(image_bgr, cv2.INTER_LINEAR)
            gt_mask_512 = resize_image(gt_mask, cv2.INTER_NEAREST)

            # 1) Generate centroid points on resized GT
            points = extract_centroids_from_gt(gt_mask_512)
            if len(points) == 0:
                print("[WARN] No connected components in GT after resize, use empty prediction.")
                pred_bool = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)
            else:
                txt_path = os.path.join(POINT_TXT_DIR, f"{stem}.txt")
                save_points_txt(points, txt_path)
                print(f"[SAVED] points: {txt_path}")
                print(f"[INFO] point count: {len(points)}")

                # 2) Predict with SAM
                pred_bool = predict_mask_from_points(
                    predictor=predictor,
                    image_bgr_512=image_bgr_512,
                    points=points,
                    multimask_output=MULTIMASK_OUTPUT,
                )

            # 3) Edge filter for consistency with current eval style
            pred_bool = edge_filter(pred_bool, boundary_margin=BOUNDARY_MARGIN)
            gt_bool = edge_filter(to_bool(gt_mask_512), boundary_margin=BOUNDARY_MARGIN)

            # 4) Save mask
            mask_save_path = os.path.join(MASK_DIR, f"{stem}.png")
            save_binary_mask(pred_bool, mask_save_path)
            print(f"[SAVED] mask: {mask_save_path}")

            # 5) Save overlap
            overlap_rgb = make_overlap_image(pred_bool)
            overlap_save_path = os.path.join(OVERLAP_DIR, f"{stem}.png")
            save_rgb_image(overlap_rgb, overlap_save_path)
            print(f"[SAVED] overlap: {overlap_save_path}")

            # 6) Save difference
            diff_rgb = make_diff(pred_bool, gt_bool)
            diff_save_path = os.path.join(DIFF_DIR, f"{stem}.png")
            save_rgb_image(diff_rgb, diff_save_path)
            print(f"[SAVED] difference: {diff_save_path}")

            # 7) Save point overlay (used for montage only)
            point_vis_bgr = draw_points_on_image(image_bgr_512, points if len(points) > 0 else [])
            point_vis_rgb = cv2.cvtColor(point_vis_bgr, cv2.COLOR_BGR2RGB)
            point_overlay_path = os.path.join(point_overlay_dir, f"{stem}.png")
            save_rgb_image(point_vis_rgb, point_overlay_path)

            # 8) Metrics
            pred_u8 = (pred_bool.astype(np.uint8) * 255)
            gt_u8 = (gt_bool.astype(np.uint8) * 255)
            metrics = compute_metrics(pred_u8, gt_u8, DEVICE)
            records.append({"image": image_name, **metrics})
            processed += 1

            print(f"[INFO] Dice = {metrics['Dice']:.6f}, IoU = {metrics['IoU']:.6f}")

        except Exception as e:
            print(f"[ERROR] Failed on {image_name}: {e}")
            failed += 1
            continue

    if len(records) == 0:
        raise RuntimeError("No valid records generated.")

    # 9) Save metrics CSV
    df = save_metrics_csv_with_mean_std(records, METRICS_CSV)
    print(f"\n[SAVED] metrics csv: {METRICS_CSV}")

    # 10) Save visualization montages.
    save_pair_montages(
        image_names=image_files,
        point_overlay_dir=point_overlay_dir,
        diff_dir=DIFF_DIR,
        save_dir=DIFF_DIR,
    )

    mean_row = df[df["image"] == "mean"].iloc[0]
    std_row = df[df["image"] == "std"].iloc[0]

    print("\n" + "=" * 100)
    print("[SIMPLE POINT DONE]")
    print(f"[SUMMARY] processed = {processed}")
    print(f"[SUMMARY] failed    = {failed}")
    print(f"[SUMMARY] mean Dice = {mean_row['Dice']}")
    print(f"[SUMMARY] std  Dice = {std_row['Dice']}")
    print(f"[SUMMARY] mean IoU  = {mean_row['IoU']}")
    print(f"[SUMMARY] std  IoU  = {std_row['IoU']}")
    print("=" * 100)

    return {
        "metrics_csv": METRICS_CSV,
        "mask_dir": MASK_DIR,
        "overlap_dir": OVERLAP_DIR,
        "diff_dir": DIFF_DIR,
        "point_txt_dir": POINT_TXT_DIR,
        "processed": processed,
        "failed": failed,
    }


def main():
    run_simple_point_experiment()


if __name__ == "__main__":
    main()