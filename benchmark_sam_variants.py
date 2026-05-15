import math
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from metrics import compute_metrics
from prompt_flow import run_prompt_flow_experiment

warnings.filterwarnings("ignore")


# =========================
# Default Config
# =========================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "/home/ubuntu/nanoseg/data/ptsn/images"
GT_DIR = "/home/ubuntu/nanoseg/data/ptsn/GT"
BBOX_UNET_ALL_DIR = "/home/ubuntu/nanoseg/data/ptsn/bbox_unet_all"

# Only use one reference
REFERENCE_ID = "28"

# Final outputs
REF_SELECT_ROOT = "/home/ubuntu/nanoseg/result/ref_select"
MASK_ROOT = "/home/ubuntu/nanoseg/result/masks"
DIFF_ROOT = "/home/ubuntu/nanoseg/result/difference"

# Temporary root for intermediate files; will be deleted automatically
TEMP_ROOT = "/home/ubuntu/nanoseg/result/_variant_tmp"

# Variant models to compare
MODEL_VARIANTS = [
    {
        "model_name": "microsam",
        "display_name": "MicroSAM",
        "model_tag": "microsam_vit_b",
        "sam_checkpoint": "/home/ubuntu/nanoseg/checkpoints/microsam_vit_b.pth",
        "model_type": "vit_b",
        "summary_filename": "summary_microsam.csv",
        "per_ref_csv_name": "MicroSAM.csv",
    },
    {
        "model_name": "medsam",
        "display_name": "MedSAM",
        "model_tag": "medsam_vit_b",
        "sam_checkpoint": "/home/ubuntu/nanoseg/checkpoints/medsam_vit_b.pth",
        "model_type": "vit_b",
        "summary_filename": "summary_medsam.csv",
        "per_ref_csv_name": "MedSAM.csv",
    },
]

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

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
TARGET_SIZE = 512
BOUNDARY_MARGIN = 5

# =========================
# Difference image colors
# =========================
# TP / FP / FN as requested
TP_COLOR = (220, 220, 220)
FP_COLOR = (241, 198, 81)
FN_COLOR = (0, 128, 255)
BG_COLOR = (255, 255, 255)

# =========================
# Montage config
# =========================
MONTAGE_1_COLS = 4
MONTAGE_1_ROWS = 5   # first 20 images

MONTAGE_2_COLS = 4
CELL_GAP = 16
OUTER_PAD = 24

LABEL_FONT_SIZE = 28
LABEL_MARGIN_X = 12
LABEL_MARGIN_Y = 8


# =========================
# Helpers
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_rmtree(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def list_valid_images(image_dir: str) -> List[str]:
    return sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(VALID_EXTS)
    ])


def resize(img: np.ndarray, interp: int) -> np.ndarray:
    return cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=interp)


def to_bool(mask: np.ndarray) -> np.ndarray:
    return mask > 0


def edge_filter(mask: np.ndarray, boundary_margin: int = BOUNDARY_MARGIN) -> np.ndarray:
    from skimage.measure import label, regionprops

    labeled = label(mask)
    out = np.zeros_like(mask, dtype=np.uint8)

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


def save_metrics_csv_with_mean_std(records: List[Dict], out_path: str) -> pd.DataFrame:
    if len(records) == 0:
        raise ValueError(f"No records available to save metrics csv: {out_path}")

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
    df.to_csv(out_path, index=False)
    return df


def build_summary_csv(records: List[Dict], out_path: str) -> pd.DataFrame:
    if len(records) == 0:
        raise ValueError("No records available to build summary csv.")

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
    df.to_csv(out_path, index=False)
    return df


def save_binary_mask(mask_bool: np.ndarray, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(out_path, mask_u8)


def make_diff(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    img = np.ones((*gt.shape, 3), dtype=np.uint8) * 255

    tp = pred & gt
    fp = pred & (~gt)
    fn = (~pred) & gt

    img[tp] = TP_COLOR
    img[fp] = FP_COLOR
    img[fn] = FN_COLOR

    return img


def save_difference_image(pred_bool: np.ndarray, gt_bool: np.ndarray, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    diff_img = make_diff(pred_bool, gt_bool)
    Image.fromarray(diff_img).save(out_path)


# =========================
# Montage helpers
# =========================
def try_parse_int_stem(path: str) -> Tuple[int, object]:
    stem = Path(path).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem.lower())


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


def load_diff_images(diff_root: str) -> List[str]:
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = []

    if not os.path.exists(diff_root):
        return image_paths

    for name in os.listdir(diff_root):
        p = os.path.join(diff_root, name)
        if not os.path.isfile(p):
            continue
        if name.lower() in {"montage.png", "montage_1.png", "montage_2.png"}:
            continue
        if Path(name).suffix.lower() in valid_exts:
            image_paths.append(p)

    return sorted(image_paths, key=try_parse_int_stem)


def draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font) -> None:
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def paste_grid(
    canvas: Image.Image,
    image_paths: List[str],
    start_xy: Tuple[int, int],
    cols: int,
    rows: int,
    cell_w: int,
    cell_h: int,
    font
) -> None:
    draw = ImageDraw.Draw(canvas)
    start_x, start_y = start_xy

    max_slots = cols * rows
    image_paths = image_paths[:max_slots]

    for idx, img_path in enumerate(image_paths):
        r = idx // cols
        c = idx % cols

        cell_x = start_x + c * (cell_w + CELL_GAP)
        cell_y = start_y + r * (cell_h + CELL_GAP)

        img = Image.open(img_path).convert("RGB")

        x_offset = cell_x + (cell_w - img.width) // 2
        y_offset = cell_y + (cell_h - img.height) // 2

        canvas.paste(img, (x_offset, y_offset))

        label = Path(img_path).stem
        draw_label(
            draw,
            label,
            x_offset + LABEL_MARGIN_X,
            y_offset + LABEL_MARGIN_Y,
            font
        )


def save_single_montage(
    image_paths: List[str],
    save_path: str,
    cols: int,
    rows: int,
    font,
    cell_w: int,
    cell_h: int,
) -> None:
    if len(image_paths) == 0:
        return

    grid_w = cols * cell_w + (cols - 1) * CELL_GAP
    grid_h = rows * cell_h + (rows - 1) * CELL_GAP

    canvas_w = grid_w + 2 * OUTER_PAD
    canvas_h = grid_h + 2 * OUTER_PAD

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    paste_grid(
        canvas=canvas,
        image_paths=image_paths,
        start_xy=(OUTER_PAD, OUTER_PAD),
        cols=cols,
        rows=rows,
        cell_w=cell_w,
        cell_h=cell_h,
        font=font,
    )

    canvas.save(save_path)
    print(f"[SAVED] {save_path}")


def make_montages_for_dir(diff_dir: str) -> None:
    image_paths = load_diff_images(diff_dir)
    if len(image_paths) == 0:
        print(f"[WARN] No difference images found in: {diff_dir}")
        return

    first_group = image_paths[:20]
    second_group = image_paths[20:]

    sizes = []
    for p in image_paths:
        with Image.open(p) as im:
            sizes.append(im.size)

    max_w = max(w for w, h in sizes)
    max_h = max(h for w, h in sizes)
    font = load_font(LABEL_FONT_SIZE)

    montage_1_path = os.path.join(diff_dir, "montage_1.png")
    save_single_montage(
        image_paths=first_group,
        save_path=montage_1_path,
        cols=MONTAGE_1_COLS,
        rows=MONTAGE_1_ROWS,
        font=font,
        cell_w=max_w,
        cell_h=max_h,
    )

    montage_2_path = os.path.join(diff_dir, "montage_2.png")
    if len(second_group) > 0:
        rows_2 = int(math.ceil(len(second_group) / MONTAGE_2_COLS))
        save_single_montage(
            image_paths=second_group,
            save_path=montage_2_path,
            cols=MONTAGE_2_COLS,
            rows=rows_2,
            font=font,
            cell_w=max_w,
            cell_h=max_h,
        )


# =========================
# Evaluation
# =========================
def evaluate_box_only_experiment(
    image_dir: str,
    gt_dir: str,
    box_mask_dir: str,
    reference_id: str,
    out_csv_path: str,
    mask_save_dir: str,
    diff_save_dir: str,
    device: torch.device,
    boundary_margin: int = BOUNDARY_MARGIN,
) -> Dict:
    """
    Evaluate box-only results and save one csv.
    Exclude the reference image itself.

    Added outputs:
    - binary masks under mask_save_dir
    - difference images under diff_save_dir
    - montage_1.png / montage_2.png under diff_save_dir
    """
    files = list_valid_images(image_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No valid images found in: {image_dir}")

    ensure_dir(mask_save_dir)
    ensure_dir(diff_save_dir)

    records = []
    processed = 0
    skipped = 0
    failed = 0

    print(f"[INFO] BOX_MASK_DIR   : {box_mask_dir}")
    print(f"[INFO] OUT_CSV_PATH   : {out_csv_path}")
    print(f"[INFO] MASK_SAVE_DIR  : {mask_save_dir}")
    print(f"[INFO] DIFF_SAVE_DIR  : {diff_save_dir}")
    print(f"[INFO] REFERENCE_ID   : {reference_id}")

    for f in files:
        stem = os.path.splitext(f)[0]

        if stem == str(reference_id):
            print(f"[SKIP] Reference image itself: {f}")
            skipped += 1
            continue

        box_path = os.path.join(box_mask_dir, f)
        gt_path = os.path.join(gt_dir, f)

        if not os.path.exists(box_path):
            print(f"[WARN] Missing box mask, skip: {box_path}")
            skipped += 1
            continue

        if not os.path.exists(gt_path):
            print(f"[WARN] Missing GT, skip: {gt_path}")
            skipped += 1
            continue

        try:
            box = cv2.imread(box_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if box is None or gt is None:
                print(f"[WARN] Failed to read one or more files for {f}, skip.")
                skipped += 1
                continue

            box = resize(box, cv2.INTER_NEAREST)
            gt = resize(gt, cv2.INTER_NEAREST)

            box_bool = edge_filter(to_bool(box), boundary_margin=boundary_margin)
            gt_bool = edge_filter(to_bool(gt), boundary_margin=boundary_margin)

            # save final mask
            mask_out_path = os.path.join(mask_save_dir, f)
            save_binary_mask(box_bool, mask_out_path)

            # save difference
            diff_out_path = os.path.join(diff_save_dir, f)
            save_difference_image(box_bool, gt_bool, diff_out_path)

            # metrics
            box_u8 = (box_bool.astype(np.uint8) * 255)
            gt_u8 = (gt_bool.astype(np.uint8) * 255)
            metrics = compute_metrics(box_u8, gt_u8, device)

            records.append({"image": f, **metrics})
            processed += 1

        except Exception as e:
            print(f"[ERROR] Failed on {f}: {e}")
            failed += 1
            continue

    if len(records) == 0:
        raise RuntimeError(f"No valid evaluation records generated for reference {reference_id}.")

    df = save_metrics_csv_with_mean_std(records, out_csv_path)

    # difference montage
    make_montages_for_dir(diff_save_dir)

    mean_row = df[df["image"] == "mean"].iloc[0]
    std_row = df[df["image"] == "std"].iloc[0]

    print(f"[SAVED] {out_csv_path}")
    print(f"[INFO] processed = {processed}, skipped = {skipped}, failed = {failed}")
    print(f"[INFO] mean Dice = {mean_row['Dice']}")
    print(f"[INFO] std  Dice = {std_row['Dice']}")

    return {
        "csv_path": out_csv_path,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "mean_dice": float(mean_row["Dice"]),
        "std_dice": float(std_row["Dice"]),
    }


# =========================
# Single variant runner
# =========================
def run_one_variant_summary(
    image_dir: str,
    gt_dir: str,
    bbox_unet_all_dir: str,
    ref_select_root: str,
    mask_root: str,
    diff_root: str,
    temp_root: str,
    reference_id: str,
    model_name: str,
    display_name: str,
    model_tag: str,
    sam_checkpoint: str,
    model_type: str,
    summary_filename: str,
    per_ref_csv_name: str,
    device: torch.device,
) -> Dict:
    """
    Run evaluation for one model variant using one fixed reference only.

    Final saved files:
    - ref_select/{DisplayName}.csv
    - ref_select/{summary_filename}
    - masks/{DisplayName}/*.png
    - difference/{DisplayName}/*.png
    - difference/{DisplayName}/montage_1.png
    - difference/{DisplayName}/montage_2.png

    Temporary outputs:
    - prompt_flow box masks under temp_root
    - deleted after run
    """
    ensure_dir(ref_select_root)
    ensure_dir(mask_root)
    ensure_dir(diff_root)
    ensure_dir(temp_root)

    prompt_dir = os.path.join(bbox_unet_all_dir, str(reference_id))
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"Reference prompt folder not found: {prompt_dir}")

    print("\n" + "=" * 120)
    print(f"[VARIANT START] {model_name}")
    print(f"[INFO] DISPLAY_NAME      : {display_name}")
    print(f"[INFO] MODEL_TAG         : {model_tag}")
    print(f"[INFO] CHECKPOINT        : {sam_checkpoint}")
    print(f"[INFO] MODEL_TYPE        : {model_type}")
    print(f"[INFO] REFERENCE_ID      : {reference_id}")
    print("=" * 120)

    variant_temp_root = os.path.join(temp_root, model_name)
    safe_rmtree(variant_temp_root)
    ensure_dir(variant_temp_root)

    final_csv_path = os.path.join(ref_select_root, per_ref_csv_name)
    summary_csv_path = os.path.join(ref_select_root, summary_filename)

    variant_mask_root = os.path.join(mask_root, display_name)
    variant_diff_root = os.path.join(diff_root, display_name)

    # remove old outputs to match the new flat structure
    safe_rmtree(variant_mask_root)
    safe_rmtree(variant_diff_root)
    ensure_dir(variant_mask_root)
    ensure_dir(variant_diff_root)

    # remove stale csv files
    if os.path.exists(final_csv_path):
        os.remove(final_csv_path)
    if os.path.exists(summary_csv_path):
        os.remove(summary_csv_path)

    try:
        # 1. prompt flow only
        prompt_result = run_prompt_flow_experiment(
            image_dir=image_dir,
            gt_dir=gt_dir,
            prompt_dir=prompt_dir,
            output_root=variant_temp_root,
            reference_id=reference_id,
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            model_tag=model_tag,
            device=device,
        )

        # 2. box-only evaluation + save masks/difference/montages
        evaluate_box_only_experiment(
            image_dir=image_dir,
            gt_dir=gt_dir,
            box_mask_dir=prompt_result["box_mask_dir"],
            reference_id=reference_id,
            out_csv_path=final_csv_path,
            mask_save_dir=variant_mask_root,
            diff_save_dir=variant_diff_root,
            device=device,
        )

        # 3. build one-line summary
        df = pd.read_csv(final_csv_path)
        mean_row = df[df["image"] == "mean"].iloc[0].to_dict()

        summary_record = {"image": str(reference_id)}
        for col in METRIC_COLUMNS:
            summary_record[col] = mean_row[col]

        summary_df = build_summary_csv([summary_record], summary_csv_path)

        print(f"[DONE {model_name}] reference = {reference_id}")
        print(f"[INFO] mean Dice = {summary_record['Dice']}")
        print(f"[INFO] mean IoU  = {summary_record['IoU']}")

    finally:
        safe_rmtree(variant_temp_root)

    print("\n" + "=" * 120)
    print(f"[VARIANT DONE] {model_name}")
    print(f"[SAVED] {final_csv_path}")
    print(f"[SAVED] {summary_csv_path}")
    print("=" * 120)

    return {
        "model_name": model_name,
        "display_name": display_name,
        "model_tag": model_tag,
        "summary_csv": summary_csv_path,
    }


# =========================
# Main
# =========================
def main():
    ensure_dir(REF_SELECT_ROOT)
    ensure_dir(MASK_ROOT)
    ensure_dir(DIFF_ROOT)
    ensure_dir(TEMP_ROOT)

    all_results = []

    for cfg in MODEL_VARIANTS:
        result = run_one_variant_summary(
            image_dir=IMAGE_DIR,
            gt_dir=GT_DIR,
            bbox_unet_all_dir=BBOX_UNET_ALL_DIR,
            ref_select_root=REF_SELECT_ROOT,
            mask_root=MASK_ROOT,
            diff_root=DIFF_ROOT,
            temp_root=TEMP_ROOT,
            reference_id=REFERENCE_ID,
            model_name=cfg["model_name"],
            display_name=cfg["display_name"],
            model_tag=cfg["model_tag"],
            sam_checkpoint=cfg["sam_checkpoint"],
            model_type=cfg["model_type"],
            summary_filename=cfg["summary_filename"],
            per_ref_csv_name=cfg["per_ref_csv_name"],
            device=DEVICE,
        )
        all_results.append(result)

    safe_rmtree(TEMP_ROOT)

    print("\n" + "=" * 120)
    print("[ALL VARIANTS FINISHED]")
    for r in all_results:
        print(f"{r['model_name']}: {r['summary_csv']}")
    print("=" * 120)


if __name__ == "__main__":
    main()