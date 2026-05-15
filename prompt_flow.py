"""
Prompt Flow Segmentation Pipeline

This script performs prompt-guided nanoparticle segmentation using
Segment Anything Model (SAM) with bounding-box prompts.

Main features
-------------
- Bounding-box guided SAM segmentation
- Batched prompt inference
- Connected-component analysis
- Automatic area-range extraction
- Binary mask export for downstream processing

The pipeline is designed for one-shot segmentation workflows and
large-scale reproducible experiments.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from postprocess import postprocess_mask
from segment_anything import sam_model_registry, SamPredictor
from utils import read_labels_txt, clip_box_to_image

# ----------------------------------------
# Default Config
# These defaults are only for standalone testing.
# Batch traversal should call run_prompt_flow_experiment(...)
# with explicit arguments.
# ----------------------------------------
DEFAULT_SAM_CHECKPOINT = "path/to/checkpoints/sam_vit_b.pth"
DEFAULT_MODEL_TYPE = "vit_b"
DEFAULT_MODEL_TAG = "sam_vit_b"

DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_IMAGE_DIR = "path/to/dataset/images"
DEFAULT_GT_DIR = "path/to/dataset/masks"

# Example only: one bbox_unet subfolder for standalone testing
DEFAULT_PROMPT_DIR = "path/to/prompts/reference_01"
DEFAULT_REFERENCE_ID = "1"

# Example only: one output folder for standalone testing
DEFAULT_OUTPUT_ROOT = "path/to/output"

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MULTIMASK_OUTPUT = True
BATCH_SIZE = 2

# Unified processing size
TARGET_SIZE = (512, 512)   # (W, H)


# ----------------------------------------
# Model
# ----------------------------------------
def build_predictor(
    sam_checkpoint: str,
    model_type: str,
    device: torch.device
) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    return predictor


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


def resize_image_and_gt(
    image_bgr: np.ndarray,
    gt_mask: np.ndarray,
    target_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    target_w, target_h = target_size

    image_resized = cv2.resize(
        image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )
    gt_resized = cv2.resize(
        gt_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )

    return image_resized, gt_resized


def list_valid_images(image_dir: str) -> List[str]:
    image_files = [
        f for f in sorted(os.listdir(image_dir))
        if f.lower().endswith(VALID_EXTS)
    ]
    return image_files


def get_connected_components(mask_bool: np.ndarray) -> List[Dict]:
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    components = []
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        comp_mask = labels == lab
        components.append({
            "label": lab,
            "area": area,
            "mask": comp_mask
        })
    return components


def get_area_range_from_components(
    mask_bool: np.ndarray
) -> Tuple[Optional[int], Optional[int], List[Dict]]:
    comps = get_connected_components(mask_bool)
    if len(comps) == 0:
        return None, None, comps
    areas = [c["area"] for c in comps]
    return min(areas), max(areas), comps


def write_area_range_txt(area_lines: List[str], save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("image_name min_area max_area\n")
        f.writelines(area_lines)


# ----------------------------------------
# Box helpers
# ----------------------------------------
def scale_boxes_to_target(
    boxes: List[Tuple[int, float, float, float, float]],
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int
) -> List[Tuple[int, float, float, float, float]]:
    """
    Scale bbox coordinates from original image size to target size.
    Input format:
    [(cls, x1, y1, x2, y2), ...]
    """
    sx = target_w / orig_w
    sy = target_h / orig_h

    scaled_boxes = []
    for cls, x1, y1, x2, y2 in boxes:
        x1_new = x1 * sx
        y1_new = y1 * sy
        x2_new = x2 * sx
        y2_new = y2 * sy
        scaled_boxes.append((cls, x1_new, y1_new, x2_new, y2_new))

    return scaled_boxes


def predict_single_box(
    predictor: SamPredictor,
    box_xyxy: np.ndarray
) -> np.ndarray:
    masks, scores, _ = predictor.predict(
        box=box_xyxy,
        multimask_output=MULTIMASK_OUTPUT,
        return_logits=False
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx] > 0.5


def predict_in_batches(
    predictor: SamPredictor,
    all_boxes: List[List[float]],
    H: int,
    W: int,
    batch_size: int,
    device: torch.device
) -> np.ndarray:
    merged_mask = np.zeros((H, W), dtype=bool)

    for i in range(0, len(all_boxes), batch_size):
        batch_boxes = all_boxes[i:i + batch_size]
        boxes_tensor = torch.tensor(
            batch_boxes, dtype=torch.float32, device=device
        )
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_tensor, (H, W)
        )

        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=MULTIMASK_OUTPUT
        )

        best_indices = torch.argmax(scores, dim=1)
        for j, idx in enumerate(best_indices):
            best_mask = masks[j, idx].detach().cpu().numpy() > 0.5
            merged_mask |= best_mask

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return merged_mask


def run_prompt_flow_on_image(
    predictor: SamPredictor,
    image_bgr_512: np.ndarray,
    prompt_path: str,
    orig_w: int,
    orig_h: int,
    batch_size: int,
    device: torch.device
) -> np.ndarray:
    """
    The input image is already resized to 512x512.
    Bboxes in prompt txt are assumed to be in the original image coordinate
    system and will be scaled to the target size.
    """
    H, W = image_bgr_512.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr_512, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    boxes = read_labels_txt(prompt_path)
    if len(boxes) == 0:
        raise ValueError(f"No bbox found in prompt file: {prompt_path}")

    boxes = scale_boxes_to_target(
        boxes=boxes,
        orig_w=orig_w,
        orig_h=orig_h,
        target_w=W,
        target_h=H
    )

    merged_mask = np.zeros((H, W), dtype=bool)

    if batch_size == 1:
        for cls, x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = clip_box_to_image(x1, y1, x2, y2, W, H)
            box_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
            best_mask = predict_single_box(predictor, box_xyxy)
            merged_mask |= best_mask
    else:
        all_boxes = []
        for cls, x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = clip_box_to_image(x1, y1, x2, y2, W, H)
            all_boxes.append([x1, y1, x2, y2])

        if len(all_boxes) > 0:
            merged_mask = predict_in_batches(
                predictor=predictor,
                all_boxes=all_boxes,
                H=H,
                W=W,
                batch_size=batch_size,
                device=device
            )

    merged_mask_uint8 = (merged_mask.astype(np.uint8) * 255)
    final_mask = postprocess_mask(merged_mask_uint8, image_bgr_512)

    if final_mask.shape[:2] != (H, W):
        final_mask = cv2.resize(final_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    return to_binary_mask(final_mask)


# ----------------------------------------
# Main experiment function
# ----------------------------------------
def run_prompt_flow_experiment(
    image_dir: str,
    gt_dir: str,
    prompt_dir: str,
    output_root: str,
    reference_id: str,
    sam_checkpoint: str = DEFAULT_SAM_CHECKPOINT,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_tag: str = DEFAULT_MODEL_TAG,
    device: torch.device = DEFAULT_DEVICE,
    target_size: Tuple[int, int] = TARGET_SIZE,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Run the full prompt flow for one bbox_unet subfolder.

    Parameters
    ----------
    image_dir : image directory
    gt_dir : GT directory
    prompt_dir : one bbox_unet subfolder, e.g. /.../bbox_unet_all/13
    output_root : current experiment output root, e.g. /.../ref_select/13
    reference_id : current reference id, e.g. "13"
    sam_checkpoint : checkpoint path, allows switching among vit_b variants
    model_type : SAM model type such as "vit_b"
    model_tag : a user-defined model tag for later batch management
    """
    box_mask_dir = os.path.join(output_root, "mask", "box")
    area_range_txt = os.path.join(output_root, "area_range.txt")
    os.makedirs(box_mask_dir, exist_ok=True)

    predictor = build_predictor(
        sam_checkpoint=sam_checkpoint,
        model_type=model_type,
        device=device
    )

    image_files = list_valid_images(image_dir)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No valid images found in {image_dir}")

    print(f"[INFO] DEVICE         : {device}")
    print(f"[INFO] MODEL_TYPE     : {model_type}")
    print(f"[INFO] MODEL_TAG      : {model_tag}")
    print(f"[INFO] CHECKPOINT     : {sam_checkpoint}")
    print(f"[INFO] IMAGE_DIR      : {image_dir}")
    print(f"[INFO] GT_DIR         : {gt_dir}")
    print(f"[INFO] PROMPT_DIR     : {prompt_dir}")
    print(f"[INFO] OUTPUT_ROOT    : {output_root}")
    print(f"[INFO] BOX_MASK_DIR   : {box_mask_dir}")
    print(f"[INFO] AREA_RANGE_TXT : {area_range_txt}")
    print(f"[INFO] REFERENCE_ID   : {reference_id}")
    print(f"[INFO] TARGET_SIZE    : {target_size}")
    print(f"[INFO] BATCH_SIZE     : {batch_size}")
    print(f"[INFO] Total images   : {len(image_files)}")

    processed = 0
    skipped = 0
    failed = 0
    area_lines = []

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
        prompt_path = os.path.join(prompt_dir, stem + ".txt")

        print("\n" + "=" * 80)
        print(f"[INFO] Processing image : {image_name}")
        print(f"[INFO] IMAGE  : {img_path}")
        print(f"[INFO] GT     : {gt_path}")
        print(f"[INFO] PROMPT : {prompt_path}")

        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found, skip: {gt_path}")
            skipped += 1
            continue

        if not os.path.exists(prompt_path):
            print(f"[WARN] Prompt txt not found, skip: {prompt_path}")
            skipped += 1
            continue

        try:
            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if image_bgr is None:
                raise ValueError(f"Failed to read image: {img_path}")
            if gt_mask is None:
                raise ValueError(f"Failed to read GT: {gt_path}")

            orig_h, orig_w = image_bgr.shape[:2]

            image_bgr_512, gt_mask_512 = resize_image_and_gt(
                image_bgr=image_bgr,
                gt_mask=gt_mask,
                target_size=target_size
            )

            # Preserve preprocessing consistency.
            _ = to_binary_mask(gt_mask_512)

            print("[RUN] Prompt flow (512x512)")
            box_mask_bool = run_prompt_flow_on_image(
                predictor=predictor,
                image_bgr_512=image_bgr_512,
                prompt_path=prompt_path,
                orig_w=orig_w,
                orig_h=orig_h,
                batch_size=batch_size,
                device=device
            )

            min_area, max_area, components = get_area_range_from_components(box_mask_bool)
            if min_area is None or max_area is None:
                print("[WARN] Prompt flow produced no connected components, skip this image.")
                skipped += 1
                continue

            print(f"[INFO] Prompt flow components: {len(components)}")
            print(f"[INFO] Prompt flow area range (512x512): min={min_area}, max={max_area}")

            mask_path = os.path.join(box_mask_dir, stem + ".png")
            save_binary_mask(box_mask_bool, mask_path)
            print(f"[SAVED] mask: {mask_path}")

            area_lines.append(f"{image_name} {int(min_area)} {int(max_area)}\n")
            processed += 1

        except Exception as e:
            print(f"[ERROR] Failed on {image_name}: {e}")
            failed += 1
            continue

    write_area_range_txt(area_lines, area_range_txt)

    print("\n" + "=" * 80)
    print("[DONE] Prompt flow finished.")
    print(f"[SUMMARY] processed = {processed}")
    print(f"[SUMMARY] skipped   = {skipped}")
    print(f"[SUMMARY] failed    = {failed}")
    print(f"[SUMMARY] area txt  = {area_range_txt}")

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
# Standalone direct run
# ----------------------------------------
def main():
    """
    Standalone example entry point.
    """
    run_prompt_flow_experiment(
        image_dir=DEFAULT_IMAGE_DIR,
        gt_dir=DEFAULT_GT_DIR,
        prompt_dir=DEFAULT_PROMPT_DIR,
        output_root=DEFAULT_OUTPUT_ROOT,
        reference_id=DEFAULT_REFERENCE_ID,
        sam_checkpoint=DEFAULT_SAM_CHECKPOINT,
        model_type=DEFAULT_MODEL_TYPE,
        model_tag=DEFAULT_MODEL_TAG,
        device=DEFAULT_DEVICE,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE
    )


if __name__ == "__main__":
    main()