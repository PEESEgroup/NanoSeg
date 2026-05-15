"""
Batch NanoSeg Experiment Runner

This script orchestrates the complete NanoSeg one-shot segmentation
pipeline across multiple reference selections.

Main stages
-----------
1. Prompt-flow segmentation
2. Global point-flow segmentation
3. Evaluation and mask merging
4. Metric aggregation and summary generation
5. Best-reference selection
6. Full-result regeneration for the optimal reference

The script is designed for reproducible large-scale benchmarking
and automated reference-image selection.
"""

import os
from typing import Dict, List

import pandas as pd
import torch

from evaluate_and_merge import run_evaluate_and_merge_experiment
from global_point_flow import run_global_point_flow_experiment
from prompt_flow import run_prompt_flow_experiment

# ----------------------------------------
# Default Config
# ----------------------------------------
DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data roots
IMAGE_DIR = "path/to/dataset/images"
GT_DIR = "path/to/dataset/masks"
BBOX_UNET_ALL_DIR = "path/to/prompts"

# Result roots
REF_SELECT_ROOT = "path/to/output/ref_select"
FULL_RESULT_ROOT = "path/to/output"

# Model config
SAM_CHECKPOINT = "path/to/checkpoints/sam_vit_b.pth"
MODEL_TYPE = "vit_b"
MODEL_TAG = "sam_vit_b"

# Metrics columns, aligned with evaluate_and_merge.py
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


# ----------------------------------------
# Helpers
# ----------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_reference_ids(bbox_unet_all_dir: str) -> List[str]:
    """
    Return numeric reference subfolders sorted in ascending order.
    """
    ref_ids = []
    for name in os.listdir(bbox_unet_all_dir):
        full_path = os.path.join(bbox_unet_all_dir, name)
        if os.path.isdir(full_path) and name.isdigit():
            ref_ids.append(name)
    ref_ids = sorted(ref_ids, key=lambda x: int(x))
    return ref_ids


def load_mean_row_from_csv(csv_path: str) -> Dict:
    """
    Load mean metrics from one CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrics csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    mean_df = df[df["image"] == "mean"]
    if len(mean_df) != 1:
        raise ValueError(f"'mean' row not found or duplicated in: {csv_path}")

    row = mean_df.iloc[0].to_dict()
    return row


def build_summary_csv(records: List[Dict], out_path: str) -> pd.DataFrame:
    """
    Build summary metrics across all references.

    Row layout:
    - one row per reference id (image column stores the ref id)
    - final two rows: mean, std

    Column layout:
    - same as existing metrics csv: image + METRIC_COLUMNS
    """
    if len(records) == 0:
        raise ValueError("No records available to build summary.csv")

    df = pd.DataFrame(records)
    df = df[["image"] + METRIC_COLUMNS]

    # Sort rows by numeric reference id, but keep mean/std for appending later
    df["image"] = df["image"].astype(str)
    df = df.sort_values(by="image", key=lambda s: s.astype(int)).reset_index(drop=True)

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


def select_best_reference(summary_records: List[Dict]) -> Dict:
    """
    Select the best reference using Dice, IoU, and reference ID.
    """
    if len(summary_records) == 0:
        raise ValueError("No summary records available for best-reference selection.")

    def sort_key(r: Dict):
        return (-float(r["Dice"]), -float(r["IoU"]), int(r["image"]))

    best = sorted(summary_records, key=sort_key)[0]
    return best


# ----------------------------------------
# Main batch runner
# ----------------------------------------
def run_batch_experiments(
    image_dir: str = IMAGE_DIR,
    gt_dir: str = GT_DIR,
    bbox_unet_all_dir: str = BBOX_UNET_ALL_DIR,
    ref_select_root: str = REF_SELECT_ROOT,
    full_result_root: str = FULL_RESULT_ROOT,
    sam_checkpoint: str = SAM_CHECKPOINT,
    model_type: str = MODEL_TYPE,
    model_tag: str = MODEL_TAG,
    device: torch.device = DEFAULT_DEVICE,
) -> Dict:
    """
    Batch pipeline:
    1. traverse all bbox_unet subfolders
    2. run prompt_flow
    3. run global_point_flow
    4. run evaluate_and_merge
    5. collect nanoseg mean metrics into summary.csv
    6. select best reference
    7. rerun best reference with full visual outputs to overwrite standard result dirs
    """
    ensure_dir(ref_select_root)

    ref_ids = list_reference_ids(bbox_unet_all_dir)
    if len(ref_ids) == 0:
        raise FileNotFoundError(f"No numeric bbox_unet subfolders found in: {bbox_unet_all_dir}")

    print("=" * 100)
    print("[BATCH START]")
    print(f"[INFO] DEVICE             : {device}")
    print(f"[INFO] IMAGE_DIR          : {image_dir}")
    print(f"[INFO] GT_DIR             : {gt_dir}")
    print(f"[INFO] BBOX_UNET_ALL_DIR  : {bbox_unet_all_dir}")
    print(f"[INFO] REF_SELECT_ROOT    : {ref_select_root}")
    print(f"[INFO] FULL_RESULT_ROOT   : {full_result_root}")
    print(f"[INFO] SAM_CHECKPOINT     : {sam_checkpoint}")
    print(f"[INFO] MODEL_TYPE         : {model_type}")
    print(f"[INFO] MODEL_TAG          : {model_tag}")
    print(f"[INFO] Total references   : {len(ref_ids)}")
    print("=" * 100)

    summary_records = []
    failed_refs = []

    for ref_id in ref_ids:
        print("\n" + "#" * 100)
        print(f"[RUNNING REFERENCE] {ref_id}")
        print("#" * 100)

        prompt_dir = os.path.join(bbox_unet_all_dir, ref_id)
        output_root = os.path.join(ref_select_root, ref_id)

        try:
            # -------------------------
            # 1. prompt flow
            # -------------------------
            prompt_result = run_prompt_flow_experiment(
                image_dir=image_dir,
                gt_dir=gt_dir,
                prompt_dir=prompt_dir,
                output_root=output_root,
                reference_id=ref_id,
                sam_checkpoint=sam_checkpoint,
                model_type=model_type,
                model_tag=model_tag,
                device=device,
            )

            # -------------------------
            # 2. global point flow
            # -------------------------
            point_result = run_global_point_flow_experiment(
                image_dir=image_dir,
                gt_dir=gt_dir,
                output_root=output_root,
                reference_id=ref_id,
                area_range_txt=prompt_result["area_range_txt"],
                sam_checkpoint=sam_checkpoint,
                model_type=model_type,
                model_tag=model_tag,
                device=device,
            )

            # -------------------------
            # 3. evaluate and merge
            # -------------------------
            eval_result = run_evaluate_and_merge_experiment(
                image_dir=image_dir,
                gt_dir=gt_dir,
                output_root=output_root,
                reference_id=ref_id,
                device=device,
                save_full_visuals=False,
                full_result_root=full_result_root,
            )

            # -------------------------
            # 4. collect nanoseg mean row for summary.csv
            # -------------------------
            mean_row = load_mean_row_from_csv(eval_result["nanoseg_csv"])

            # Replace mean-row label with the reference ID.
            summary_record = {"image": str(ref_id)}
            for col in METRIC_COLUMNS:
                summary_record[col] = mean_row[col]

            summary_records.append(summary_record)

            print(f"[DONE REFERENCE] {ref_id}")
            print(f"[INFO] mean Dice = {summary_record['Dice']}")
            print(f"[INFO] mean IoU  = {summary_record['IoU']}")

        except Exception as e:
            print(f"[ERROR] Reference {ref_id} failed: {e}")
            failed_refs.append(ref_id)
            continue

    # -------------------------
    # 5. Write summary.csv
    # -------------------------
    if len(summary_records) == 0:
        raise RuntimeError("All references failed. No summary can be generated.")

    summary_csv_path = os.path.join(ref_select_root, "summary.csv")
    summary_df = build_summary_csv(summary_records, summary_csv_path)

    print("\n" + "=" * 100)
    print(f"[SAVED] summary.csv -> {summary_csv_path}")
    print("=" * 100)

    # -------------------------
    # 6. Select best reference
    # -------------------------
    best_record = select_best_reference(summary_records)
    best_ref_id = str(best_record["image"])

    print("\n" + "=" * 100)
    print("[BEST REFERENCE SELECTED]")
    print(f"[INFO] best_ref_id = {best_ref_id}")
    print(f"[INFO] Dice        = {best_record['Dice']}")
    print(f"[INFO] IoU         = {best_record['IoU']}")
    print("=" * 100)

    # -------------------------
    # 7. Rerun best reference with full outputs
    # -------------------------
    print("\n" + "=" * 100)
    print(f"[RERUN BEST REFERENCE FOR FULL OUTPUT] {best_ref_id}")
    print("=" * 100)

    best_prompt_dir = os.path.join(bbox_unet_all_dir, best_ref_id)
    best_output_root = os.path.join(ref_select_root, best_ref_id)

    # Rerun prompt flow
    best_prompt_result = run_prompt_flow_experiment(
        image_dir=image_dir,
        gt_dir=gt_dir,
        prompt_dir=best_prompt_dir,
        output_root=best_output_root,
        reference_id=best_ref_id,
        sam_checkpoint=sam_checkpoint,
        model_type=model_type,
        model_tag=model_tag,
        device=device,
    )

    # Rerun point flow
    best_point_result = run_global_point_flow_experiment(
        image_dir=image_dir,
        gt_dir=gt_dir,
        output_root=best_output_root,
        reference_id=best_ref_id,
        area_range_txt=best_prompt_result["area_range_txt"],
        sam_checkpoint=sam_checkpoint,
        model_type=model_type,
        model_tag=model_tag,
        device=device,
    )

    # Rerun evaluation with full visuals, overwrite standard result dirs
    best_eval_result = run_evaluate_and_merge_experiment(
        image_dir=image_dir,
        gt_dir=gt_dir,
        output_root=best_output_root,
        reference_id=best_ref_id,
        device=device,
        save_full_visuals=True,
        full_result_root=full_result_root,
    )

    print("\n" + "=" * 100)
    print("[BATCH FINISHED]")
    print(f"[SUMMARY] total refs     = {len(ref_ids)}")
    print(f"[SUMMARY] failed refs    = {len(failed_refs)}")
    if len(failed_refs) > 0:
        print(f"[SUMMARY] failed list    = {failed_refs}")
    print(f"[SUMMARY] best ref id    = {best_ref_id}")
    print(f"[SUMMARY] summary.csv    = {summary_csv_path}")
    print(f"[SUMMARY] full result dir= {full_result_root}")
    print("=" * 100)

    return {
        "summary_csv": summary_csv_path,
        "best_ref_id": best_ref_id,
        "failed_refs": failed_refs,
        "summary_df": summary_df,
    }


# ----------------------------------------
# Standalone direct run
# ----------------------------------------
def main():
    run_batch_experiments(
        image_dir=IMAGE_DIR,
        gt_dir=GT_DIR,
        bbox_unet_all_dir=BBOX_UNET_ALL_DIR,
        ref_select_root=REF_SELECT_ROOT,
        full_result_root=FULL_RESULT_ROOT,
        sam_checkpoint=SAM_CHECKPOINT,
        model_type=MODEL_TYPE,
        model_tag=MODEL_TAG,
        device=DEFAULT_DEVICE,
    )


if __name__ == "__main__":
    main()