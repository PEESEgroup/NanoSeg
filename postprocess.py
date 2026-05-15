"""
Segmentation Postprocessing Utilities

This module provides postprocessing operations for nanoparticle
segmentation masks.

Main features
-------------
- Contrast-based mask refinement
- Bounding-box background comparison
- Edge-touching object removal
- Final mask cleanup pipeline

The module is used to suppress low-confidence regions and improve
segmentation quality before evaluation.
"""

import cv2
import numpy as np
from skimage.measure import label, regionprops


def process_mask(mask: np.ndarray, image: np.ndarray,
                 expand_ratio: float = 1.8,
                 contrast_thresh: float = 20.0) -> np.ndarray:
    """
    Filter mask regions by contrast with expanded bounding box.

    Args:
        mask: binary mask (uint8, 0/255)
        image: corresponding RGB image (uint8)
        expand_ratio: factor to expand the bounding box
        contrast_thresh: intensity difference threshold
    Returns:
        filtered mask (uint8, 0/255)
    """
    # Ensure binary mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks_to_keep = np.zeros_like(mask, dtype=np.uint8)
    H, W = mask.shape[:2]

    for contour in contours:
        single_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)

        x, y, w, h = cv2.boundingRect(contour)
        new_w, new_h = int(w * expand_ratio), int(h * expand_ratio)
        cx, cy = x + w // 2, y + h // 2
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(W, cx + new_w // 2)
        y2 = min(H, cy + new_h // 2)

        mask_pixels = image[single_mask > 0]
        mask_mean = np.mean(mask_pixels) if len(mask_pixels) > 0 else 0

        bbox_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
        bbox_minus_mask = cv2.bitwise_and(image, image, mask=bbox_mask)
        bbox_minus_mask[single_mask > 0] = 0
        valid_pixels = bbox_minus_mask[bbox_minus_mask > 0]
        bbox_minus_mean = np.mean(valid_pixels) if len(valid_pixels) > 0 else 0

        if (mask_mean - bbox_minus_mean) >= contrast_thresh:
            cv2.drawContours(masks_to_keep, [contour], -1, 255, thickness=cv2.FILLED)

    return masks_to_keep


def edge_filter(mask: np.ndarray, boundary_margin: int = 2,
                target_size: int = 1024) -> np.ndarray:
    """
    Remove objects touching the border of the image.

    Args:
        mask: binary mask (uint8, 0/255)
        boundary_margin: safe margin from border
        target_size: assume input is resized to this size
    Returns:
        filtered mask (uint8, 0/255)
    """
    labeled_mask = label(mask)
    filtered_mask = np.zeros_like(mask)

    inner_x_min, inner_x_max = boundary_margin, target_size - boundary_margin
    inner_y_min, inner_y_max = boundary_margin, target_size - boundary_margin

    for region in regionprops(labeled_mask):
        min_row, min_col, max_row, max_col = region.bbox
        if (min_row >= inner_y_min and max_row <= inner_y_max and
                min_col >= inner_x_min and max_col <= inner_x_max):
            filtered_mask[labeled_mask == region.label] = 255

    return filtered_mask


def postprocess_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Complete postprocessing pipeline: process + edge filter.
    """
    processed = process_mask(mask, image)
    final_mask = edge_filter(processed, boundary_margin=2, target_size=mask.shape[0])
    return final_mask
