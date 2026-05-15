"""
Utility Functions for NanoSeg Pipeline

This module provides lightweight helper utilities used across
the NanoSeg segmentation pipeline.

Main features
-------------
- Bounding-box prompt parsing
- Bounding-box boundary clipping
- Coordinate sanitization utilities
"""

import os
from typing import List, Tuple

def read_labels_txt(txt_path: str) -> List[Tuple[int, int, int, int, int]]:
    """Load bounding-box prompts from a text file."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x1 = int(float(parts[1]))
            y1 = int(float(parts[2]))
            x2 = int(float(parts[3]))
            y2 = int(float(parts[4]))
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def clip_box_to_image(x1, y1, x2, y2, w, h):
    """Clip bounding-box coordinates to image boundaries."""
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2
