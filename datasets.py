"""
Dataset and DataLoader Utilities for Segmentation Training

This module provides dataset definitions and dataloader builders for
nanoparticle segmentation tasks.

Main features
-------------
- Paired image-mask loading
- Albumentations preprocessing
- Binary-mask conversion
- Fixed training-image selection
- Train/validation dataloader generation

The module is designed for U-Net and NanoSeg training pipelines.
"""

import os
from glob import glob
from typing import List, Tuple

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import cv2
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2


VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------
# Dataset
# ----------------------------------------
class SegmentationDataset(data.Dataset):
    """Segmentation dataset with paired image-mask loading."""

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transforms=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask[mask > 0] = 1

        return image, mask


# ----------------------------------------
# Transforms
# ----------------------------------------
def build_train_transforms(image_size: int):
    """Build training augmentations."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])


def build_val_transforms(image_size: int):
    """Build validation transforms."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2(),
    ])


# ----------------------------------------
# Dataset utilities
# ----------------------------------------
def list_image_paths(image_dir: str):
    """List valid image paths."""
    paths = []

    for ext in VALID_EXTS:
        paths.extend(glob(os.path.join(image_dir, f"*{ext}")))

    return sorted(paths)


def split_train_val(
    image_paths: List[str],
    mask_paths: List[str],
    train_img_names: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split dataset into fixed training and validation subsets."""
    train_image_paths = [
        path for path in image_paths
        if os.path.basename(path) in train_img_names
    ]

    train_mask_paths = [
        path for path in mask_paths
        if os.path.basename(path) in train_img_names
    ]

    val_image_paths = [
        path for path in image_paths
        if os.path.basename(path) not in train_img_names
    ]

    val_mask_paths = [
        path for path in mask_paths
        if os.path.basename(path) not in train_img_names
    ]

    return (
        train_image_paths,
        train_mask_paths,
        val_image_paths,
        val_mask_paths,
    )


# ----------------------------------------
# Dataloader builder
# ----------------------------------------
def get_data(
    data_root: str,
    batch_size: int = 32,
    train_img_names: List[str] = None,
    image_size: int = 1024,
    num_workers: int = 0,
):
    """
    Build train and validation dataloaders.

    Dataset structure
    -----------------
    data_root/
        image/
        label/
    """
    image_dir = os.path.join(data_root, "image")
    mask_dir = os.path.join(data_root, "label")

    image_paths = list_image_paths(image_dir)
    mask_paths = list_image_paths(mask_dir)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in: {image_dir}")

    if len(mask_paths) == 0:
        raise ValueError(f"No masks found in: {mask_dir}")

    if train_img_names is None or len(train_img_names) == 0:
        raise ValueError("train_img_names must be provided.")

    (
        train_image_paths,
        train_mask_paths,
        val_image_paths,
        val_mask_paths,
    ) = split_train_val(
        image_paths=image_paths,
        mask_paths=mask_paths,
        train_img_names=train_img_names,
    )

    if len(train_image_paths) == 0:
        raise ValueError("No training images selected.")

    if len(val_image_paths) == 0:
        raise ValueError("No validation images available.")

    train_dataset = SegmentationDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        transforms=build_train_transforms(image_size),
    )

    val_dataset = SegmentationDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        transforms=build_val_transforms(image_size),
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
