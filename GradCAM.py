"""
Grad-CAM Visualization for Segmentation Models

This script generates Grad-CAM visualizations for a trained U-Net
segmentation model.

Main features
-------------
- Load trained segmentation checkpoint
- Run foreground segmentation prediction
- Generate semantic-segmentation Grad-CAM maps
- Overlay CAM heatmaps on input images
- Batch-process image folders

The script is designed for model interpretability analysis in
nanoparticle segmentation tasks.
"""

import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from torchvision import transforms

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------
# Model utilities
# ----------------------------------------
def build_model(
    encoder_name: str = "resnet18",
    num_classes: int = 2,
    in_channels: int = 3,
) -> torch.nn.Module:
    """Build the segmentation model architecture."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model weights from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict = {
            k.replace("module.", "") if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
    elif isinstance(checkpoint, dict):
        state_dict = {
            k.replace("module.", "") if k.startswith("module.") else k: v
            for k, v in checkpoint.items()
        }
        model.load_state_dict(state_dict, strict=True)
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    model = model.float().to(device)
    model.eval()
    return model


def get_target_layer(model: torch.nn.Module, layer_name: str):
    """Select the target layer used for Grad-CAM."""
    if layer_name == "decoder_last":
        return model.decoder.blocks[-1].conv1
    if layer_name == "decoder_mid":
        return model.decoder.blocks[len(model.decoder.blocks) // 2].conv1
    if layer_name == "encoder_last":
        return model.encoder.layer4[-1]

    raise ValueError(
        f"Unsupported target layer: {layer_name}. "
        "Choose from: decoder_last, decoder_mid, encoder_last."
    )


# ----------------------------------------
# Image utilities
# ----------------------------------------
def list_images(image_dir: str):
    """List valid images in a folder."""
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(VALID_EXTS)]
    return sorted(files)


def read_image_rgb_float(image_path: str, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read image, resize, and return both BGR uint8 and RGB float versions."""
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image_bgr = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    image_rgb_float = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    return image_bgr, image_rgb_float


def build_input_tensor(image_rgb_float: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert RGB float image to model input tensor."""
    preprocess = transforms.Compose([transforms.ToTensor()])
    return preprocess(image_rgb_float).unsqueeze(0).to(device).float()


# ----------------------------------------
# Grad-CAM generation
# ----------------------------------------
def generate_segmentation_cam(
    model: torch.nn.Module,
    cam: GradCAM,
    input_tensor: torch.Tensor,
    image_rgb_float: np.ndarray,
    target_class: int,
) -> np.ndarray:
    """Generate one Grad-CAM overlay for semantic segmentation."""
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = output.argmax(dim=1).squeeze().detach().cpu().numpy()

    target_mask = np.zeros(pred_mask.shape, dtype=np.uint8)
    target_mask[pred_mask == target_class] = 1

    if target_mask.sum() == 0:
        grayscale_cam = np.zeros_like(target_mask, dtype=np.float32)
    else:
        targets = [SemanticSegmentationTarget(target_class, target_mask)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    cam_image = show_cam_on_image(image_rgb_float, grayscale_cam, use_rgb=True)
    return cam_image


def run_gradcam_batch(
    image_dir: str,
    checkpoint_path: str,
    output_dir: str,
    encoder_name: str = "resnet18",
    num_classes: int = 2,
    in_channels: int = 3,
    target_size: int = 1024,
    target_class: int = 1,
    target_layer_name: str = "decoder_last",
    device: torch.device = None,
) -> None:
    """Run Grad-CAM visualization for all images in a folder."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    os.makedirs(output_dir, exist_ok=True)

    model = build_model(
        encoder_name=encoder_name,
        num_classes=num_classes,
        in_channels=in_channels,
    )
    model = load_checkpoint(model, checkpoint_path, device)

    target_layer = get_target_layer(model, target_layer_name)
    cam = GradCAM(model=model, target_layers=[target_layer])

    image_files = list_images(image_dir)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No valid images found in: {image_dir}")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Image directory: {image_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Target layer: {target_layer_name}")
    print(f"[INFO] Target class: {target_class}")
    print(f"[INFO] Total images: {len(image_files)}")

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)

        try:
            _, image_rgb_float = read_image_rgb_float(image_path, target_size)
            input_tensor = build_input_tensor(image_rgb_float, device)

            cam_image = generate_segmentation_cam(
                model=model,
                cam=cam,
                input_tensor=input_tensor,
                image_rgb_float=image_rgb_float,
                target_class=target_class,
            )

            save_path = os.path.join(output_dir, image_name)
            cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            print(f"[SAVED] {save_path}")

        except Exception as exc:
            print(f"[WARN] Failed on {image_name}: {exc}")
            continue

    print("[DONE] Grad-CAM visualization finished.")


# ----------------------------------------
# Command-line interface
# ----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for a trained U-Net segmentation model."
    )

    parser.add_argument("--image_dir", default="path/to/images")
    parser.add_argument("--checkpoint", default="path/to/model.pt")
    parser.add_argument("--output_dir", default="path/to/output/gradcam")

    parser.add_argument("--encoder_name", default="resnet18")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--target_size", type=int, default=1024)
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument(
        "--target_layer",
        default="decoder_last",
        choices=["decoder_last", "decoder_mid", "encoder_last"],
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_gradcam_batch(
        image_dir=args.image_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        encoder_name=args.encoder_name,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        target_size=args.target_size,
        target_class=args.target_class,
        target_layer_name=args.target_layer,
        device=device,
    )


if __name__ == "__main__":
    main()
