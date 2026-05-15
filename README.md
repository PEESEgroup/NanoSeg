# NanoSeg

NanoSeg is a one-shot nanoparticle segmentation framework for electron microscopy images based on the Segment Anything Model (SAM). The pipeline integrates automatic prompt generation, prompt-guided segmentation, postprocessing, and particle-level descriptor extraction for reproducible catalyst microscopy analysis.

The framework supports:
- one-shot SAM segmentation
- automatic bbox prompt generation
- reference image selection
- segmentation benchmarking
- particle descriptor extraction

---

## Installation

All dependencies are provided in `requirements.txt`.

Create a new environment and install dependencies:

```bash
conda create -n nanoseg python=3.10
conda activate nanoseg

pip install -r requirements.txt
```

---

## Project Structure

```text
NanoSeg/
├── train_integrated_clean.py
├── datasets_clean.py
├── prompt_flow.py
├── global_point_flow.py
├── bbox_generation.py
├── GradCAM.py
├── benchmark_unet.py
├── benchmark_ham.py
├── benchmark_sam_variants.py
├── simple_point.py
├── ref_pick.py
├── particle_db.py
├── metrics.py
├── postprocess.py
├── utils.py
├── requirements.txt
└── data/
```

---

## Main Pipeline

### 1. Train U-Net

```bash
python train_integrated_clean.py \
    --data_root path/to/dataset \
    --results_dir path/to/output
```

### 2. Generate Grad-CAM

```bash
python GradCAM.py \
    --image_dir path/to/images \
    --checkpoint path/to/checkpoint.pt \
    --output_dir path/to/cam_output
```

### 3. Generate Bounding-Box Prompts

```bash
python bbox_generation.py \
    --image_dir path/to/cam_images \
    --output_dir path/to/prompts
```

### 4. Run Prompt-Guided SAM Segmentation

```bash
python prompt_flow.py
```

### 5. Run Full NanoSeg Pipeline

```bash
python run_batch.py
```

### 6. Extract Particle Descriptors

```bash
python particle_db.py \
    --mask_dir path/to/masks \
    --output_dir path/to/output
```

---

## Benchmark Scripts

```text
benchmark_unet.py          # One-shot U-Net baseline
benchmark_ham.py           # Fully supervised baseline
benchmark_sam_variants.py  # MicroSAM / MedSAM comparison
simple_point.py            # Single-point SAM baseline
artifacts_augmentation.py  # Robustness evaluation
```

---

## Citation

If you use NanoSeg in your research, please cite the corresponding work.