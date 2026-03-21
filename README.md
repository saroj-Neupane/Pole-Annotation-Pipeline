# Pole Calibration & Annotation Pipeline

A multi-stage computer vision system for automated detection, calibration, and annotation of electric utility pole infrastructure from field photographs. The pipeline combines YOLOv11 object detection with HRNet-W32 keypoint localization across 13+ specialized models to measure pole heights, detect equipment, and localize wire attachments.

**[Live Demo](https://pole-annotation-app-gyq2qukkaq-uc.a.run.app/)** — External deployment; app source is not included in this repository

## Sample Results

<p align="center">
  <img src="assets/Pole_Calibration.jpg" width="270" alt="Calibration: ruler markings + pole top detection"/>
  <img src="assets/Pole_Annotation.jpg" width="270" alt="Annotation: equipment + attachment detection"/>
  <img src="assets/Midspan_Calibration.jpg" width="270" alt="Midspan ruler calibration"/>
</p>
<p align="center">
  <em>Left: Calibration pipeline — ruler markings (2.5–16.5 ft) and pole top keypoints. Center: Annotation pipeline — equipment and attachment detection with keypoints. Right: Midspan calibration on a different photo type.</em>
</p>

## Pipeline Architecture

```
                           ┌─────────────────────────────────────────┐
                           │           Input: Pole Photo             │
                           └──────────────────┬──────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
        ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
        │   CALIBRATION     │   │    EQUIPMENT       │   │    ATTACHMENT      │
        │   PIPELINE        │   │    PIPELINE        │   │    PIPELINE        │
        │                   │   │                    │   │                    │
        │ 1. Pole Detection │   │ 1. Pole Detection  │   │ 1. Pole Detection  │
        │    (YOLOv11s)     │   │    (YOLOv11s)      │   │    (YOLOv11s)      │
        │        │          │   │        │           │   │        │           │
        │ 2. Ruler Detection│   │ 2. Upper 70% Crop  │   │ 2. Upper 70% Crop  │
        │    (YOLOv11s)     │   │    (2:5 aspect)    │   │    (2:5 aspect)    │
        │        │          │   │        │           │   │        │           │
        │ 3. Ruler Markings │   │ 3. Equipment Det.  │   │ 3. Attachment Det. │
        │    (HRNet-W32)    │   │    (YOLOv11s)      │   │    (YOLOv11s)      │
        │    5 keypoints    │   │    4 classes        │   │    6 classes        │
        │        │          │   │        │           │   │        │           │
        │ 4. Pole Top       │   │ 4. Per-Equipment   │   │ 4. Per-Attachment  │
        │    (HRNet-W32)    │   │    Keypoints       │   │    Keypoints       │
        │    1 keypoint     │   │    (HRNet-W32)     │   │    (HRNet-W32)     │
        │        │          │   │                    │   │                    │
        │ 5. PPI → Height   │   └────────────────────┘   └────────────────────┘
        └───────────────────┘
```

### Key Design Decisions

- **Hierarchical crop strategy**: Upper 70% of pole in 2:5 aspect ratio — balances field-of-view with resolution for equipment/attachment detection
- **Per-class threshold optimization**: F1-maximizing confidence thresholds via automated sweep (not a single global threshold)
- **Task-specific augmentation**: Minimal augmentation for scale-sensitive calibration models; aggressive augmentation (mosaic, mixup, copy-paste) for sparse equipment classes
- **Keypoint interpolation**: Polynomial fitting on confident ruler markings with linear fallback for occluded points

## Results

The metrics below are reported results from prior training/evaluation runs. The
repository currently does **not** include datasets, trained weights, or saved
evaluation outputs, so these numbers are documentation rather than artifacts
that can be verified from the checked-in files alone.

### Calibration Pipeline

| Component | Metric | Value |
|-----------|--------|-------|
| Pole Detection | mAP@0.5 | 0.908 |
| Pole Detection | Detection Rate | 99.2% |
| Ruler Detection | mAP@0.5 | 0.908 |
| Ruler Detection | Detection Rate | 99.8% |
| Ruler Markings (5 keypoints) | Mean Error | **0.335 inches** |
| Ruler Markings | PCK@1 inch | 98.4% |
| Pole Top | Mean Error | **1.49 inches** |
| Pole Top | PCK@3 inches | 95.0% |

### Equipment Detection

| Class | F1 Score | Confidence Threshold |
|-------|----------|---------------------|
| Transformer | 0.965 | 0.361 |
| Street Light | 0.950 | 0.101 |
| Riser | 0.757 | 0.288 |
| Secondary Drip Loop | 0.718 | 0.300 |
| **Overall mAP@0.5** | **0.865** | |

### Attachment Detection

| Class | F1 Score | Confidence Threshold |
|-------|----------|---------------------|
| Comm | 0.882 | 0.213 |
| Neutral | 0.834 | 0.281 |
| Primary | 0.778 | 0.192 |
| Secondary | 0.642 | 0.131 |
| Down Guy | 0.628 | 0.141 |
| Guy | 0.493 | 0.211 |
| **Overall mAP@0.5** | **0.744** | |

> **Note**: Wire classes (guy, down guy) remain challenging due to thin visual profiles, high occlusion, and limited training data. Cable classes (comm, primary, neutral) perform significantly better.

## Models

| Model | Architecture | Task |
|-------|-------------|------|
| `pole_detection` | YOLOv11s | Bounding box detection of utility poles |
| `ruler_detection` | YOLOv11s | Bounding box detection of ruler scale bars |
| `ruler_marking_detection` | HRNet-W32 | 5 keypoints on ruler markings (2.5, 6.5, 10.5, 14.5, 16.5 ft) |
| `pole_top_detection` | HRNet-W32 | Single keypoint at pole top |
| `equipment_detection` | YOLOv11s | 4-class detection (riser, transformer, street light, secondary drip loop) |
| `riser_keypoint_detection` | HRNet-W32 | 1 keypoint (top) |
| `transformer_keypoint_detection` | HRNet-W32 | 2 keypoints (top bolt, bottom) |
| `street_light_keypoint_detection` | HRNet-W32 | 3 keypoints (upper bracket, lower bracket, drip loop) |
| `secondary_drip_loop_keypoint_detection` | HRNet-W32 | 1 keypoint (lowest point) |
| `attachment_detection` | YOLOv11s | 6-class detection (comm, down guy, primary, secondary, neutral, guy) |
| `comm_keypoint_detection` | HRNet-W32 | 1 keypoint (center) |
| `down_guy_keypoint_detection` | HRNet-W32 | 1 keypoint (center) |
| `primary_keypoint_detection` | HRNet-W32 | 1 keypoint (center) |
| `secondary_keypoint_detection` | HRNet-W32 | 1 keypoint (center) |
| `neutral_keypoint_detection` | HRNet-W32 | 1 keypoint (center) |
| `guy_keypoint_detection` | HRNet-W32 | 1 keypoint (center) |

## Project Structure

```
├── src/
│   ├── config.py                 # Single source of truth: paths, models, thresholds
│   ├── models.py                 # HRNet-W32 keypoint architecture
│   ├── training_utils.py         # Training loops for YOLO and keypoint models
│   ├── inference.py              # High-level inference API
│   ├── inference_utils.py        # Low-level detection and keypoint inference
│   ├── inference_pipelines.py    # Pre-configured pipelines (calibration, equipment, attachment)
│   ├── datasets.py               # PyTorch dataset classes with augmentation
│   ├── evaluation_utils.py       # mAP, PCK, IoU calculation
│   ├── evaluation_charts.py      # Metric visualization
│   ├── threshold_utils.py        # F1-maximizing threshold sweep
│   ├── visualization.py          # Bounding box and keypoint drawing
│   ├── losses.py                 # Focal heatmap loss
│   └── ...
├── scripts/
│   ├── prepare_dataset.py        # Convert raw labels to YOLO format
│   ├── evaluate_models.py        # Run E2E evaluation
│   ├── threshold_sweep.py        # F1-maximize per-class thresholds
│   ├── train_all_overnight.py    # Automated multi-model training
│   └── ...
├── notebooks/
│   ├── calibration/              # Data exploration, training, evaluation, inference
│   ├── equipment/                # Same structure
│   ├── attachment/               # Same structure
│   └── E2E_Production.ipynb      # Full pipeline demo
├── train.py                      # Unified training entry point
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the core runtime stack for training/inference, but
several scripts and notebooks import additional packages that are not currently
pinned there, including:

```bash
pip install matplotlib pandas scikit-learn tqdm pyyaml
```

Download pretrained HRNet weights:
```bash
python scripts/download_hrnet_pretrained.py
```

## Repository State

This repository contains the pipeline code, notebooks, and helper scripts, but
large artifacts are intentionally not versioned:

- `data/` is a placeholder in git; raw pole/midspan photos and labels are not included
- `models/` is a placeholder in git; pretrained/downloaded weights are not included
- `datasets/`, `runs/`, and `results/` are generated locally and are not present in a fresh clone

Most training, dataset-preparation, and evaluation commands in this README
assume you already have the source data available locally under `data/`.

## Training

```bash
# Train any model by name
python train.py --model pole_detection
python train.py --model ruler_marking_detection --epochs 150 --batch-size 64
python train.py --model equipment_detection --warm-start --epochs 50

# Train all models sequentially
python scripts/train_all_overnight.py
```

Prepared datasets are expected under `datasets/<model_name>/` with YOLO format
(for detection) or the repository's custom keypoint format (for HRNet models).
Use `scripts/prepare_dataset.py` to build them from local raw labels once the
source data exists under `data/`.

Weights are saved to `runs/<model_name>/weights/best.pt` (YOLO) or `best.pth` (HRNet).

## Inference

```python
from pathlib import Path
from src.inference import load_all_models, run_end_to_end_inference_simple

models = load_all_models()
results = run_end_to_end_inference_simple(
    Path("path/to/image.jpg"), models, use_tta=True, show_visualization=True
)
```

## Evaluation

```bash
# Evaluate calibration pipeline
python scripts/evaluate_models.py --calibration

# Evaluate equipment detection
python scripts/evaluate_models.py --equipment

# Evaluate attachment detection
python scripts/evaluate_models.py --attachment

# Run threshold sweep to optimize per-class confidence
python scripts/threshold_sweep.py --update-config
```

## Notebooks

Each pipeline has 4 notebooks for the full ML workflow:

| Notebook | Purpose |
|----------|---------|
| `001_Data_Exploration.ipynb` | Dataset statistics, class distribution, sample visualization |
| `002_Model_Training.ipynb` | Training with hyperparameter exploration |
| `003_Evaluation.ipynb` | Metrics computation, error analysis, threshold tuning |
| `004_Inference.ipynb` | End-to-end inference with visualization |

## Tech Stack

- **Detection**: [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) (small variant)
- **Keypoints**: Custom [HRNet-W32](https://arxiv.org/abs/1902.09212) with Gaussian heatmap regression
- **Training**: PyTorch, TensorBoard
- **Evaluation**: Custom mAP, PCK, IoU implementations

## License

MIT License - see [LICENSE](LICENSE) for details.
