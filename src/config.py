"""
Configuration constants for Pole Annotation.

Per-model sections: training config, augmentation, inference weights and thresholds.
Inference confidence: single source of truth per model. Per-class thresholds are
F1-maximizing from threshold sweep. Update via: evaluate_models.py --equipment/--attachment
or scripts/threshold_sweep.py --update-config.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

# =============================================================================
# Project & Dataset Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent


def path_relative_to_project(path) -> str:
    """Return path as string relative to project root for clean display."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)


BASE_DIR_POLE = PROJECT_ROOT / "data" / "data_pole"
BASE_DIR_MIDSPAN = PROJECT_ROOT / "data" / "data_midspan"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Dataset names
POLE_DETECTION = "pole_detection"
RULER_DETECTION = "ruler_detection"
RULER_MARKING_DETECTION = "ruler_marking_detection"
POLE_TOP_DETECTION = "pole_top_detection"
RULER_MARKING_DETECTION_MIDSPAN = "ruler_marking_detection_midspan"
EQUIPMENT_DETECTION = "equipment_detection"
ATTACHMENT_DETECTION = "attachment_detection"
RISER_KEYPOINT_DETECTION = "riser_keypoint_detection"
TRANSFORMER_KEYPOINT_DETECTION = "transformer_keypoint_detection"
STREET_LIGHT_KEYPOINT_DETECTION = "street_light_keypoint_detection"
SECONDARY_DRIP_LOOP_KEYPOINT_DETECTION = "secondary_drip_loop_keypoint_detection"

# Derived paths
EQUIPMENT_DATASET_DIR = DATASETS_DIR / "equipment_detection"
ATTACHMENT_DATASET_DIR = DATASETS_DIR / "attachment_detection"
POLE_LABELS_DIR = BASE_DIR_POLE / "Labels"
MIDSPAN_LABELS_DIR = BASE_DIR_MIDSPAN / "Labels"

# Single source of truth: dataset dirs for all trainable models
DATASET_DIRS = {
    POLE_DETECTION: DATASETS_DIR / "pole_detection",
    RULER_DETECTION: DATASETS_DIR / "ruler_detection",
    RULER_MARKING_DETECTION: DATASETS_DIR / "ruler_marking_detection",
    POLE_TOP_DETECTION: DATASETS_DIR / "pole_top_detection",
    EQUIPMENT_DETECTION: EQUIPMENT_DATASET_DIR,
    ATTACHMENT_DETECTION: ATTACHMENT_DATASET_DIR,
    "riser_keypoint_detection": DATASETS_DIR / "riser_keypoint_detection",
    "transformer_keypoint_detection": DATASETS_DIR / "transformer_keypoint_detection",
    "street_light_keypoint_detection": DATASETS_DIR / "street_light_keypoint_detection",
    "comm_keypoint_detection": DATASETS_DIR / "comm_keypoint_detection",
    "down_guy_keypoint_detection": DATASETS_DIR / "down_guy_keypoint_detection",
    "primary_keypoint_detection": DATASETS_DIR / "primary_keypoint_detection",
    "secondary_keypoint_detection": DATASETS_DIR / "secondary_keypoint_detection",
    "neutral_keypoint_detection": DATASETS_DIR / "neutral_keypoint_detection",
    "guy_keypoint_detection": DATASETS_DIR / "guy_keypoint_detection",
    "secondary_drip_loop_keypoint_detection": DATASETS_DIR / "secondary_drip_loop_keypoint_detection",
}

# =============================================================================
# Shared Constants (used by multiple models)
# =============================================================================

HRNET_WEIGHTS_PATH = 'models/hrnet_w32.pth'
YOLO_MODELS_DIR = Path('models')
YOLO_MODEL_PATHS = {
    'nano': YOLO_MODELS_DIR / 'yolo11n.pt',
    'small': YOLO_MODELS_DIR / 'yolo11s.pt',
    'medium': YOLO_MODELS_DIR / 'yolo11m.pt',
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference path resolution (runs/ for dev, models/production/ when USE_PRODUCTION_MODELS=true)
def _model_weights_path(model_name: str, extension: str) -> Path:
    use_production = os.environ.get('USE_PRODUCTION_MODELS', '').lower() in ('true', '1', 'yes')
    if use_production:
        return PROJECT_ROOT / 'models' / 'production' / model_name / 'production' / f'model{extension}'
    return PROJECT_ROOT / 'runs' / model_name / 'weights' / f'best{extension}'

# =============================================================================
# YOLO Default Builder
# =============================================================================

def _yolo_defaults(epochs=100, **overrides):
    base = {
        'batch_size': 16,
        'epochs': epochs,
        'patience': 20,
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'amp': True,
        'dropout': 0.1,
        'imgsz': 960,
        'use_rect': True,
        'model_size': 'small',
    }
    base.update(overrides)
    return base

# =============================================================================
# POLE DETECTION
# =============================================================================

POLE_DETECTION_CONFIG = _yolo_defaults()
POLE_AUGMENT_PARAMS = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=1.0, translate=0.0, scale=0.0, shear=0.0,
    perspective=0.001, fliplr=0.2, flipud=0.0, mosaic=0.0, mixup=0.0,
)
INFERENCE_POLE_WEIGHTS = _model_weights_path('pole_detection', '.pt')
INFERENCE_POLE_CONF_THRESHOLD = 0.01  # catch all poles (critical infrastructure)
# Threshold sweep: python scripts/threshold_sweep.py [--update-config]
# Results saved to runs/threshold_sweep_results.json

# =============================================================================
# RULER DETECTION (Pole photos)
# =============================================================================

RULER_DETECTION_CONFIG = _yolo_defaults()
RULER_AUGMENT_PARAMS = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=5.0, translate=0.05, scale=0.1, shear=2.0,
    perspective=0.001, fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0,
)
INFERENCE_RULER_WEIGHTS = _model_weights_path('ruler_detection', '.pt')
INFERENCE_RULER_CONF_THRESHOLD = 0.01  # permissive: catch all rulers (critical for calibration)

# =============================================================================
# RULER MARKING (Keypoints)
# =============================================================================

KEYPOINT_NAMES = ['2.5', '6.5', '10.5', '14.5', '16.5']
NUM_KEYPOINTS = len(KEYPOINT_NAMES)
RESIZE_HEIGHT, RESIZE_WIDTH = 1440, 96
HEATMAP_HEIGHT, HEATMAP_WIDTH = 1440, 96
GAUSSIAN_SIGMA_X = HEATMAP_WIDTH / 8
GAUSSIAN_SIGMA_Y = HEATMAP_HEIGHT / 32

RULER_MARKING_DETECTION_CONFIG = dict(
    batch_size=32,
    epochs=100,
    patience=40,
    learning_rate=1e-3,
    use_focal_loss=False,
    resize_height=RESIZE_HEIGHT,
    resize_width=RESIZE_WIDTH,
    heatmap_height=HEATMAP_HEIGHT,
    heatmap_width=HEATMAP_WIDTH,
    min_visible_keypoints=5,
    augmentation_params={'brightness': 0.25, 'contrast': 0.25, 'saturation': 0.25},
    geometric_augmentations={'translate_x': 0.05, 'translate_y': 0.05, 'scale_min': 0.97, 'scale_max': 1.05, 'rotate': 5.0},
)
INFERENCE_RULER_MARKING_WEIGHTS = _model_weights_path('ruler_marking_detection', '.pth')

# =============================================================================
# POLE TOP DETECTION
# =============================================================================

POLE_TOP_RESIZE_HEIGHT, POLE_TOP_RESIZE_WIDTH = 256, 192
POLE_TOP_HEATMAP_HEIGHT, POLE_TOP_HEATMAP_WIDTH = 256, 192
POLE_TOP_NUM_KEYPOINTS = 1

POLE_TOP_DETECTION_CONFIG = dict(
    batch_size=96,
    epochs=100,
    patience=40,
    learning_rate=1e-3,
    use_focal_loss=False,
    resize_height=POLE_TOP_RESIZE_HEIGHT,
    resize_width=POLE_TOP_RESIZE_WIDTH,
    heatmap_height=POLE_TOP_HEATMAP_HEIGHT,
    heatmap_width=POLE_TOP_HEATMAP_WIDTH,
    augmentation_params={'brightness': 0.25, 'contrast': 0.25, 'saturation': 0.25},
    geometric_augmentations={'translate_x': 0.10, 'translate_y': 0.30, 'scale_min': 0.95, 'scale_max': 1.05, 'rotate': 5.0},
)
POLE_TOP_AUGMENT_PARAMS = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=3.0, translate=0.03, scale=0.05, shear=1.0,
    perspective=0.0005, fliplr=0.5, flipud=0.0, mosaic=0.0, mixup=0.0,
)
INFERENCE_POLE_TOP_WEIGHTS = _model_weights_path('pole_top_detection', '.pth')

# =============================================================================
# EQUIPMENT DETECTION (Riser, Transformer, Street Light)
# =============================================================================

EQUIPMENT_CLASSES = {'riser': 0, 'transformer': 1, 'street_light': 2, 'secondary_drip_loop': 3}
EQUIPMENT_CLASS_NAMES = [k for k, _ in sorted(EQUIPMENT_CLASSES.items(), key=lambda x: x[1])]

EQUIPMENT_DETECTION_CONFIG = _yolo_defaults(
    epochs=100,
    patience=40,
    dropout=0.15,
    weight_decay=0.003,
    batch_size=48,
    cls=1.5,              # 1.0→1.5: higher cls weight for riser/SDL classification
    lr0=0.0005,           # 0.001→0.0005: lower LR for data-limited riser class
)
EQUIPMENT_AUGMENT_PARAMS = dict(
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,   # hsv_h: 0.015→0.02 (weathered/rusted equipment)
    degrees=5.0, translate=0.08,
    scale=0.5,                            # 0.4→0.5: riser/SDL vary in apparent size
    shear=2.0, perspective=0.001,
    fliplr=0.5, flipud=0.0,
    mosaic=1.0,                           # 0.8→1.0: always-on mosaic for sparse classes
    mixup=0.10,                           # 0.05→0.10
    copy_paste=0.3,                       # 0→0.3: synthetic placement improves riser/SDL recall
)
INFERENCE_EQUIPMENT_WEIGHTS = _model_weights_path('equipment_detection', '.pt')

# Confidence thresholds: F1-maximizing per class from threshold sweep.
# Update via: python scripts/evaluate_models.py --equipment or scripts/threshold_sweep.py --update-config
INFERENCE_EQUIPMENT_CONF_THRESHOLD = 0.2983  # fallback for unknown classes
INFERENCE_EQUIPMENT_CONF_PER_CLASS = {
    'riser': 0.2883,
    'transformer': 0.3614,
    'street_light': 0.1011,
    'secondary_drip_loop': 0.3003
}
INFERENCE_EQUIPMENT_MIN_BBOX_AREA_FRAC = 0.001  # min bbox area as fraction of crop area
INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET = 1  # max detections for secondary_drip_loop class

# =============================================================================
# ATTACHMENT DETECTION (Comm, Down Guy)
# =============================================================================

ATTACHMENT_CLASSES = {
    'comm': 0,
    'down_guy': 1,
    'primary': 2,
    'secondary': 3,
    'neutral': 4,  # includes open_secondary
    'guy': 5,      # includes power_guy
}
ATTACHMENT_CLASS_NAMES = [k for k, _ in sorted(ATTACHMENT_CLASSES.items(), key=lambda x: x[1])]

ATTACHMENT_DETECTION_CONFIG = _yolo_defaults(
    epochs=100,
    patience=40,
    dropout=0.15,
    weight_decay=0.005,             # 0.001→0.005: reduce overfitting (val cls_loss was 2x train)
    batch_size=48,
    imgsz=960,
    cls=1.5,
    lr0=0.0003,                     # 0.0005→0.0003: finer optimization for subtle down_guy
)
ATTACHMENT_AUGMENT_PARAMS = dict(
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,   # hsv_h: 0.015→0.02 (weathered hardware)
    degrees=5.0, translate=0.10,          # translate: 0.08→0.10 (comm/down_guy vary in height on pole)
    scale=0.6,                            # 0.5→0.6: down_guy size varies enormously
    shear=2.0, perspective=0.001,
    fliplr=0.5, flipud=0.0,
    mosaic=1.0,                           # 0.8→1.0: always-on mosaic for subtle down_guy
    mixup=0.1,                            # 0.0→0.1: helps subtle class separation
    copy_paste=0.3,                       # 0→0.3: synthetic down_guy placements improve recall
)
INFERENCE_ATTACHMENT_WEIGHTS = _model_weights_path('attachment_detection', '.pt')
# F1-maximizing per class. Update via: evaluate_models.py --attachment or threshold_sweep.py --update-config
INFERENCE_ATTACHMENT_CONF_THRESHOLD = 0.1752  # fallback
INFERENCE_ATTACHMENT_CONF_PER_CLASS = {
    'comm': 0.2132,
    'down_guy': 0.1411,
    'primary': 0.1922,
    'secondary': 0.1311,
    'neutral': 0.2813,
    'guy': 0.2112
}
INFERENCE_ATTACHMENT_MIN_BBOX_AREA_FRAC = 0.001  # min bbox area as fraction of crop area

# =============================================================================
# ATTACHMENT KEYPOINT DETECTION (1 keypoint: center) - Factory to reduce duplication
# =============================================================================

def _attachment_keypoint_config(resize_height: int, resize_width: int, **overrides) -> dict:
    """Build attachment keypoint config. Single keypoint (center) with shared defaults."""
    base = dict(
        batch_size=32,
        epochs=100,
        patience=40,
        learning_rate=1e-3,
        use_focal_loss=False,
        resize_height=resize_height,
        resize_width=resize_width,
        heatmap_height=resize_height,
        heatmap_width=resize_width,
        augmentation_params={'brightness': 0.25, 'contrast': 0.25, 'saturation': 0.25},
        geometric_augmentations={'translate_x': 0.10, 'translate_y': 0.10, 'scale_min': 0.90, 'scale_max': 1.10, 'rotate': 10.0},
    )
    base.update(overrides)
    return base


COMM_KEYPOINT_DETECTION = "comm_keypoint_detection"
COMM_NUM_KEYPOINTS = 1
COMM_KEYPOINT_NAMES = ['attachment']
COMM_KEYPOINT_DETECTION_CONFIG = _attachment_keypoint_config(192, 384)
INFERENCE_COMM_KEYPOINT_WEIGHTS = _model_weights_path('comm_keypoint_detection', '.pth')

DOWN_GUY_KEYPOINT_DETECTION = "down_guy_keypoint_detection"
DOWN_GUY_NUM_KEYPOINTS = 1
DOWN_GUY_KEYPOINT_NAMES = ['attachment']
DOWN_GUY_KEYPOINT_DETECTION_CONFIG = _attachment_keypoint_config(
    512, 256,
    use_focal_loss=False,                # focal loss for hard/ambiguous attachment point localization
    # High-rotation augmentation: down guys are diagonal wires at varying angles (up to 45°)
    augmentation_params={'brightness': 0.40, 'contrast': 0.40, 'saturation': 0.30, 'hue': 0.05, 'erasing_prob': 0.00},
    geometric_augmentations={'translate_x': 0.30, 'translate_y': 0.30, 'scale_min': 0.70, 'scale_max': 1.30, 'rotate': 30.0},
)
INFERENCE_DOWN_GUY_KEYPOINT_WEIGHTS = _model_weights_path('down_guy_keypoint_detection', '.pth')

# primary, secondary, neutral, guy: same config as comm (1ft×2ft bbox)
PRIMARY_KEYPOINT_DETECTION = "primary_keypoint_detection"
SECONDARY_KEYPOINT_DETECTION = "secondary_keypoint_detection"
NEUTRAL_KEYPOINT_DETECTION = "neutral_keypoint_detection"
GUY_KEYPOINT_DETECTION = "guy_keypoint_detection"
INFERENCE_PRIMARY_KEYPOINT_WEIGHTS = _model_weights_path('primary_keypoint_detection', '.pth')
INFERENCE_SECONDARY_KEYPOINT_WEIGHTS = _model_weights_path('secondary_keypoint_detection', '.pth')
INFERENCE_NEUTRAL_KEYPOINT_WEIGHTS = _model_weights_path('neutral_keypoint_detection', '.pth')
INFERENCE_GUY_KEYPOINT_WEIGHTS = _model_weights_path('guy_keypoint_detection', '.pth')

ATTACHMENT_KEYPOINT_CONFIGS = {
    'comm': (COMM_KEYPOINT_DETECTION_CONFIG, COMM_NUM_KEYPOINTS, INFERENCE_COMM_KEYPOINT_WEIGHTS),
    'down_guy': (DOWN_GUY_KEYPOINT_DETECTION_CONFIG, DOWN_GUY_NUM_KEYPOINTS, INFERENCE_DOWN_GUY_KEYPOINT_WEIGHTS),
    'primary': (_attachment_keypoint_config(192, 384), 1, INFERENCE_PRIMARY_KEYPOINT_WEIGHTS),
    'secondary': (_attachment_keypoint_config(192, 384), 1, INFERENCE_SECONDARY_KEYPOINT_WEIGHTS),
    'neutral': (_attachment_keypoint_config(192, 384), 1, INFERENCE_NEUTRAL_KEYPOINT_WEIGHTS),
    'guy': (_attachment_keypoint_config(192, 384), 1, INFERENCE_GUY_KEYPOINT_WEIGHTS),
}

# =============================================================================
# RISER KEYPOINT DETECTION
# =============================================================================

RISER_NUM_KEYPOINTS = 1
RISER_KEYPOINT_NAMES = ['top']

RISER_KEYPOINT_DETECTION_CONFIG = dict(
    batch_size=64,
    epochs=100,
    patience=40,
    learning_rate=1e-3,
    use_focal_loss=False,             
    resize_height=384,
    resize_width=144,
    heatmap_height=384,
    heatmap_width=144,
    augmentation_params={'brightness': 0.35, 'contrast': 0.35, 'saturation': 0.30, 'hue': 0.05, 'erasing_prob': 0.00},
    geometric_augmentations={'translate_x': 0.10, 'translate_y': 0.05, 'scale_min': 0.80, 'scale_max': 1.20, 'rotate': 15.0},
)
INFERENCE_RISER_KEYPOINT_WEIGHTS = _model_weights_path('riser_keypoint_detection', '.pth')

# =============================================================================
# TRANSFORMER KEYPOINT DETECTION
# =============================================================================

TRANSFORMER_NUM_KEYPOINTS = 2
TRANSFORMER_KEYPOINT_NAMES = ['top_bolt', 'bottom']

TRANSFORMER_KEYPOINT_DETECTION_CONFIG = dict(
    batch_size=48,
    epochs=100,
    patience=40,
    learning_rate=1e-3,
    use_focal_loss=False,              
    resize_height=384,
    resize_width=288,
    heatmap_height=384,
    heatmap_width=288,
    # Stronger augmentation: transformers can be tilted, mounted at angles, varying lighting
    augmentation_params={'brightness': 0.35, 'contrast': 0.35, 'saturation': 0.30, 'hue': 0.05, 'erasing_prob': 0.00},
    geometric_augmentations={'translate_x': 0.15, 'translate_y': 0.15, 'scale_min': 0.80, 'scale_max': 1.20, 'rotate': 15.0},
)
INFERENCE_TRANSFORMER_KEYPOINT_WEIGHTS = _model_weights_path('transformer_keypoint_detection', '.pth')

# =============================================================================
# STREET LIGHT KEYPOINT DETECTION
# =============================================================================

STREET_LIGHT_NUM_KEYPOINTS = 3
STREET_LIGHT_KEYPOINT_NAMES = ['upper_bracket', 'lower_bracket', 'drip_loop']

STREET_LIGHT_KEYPOINT_DETECTION_CONFIG = dict(
    batch_size=16,
    epochs=100,
    patience=40,
    learning_rate=1e-3,
    use_focal_loss=False,              # inst_PCK@1"=22.1%: focal loss for hard 3-keypoint alignment
    resize_height=512,
    resize_width=384,
    heatmap_height=512,
    heatmap_width=384,
    # translate reduced 0.30→0.20: over-augmentation was destroying spatial context for 3-keypoint model
    augmentation_params={'brightness': 0.40, 'contrast': 0.40, 'saturation': 0.40, 'hue': 0.05, 'erasing_prob': 0.00},
    geometric_augmentations={'translate_x': 0.20, 'translate_y': 0.10, 'scale_min': 0.75, 'scale_max': 1.25, 'rotate': 5.0},
)
INFERENCE_STREET_LIGHT_KEYPOINT_WEIGHTS = _model_weights_path('street_light_keypoint_detection', '.pth')

# =============================================================================
# SECONDARY DRIP LOOP KEYPOINT DETECTION
# =============================================================================

SECONDARY_DRIP_LOOP_NUM_KEYPOINTS = 1
SECONDARY_DRIP_LOOP_KEYPOINT_NAMES = ['lowest_point']

SECONDARY_DRIP_LOOP_KEYPOINT_DETECTION_CONFIG = dict(
    batch_size=32,
    epochs=100,
    patience=40,
    learning_rate=1e-3,
    use_focal_loss=False,              
    resize_height=512,
    resize_width=384,
    heatmap_height=512,
    heatmap_width=384,
    # translate_y 0.15→0.30: lowest_point shifts primarily in Y with wire slack variation
    augmentation_params={'brightness': 0.35, 'contrast': 0.35, 'saturation': 0.30, 'hue': 0.05, 'erasing_prob': 0.00},
    geometric_augmentations={'translate_x': 0.10, 'translate_y': 0.10, 'scale_min': 0.75, 'scale_max': 1.25, 'rotate': 5.0},
)
INFERENCE_SECONDARY_DRIP_LOOP_KEYPOINT_WEIGHTS = _model_weights_path('secondary_drip_loop_keypoint_detection', '.pth')

# Equipment keypoint lookup (used by load_keypoint_detector, inference)
EQUIPMENT_KEYPOINT_CONFIGS = {
    'riser': (RISER_KEYPOINT_DETECTION_CONFIG, RISER_NUM_KEYPOINTS, INFERENCE_RISER_KEYPOINT_WEIGHTS),
    'transformer': (TRANSFORMER_KEYPOINT_DETECTION_CONFIG, TRANSFORMER_NUM_KEYPOINTS, INFERENCE_TRANSFORMER_KEYPOINT_WEIGHTS),
    'street_light': (STREET_LIGHT_KEYPOINT_DETECTION_CONFIG, STREET_LIGHT_NUM_KEYPOINTS, INFERENCE_STREET_LIGHT_KEYPOINT_WEIGHTS),
    'secondary_drip_loop': (SECONDARY_DRIP_LOOP_KEYPOINT_DETECTION_CONFIG, SECONDARY_DRIP_LOOP_NUM_KEYPOINTS, INFERENCE_SECONDARY_DRIP_LOOP_KEYPOINT_WEIGHTS),
}

# Unified keypoint configs for training (equipment + attachment)
KEYPOINT_DETECTION_CONFIGS = {**EQUIPMENT_KEYPOINT_CONFIGS, **ATTACHMENT_KEYPOINT_CONFIGS}

# Mapping from train.py model name to keypoint_type for KEYPOINT_DETECTION_CONFIGS
KEYPOINT_MODEL_TO_TYPE = {
    'riser_keypoint_detection': 'riser',
    'transformer_keypoint_detection': 'transformer',
    'street_light_keypoint_detection': 'street_light',
    'secondary_drip_loop_keypoint_detection': 'secondary_drip_loop',
    'comm_keypoint_detection': 'comm',
    'down_guy_keypoint_detection': 'down_guy',
    'primary_keypoint_detection': 'primary',
    'secondary_keypoint_detection': 'secondary',
    'neutral_keypoint_detection': 'neutral',
    'guy_keypoint_detection': 'guy',
}

# Keypoint dataset prep: (type, dataset_dir, prep_fn_name)
# prep_fn_name: 'equipment' -> prepare_keypoint_detection_dataset, 'attachment' -> prepare_attachment_keypoint_dataset
KEYPOINT_PREPARE_SPECS = [
    ('riser', 'riser_keypoint_detection', 'equipment'),
    ('transformer', 'transformer_keypoint_detection', 'equipment'),
    ('street_light', 'street_light_keypoint_detection', 'equipment'),
    ('secondary_drip_loop', 'secondary_drip_loop_keypoint_detection', 'equipment'),
    ('comm', 'comm_keypoint_detection', 'attachment'),
    ('down_guy', 'down_guy_keypoint_detection', 'attachment'),
    ('primary', 'primary_keypoint_detection', 'attachment'),
    ('secondary', 'secondary_keypoint_detection', 'attachment'),
    ('neutral', 'neutral_keypoint_detection', 'attachment'),
    ('guy', 'guy_keypoint_detection', 'attachment'),
]

# =============================================================================
# Equipment & Attachment Domain (bbox sizes in feet - used by data prep)
# =============================================================================

RISER_BBOX_HEIGHT_FEET, RISER_BBOX_WIDTH_FEET = 4.0, 1.5
TRANSFORMER_BBOX_HEIGHT_FEET, TRANSFORMER_BBOX_WIDTH_FEET = 4.0, 3.0
STREET_LIGHT_BBOX_HEIGHT_FEET, STREET_LIGHT_BBOX_WIDTH_FEET = 8.0, 6.0
SECONDARY_DRIP_LOOP_BBOX_HEIGHT_FEET, SECONDARY_DRIP_LOOP_BBOX_WIDTH_FEET = 4.0, 3.0
ATTACHMENT_BBOX_HEIGHT_FEET, ATTACHMENT_BBOX_WIDTH_FEET = 1.0, 2.0
DOWN_GUY_BBOX_HEIGHT_FEET, DOWN_GUY_BBOX_WIDTH_FEET = 4.0, 2.0

# =============================================================================
# Inference Settings
# =============================================================================

INFERENCE_MAX_DETECTIONS = 1
INFERENCE_USE_TTA = True
INFERENCE_USE_INTERPOLATION = False

# =============================================================================
# Review Pipeline Config
# =============================================================================

REVIEW_PIPELINE_CONFIG = {
    "ls_url":            "http://localhost:8080",
    "ls_token":          os.getenv("LS_TOKEN", ""),
    "yolo_batch_size":   32,    # images per GPU batch for all YOLO detectors
    "det_conf":          0.01,  # low threshold — all candidates shown as suggestions
    "no_detection_loss": 0.85,  # loss when pole found but nothing else detected
    "top_n":             200,   # default number of samples to upload
    "crops_dir":         str(PROJECT_ROOT / "data" / "e2e_crops"),
}

# =============================================================================
# Output Directories
# =============================================================================

RUNS_DIR = PROJECT_ROOT / 'runs'
RESULTS_DIR = PROJECT_ROOT / 'results'
# Results by domain: calibration, attachment, equipment
RESULTS_CALIBRATION_DIR = RESULTS_DIR / 'calibration'
RESULTS_ATTACHMENT_DIR = RESULTS_DIR / 'attachment'
RESULTS_EQUIPMENT_DIR = RESULTS_DIR / 'equipment'
# Legacy; evaluation_utils saves to domain-specific dirs
EVALUATION_RESULTS_DIR = RESULTS_CALIBRATION_DIR
FROZEN_MANIFEST_FILENAME = 'frozen_manifest.json'

# Master split manifest: single source of truth for train/val/test across all datasets.
# Ensures test images in one dataset are never in train for another.
SPLIT_MANIFEST_PATH = DATASETS_DIR / "split_manifest.json"
SPLIT_MANIFEST_RANDOM_STATE = 42

# Inference paths (used by notebooks - no path construction in notebooks)
POLE_PHOTOS_DIR = BASE_DIR_POLE / "Photos"
MIDSPAN_PHOTOS_DIR = BASE_DIR_MIDSPAN / "Photos"
EQUIPMENT_DETECTION_IMAGES_VAL = EQUIPMENT_DATASET_DIR / "images" / "val"
ATTACHMENT_DETECTION_IMAGES_VAL = ATTACHMENT_DATASET_DIR / "images" / "val"
RISER_KEYPOINT_IMAGES_VAL = DATASET_DIRS["riser_keypoint_detection"] / "images" / "val"
TRANSFORMER_KEYPOINT_IMAGES_VAL = DATASET_DIRS["transformer_keypoint_detection"] / "images" / "val"
STREET_LIGHT_KEYPOINT_IMAGES_VAL = DATASET_DIRS["street_light_keypoint_detection"] / "images" / "val"
SECONDARY_DRIP_LOOP_KEYPOINT_IMAGES_VAL = DATASET_DIRS["secondary_drip_loop_keypoint_detection"] / "images" / "val"
COMM_KEYPOINT_IMAGES_VAL = DATASET_DIRS["comm_keypoint_detection"] / "images" / "val"
DOWN_GUY_KEYPOINT_IMAGES_VAL = DATASET_DIRS["down_guy_keypoint_detection"] / "images" / "val"
PRIMARY_KEYPOINT_IMAGES_VAL = DATASET_DIRS["primary_keypoint_detection"] / "images" / "val"
SECONDARY_KEYPOINT_IMAGES_VAL = DATASET_DIRS["secondary_keypoint_detection"] / "images" / "val"
NEUTRAL_KEYPOINT_IMAGES_VAL = DATASET_DIRS["neutral_keypoint_detection"] / "images" / "val"
GUY_KEYPOINT_IMAGES_VAL = DATASET_DIRS["guy_keypoint_detection"] / "images" / "val"
# E2E evaluation: use TEST split only (data model has never seen).
# Derived from prepared datasets (equipment/attachment) which split with random_state=42.
# Run prepare_dataset.py before E2E eval so test split exists.
EQUIPMENT_E2E_IMAGES_DIR = POLE_PHOTOS_DIR  # Source dir; eval filters by test stems
ATTACHMENT_E2E_IMAGES_DIR = POLE_PHOTOS_DIR
E2E_USE_TEST_SPLIT_ONLY = True  # If True, only evaluate on test split (unseen data)
EVALUATION_YOLO_BATCH_SIZE = 64  # Images per batch for calibration YOLO inference (pole/ruler detection)
KEYPOINT_CHECKPOINTS_DIR = RUNS_DIR / 'keypoint_detection' / 'checkpoints'  # Legacy; trainers use RUNS_DIR/{model}/weights
ANNOTATED_PHOTOS_SUBDIR = 'annotated_photos'
LABELS_SUBDIR = 'labels'

# =============================================================================
# Visualization Colors (RGB tuples - single source of truth)
# =============================================================================
# Semantic roles: GT vs Pred vs Overlap. Per-class: equipment, attachment, keypoints.
# All visualization code imports from here for consistency.

# Ground truth vs Prediction vs Overlap
COLOR_GT = (0, 180, 0)          # Green
COLOR_PRED = (220, 50, 50)      # Red
COLOR_OVERLAP = (255, 220, 0)   # Yellow
COLOR_POLE = (100, 200, 255)   # Blue
COLOR_RULER = (255, 165, 0)     # Orange

# Per-class: equipment (riser, transformer, street_light, secondary_drip_loop)
EQUIPMENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    'riser': (255, 80, 80),
    'transformer': (80, 200, 80),
    'street_light': (80, 120, 255),
    'secondary_drip_loop': (200, 150, 100),
}

# Per-class: attachment (comm, down_guy)
ATTACHMENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    'comm': (100, 200, 255),
    'down_guy': (255, 200, 80),
    'primary': (255, 100, 100),
    'secondary': (255, 180, 80),
    'neutral': (200, 200, 200),
    'guy': (150, 150, 255),
    # Backward compat for merged classes (viz of old labels)
    'open_secondary': (200, 200, 200),
    'power_guy': (150, 150, 255),
}

# Per-keypoint: equipment & attachment keypoint names
KEYPOINT_COLORS: Dict[str, Tuple[int, int, int]] = {
    'attachment': (255, 255, 0),
    'top_bolt': (255, 0, 255),
    'bottom': (0, 255, 255),
    'upper_bracket': (255, 128, 0),
    'lower_bracket': (0, 255, 128),
    'drip_loop': (200, 150, 100),
    'riser_top': (255, 200, 100),
    'lowest_point': (200, 150, 100),
}

# Unified color lookup: bbox AND keypoints share the same color per object class
OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    'pole': COLOR_POLE,
    'ruler': COLOR_RULER,
    **EQUIPMENT_COLORS,
    **ATTACHMENT_COLORS,
}

# Fallback for unknown classes
DEFAULT_UNKNOWN_COLOR = (128, 128, 128)

# Keypoint line/label color for dataset exploration viz (bright yellow, visible on any background)
KEYPOINT_VIZ_LINE_COLOR: Tuple[int, int, int] = (255, 255, 0)

# Fallback for unknown keypoints (deterministic from name hash)
FALLBACK_KEYPOINT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
)

# Ruler marking heights (matplotlib hex for pole/midspan viz)
RULER_MARKING_COLOR_MAP: Dict[str, str] = {
    '0.0': '#FF0000', '2.5': '#0066CC', '6.5': '#00AA00',
    '10.5': '#FF6123', '14.5': '#9900CC', '16.5': '#CC0066', '17.0': '#FFD700',
}

# Chart colors - single source of truth for evaluation charts (matplotlib hex)
CHART_COLORS: Dict[str, str] = {
    'blue': '#3498db',
    'magenta': '#9b59b6',
    'orange': '#f39c12',
    'green': '#2ecc71',
    'red': '#e74c3c',
}
CHART_COLORS_LIST = [CHART_COLORS['green'], CHART_COLORS['blue'], CHART_COLORS['magenta'],
                     CHART_COLORS['red'], CHART_COLORS['orange'], '#1abc9c']

# Figure/style defaults for consistent matplotlib viz
VIZ_FIG_DEFAULTS = {
    'facecolor': 'white',
    'dpi': 100,
    'title_fontsize': 14,
    'title_fontsize_large': 18,  # single-panel detection viz
    'title_fontweight': 'bold',
}
VIZ_BBOX_THICKNESS = 3
VIZ_DETECTION_BBOX_THICKNESS = 4  # pole/ruler single-detection viz
VIZ_LINE_LENGTH_FRAC = 0.08  # line_len = int(w * VIZ_LINE_LENGTH_FRAC)
VIZ_FONT_SCALE_DENOM = 1600  # font_scale = max(0.5, w / VIZ_FONT_SCALE_DENOM)
VIZ_FONT_THICK_DENOM = 800   # font_thick = max(1, int(w / VIZ_FONT_THICK_DENOM))

# =============================================================================
# Visualization & Evaluation
# =============================================================================

VISUALIZATION_DATASETS_CONFIG = {
    "Pole detection": {
        'images_dir': DATASETS_DIR / "pole_detection" / "images" / "val",
        'labels_dir': DATASETS_DIR / "pole_detection" / "labels" / "val",
        'type': 'yolo_bbox',
        'class_names': ["pole"],
    },
    "Ruler detection": {
        'images_dir': DATASETS_DIR / "ruler_detection" / "images" / "val",
        'labels_dir': DATASETS_DIR / "ruler_detection" / "labels" / "val",
        'type': 'yolo_bbox',
        'class_names': ["ruler"],
    },
    "Ruler marking": {
        'images_dir': DATASETS_DIR / "ruler_marking_detection" / "images" / "val",
        'labels_dir': DATASETS_DIR / "ruler_marking_detection" / "labels" / "val",
        'type': 'keypoints',
        'keypoint_names': KEYPOINT_NAMES,
    },
    "Pole top": {
        'images_dir': DATASETS_DIR / "pole_top_detection" / "images" / "val",
        'labels_dir': DATASETS_DIR / "pole_top_detection" / "labels" / "val",
        'type': 'keypoints',
        'keypoint_names': ["pole_top"],
    },
}

INFERENCE_POLE_IMAGES_DIR = PROJECT_ROOT / "inference" / "pole" / "images"
INFERENCE_POLE_OUTPUT_DIR = PROJECT_ROOT / "inference" / "pole"
INFERENCE_MIDSPAN_IMAGES_DIR = PROJECT_ROOT / "inference" / "midspan" / "images"
INFERENCE_MIDSPAN_OUTPUT_DIR = PROJECT_ROOT / "inference" / "midspan"

EVALUATION_DATASETS_CONFIG = {
    "pole_detection": {
        'images_dir': DATASETS_DIR / "pole_detection" / "images" / "test",
        'pole_labels_dir': DATASETS_DIR / "pole_detection" / "labels" / "test",
        'pole_top_labels_dir': DATASETS_DIR / "pole_top_detection" / "labels" / "test",
        'location_files_dir': BASE_DIR_POLE / "Labels",
    },
    "ruler_detection": {
        'images_dir': DATASETS_DIR / "ruler_detection" / "images" / "test",
        'ruler_labels_dir': DATASETS_DIR / "ruler_detection" / "labels" / "test",
        'ruler_marking_labels_dir': DATASETS_DIR / "ruler_marking_detection" / "labels" / "test",
        'location_files_dir': BASE_DIR_MIDSPAN / "Labels",
    },
}

# Attachment: all 6 classes (eval runs per-class on attachment_detection)
ATTACHMENT_EVALUATION_CONFIG = {
    'comm_detection': {
        'class_id': 0,
        'class_name': 'comm',
        'images_dir': ATTACHMENT_DATASET_DIR / "images" / "test",
        'labels_dir': ATTACHMENT_DATASET_DIR / "labels" / "test",
        'keypoint_dataset': DATASETS_DIR / "comm_keypoint_detection",
    },
    'down_guy_detection': {
        'class_id': 1,
        'class_name': 'down_guy',
        'images_dir': ATTACHMENT_DATASET_DIR / "images" / "test",
        'labels_dir': ATTACHMENT_DATASET_DIR / "labels" / "test",
        'keypoint_dataset': DATASETS_DIR / "down_guy_keypoint_detection",
    },
    'primary_detection': {
        'class_id': 2,
        'class_name': 'primary',
        'images_dir': ATTACHMENT_DATASET_DIR / "images" / "test",
        'labels_dir': ATTACHMENT_DATASET_DIR / "labels" / "test",
        'keypoint_dataset': DATASETS_DIR / "primary_keypoint_detection",
    },
    'secondary_detection': {
        'class_id': 3,
        'class_name': 'secondary',
        'images_dir': ATTACHMENT_DATASET_DIR / "images" / "test",
        'labels_dir': ATTACHMENT_DATASET_DIR / "labels" / "test",
        'keypoint_dataset': DATASETS_DIR / "secondary_keypoint_detection",
    },
    'neutral_detection': {
        'class_id': 4,
        'class_name': 'neutral',
        'images_dir': ATTACHMENT_DATASET_DIR / "images" / "test",
        'labels_dir': ATTACHMENT_DATASET_DIR / "labels" / "test",
        'keypoint_dataset': DATASETS_DIR / "neutral_keypoint_detection",
    },
    'guy_detection': {
        'class_id': 5,
        'class_name': 'guy',
        'images_dir': ATTACHMENT_DATASET_DIR / "images" / "test",
        'labels_dir': ATTACHMENT_DATASET_DIR / "labels" / "test",
        'keypoint_dataset': DATASETS_DIR / "guy_keypoint_detection",
    },
}

# Equipment: streetlight_detection, transformer_detection, riser_detection, secondary_drip_loop_detection (eval runs per-class)
EQUIPMENT_EVALUATION_CONFIG = {
    'streetlight_detection': {
        'class_id': 2,
        'class_name': 'street_light',
        'images_dir': EQUIPMENT_DATASET_DIR / "images" / "test",
        'labels_dir': EQUIPMENT_DATASET_DIR / "labels" / "test",
        'keypoint_dataset': DATASETS_DIR / "street_light_keypoint_detection",
    },
    'secondary_drip_loop_detection': {
        'class_id': 3,
        'class_name': 'secondary_drip_loop',
        'images_dir': EQUIPMENT_DATASET_DIR / "images" / "test",
        'labels_dir': EQUIPMENT_DATASET_DIR / "labels" / "test",
        'keypoint_dataset': DATASETS_DIR / "secondary_drip_loop_keypoint_detection",
    },
    'transformer_detection': {
        'class_id': 1,
        'class_name': 'transformer',
        'images_dir': EQUIPMENT_DATASET_DIR / "images" / "test",
        'labels_dir': EQUIPMENT_DATASET_DIR / "labels" / "test",
        'keypoint_dataset': DATASETS_DIR / "transformer_keypoint_detection",
    },
    'riser_detection': {
        'class_id': 0,
        'class_name': 'riser',
        'images_dir': EQUIPMENT_DATASET_DIR / "images" / "test",
        'labels_dir': EQUIPMENT_DATASET_DIR / "labels" / "test",
        'keypoint_dataset': DATASETS_DIR / "riser_keypoint_detection",
    },
}

# =============================================================================
# Confidence Weights (Weighted Confidence Metric)
# =============================================================================
# Based on Pearson correlation analysis - Date: 2026-02-07

RULER_MARKING_WEIGHTS = {
    '10.5': 0.3143, '16.5': 0.2776, '2.5': 0.2213,
    '14.5': 0.1559, '6.5': 0.0309,
}
POLE_TOP_WEIGHT_ALONE = 1.0
POLE_PHOTO_CONFIDENCE_WEIGHTS = {'pole_top': 0.5, 'ruler_marking': 0.5}
CONFIDENCE_WEIGHTS_METADATA = {
    'created_date': '2026-02-07',
    'test_set_size': {'ruler_marking': 431, 'pole_top': 239},
    'improvement_vs_average': '+17.93% (ruler markings)',
    'method': 'Weighted average using Pearson correlation magnitude as weights',
}

# Backward compatibility
AUGMENT_PARAMS = RULER_AUGMENT_PARAMS
