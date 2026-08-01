"""
Configuration constants for Pole Annotation.

Per-model sections: training config, augmentation, inference weights and thresholds.
Inference confidence: single source of truth per model. Per-class thresholds are
F1-maximizing from threshold sweep. Update via: evaluate_models.py --equipment/--attachment
or scripts/eval/threshold_sweep.py --update-config.
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
MIDSPAN_WIRE_STRIP_DETECTION = "midspan_wire_strip_detection"
EQUIPMENT_DETECTION = "equipment_detection"
ATTACHMENT_DETECTION = "attachment_detection"
UNIFIED_POLE_DETECTION = "unified_pole_detection"
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
    MIDSPAN_WIRE_STRIP_DETECTION: DATASETS_DIR / "midspan_wire_strip_detection",
    EQUIPMENT_DETECTION: EQUIPMENT_DATASET_DIR,
    ATTACHMENT_DETECTION: ATTACHMENT_DATASET_DIR,
    UNIFIED_POLE_DETECTION: DATASETS_DIR / "unified_pole_detection",
    "riser_keypoint_detection": DATASETS_DIR / "riser_keypoint_detection",
    "transformer_keypoint_detection": DATASETS_DIR / "transformer_keypoint_detection",
    "street_light_keypoint_detection": DATASETS_DIR / "street_light_keypoint_detection",
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
# Threshold sweep: python scripts/eval/threshold_sweep.py [--update-config]
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
# MIDSPAN WIRE STRIP DETECTION
# =============================================================================

# Full-height ruler column strip → 1D wire height heatmap (tall × narrow)
WIRE_STRIP_RESIZE_HEIGHT, WIRE_STRIP_RESIZE_WIDTH = 3480, 96
WIRE_STRIP_HEATMAP_HEIGHT, WIRE_STRIP_HEATMAP_WIDTH = 3480, 96
WIRE_STRIP_GAUSSIAN_SIGMA_X = WIRE_STRIP_HEATMAP_WIDTH / 8
WIRE_STRIP_GAUSSIAN_SIGMA_Y = WIRE_STRIP_HEATMAP_HEIGHT / 128
# Peak extraction: central-band mean column profile + scipy.find_peaks
# (height + distance + prominence). Prominence rejects spurious bumps on an elevated
# baseline; the central-band mean suppresses edge noise. Tuned for max F1 on the val
# split (test F1@2in 0.366 greedy -> 0.824 here). See scripts/eval_midspan_strip_f1.py.
WIRE_STRIP_PEAK_MIN_DISTANCE = 12  # min heatmap-rows between peaks (~2 inch at typical PPI)
WIRE_STRIP_PEAK_HEIGHT = 0.6       # find_peaks min profile height
WIRE_STRIP_PEAK_PROMINENCE = 0.05  # find_peaks min prominence (the spurious-peak killer)
WIRE_STRIP_PROFILE_BAND = 16       # half-width of central column band for the 1-D profile
WIRE_STRIP_PEAK_THRESHOLD = 0.25   # legacy greedy-scan threshold (kept for back-compat)

MIDSPAN_WIRE_STRIP_DETECTION_CONFIG = dict(
    batch_size=16,
    epochs=100,
    patience=40,
    learning_rate=1e-3,
    use_focal_loss=False,
    resize_height=WIRE_STRIP_RESIZE_HEIGHT,
    resize_width=WIRE_STRIP_RESIZE_WIDTH,
    heatmap_height=WIRE_STRIP_HEATMAP_HEIGHT,
    heatmap_width=WIRE_STRIP_HEATMAP_WIDTH,
    min_wires=1,
    augmentation_params={'brightness': 0.25, 'contrast': 0.25, 'saturation': 0.25},
    geometric_augmentations={'translate_x': 0.02, 'translate_y': 0.03, 'scale_min': 0.98, 'scale_max': 1.02, 'rotate': 2.0},
)
INFERENCE_MIDSPAN_WIRE_STRIP_WEIGHTS = _model_weights_path('midspan_wire_strip_detection', '.pth')

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
# Update via: python scripts/eval/evaluate_models.py --equipment or scripts/eval/threshold_sweep.py --update-config
INFERENCE_EQUIPMENT_CONF_THRESHOLD = 0.3413  # fallback for unknown classes
INFERENCE_EQUIPMENT_CONF_PER_CLASS = {
    'riser': 0.2943,
    'transformer': 0.3874,
    'street_light': 0.5035,
    'secondary_drip_loop': 0.3123
}
INFERENCE_EQUIPMENT_MIN_BBOX_AREA_FRAC = 0.001  # min bbox area as fraction of crop area
INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET = 1  # max detections for secondary_drip_loop class

# --- Single YOLO-pose equipment model (box + keypoints in one shot) ---
# Replaces the two-stage YOLO-box + per-class HRNet pipeline. YOLO-pose needs a
# fixed kpt_shape across all classes, so all classes pad to the max keypoint count
# (street_light = 3). Slot order matches parse_equipment_with_keypoints kp0/kp1/kp2,
# so each slot keeps the same semantic as the per-class HRNet index (apples-to-apples
# with the EQUIPMENT eval). Missing slots are written with visibility=0 (masked from
# the pose loss). Equipment keypoints are vertical (top/bottom/bracket) with no
# left-right pairs, so the horizontal-flip map is the identity.
# Real (non-padded) keypoint count per class — used to truncate pose predictions at eval.
# (riser=1, transformer=2, street_light=3, secondary_drip_loop=1; see *_NUM_KEYPOINTS below.)
EQUIPMENT_POSE_CLASS_NUM_KP = {
    'riser': 1,
    'transformer': 2,
    'street_light': 3,
    'secondary_drip_loop': 1,
}

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
# Wire attachment HARDWARE tokens (insulator_spec vocabulary)
# =============================================================================
# Hardware is the VISUAL proxy for wire tier (spool→secondary, three_bolt→comm,
# pin/post/davit→power) and yields a deadend dustbin signal. The token machinery
# below is consumed by extract_height (location-file `_hw` lines) and the unified
# joint-class encoder (unified_joint_class) — the standalone hw detector is legacy.
#
# Two classes are assigned by attachment NAME, not insulator_spec (guys carry no
# insulator): 'guy' (aerial/head/power guy — crosses spans, appears at midspan) and
# 'down_guy' (anchor guy to ground — pole-only, NEVER at midspan → matcher dustbin
# signal).
WIRE_HW_CLASS_NAMES = ['spool', 'three_bolt', 'pin', 'post', 'deadend', 'davit', 'guy', 'down_guy']
# Raw insulator tokens merged into one canonical token (location files keep the raw
# token, so the split is reversible; single_bolt is visually near-identical + rare).
WIRE_HW_CLASS_MERGE = {'single_bolt': 'three_bolt'}
# Classes assigned by attachment NAME (no insulator). They ride along on photos that
# already qualify via an insulator; a guy ALONE does not qualify a photo for inclusion.
WIRE_HW_GUY_CLASSES = ('guy', 'down_guy')
# hardware token -> coarse tier (downstream derivation); deadend = power-terminating
WIRE_HW_TO_TIER = {
    'spool': 'secondary', 'three_bolt': 'comm',
    'pin': 'power', 'post': 'power', 'deadend': 'power', 'davit': 'power',
    'guy': 'guy', 'down_guy': 'guy',
}
WIRE_HW_DEADEND_TOKENS = ('deadend',)

def normalize_hardware_spec(spec):
    """Map a raw Katapult insulator_spec to a canonical hardware token, or None.

    Robust to size/voltage suffixes: 'Pin Insulator - 5 kV' -> pin, 'Deadend 12.75"'
    -> deadend, 'Spool 3"' -> spool, 'Three Bolt' -> three_bolt, etc.
    """
    if not spec:
        return None
    s = str(spec).strip().lower()
    if 'spool' in s:
        return 'spool'
    if 'deadend' in s or 'dead end' in s or 'dead-end' in s:
        return 'deadend'
    if 'three bolt' in s or 'three-bolt' in s or '3 bolt' in s:
        return 'three_bolt'
    if 'single bolt' in s or 'single-bolt' in s or '1 bolt' in s:
        return 'single_bolt'
    if 'pin' in s:
        return 'pin'
    if 'post' in s:
        return 'post'
    if 'davit' in s:
        return 'davit'
    return None


def hardware_token_for_spec(spec):
    """normalize_hardware_spec + training-time class merges (single_bolt -> three_bolt)."""
    tok = normalize_hardware_spec(spec)
    return WIRE_HW_CLASS_MERGE.get(tok, tok)


def hardware_tier_for_spec(spec):
    """Coarse tier ('power'|'comm'|'secondary'|'guy') for a raw insulator_spec, else None."""
    return WIRE_HW_TO_TIER.get(hardware_token_for_spec(spec))


def spec_is_deadend(spec):
    """True if an insulator_spec is a deadend (power-terminating → matcher dustbin prior)."""
    return hardware_token_for_spec(spec) in WIRE_HW_DEADEND_TOKENS


# cable_type (a trace's tier label) -> the SAME coarse 4-tier space as WIRE_HW_TO_TIER.
# Hardware is the visual proxy for this tier, so the matcher can compare a pole's
# hardware-derived tier against a midspan marker's cable_type tier. (Spool carries
# secondary+neutral; Three Bolt carries CATV/Fiber/Telco/ADSS; Pin/Post/Davit/Deadend
# carry Primary.) 'Traffic Cable' and unknowns map to None (no tier signal).
CABLE_TYPE_TO_TIER = {
    'Primary': 'power',
    'Secondary': 'secondary', 'Open Secondary': 'secondary', 'Neutral': 'secondary',
    'CATV': 'comm', 'Fiber': 'comm', 'Telco': 'comm', 'ADSS': 'comm',
    'Guy': 'guy', 'Power Guy': 'guy',
}


def tier_for_cable_type(ct):
    """Coarse tier for a midspan marker's cable_type, or None if unmapped."""
    return CABLE_TYPE_TO_TIER.get(ct)


# 3-class MIDSPAN tier space (bare/multiplex/comm) — the VISUAL form of the conductor at midspan:
#   bare      = a single bare conductor: Primary, Neutral, AND Open Secondary (open-wire secondary
#               = individual bare conductors, NOT a bundle).
#   multiplex = Secondary only (triplex/quadruplex twisted service bundle — the one thick bundle).
#   comm      = CATV / Telco / Fiber.
# DISTINCT from the coarse CABLE_TYPE_TO_TIER above (which lumps Neutral+Secondary+Open Secondary
# into 'secondary'). Two maps land BOTH ends in the same 3 classes: cable_type (midspan GT / strip
# labels) and the FINE unified pole class name (pole side — must use the class name, NOT wire_class,
# because _UNIFIED_WIRE_CLASS collapses open_secondary->secondary and would lose bare-vs-multiplex).
MIDSPAN_TIER3 = ('bare', 'multiplex', 'comm')
CABLE_TYPE_TO_TIER3 = {
    'Primary': 'bare', 'Neutral': 'bare', 'Open Secondary': 'bare',
    'Secondary': 'multiplex',
    'CATV': 'comm', 'Fiber': 'comm', 'Telco': 'comm', 'ADSS': 'comm',
}
# unified pole class NAME -> tier3. Power hardware carries a (bare) Primary conductor.
UNIFIED_CLASS_TO_TIER3 = {
    'pin': 'bare', 'post': 'bare', 'davit': 'bare', 'deadend': 'bare',
    'arm2': 'bare', 'arm3': 'bare', 'arm4plus': 'bare', 'primary': 'bare',
    'open_secondary': 'bare', 'neutral': 'bare',
    'secondary': 'multiplex',
    'catv': 'comm', 'telco': 'comm', 'fiber': 'comm', 'comm': 'comm',
}


def cable_type_to_tier3(ct):
    """3-class midspan tier (bare/multiplex/comm) for a cable_type, or None if unmapped.

    Falls back to normalize_cable_type for raw variants ('CATV Com', 'Telco Com', ...) —
    exact-match alone silently dropped ~90 balanced-eval chains to tier None."""
    exact = CABLE_TYPE_TO_TIER3.get(ct)
    if exact is not None:
        return exact
    return UNIFIED_CLASS_TO_TIER3.get(normalize_cable_type(ct))


def unified_class_to_tier3(name):
    """3-class midspan tier for a unified pole class NAME, or None (guy/down_guy/unspecified)."""
    return UNIFIED_CLASS_TO_TIER3.get(name)


# =============================================================================
# UNIFIED POLE DETECTION (single YOLO-pose model: hardware + cable_type joint class)
# =============================================================================
# One detector per attachment keypoint whose CLASS jointly encodes the supporting
# hardware AND the cable type. Hardware and cable_type are tightly tier-coupled
# (power hw -> Primary; spool -> secondary tier; three_bolt -> comm tier), so the
# joint space is small (~17). Decoding a class yields (hardware, cable_type, K):
#   - Power tier (cable_type == Primary): class encodes the HARDWARE sub-type
#     (pin/post/davit/deadend), the crossarm wire-count (arm2/arm3/arm4plus, POWER
#     arms only per project convention), or a generic `primary` when hw is unread.
#   - Secondary tier (hardware == spool): class encodes the CABLE_TYPE
#     (secondary/open_secondary/neutral) -- hardware is redundant.
#   - Comm tier (hardware == three_bolt): class encodes the CABLE_TYPE
#     (catv/telco/fiber) -- hardware is redundant.
#   - Guys carry no insulator -> guy / down_guy.
#   - `unspecified` = a recognized conductor whose tier is unknown (recovers the
#     pole-top conductors the old extractor dropped: ADSS/empty/traffic cable).
# Trained on non-MI jobs (clean crossarm K). Crossarm = ONE keypoint; K predicted
# from the arm's appearance via the arm{2,3,4plus} classes (no coincident keypoints).

UNIFIED_POLE_DETECTION_CLASS_NAMES = [
    # power tier (cable_type = Primary; class = hardware sub-type / arm wire-count)
    'pin', 'post', 'davit', 'deadend', 'arm2', 'arm3', 'arm4plus', 'primary',
    # secondary tier (hardware = spool; class = cable_type)
    'secondary', 'open_secondary', 'neutral',
    # comm tier (hardware = three_bolt; class = cable_type)
    'catv', 'telco', 'fiber',
    # guys (no insulator)
    'guy', 'down_guy',
    # recognized conductor, tier unknown (pole-top recovery)
    'unspecified',
]
UNIFIED_POLE_DETECTION_CLASSES = {n: i for i, n in enumerate(UNIFIED_POLE_DETECTION_CLASS_NAMES)}
UNIFIED_POLE_DETECTION_NUM_KEYPOINTS = 1
UNIFIED_POLE_DETECTION_KEYPOINT_NAMES = ['attachment']
# same 1ft x 2ft (H x W) attachment box as wire_detection / wire_hw
UNIFIED_POLE_DETECTION_BBOX_HEIGHT_FEET = 1.0
UNIFIED_POLE_DETECTION_BBOX_WIDTH_FEET = 2.0
# CROSSARM multiplicity (arm{2,3,4plus}) only for POWER arms (matches the wire_tracer
# convention CROSSARM_HW); spool/three_bolt arms are capped at 1 wire.
UNIFIED_CROSSARM_POWER_HW = ('pin', 'post', 'davit', 'deadend')

# location-file lines (backward-compatible additions written by extract_height --pole):
#   `<prefix>_ct,<raw cable_type>`   e.g. primary1_ct,Primary  /  comm2_ct,CATV
#   `<prefix>_arm,<K>`               e.g. primary1_arm,3   (arm attachments only; K=wire count)
UNIFIED_POLE_DETECTION_CT_SUFFIX = '_ct'
UNIFIED_POLE_DETECTION_ARM_SUFFIX = '_arm'


def normalize_cable_type(raw):
    """Map a raw Katapult cable_type/company string to a canonical token used by the
    unified classes, or None. Handles pole-job short names and MI ' Com' suffixes."""
    if not raw:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if 'down' in s and 'guy' in s:
        return 'down_guy'
    if 'guy' in s:                      # power guy / overhead guy / guy
        return 'guy'
    if 'open' in s and 'sec' in s:
        return 'open_secondary'
    if 'neutral' in s:
        return 'neutral'
    if 'secondary' in s:
        return 'secondary'
    if 'primary' in s:
        return 'primary'
    if 'catv' in s:
        return 'catv'
    if 'telco' in s or 'telephone' in s:
        return 'telco'
    if 'fiber' in s or 'adss' in s:     # ADSS = all-dielectric self-supporting fiber
        return 'fiber'
    return None


# unified class name -> (hardware token|None, canonical cable_type|None, K|None, display)
UNIFIED_POLE_DECODE = {
    'pin':            ('pin', 'primary', 1, 'Pin Insulator'),
    'post':           ('post', 'primary', 1, 'Post Insulator'),
    'davit':          ('davit', 'primary', 1, 'Davit Arm'),
    'deadend':        ('deadend', 'primary', 1, 'Deadend'),
    'arm2':           ('arm', 'primary', 2, 'Crossarm x2'),
    'arm3':           ('arm', 'primary', 3, 'Crossarm x3'),
    'arm4plus':       ('arm', 'primary', 4, 'Crossarm x4+'),
    'primary':        (None, 'primary', 1, 'Primary (hardware unread)'),
    'secondary':      ('spool', 'secondary', 1, 'Spool (Secondary)'),
    'open_secondary': ('spool', 'open_secondary', 1, 'Spool (Open Secondary)'),
    'neutral':        ('spool', 'neutral', 1, 'Spool (Neutral)'),
    'catv':           ('three_bolt', 'catv', 1, 'Three-Bolt (CATV)'),
    'telco':          ('three_bolt', 'telco', 1, 'Three-Bolt (Telco)'),
    'fiber':          ('three_bolt', 'fiber', 1, 'Three-Bolt (Fiber)'),
    'guy':            (None, 'guy', 1, 'Guy'),
    'down_guy':       (None, 'down_guy', 1, 'Down Guy'),
    'unspecified':    (None, None, 1, 'Unspecified Wire'),
    'comm':           ('three_bolt', 'comm', 1, 'Three-Bolt (Comm)'),  # 14-class merge variant
    # hardware-first 10-class variant: cable-type consolidated into the hardware token
    'arm3plus':       ('arm', 'primary', 3, 'Crossarm x3+'),
    'spool':          ('spool', 'secondary', 1, 'Spool'),         # sec/neutral indistinguishable -> user assigns
    'three_bolt':     ('three_bolt', 'comm', 1, 'Three-Bolt (Comm)'),
}

# -----------------------------------------------------------------------------
# 14-CLASS MERGE VARIANT (idea #1): fold the within-tier-confused fine classes.
#   open_secondary -> neutral   (the two are visually near-identical, co-occur on the
#                                secondary rack, and are mutually mislabeled)
#   catv/telco/fiber -> comm     (the model does not reliably split comm; user refines)
# Kept ALONGSIDE the deployed 17-class set above (that set is unchanged). Used only by the
# `*_merged` dataset/model when explicitly selected (decode via unified_class_names). The
# raw `_ct` location-file lines are preserved, so the split stays reversible. Diagnostics:
# open_sec(14.4%)+neutral(14.3%) and catv/telco/fiber(22.3%) are the confused buckets; the
# secondary TIER recall is already 0.913, so the merge mainly recovers fine-class fidelity.
UNIFIED_POLE_DETECTION_CLASS_NAMES_MERGED = [
    'pin', 'post', 'davit', 'deadend', 'arm2', 'arm3', 'arm4plus', 'primary',
    'secondary', 'neutral', 'comm', 'guy', 'down_guy', 'unspecified',
]
UNIFIED_CABLE_TYPE_MERGE = {
    'open_secondary': 'neutral', 'catv': 'comm', 'telco': 'comm', 'fiber': 'comm',
}
# old 17-class id -> new 14-class id (remaps existing YOLO labels; no re-extraction needed)
UNIFIED_MERGE_CLASS_ID_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    9: 9, 10: 9,            # open_secondary, neutral -> neutral
    11: 10, 12: 10, 13: 10,  # catv, telco, fiber -> comm
    14: 11, 15: 12, 16: 13,  # guy, down_guy, unspecified
}

# HARDWARE-FIRST 10-class scheme: consolidate cable-type variants of the SAME hardware
# (spool = secondary|open_secondary|neutral; three_bolt = catv|telco|fiber) so the keypoint
# detector gets undiluted per-hardware signal (node recall = the e2e bottleneck). Crossarm-K is
# preserved (arm3plus folds the degenerate arm4plus); the dead classes (primary 6, unspecified 7)
# are dropped. Cable_type becomes a tier-derived / user-assigned attribute, not a detection split
# (cable granularity is matcher-invisible at e2e). decode handled by the arm3plus/spool/three_bolt
# entries added to UNIFIED_POLE_DECODE, so _unified_point needs no change.
UNIFIED_POLE_DETECTION_CLASS_NAMES_HWFIRST = [
    'pin', 'post', 'davit', 'deadend', 'arm2', 'arm3plus',
    'spool', 'three_bolt', 'guy', 'down_guy',
]
# old 17-class id -> hardware-first id (None = drop the label line: hardware-unread/unspecified)
UNIFIED_HWFIRST_CLASS_ID_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,   # pin, post, davit, deadend, arm2
    5: 5, 6: 5,                      # arm3, arm4plus -> arm3plus
    7: None,                         # primary (hardware-unread, 6 inst) -> drop
    8: 6, 9: 6, 10: 6,              # secondary, open_secondary, neutral -> spool
    11: 7, 12: 7, 13: 7,           # catv, telco, fiber -> three_bolt
    14: 8, 15: 9,                  # guy, down_guy
    16: None,                       # unspecified (7 inst) -> drop
}


def unified_joint_class(hw_token=None, cable_type=None, is_arm=False, arm_k=None):
    """Encode (hardware token, raw cable_type, is_arm, K) -> a unified class name.

    Single source of truth shared by the dataset prep (GT labels) and the eval
    harness (per-pole GT). Returns None only when there is no conductor at all.
    """
    ct = normalize_cable_type(cable_type)
    hw = WIRE_HW_CLASS_MERGE.get(hw_token, hw_token) if hw_token else None
    # guys carry no insulator (decide by cable_type OR hw token)
    if ct == 'down_guy' or hw == 'down_guy':
        return 'down_guy'
    if ct == 'guy' or hw == 'guy':
        return 'guy'
    # POWER crossarm: one keypoint, K wires -> wire-count class (non-power arms fall
    # through and collapse to a single cable_type class, per project convention)
    if is_arm and (ct == 'primary' or hw in UNIFIED_CROSSARM_POWER_HW):
        k = arm_k or 0
        if k >= 4:
            return 'arm4plus'
        if k == 3:
            return 'arm3'
        return 'arm2'                   # K<=2 (K=1 arms are rare; treat as arm2)
    # secondary / comm tiers: cable_type IS the discriminator (hardware redundant)
    if ct in ('secondary', 'open_secondary', 'neutral', 'catv', 'telco', 'fiber'):
        return ct
    # power tier (Primary): hardware sub-type is the discriminator
    if ct == 'primary':
        if hw in ('pin', 'post', 'davit', 'deadend'):
            return hw
        return 'primary'                # generic / hardware unread
    # recognized conductor with unknown/empty tier (pole-top recovery)
    return 'unspecified'


def decode_unified_class(name):
    """Decode a unified class name into (hardware, cable_type, K, display_label)."""
    return UNIFIED_POLE_DECODE.get(name)


UNIFIED_POLE_DETECTION_CONFIG = _yolo_defaults(
    epochs=100,
    patience=40,
    dropout=0.15,
    weight_decay=0.005,
    batch_size=48,
    imgsz=960,
    cls=1.5,
    lr0=0.0003,
)
UNIFIED_POLE_DETECTION_AUGMENT_PARAMS = dict(
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
    degrees=5.0, translate=0.10,
    scale=0.6, shear=2.0, perspective=0.001,
    fliplr=0.5, flipud=0.0,
    mosaic=1.0, mixup=0.1, copy_paste=0.3,
)
INFERENCE_UNIFIED_POLE_DETECTION_WEIGHTS = _model_weights_path('unified_pole_detection', '.pt')
INFERENCE_UNIFIED_POLE_DETECTION_CONF_THRESHOLD = 0.20


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

# comm/down_guy are the two attachment keypoint models the annotation e2e path
# (evaluation_attachment_equipment) still loads; the 6-set legacy wire keypoint
# stack (primary/secondary/neutral/guy) was removed 2026-07-29.
ATTACHMENT_KEYPOINT_CONFIGS = {
    'comm': (COMM_KEYPOINT_DETECTION_CONFIG, COMM_NUM_KEYPOINTS, INFERENCE_COMM_KEYPOINT_WEIGHTS),
    'down_guy': (DOWN_GUY_KEYPOINT_DETECTION_CONFIG, DOWN_GUY_NUM_KEYPOINTS, INFERENCE_DOWN_GUY_KEYPOINT_WEIGHTS),
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

# Mapping from train.py model name to keypoint_type (EQUIPMENT_KEYPOINT_CONFIGS)
# PRODUCTION keypoint models = the 4 equipment HRNet sets ONLY (the 6 legacy
# wire-attachment keypoint sets were removed 2026-07-29 with the legacy stacks).
KEYPOINT_MODEL_TO_TYPE = {
    'riser_keypoint_detection': 'riser',
    'transformer_keypoint_detection': 'transformer',
    'street_light_keypoint_detection': 'street_light',
    'secondary_drip_loop_keypoint_detection': 'secondary_drip_loop',
}

# Keypoint dataset prep: (type, dataset_dir, prep_kind)
KEYPOINT_PREPARE_SPECS = [
    ('riser', 'riser_keypoint_detection', 'equipment'),
    ('transformer', 'transformer_keypoint_detection', 'equipment'),
    ('street_light', 'street_light_keypoint_detection', 'equipment'),
    ('secondary_drip_loop', 'secondary_drip_loop_keypoint_detection', 'equipment'),
]

# =============================================================================
# Equipment & Attachment Domain (bbox sizes in feet - used by data prep)
# =============================================================================

RISER_BBOX_HEIGHT_FEET, RISER_BBOX_WIDTH_FEET = 4.0, 1.5
TRANSFORMER_BBOX_HEIGHT_FEET, TRANSFORMER_BBOX_WIDTH_FEET = 4.0, 3.0
STREET_LIGHT_BBOX_HEIGHT_FEET, STREET_LIGHT_BBOX_WIDTH_FEET = 8.0, 6.0
SECONDARY_DRIP_LOOP_BBOX_HEIGHT_FEET, SECONDARY_DRIP_LOOP_BBOX_WIDTH_FEET = 4.0, 3.0
ATTACHMENT_BBOX_HEIGHT_FEET, ATTACHMENT_BBOX_WIDTH_FEET = 1.0, 2.0
WIRE_BBOX_WIDTH_FEET, WIRE_BBOX_HEIGHT_FEET = 2.0, 1.0
DOWN_GUY_BBOX_HEIGHT_FEET, DOWN_GUY_BBOX_WIDTH_FEET = 4.0, 2.0
# MI-regime jobs (UtilityCo-MI, bare-wire annotation): one wire marker for multiple
# primaries. Detection is CONTENT-based (data_utils.mi_like_jobs: CE-dominant company or
# zero insulator_spec) — this prefix list is only the legacy disk-stem fallback. Midspan
# training drops only PRIMARY-BEARING MI photos (mi_dirty_midspan_pids); primary-free MI
# midspan photos are kept, mirroring the pole-side include_mi_clean policy.
MIDSPAN_WIRE_EXCLUDED_JOB_PREFIXES = ('MI',)

# =============================================================================
# Wire tracing (Stage-0 extractor): per-span pole↔midspan↔pole correspondence
# =============================================================================
# Built from raw Katapult job JSONs in BASE_DIR_MIDSPAN (each self-contains pole +
# midspan markers linked by shared trace ids). Output is the matcher GT dataset.
WIRE_TRACING_DATASET = "wire_tracing_dataset"
WIRE_TRACING_DATASET_DIR = DATASETS_DIR / "wire_tracing_dataset"
# Raw Katapult job-JSON source for the wire-tracing builder. Repointed (2026-06-21) from the legacy
# data/data_midspan/*.json (30 jobs, an import artifact) to the migrated AUTHORITATIVE deduped set
# data/jobs/*.json (116 jobs, richer-wins on conflict). For the 23 jobs that overlap this is a
# byte-identical superset except one job's sub-meter richer-wins node coords (MNMW029); the extra
# jobs add GT-only spans whose midspan photos aren't on disk (the e2e harness drops them, so the
# e2e baseline is unchanged). Falls back to BASE_DIR_MIDSPAN if data/jobs is absent. This is the
# read that must be repointed BEFORE data/data_midspan can be deleted.
_WIRE_TRACING_JOBS_DIR = PROJECT_ROOT / "data" / "jobs"
WIRE_TRACING_JOB_SOURCE_DIR = _WIRE_TRACING_JOBS_DIR if _WIRE_TRACING_JOBS_DIR.exists() else BASE_DIR_MIDSPAN
# Regime guard: MI-style jobs annotate multi-primary crossarms as ONE collapsed wire
# and carry zero insulator markers — content signal is far more robust than the name.
WIRE_TRACING_MI_MAX_INSULATORS = 0          # job with <= this many insulators -> MI regime -> excluded
# Span scope: only spans that physically carry trace-able wires pole-to-pole.
WIRE_TRACING_IN_SCOPE_CONNECTION_TYPES = ('aerial cable',)
# Both span endpoints must be one of these node types; otherwise it is a service drop /
# tap / anchor / reference span and its pole-side attachments fall to the matcher dustbin.
WIRE_TRACING_POLE_NODE_TYPES = ('pole', 'break point')
# Single-midspan only: a connection with >1 section-with-photos (pole-mid-mid-...-pole) is
# AMBIGUOUS — the SCID-pair photo naming "(A)-to-(B)" can't say which section a photo/GT
# belongs to, so detection runs on the wrong midspan. Keep only single-section (pole-mid-pole).
WIRE_TRACING_SINGLE_SECTION_ONLY = True
# Multi-section spans (pole -> M1 -> ... -> Mk -> pole). When True, build_span_sample emits the
# ordered per-section structure (sides.M_sections + gt.chains_multi) for ALL spans and INCLUDES
# multi-section connections (the photo<->section ambiguity is now resolved by the ruler-keypoint
# re-keying, scripts/data/resolve_multisection_midspan.py). The legacy single-M fields
# (sides.M, gt.chains, gt.dustbin) are still emitted against a SPINE section (midpoint_section
# when photo-bearing, else nearest-A) so single-section output stays byte-identical and existing
# consumers keep working. Default ON (2026-06-21): always keep multi-section spans + emit the
# per-section path (real e2e chain acc 0.581 on 638 spans; the wire_tracer auto-dispatches them).
# Set False for a byte-identical legacy single-section build.
WIRE_TRACING_MULTI_SECTION = True

# =============================================================================
# Inference Settings
# =============================================================================

INFERENCE_MAX_DETECTIONS = 1
INFERENCE_USE_TTA = True
INFERENCE_USE_INTERPOLATION = False

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
