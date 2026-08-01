"""
Constants mirrored from the training repo's src/config.py for numerical parity.

Keep in sync with src/config.py equipment section when upstream values change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

_SDK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLE_WEIGHTS_PATH = (
    # calibration_sdk is versioned (v1/v2); pair with v2's pole detector (2026-07-04 fix —
    # the old flat path broke when calibration_sdk gained version dirs).
    _SDK_ROOT.parent.parent / "calibration_sdk" / "v2" / "calibration" / "weights" / "pole_detection.onnx"
)

# -----------------------------------------------------------------------------
# Normalization (ImageNet)
# -----------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Pole detection (shared ONNX from calibration_sdk)
# -----------------------------------------------------------------------------
POLE_INPUT_SIZE = 960
POLE_CONF_THRESHOLD = 0.01
NMS_IOU_THRESHOLD = 0.7
POLE_MAX_DETECTIONS = 1

# -----------------------------------------------------------------------------
# Equipment crop geometry
# -----------------------------------------------------------------------------
EQUIPMENT_CROP_HEIGHT_FRACTION = 0.70
EQUIPMENT_CROP_ASPECT_W_OVER_H = 2 / 5

# -----------------------------------------------------------------------------
# Equipment YOLO detection (4 classes)
# -----------------------------------------------------------------------------
EQUIPMENT_CLASS_NAMES: Tuple[str, ...] = (
    "riser",
    "transformer",
    "street_light",
    "secondary_drip_loop",
)
EQUIPMENT_INPUT_SIZE = 960
EQUIPMENT_CONF_THRESHOLD = 0.2983
EQUIPMENT_CONF_PER_CLASS: Dict[str, float] = {
    "riser": 0.2883,
    "transformer": 0.3614,
    "street_light": 0.1011,
    "secondary_drip_loop": 0.3003,
}
EQUIPMENT_BASE_CONF = min(EQUIPMENT_CONF_PER_CLASS.values())
EQUIPMENT_MAX_DETECTIONS = 20
EQUIPMENT_MIN_BBOX_AREA_FRAC = 0.001
SECONDARY_DRIP_LOOP_MAX_DET = 1

# -----------------------------------------------------------------------------
# Per-class HRNet keypoint models
# Each entry: onnx filename, (input_h, input_w), num_keypoints, keypoint names
# -----------------------------------------------------------------------------
EQUIPMENT_KEYPOINT_SPECS: Dict[str, Tuple[str, Tuple[int, int], int, Tuple[str, ...]]] = {
    "riser": (
        "riser_keypoint_detection.onnx",
        (384, 144),
        1,
        ("top",),
    ),
    "transformer": (
        "transformer_keypoint_detection.onnx",
        (384, 288),
        2,
        ("top_bolt", "bottom"),
    ),
    "street_light": (
        "street_light_keypoint_detection.onnx",
        (512, 384),
        3,
        ("upper_bracket", "lower_bracket", "drip_loop"),
    ),
    "secondary_drip_loop": (
        "secondary_drip_loop_keypoint_detection.onnx",
        (512, 384),
        1,
        ("lowest_point",),
    ),
}

# Visualization colors (RGB)
EQUIPMENT_BOX_COLORS: Dict[str, Tuple[int, int, int]] = {
    "riser": (255, 165, 0),
    "transformer": (0, 200, 255),
    "street_light": (200, 0, 200),
    "secondary_drip_loop": (0, 255, 128),
}
CROP_BOX_COLOR = (128, 128, 128)
POLE_BOX_COLOR = (0, 200, 0)
