"""
Constants mirrored from the training repo's src/config.py for numerical parity.

This is intentionally a tiny, dependency-free copy. Keep these in sync with
the training repo if any of the upstream values change.
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# -----------------------------------------------------------------------------
# Normalization (ImageNet)
# -----------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# YOLO (pole + ruler) detection
# -----------------------------------------------------------------------------
POLE_INPUT_SIZE = 960          # imgsz used for both training and ONNX export
RULER_INPUT_SIZE = 960
POLE_CONF_THRESHOLD = 0.01     # INFERENCE_POLE_CONF_THRESHOLD
RULER_CONF_THRESHOLD = 0.01    # INFERENCE_RULER_CONF_THRESHOLD
NMS_IOU_THRESHOLD = 0.7        # ultralytics default
MAX_DETECTIONS = 1             # INFERENCE_MAX_DETECTIONS

# -----------------------------------------------------------------------------
# Ruler-marking keypoints (HRNet, 5 keypoints @ feet positions)
# -----------------------------------------------------------------------------
RULER_KEYPOINT_NAMES = ("2.5", "6.5", "10.5", "14.5", "16.5")
RULER_NUM_KEYPOINTS = len(RULER_KEYPOINT_NAMES)
RULER_INPUT_HW = (1440, 96)    # (height, width) — model input
RULER_HEATMAP_HW = (1440, 96)

# Confidence weights from a Pearson correlation analysis on the ruler-marking
# keypoints. Used to compute a single weighted_conf score per ruler crop.
RULER_MARKING_WEIGHTS = {
    "10.5": 0.3143,
    "16.5": 0.2776,
    "2.5":  0.2213,
    "14.5": 0.1559,
    "6.5":  0.0309,
}

# -----------------------------------------------------------------------------
# Pole-top keypoint (HRNet, 1 keypoint)
# -----------------------------------------------------------------------------
POLE_TOP_NUM_KEYPOINTS = 1
POLE_TOP_INPUT_HW = (256, 192)   # (height, width) — model input
POLE_TOP_HEATMAP_HW = (256, 192)
POLE_TOP_CROP_FRACTION = 0.10    # crop upper 10% of pole bbox before inference

# -----------------------------------------------------------------------------
# TTA
# -----------------------------------------------------------------------------
TTA_VERTICAL_SHIFTS = (-2, 0, 2)
