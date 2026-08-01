"""Upper-70% 2:5 pole crop (mirrors src/data_utils.py::_compute_pole_upper70_2x5_crop)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .constants import POLE_CROP_ASPECT_W_OVER_H, POLE_CROP_HEIGHT_FRACTION


def compute_pole_upper70_2x5_crop(
    img: np.ndarray,
    pole_bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[np.ndarray, int, int, int, int, int, int]]:
    """Crop to pole bbox, upper 70%, expanded horizontally to a 2:5 aspect ratio.

    Returns (crop, x1_new, y1, x2_new, crop_y2, crop_w_actual, crop_h_actual) or None.
    Byte-for-byte identical to the training-repo helper.
    """
    x1, y1, x2, y2 = pole_bbox
    crop_h_full = y2 - y1
    crop_h = int(crop_h_full * POLE_CROP_HEIGHT_FRACTION)
    if crop_h < 10 or (x2 - x1) < 10:
        return None
    target_width = int(crop_h * POLE_CROP_ASPECT_W_OVER_H)
    center_x = (x1 + x2) / 2
    x1_new = max(0, int(center_x - target_width / 2))
    x2_new = min(img_w, int(center_x + target_width / 2))
    if x2_new - x1_new < 10:
        return None
    crop = img[y1 : y1 + crop_h, x1_new:x2_new]
    crop_h_actual, crop_w_actual = crop.shape[:2]
    crop_y2 = y1 + crop_h
    return crop, x1_new, y1, x2_new, crop_y2, crop_w_actual, crop_h_actual
