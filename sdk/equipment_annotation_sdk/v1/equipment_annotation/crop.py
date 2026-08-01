"""Equipment crop extraction — upper 70% of pole bbox at 2:5 aspect ratio."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .constants import EQUIPMENT_CROP_ASPECT_W_OVER_H, EQUIPMENT_CROP_HEIGHT_FRACTION


def extract_equipment_crop(
    img_bgr: np.ndarray,
    pole_bbox: Tuple[int, int, int, int],
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Extract upper 70% 2:5 crop from pole bbox.

    Returns (crop_bgr, (x1, y1, x2, y2)) in full-image coordinates, or (None, None).
    Mirrors src/evaluation_attachment_equipment.py::_extract_equipment_crop.
    """
    x1, y1, x2, y2 = pole_bbox
    crop_h_full = y2 - y1
    crop_h = int(crop_h_full * EQUIPMENT_CROP_HEIGHT_FRACTION)
    if crop_h < 10 or (x2 - x1) < 10:
        return None, None
    target_width = int(crop_h * EQUIPMENT_CROP_ASPECT_W_OVER_H)
    center_x = (x1 + x2) / 2
    x1_new = max(0, int(center_x - target_width / 2))
    x2_new = min(img_bgr.shape[1], int(center_x + target_width / 2))
    if x2_new - x1_new < 10:
        return None, None
    crop = img_bgr[y1 : y1 + crop_h, x1_new:x2_new]
    return crop, (x1_new, y1, x2_new, y1 + crop_h)
