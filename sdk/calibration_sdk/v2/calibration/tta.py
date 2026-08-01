"""
Test-time augmentation: vertical-shift heatmap averaging.

Mirrors src/inference_utils.py::infer_keypoints_on_crop and
infer_pole_top_on_crop in the upstream training repo. We:

    1. Apply small integer vertical shifts to the *input crop* (REFLECT_101).
    2. Run the HRNet ONNX session on each shifted version.
    3. Shift the resulting heatmap back into the unshifted coordinate space
       (CONSTANT 0 padding for the boundary).
    4. Average heatmaps across shifts before argmax decoding.

This gives noticeably tighter keypoint localisation at the cost of N x latency
(default N = 3 shifts).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .constants import TTA_VERTICAL_SHIFTS
from .hrnet_onnx import HrnetOnnxKeypointer


def heatmaps_with_vertical_shift_tta(
    keypointer: HrnetOnnxKeypointer,
    crop_rgb: np.ndarray,
    shifts: Tuple[int, ...] = TTA_VERTICAL_SHIFTS,
) -> np.ndarray:
    """Return averaged sigmoid heatmaps of shape (K, Hh, Wh)."""
    h_crop, w_crop = crop_rgb.shape[:2]
    h_in, w_in = keypointer.input_hw

    accumulated = None
    n = 0
    for shift in shifts:
        if shift == 0:
            shifted = crop_rgb
        else:
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            shifted = cv2.warpAffine(
                crop_rgb, M, (w_crop, h_crop),
                borderMode=cv2.BORDER_REFLECT_101,
                flags=cv2.INTER_LINEAR,
            )

        heatmaps = keypointer.run_heatmaps(shifted)  # (K, Hh, Wh)

        if shift != 0:
            # Shift heatmaps back into the unshifted coordinate space.
            # The shift in the original crop maps linearly to the heatmap.
            shift_px_in_resized = shift / h_crop * h_in
            M_back = np.float32([[1, 0, 0], [0, 1, -shift_px_in_resized]])
            aligned = np.empty_like(heatmaps)
            for k in range(heatmaps.shape[0]):
                aligned[k] = cv2.warpAffine(
                    heatmaps[k], M_back, (heatmaps.shape[2], heatmaps.shape[1]),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
                    flags=cv2.INTER_LINEAR,
                )
            heatmaps = aligned

        accumulated = heatmaps.copy() if accumulated is None else accumulated + heatmaps
        n += 1

    assert accumulated is not None
    return accumulated / max(n, 1)
