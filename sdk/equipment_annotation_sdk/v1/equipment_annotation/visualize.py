"""Pure-OpenCV visualization for equipment annotation results."""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from .constants import CROP_BOX_COLOR, EQUIPMENT_BOX_COLORS, POLE_BOX_COLOR


def draw_annotations(rgb: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """Return a new RGB image with pole, crop, equipment boxes, and keypoints drawn."""
    out = rgb.copy()

    pole = result.get("pole")
    if pole:
        x1, y1, x2, y2 = pole["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), POLE_BOX_COLOR, 2)
        cv2.putText(
            out, f"pole {pole['conf']:.2f}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, POLE_BOX_COLOR, 2, cv2.LINE_AA,
        )

    crop_bounds = result.get("crop_bounds")
    if crop_bounds:
        x1, y1, x2, y2 = crop_bounds
        cv2.rectangle(out, (x1, y1), (x2, y2), CROP_BOX_COLOR, 2, cv2.LINE_AA)

    for det in result.get("equipment") or []:
        cls_name = det["cls_name"]
        color = EQUIPMENT_BOX_COLORS.get(cls_name, (255, 255, 0))
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out, f"{cls_name} {det['conf']:.2f}", (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        for kp in det.get("keypoints") or []:
            x, y = int(round(kp["x"])), int(round(kp["y"]))
            cv2.circle(out, (x, y), 5, color, -1)
            cv2.circle(out, (x, y), 6, (255, 255, 255), 1)
            label = f"{kp['name']} {kp['conf']:.2f}"
            cv2.putText(
                out, label, (x + 8, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    return out
