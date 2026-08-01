"""
Pure-OpenCV visualization for calibration results. No matplotlib.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

POLE_BOX_COLOR = (0, 200, 0)        # green
RULER_BOX_COLOR = (0, 165, 255)     # orange (BGR -> RGB equivalent here is (255,165,0))
RULER_KP_COLOR = (255, 0, 0)        # red — ruler markings
POLE_TOP_COLOR = (0, 0, 255)        # blue — pole top


def draw_annotations(rgb: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """Return a new RGB image with detections + keypoints drawn on top."""
    out = rgb.copy()

    pole = result.get("pole")
    if pole:
        x1, y1, x2, y2 = pole["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), POLE_BOX_COLOR, 3)
        cv2.putText(
            out, f"pole {pole['conf']:.2f}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, POLE_BOX_COLOR, 2, cv2.LINE_AA,
        )

    ruler = result.get("ruler")
    if ruler:
        x1, y1, x2, y2 = ruler["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), RULER_BOX_COLOR, 3)
        cv2.putText(
            out, f"ruler {ruler['conf']:.2f}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RULER_BOX_COLOR, 2, cv2.LINE_AA,
        )

    kps = result.get("ruler_keypoints") or []
    for kp in kps:
        x, y = int(round(kp["x"])), int(round(kp["y"]))
        cv2.circle(out, (x, y), 6, RULER_KP_COLOR, -1)
        cv2.circle(out, (x, y), 7, (255, 255, 255), 1)
        label = f"{kp['name']}ft {kp['conf']:.2f}"
        cv2.putText(
            out, label, (x + 10, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RULER_KP_COLOR, 1, cv2.LINE_AA,
        )

    pt = result.get("pole_top")
    if pt:
        x, y = int(round(pt["x"])), int(round(pt["y"]))
        cv2.circle(out, (x, y), 8, POLE_TOP_COLOR, -1)
        cv2.circle(out, (x, y), 9, (255, 255, 255), 1)
        cv2.putText(
            out, f"top {pt['conf']:.2f}", (x + 12, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, POLE_TOP_COLOR, 2, cv2.LINE_AA,
        )

    return out
