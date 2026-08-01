"""RULER-LINE strip geometry — pure numpy/cv2 port of the training-repo helpers
(src/data_utils.extract_ruler_line_strip + src/ruler_height_model projective fit/inverse).

The strip axis is the least-squares straight line through the CALIBRATION ruler tick
anchors (the same ticks the tkinter calibration step produces / the job JSON's
anchor_calibration stores). Width = 3 ft via the projective model's local vertical
scale at mid-ruler; bottom = the projected 0.0 ft ground line; top = photo row 0.
Rectified with a single shear warp so the tick line becomes the vertical center axis.

Ticks are ``(height_ft, percent_x, percent_y)`` triples; only the real anchor heights
(2.5/6.5/10.5/14.5/16.5 ft) are used. Byte-consistent with the dataset-prep crop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .constants import RULER_ANCHOR_FEET, WIRE_STRIP_LINE_WIDTH_FT

_PY_SCALE = 100.0
_MIN_PROJECTIVE_ANCHORS = 3


def filter_ticks(ticks: Sequence[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Keep only the real anchor-height ticks (drops 0.0 ground / legacy 17.0 rows)."""
    return [(float(f), float(px), float(py)) for f, px, py in ticks if float(f) in RULER_ANCHOR_FEET]


def fit_projective(ticks: Sequence[Tuple[float, float, float]]):
    """percentY -> inches projective fit (a+b·x)/(1+c·x); linear fallback at 2 ticks.

    Mirrors src/ruler_height_model.fit_photo_height for the anchor path. Returns
    ("projective"|"linear", coef) or None."""
    pts = filter_ticks(ticks)
    if len(pts) < 2:
        return None
    xs = [py / _PY_SCALE for _f, _px, py in pts]
    hs = [f * 12.0 for f, _px, _py in pts]
    if len({round(x, 6) for x in xs}) >= _MIN_PROJECTIVE_ANCHORS:
        x = np.asarray(xs, dtype=float)
        h = np.asarray(hs, dtype=float)
        a_mat = np.column_stack([np.ones_like(x), x, -x * h])
        try:
            coef, *_ = np.linalg.lstsq(a_mat, h, rcond=None)
            a, b, c = (float(coef[0]), float(coef[1]), float(coef[2]))
            if all(np.isfinite(v) for v in (a, b, c)):
                return ("projective", (a, b, c))
        except np.linalg.LinAlgError:
            pass
    if len({round(x, 6) for x in xs}) >= 2:
        m, i = np.polyfit(xs, hs, 1)
        return ("linear", (float(m) / _PY_SCALE, float(i)))   # slope per percent
    return None


def height_in_at(fit, percent_y: float) -> Optional[float]:
    """Inches at percent_y (mirrors src/ruler_height_model.height_in_at)."""
    if fit is None:
        return None
    kind, coef = fit
    if kind == "projective":
        a, b, c = coef
        x = float(percent_y) / _PY_SCALE
        denom = 1.0 + c * x
        if denom <= 1e-9:
            return None
        val = (a + b * x) / denom
    else:
        slope, intercept = coef
        val = slope * float(percent_y) + intercept
    return float(val) if val > 0 else None


def percent_y_at_height(fit, inches: float) -> Optional[float]:
    """Inverse model: percent_y where the fit reads `inches` (0 = ground line)."""
    if fit is None:
        return None
    kind, coef = fit
    h = float(inches)
    if kind == "projective":
        a, b, c = coef
        denom = b - h * c
        if abs(denom) <= 1e-9:
            return None
        return (h - a) / denom * _PY_SCALE
    slope, intercept = coef
    if abs(slope) <= 1e-12:
        return None
    return (h - intercept) / slope


def extract_ruler_line_strip(
    img_bgr: np.ndarray,
    ticks: Sequence[Tuple[float, float, float]],
    width_ft: float = WIRE_STRIP_LINE_WIDTH_FT,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """Rectified ruler-line strip (mirrors src/data_utils.extract_ruler_line_strip).

    Returns (strip_bgr, meta) or None when the ticks/fit are unusable. meta carries the
    tick line (line_m, line_c in pixels: x = m·y + c), ground_y_px and width_px needed
    to map strip y_norm back to full-photo percent and to project x onto the tick line.
    """
    pts = filter_ticks(ticks)
    fit = fit_projective(pts)
    if fit is None or len(pts) < 2:
        return None
    img_h, img_w = img_bgr.shape[:2]
    ys = np.array([py / 100.0 * img_h for _f, _px, py in pts], dtype=float)
    xs = np.array([px / 100.0 * img_w for _f, px, _py in pts], dtype=float)
    if len({round(v, 3) for v in ys}) < 2:
        return None
    m, c = np.polyfit(ys, xs, 1)                     # x = m·y + c (pixels)

    ground_pct = percent_y_at_height(fit, 0.0)
    if ground_pct is None or ground_pct * img_h / 100.0 <= ys.max():
        ground_y = img_h
    else:
        ground_y = min(img_h, int(round(ground_pct / 100.0 * img_h)))
    if ground_y < 8:
        return None

    py_mid = float(np.mean([py for _f, _px, py in pts]))
    h1 = height_in_at(fit, py_mid - 0.3)
    h2 = height_in_at(fit, py_mid + 0.3)
    if h1 is None or h2 is None:
        return None
    in_per_pct = abs(h2 - h1) / 0.6
    if in_per_pct <= 1e-6:
        return None
    px_per_inch = (img_h / 100.0) / in_per_pct
    width_px = max(8, int(round(width_ft * 12.0 * px_per_inch)))

    M = np.array([[1.0, m, c - width_px / 2.0],
                  [0.0, 1.0, 0.0]], dtype=np.float64)
    strip = cv2.warpAffine(
        img_bgr, M, (width_px, ground_y),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    meta = {
        "line_m": float(m), "line_c": float(c),
        "ground_y_px": int(ground_y), "width_px": int(width_px),
        "px_per_inch_mid": float(px_per_inch), "width_ft": float(width_ft),
        "full_h": int(img_h), "full_w": int(img_w),
    }
    return strip, meta
