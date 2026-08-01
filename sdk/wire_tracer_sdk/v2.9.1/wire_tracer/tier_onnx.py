"""Midspan wire TIER classifier — ONNX port of src/midspan_tier.MidspanTierClassifier (v2.9).

Classifies each detected midspan crossing bare / multiplex / comm (or 'none' = veto, the
4th class absorbing false peaks) from a PPI-normalized 40"x10" photo patch (resized to
256x64, RGB/255) via a resnet18 exported to ONNX. The predicted tier feeds the matcher's
``w_mid_tier3_bonus`` agreement bonus.

PPI comes from the calibration ticks the app already passes to ``run(midspan_ticks=…)``
(same source as the ruler-line strip crop): fit the projective height model on the ticks,
take the local inch-per-pixel scale at mid-ruler. Frames without ticks get no tier
(tier3=None on every point — graceful degrade, matcher term is a no-op).

Patch geometry MUST match the training extractor (scripts/diag/probe_tier_separability.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .constants import (
    TIER3_CLASSES,
    TIER_GATES,
    TIER_PATCH_H,
    TIER_PATCH_IN_H,
    TIER_PATCH_IN_W,
    TIER_PATCH_W,
)
from .ruler_line import fit_projective, height_in_at


def ppi_from_ticks(img_h: int, ticks: Sequence[Tuple[float, float, float]]) -> Optional[float]:
    """Local pixels-per-inch at mid-ruler from (height_ft, percent_x, percent_y) ticks.

    Same construction as the repo fallback (_ppi_for_midspan_photo / build_tier_cache):
    fit the projective height model, measure inches across a 0.6%-of-height window."""
    fit = fit_projective(ticks)
    if fit is None:
        return None
    py_mid = float(np.mean([t[2] for t in ticks]))
    h1, h2 = height_in_at(fit, py_mid - 0.3), height_in_at(fit, py_mid + 0.3)
    if h1 is None or h2 is None or abs(h2 - h1) < 1e-6:
        return None
    return (img_h / 100.0) / (abs(h2 - h1) / 0.6)


class MidspanTierOnnx:
    """Attaches ``tier3`` to detected midspan points (in place); None = no signal."""

    def __init__(self, onnx_path: Path, providers: Optional[List[str]] = None,
                 gates: Sequence[float] = TIER_GATES):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.gates = tuple(gates)

    def classify_points(self, img_bgr: np.ndarray, points: List[Dict],
                        ppi: Optional[float]) -> None:
        """Set ``tier3`` on every point ({x, y} in photo percent). No-op without PPI."""
        for p in points:
            p.setdefault("tier3", None)
        if not points or not ppi:
            return
        H, W = img_bgr.shape[:2]
        half_w, half_h = TIER_PATCH_IN_W / 2.0 * ppi, TIER_PATCH_IN_H / 2.0 * ppi
        idxs, batch = [], []
        for i, p in enumerate(points):
            x_px, y_px = p["x"] / 100.0 * W, p["y"] / 100.0 * H
            x0, x1 = int(round(x_px - half_w)), int(round(x_px + half_w))
            y0, y1 = int(round(y_px - half_h)), int(round(y_px + half_h))
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 - x0 < 32 or y1 - y0 < 8:
                continue                                 # border: no tier signal
            patch = cv2.resize(img_bgr[y0:y1, x0:x1], (TIER_PATCH_W, TIER_PATCH_H),
                               interpolation=cv2.INTER_AREA)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            idxs.append(i)
            batch.append(patch.transpose(2, 0, 1))
        if not batch:
            return
        logits = self.session.run(None, {self.input_name: np.stack(batch)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        for i, pr in zip(idxs, probs):
            k = int(np.argmax(pr))
            # 4-class veto: argmax 'none' (last index) -> no tier signal for this peak
            if k < len(TIER3_CLASSES) and pr[k] >= self.gates[k]:
                points[i]["tier3"] = TIER3_CLASSES[k]
