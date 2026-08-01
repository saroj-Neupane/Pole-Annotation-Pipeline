"""
Pure-numpy / onnxruntime midspan-wire STRIP detector.

Mirrors src/inference_utils.py::infer_wires_on_strip + extract_strip_wire_peaks:

  * preprocess: full-height ruler-column RGB strip -> resize to (3480, 96) ->
    ImageNet-normalized NCHW (torchvision ToPILImage+Resize+ToTensor+Normalize).
  * model: HRNet single-channel logits -> sigmoid heatmap (3480 x 96).
  * peaks: central-band-mean column profile -> find_peaks(height, distance, prominence).

Returns wire y positions normalized to the strip height (0-1), used as the midspan
wire crossings (every wire shares the ruler-column x, so matching is height-only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from .constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    WIRE_STRIP_PEAK_HEIGHT,
    WIRE_STRIP_PEAK_MIN_DISTANCE,
    WIRE_STRIP_PEAK_PROMINENCE,
    WIRE_STRIP_PROFILE_BAND,
    WIRE_STRIP_RESIZE_HEIGHT,
    WIRE_STRIP_RESIZE_WIDTH,
)
from .numpy_ops import find_peaks


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _preprocess_strip(strip_rgb: np.ndarray) -> np.ndarray:
    """torchvision PREPROCESS (ToPILImage+Resize+ToTensor+Normalize) in numpy/PIL."""
    if strip_rgb.ndim != 3 or strip_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB strip, got {strip_rgb.shape}")
    if strip_rgb.dtype != np.uint8:
        strip_rgb = np.clip(strip_rgb, 0, 255).astype(np.uint8)
    pil = Image.fromarray(strip_rgb, mode="RGB")
    pil = pil.resize((WIRE_STRIP_RESIZE_WIDTH, WIRE_STRIP_RESIZE_HEIGHT), resample=Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return np.ascontiguousarray(arr[None, ...])


def strip_column_profile(hm_2d: np.ndarray, band: int = WIRE_STRIP_PROFILE_BAND) -> np.ndarray:
    """Central-band-mean column profile (src/inference_utils.py::strip_column_profile)."""
    w = hm_2d.shape[1]
    cen = w // 2
    lo = max(cen - band, 0)
    hi = min(cen + band, w)
    return hm_2d[:, lo:hi].mean(axis=1)


def extract_strip_wire_peaks(
    hm_2d: np.ndarray,
    min_distance: int = WIRE_STRIP_PEAK_MIN_DISTANCE,
    height: float = WIRE_STRIP_PEAK_HEIGHT,
    prominence: float = WIRE_STRIP_PEAK_PROMINENCE,
    band: int = WIRE_STRIP_PROFILE_BAND,
) -> Tuple[List[int], np.ndarray]:
    """Central-band profile + numpy find_peaks (mirrors src/inference_utils.py)."""
    profile = strip_column_profile(hm_2d, band)
    peaks, _ = find_peaks(profile, height=height, distance=min_distance, prominence=prominence)
    return peaks.tolist(), profile


class WireStripOnnx:
    """HRNet strip model through onnxruntime + numpy peak extraction."""

    def __init__(
        self,
        onnx_path: str | Path,
        peak_height: Optional[float] = None,
        peak_prominence: Optional[float] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.peak_height = WIRE_STRIP_PEAK_HEIGHT if peak_height is None else peak_height
        self.peak_prominence = WIRE_STRIP_PEAK_PROMINENCE if peak_prominence is None else peak_prominence
        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(
        self,
        strip_rgb: np.ndarray,
        min_peaks: Optional[int] = None,
        relax_heights: Tuple[float, ...] = (0.30, 0.20, 0.10),
    ) -> List[Dict[str, float]]:
        """Return [{y, y_norm, conf}] wire crossings on the strip (mirrors infer_wires_on_strip).

        min_peaks (v2.3): COUNT-GUIDED ADAPTIVE extraction — if fewer than this many peaks
        clear the height gate, re-extract at each relax_heights threshold (same heatmap, no
        extra ONNX pass) until the count is plausible. The caller supplies the count prior
        (min(#A, #B) detected pole conductors). None = fixed threshold (v2.2 behavior).
        """
        h = strip_rgb.shape[0]
        blob = _preprocess_strip(strip_rgb)
        logits = self.session.run([self.output_name], {self.input_name: blob})[0]
        heatmap = _sigmoid(logits)[0, 0]   # (Hh, Wh)
        peaks, profile = extract_strip_wire_peaks(
            heatmap, height=self.peak_height, prominence=self.peak_prominence,
        )
        if min_peaks is not None and len(peaks) < min_peaks:
            for relaxed in relax_heights:
                peaks, profile = extract_strip_wire_peaks(
                    heatmap, height=relaxed, prominence=self.peak_prominence,
                )
                if len(peaks) >= min_peaks:
                    break
        hm_h = heatmap.shape[0]
        wires: List[Dict[str, float]] = []
        for y_hm in peaks:
            y_px = y_hm / max(hm_h - 1, 1) * (h - 1) if h > 1 else float(y_hm)
            wires.append({
                "y": float(y_px),
                "y_norm": float(y_px / max(h - 1, 1)) if h > 1 else 0.0,
                "conf": float(profile[y_hm]),
            })
        return wires
