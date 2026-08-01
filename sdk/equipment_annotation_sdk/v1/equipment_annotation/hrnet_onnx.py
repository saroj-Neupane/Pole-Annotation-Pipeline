"""
Pure-numpy / onnxruntime HRNet keypoint inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from .constants import IMAGENET_MEAN, IMAGENET_STD


def preprocess_for_hrnet(
    rgb: np.ndarray,
    input_hw: Tuple[int, int],
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    """Match torchvision PREPROCESS exactly. Returns NCHW float32 array."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB, got dtype={rgb.dtype}")
    h_in, w_in = input_hw
    pil = Image.fromarray(rgb, mode="RGB")
    pil = pil.resize((w_in, h_in), resample=Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean_arr) / std_arr
    return np.ascontiguousarray(arr[None, ...])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class HrnetOnnxKeypointer:
    """Run HRNet ONNX heatmap inference and decode peaks for a crop."""

    def __init__(
        self,
        onnx_path: str | Path,
        input_hw: Tuple[int, int],
        num_keypoints: int,
        keypoint_names: Optional[Sequence[str]] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.input_hw = input_hw
        self.num_keypoints = int(num_keypoints)
        self.keypoint_names = (
            list(keypoint_names) if keypoint_names is not None else [str(i) for i in range(num_keypoints)]
        )
        if len(self.keypoint_names) != self.num_keypoints:
            raise ValueError("keypoint_names length must match num_keypoints")
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run_heatmaps(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Run the model and return sigmoid'd heatmaps with shape (K, Hh, Wh)."""
        blob = preprocess_for_hrnet(crop_rgb, self.input_hw)
        logits = self.session.run([self.output_name], {self.input_name: blob})[0]
        return _sigmoid(logits)[0]


def heatmaps_to_keypoints_in_crop(
    heatmaps: np.ndarray,
    crop_hw: Tuple[int, int],
    keypoint_names: Sequence[str],
) -> List[dict]:
    """Decode heatmaps into per-keypoint dicts in crop pixel coordinates."""
    h_crop, w_crop = crop_hw
    kps: List[dict] = []
    for idx, hm in enumerate(heatmaps):
        y_int, x_int = np.unravel_index(int(np.argmax(hm)), hm.shape)
        conf = float(hm[y_int, x_int])
        y_sub, x_sub = float(y_int), float(x_int)
        x_px = x_sub / max(hm.shape[1] - 1, 1) * (w_crop - 1) if w_crop > 1 else x_sub
        y_px = y_sub / max(hm.shape[0] - 1, 1) * (h_crop - 1) if h_crop > 1 else y_sub
        kps.append({
            "name": keypoint_names[idx],
            "x": x_px,
            "y": y_px,
            "conf": conf,
        })
    return kps
