"""
CalibrationPipeline — public entry point for the desktop app.

Pipeline stages (mirrors src/inference_utils.py::run_end_to_end_inference):

    1. Pole detection on the full image (YOLO ONNX).
    2. Ruler detection on the full image (YOLO ONNX), independent of (1).
    3. Ruler-marking keypoints on the ruler crop (HRNet ONNX, 5 keypoints).
    4. Pole-top keypoint on the upper 10 % of the pole crop (HRNet ONNX,
       1 keypoint).

All output coordinates are in original-image pixel space.

Usage::

    from calibration import CalibrationPipeline

    pipe = CalibrationPipeline()                  # auto-discovers weights/
    result = pipe.run("/path/to/photo.jpg")       # path | ndarray | PIL.Image
    print(result["ruler_keypoints"])              # list of {name,x,y,conf}

    annotated = pipe.run(img, return_annotated=True)["annotated_image"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .constants import (
    MAX_DETECTIONS,
    NMS_IOU_THRESHOLD,
    POLE_CONF_THRESHOLD,
    POLE_INPUT_SIZE,
    POLE_TOP_CROP_FRACTION,
    POLE_TOP_HEATMAP_HW,
    POLE_TOP_INPUT_HW,
    POLE_TOP_NUM_KEYPOINTS,
    RULER_CONF_THRESHOLD,
    RULER_INPUT_SIZE,
    RULER_INPUT_HW,
    RULER_KEYPOINT_NAMES,
    RULER_MARKING_WEIGHTS,
    RULER_NUM_KEYPOINTS,
    WEIGHTS_DIR,
)
from .hrnet_onnx import (
    HrnetOnnxKeypointer,
    heatmaps_to_keypoints_in_crop,
    heatmaps_to_pole_top_in_crop,
)
from .tta import heatmaps_with_vertical_shift_tta
from .yolo_onnx import Detection, YoloOnnxDetector

ImageInput = Union[str, Path, np.ndarray, "PIL.Image.Image"]  # type: ignore[name-defined]


def _load_image_rgb(image: ImageInput) -> np.ndarray:
    """Coerce path / ndarray (BGR or RGB) / PIL.Image into an HxWx3 RGB uint8 array."""
    # Path / str
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # numpy
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3/4 image array, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        # We can't auto-detect BGR vs RGB; the caller is responsible.
        # When loading via cv2.imread the caller should pass BGR converted to RGB,
        # or pass an RGB array directly. We treat ndarray inputs as RGB.
        return image

    # PIL.Image
    try:
        from PIL import Image as PILImage  # local import; PIL is already a hard dep.
    except ImportError:
        raise TypeError(f"Unsupported image type: {type(image)!r}")
    if isinstance(image, PILImage.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _weighted_conf(keypoints: Sequence[Dict[str, Any]], weights: Dict[str, float]) -> float:
    if not keypoints:
        return 0.0
    weighted_sum = 0.0
    total_weight = 0.0
    for kp in keypoints:
        w = weights.get(kp.get("name"), None)
        if w is None:
            continue
        weighted_sum += kp["conf"] * w
        total_weight += w
    if total_weight > 0:
        return weighted_sum / total_weight
    return float(np.mean([kp["conf"] for kp in keypoints]))


class CalibrationPipeline:
    """Four-stage ONNX calibration pipeline. Models are loaded lazily on first call."""

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.weights_dir = Path(weights_dir) if weights_dir else WEIGHTS_DIR
        self._providers = providers or ["CPUExecutionProvider"]
        self._pole: Optional[YoloOnnxDetector] = None
        self._ruler: Optional[YoloOnnxDetector] = None
        self._ruler_kp: Optional[HrnetOnnxKeypointer] = None
        self._pole_top: Optional[HrnetOnnxKeypointer] = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------
    def _w(self, name: str) -> Path:
        p = self.weights_dir / name
        if not p.exists():
            raise FileNotFoundError(
                f"Missing ONNX weight: {p}. Run sdk/calibration_sdk/tools/export_onnx.py."
            )
        return p

    def _get_pole(self) -> YoloOnnxDetector:
        if self._pole is None:
            self._pole = YoloOnnxDetector(
                self._w("pole_detection.onnx"),
                input_size=POLE_INPUT_SIZE,
                conf_threshold=POLE_CONF_THRESHOLD,
                iou_threshold=NMS_IOU_THRESHOLD,
                max_detections=MAX_DETECTIONS,
                providers=self._providers,
            )
        return self._pole

    def _get_ruler(self) -> YoloOnnxDetector:
        if self._ruler is None:
            self._ruler = YoloOnnxDetector(
                self._w("ruler_detection.onnx"),
                input_size=RULER_INPUT_SIZE,
                conf_threshold=RULER_CONF_THRESHOLD,
                iou_threshold=NMS_IOU_THRESHOLD,
                max_detections=MAX_DETECTIONS,
                providers=self._providers,
            )
        return self._ruler

    def _get_ruler_kp(self) -> HrnetOnnxKeypointer:
        if self._ruler_kp is None:
            self._ruler_kp = HrnetOnnxKeypointer(
                self._w("ruler_marking_detection.onnx"),
                input_hw=RULER_INPUT_HW,
                num_keypoints=RULER_NUM_KEYPOINTS,
                keypoint_names=RULER_KEYPOINT_NAMES,
                providers=self._providers,
            )
        return self._ruler_kp

    def _get_pole_top(self) -> HrnetOnnxKeypointer:
        if self._pole_top is None:
            self._pole_top = HrnetOnnxKeypointer(
                self._w("pole_top_detection.onnx"),
                input_hw=POLE_TOP_INPUT_HW,
                num_keypoints=POLE_TOP_NUM_KEYPOINTS,
                keypoint_names=("pole_top",),
                providers=self._providers,
            )
        return self._pole_top

    def warmup(self) -> None:
        """Pre-load all ONNX sessions. Optional — first run() does this lazily."""
        self._get_pole(); self._get_ruler(); self._get_ruler_kp(); self._get_pole_top()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def run(
        self,
        image: ImageInput,
        *,
        use_tta: bool = False,
        return_annotated: bool = False,
        detect_pole: bool = True,
    ) -> Dict[str, Any]:
        """Run the calibration pipeline on a single image.

        Args:
            image: file path, ndarray (treated as RGB if 3-channel) or PIL.Image.
            use_tta: enable vertical-shift TTA for HRNet stages (slower, ~2-3x).
            return_annotated: include an annotated RGB image in the result.
            detect_pole: set False to skip pole + pole-top detection (e.g. for
                midspan photos where there is no pole, only a ruler).

        Returns a dict with keys:

            pole:             {"bbox": (x1,y1,x2,y2), "conf": float} | None
            ruler:            {"bbox": (x1,y1,x2,y2), "conf": float} | None
            ruler_keypoints:  list[{"name","x","y","conf","weighted_conf"}] | None
            pole_top:         {"x","y","conf"} | None
            image_shape:      (height, width)
            annotated_image:  np.ndarray RGB (only if return_annotated=True)
        """
        rgb = _load_image_rgb(image)
        h_img, w_img = rgb.shape[:2]

        result: Dict[str, Any] = {
            "pole": None,
            "ruler": None,
            "ruler_keypoints": None,
            "pole_top": None,
            "image_shape": (h_img, w_img),
        }

        # 1. Pole detection (full image)
        pole_det: Optional[Detection] = None
        if detect_pole:
            pole_dets = self._get_pole()(rgb)
            if pole_dets:
                pole_det = pole_dets[0]
                result["pole"] = {"bbox": pole_det.bbox, "conf": pole_det.conf}

        # 2. Ruler detection (full image, independent)
        ruler_dets = self._get_ruler()(rgb)
        ruler_det: Optional[Detection] = ruler_dets[0] if ruler_dets else None
        if ruler_det is not None:
            result["ruler"] = {"bbox": ruler_det.bbox, "conf": ruler_det.conf}

        # 3. Ruler-marking keypoints (only if ruler detected)
        if ruler_det is not None:
            rx1, ry1, rx2, ry2 = ruler_det.bbox
            ruler_crop = rgb[ry1:ry2, rx1:rx2]
            if ruler_crop.size > 0:
                kp = self._get_ruler_kp()
                if use_tta:
                    heatmaps = heatmaps_with_vertical_shift_tta(kp, ruler_crop)
                else:
                    heatmaps = kp.run_heatmaps(ruler_crop)
                kps_in_crop = heatmaps_to_keypoints_in_crop(
                    heatmaps,
                    crop_hw=(ry2 - ry1, rx2 - rx1),
                    keypoint_names=RULER_KEYPOINT_NAMES,
                )
                # Crop coords -> full image coords
                kps_global: List[Dict[str, Any]] = []
                for k in kps_in_crop:
                    kps_global.append({
                        "name": k["name"],
                        "x": k["x"] + rx1,
                        "y": k["y"] + ry1,
                        "conf": k["conf"],
                    })
                wconf = _weighted_conf(kps_global, RULER_MARKING_WEIGHTS)
                for k in kps_global:
                    k["weighted_conf"] = wconf
                result["ruler_keypoints"] = kps_global

        # 4. Pole-top keypoint (only if pole detected)
        if pole_det is not None:
            px1, py1, px2, py2 = pole_det.bbox
            pole_crop = rgb[py1:py2, px1:px2]
            h_pcrop, w_pcrop = pole_crop.shape[:2]
            crop_height = max(1, int(h_pcrop * POLE_TOP_CROP_FRACTION))
            top_slice = pole_crop[0:crop_height, :]
            if top_slice.size > 0:
                pt_kp = self._get_pole_top()
                if use_tta:
                    heatmaps = heatmaps_with_vertical_shift_tta(pt_kp, top_slice)
                else:
                    heatmaps = pt_kp.run_heatmaps(top_slice)
                pole_top_in_crop = heatmaps_to_pole_top_in_crop(
                    heatmaps[0],
                    cropped_hw=top_slice.shape[:2],
                    resize_hw=POLE_TOP_INPUT_HW,
                )
                # cropped (upper-10%) coords -> pole-bbox coords -> full-image coords
                # Top-slice starts at y=0 inside pole_crop, so y in slice == y in pole_crop.
                px_global = pole_top_in_crop["x"] + px1
                py_global = pole_top_in_crop["y"] + py1
                # Clamp to pole bbox + image bounds (matches upstream).
                px_global = max(px1, min(px2, px_global))
                py_global = max(py1, min(py2, py_global))
                px_global = max(0, min(w_img - 1, px_global))
                py_global = max(0, min(h_img - 1, py_global))
                result["pole_top"] = {
                    "x": px_global,
                    "y": py_global,
                    "conf": pole_top_in_crop["conf"],
                }

        if return_annotated:
            from .visualize import draw_annotations
            result["annotated_image"] = draw_annotations(rgb, result)

        return result
