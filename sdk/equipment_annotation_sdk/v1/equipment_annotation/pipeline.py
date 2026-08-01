"""
EquipmentAnnotationPipeline — public entry point for the desktop app.

Pipeline stages (mirrors src/evaluation_attachment_equipment.py equipment path):

    1. Pole detection on the full image (YOLO ONNX, shared with calibration_sdk).
    2. Upper 70% 2:5 crop from pole bbox.
    3. Equipment detection on crop (YOLO ONNX, 4 classes).
    4. Per-detection HRNet keypoints on equipment bbox crops.

All output coordinates are in original-image pixel space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .constants import (
    DEFAULT_POLE_WEIGHTS_PATH,
    EQUIPMENT_BASE_CONF,
    EQUIPMENT_CLASS_NAMES,
    EQUIPMENT_CONF_PER_CLASS,
    EQUIPMENT_INPUT_SIZE,
    EQUIPMENT_KEYPOINT_SPECS,
    EQUIPMENT_MAX_DETECTIONS,
    EQUIPMENT_MIN_BBOX_AREA_FRAC,
    NMS_IOU_THRESHOLD,
    POLE_CONF_THRESHOLD,
    POLE_INPUT_SIZE,
    POLE_MAX_DETECTIONS,
    SECONDARY_DRIP_LOOP_MAX_DET,
    WEIGHTS_DIR,
)
from .crop import extract_equipment_crop
from .hrnet_onnx import HrnetOnnxKeypointer, heatmaps_to_keypoints_in_crop
from .yolo_onnx import Detection, YoloOnnxDetector

ImageInput = Union[str, Path, np.ndarray, "PIL.Image.Image"]  # type: ignore[name-defined]


def _load_image_rgb(image: ImageInput) -> np.ndarray:
    """Coerce path / ndarray / PIL.Image into an HxWx3 RGB uint8 array."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3/4 image array, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return image

    try:
        from PIL import Image as PILImage
    except ImportError:
        raise TypeError(f"Unsupported image type: {type(image)!r}")
    if isinstance(image, PILImage.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _apply_sdl_max_det(equipment: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep at most SECONDARY_DRIP_LOOP_MAX_DET secondary_drip_loop detections."""
    sdl = [d for d in equipment if d["cls_name"] == "secondary_drip_loop"]
    if len(sdl) <= SECONDARY_DRIP_LOOP_MAX_DET:
        return equipment
    sdl.sort(key=lambda d: d["conf"], reverse=True)
    keep_ids = {id(d) for d in sdl[:SECONDARY_DRIP_LOOP_MAX_DET]}
    return [
        d for d in equipment
        if d["cls_name"] != "secondary_drip_loop" or id(d) in keep_ids
    ]


class EquipmentAnnotationPipeline:
    """Equipment-only ONNX annotation pipeline. Models load lazily on first call."""

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        pole_weights_path: Optional[Path] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.weights_dir = Path(weights_dir) if weights_dir else WEIGHTS_DIR
        self.pole_weights_path = Path(pole_weights_path) if pole_weights_path else DEFAULT_POLE_WEIGHTS_PATH
        self._providers = providers or ["CPUExecutionProvider"]
        self._pole: Optional[YoloOnnxDetector] = None
        self._equipment: Optional[YoloOnnxDetector] = None
        self._kp_models: Dict[str, HrnetOnnxKeypointer] = {}

    def _w(self, name: str) -> Path:
        p = self.weights_dir / name
        if not p.exists():
            raise FileNotFoundError(
                f"Missing ONNX weight: {p}. Run sdk/equipment_annotation_sdk/tools/export_onnx.py."
            )
        return p

    def _get_pole(self) -> YoloOnnxDetector:
        if self._pole is None:
            if not self.pole_weights_path.exists():
                raise FileNotFoundError(
                    f"Missing pole ONNX (shared with calibration_sdk): {self.pole_weights_path}"
                )
            self._pole = YoloOnnxDetector(
                self.pole_weights_path,
                input_size=POLE_INPUT_SIZE,
                conf_threshold=POLE_CONF_THRESHOLD,
                iou_threshold=NMS_IOU_THRESHOLD,
                max_detections=POLE_MAX_DETECTIONS,
                class_aware_nms=False,
                providers=self._providers,
            )
        return self._pole

    def _get_equipment(self) -> YoloOnnxDetector:
        if self._equipment is None:
            self._equipment = YoloOnnxDetector(
                self._w("equipment_detection.onnx"),
                input_size=EQUIPMENT_INPUT_SIZE,
                conf_threshold=EQUIPMENT_BASE_CONF,
                iou_threshold=NMS_IOU_THRESHOLD,
                max_detections=EQUIPMENT_MAX_DETECTIONS,
                class_aware_nms=True,
                providers=self._providers,
            )
        return self._equipment

    def _get_kp(self, cls_name: str) -> HrnetOnnxKeypointer:
        if cls_name not in self._kp_models:
            spec = EQUIPMENT_KEYPOINT_SPECS[cls_name]
            onnx_name, input_hw, num_kp, kp_names = spec
            self._kp_models[cls_name] = HrnetOnnxKeypointer(
                self._w(onnx_name),
                input_hw=input_hw,
                num_keypoints=num_kp,
                keypoint_names=kp_names,
                providers=self._providers,
            )
        return self._kp_models[cls_name]

    def warmup(self) -> None:
        """Pre-load all ONNX sessions."""
        self._get_pole()
        self._get_equipment()
        for cls_name in EQUIPMENT_KEYPOINT_SPECS:
            self._get_kp(cls_name)

    def run(
        self,
        image: ImageInput,
        *,
        return_annotated: bool = False,
    ) -> Dict[str, Any]:
        """Run equipment annotation on a single image.

        Returns dict with keys: pole, crop_bounds, equipment, image_shape,
        and optionally annotated_image.
        """
        rgb = _load_image_rgb(image)
        h_img, w_img = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        result: Dict[str, Any] = {
            "pole": None,
            "crop_bounds": None,
            "equipment": [],
            "image_shape": (h_img, w_img),
        }

        pole_dets = self._get_pole()(rgb)
        if not pole_dets:
            if return_annotated:
                from .visualize import draw_annotations
                result["annotated_image"] = draw_annotations(rgb, result)
            return result

        pole_det = pole_dets[0]
        result["pole"] = {"bbox": pole_det.bbox, "conf": pole_det.conf}

        crop_bgr, crop_bounds = extract_equipment_crop(bgr, pole_det.bbox)
        if crop_bgr is None or crop_bounds is None:
            if return_annotated:
                from .visualize import draw_annotations
                result["annotated_image"] = draw_annotations(rgb, result)
            return result

        result["crop_bounds"] = crop_bounds
        crop_x1, crop_y1, _, _ = crop_bounds
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_h, crop_w = crop_rgb.shape[:2]
        crop_area = crop_h * crop_w
        min_bbox_area = crop_area * EQUIPMENT_MIN_BBOX_AREA_FRAC

        equip_dets = self._get_equipment()(crop_rgb)
        equipment_list: List[Dict[str, Any]] = []

        for det in equip_dets:
            cls_name = (
                EQUIPMENT_CLASS_NAMES[det.cls]
                if det.cls < len(EQUIPMENT_CLASS_NAMES)
                else "unknown"
            )
            if cls_name == "unknown":
                continue
            cls_thresh = EQUIPMENT_CONF_PER_CLASS.get(cls_name, EQUIPMENT_BASE_CONF)
            if det.conf < cls_thresh:
                continue

            ex1, ey1, ex2, ey2 = det.bbox
            x1_full = crop_x1 + ex1
            y1_full = crop_y1 + ey1
            x2_full = crop_x1 + ex2
            y2_full = crop_y1 + ey2
            bbox_area = (x2_full - x1_full) * (y2_full - y1_full)
            if bbox_area < min_bbox_area:
                continue

            entry: Dict[str, Any] = {
                "cls_id": det.cls,
                "cls_name": cls_name,
                "bbox": (x1_full, y1_full, x2_full, y2_full),
                "conf": det.conf,
                "keypoints": [],
            }

            eq_crop = rgb[y1_full:y2_full, x1_full:x2_full]
            if (
                eq_crop.shape[0] >= 10
                and eq_crop.shape[1] >= 10
                and cls_name in EQUIPMENT_KEYPOINT_SPECS
            ):
                kp_model = self._get_kp(cls_name)
                _, input_hw, _, kp_names = EQUIPMENT_KEYPOINT_SPECS[cls_name]
                heatmaps = kp_model.run_heatmaps(eq_crop)
                det_h, det_w = eq_crop.shape[:2]
                kps_in_crop = heatmaps_to_keypoints_in_crop(
                    heatmaps,
                    crop_hw=(det_h, det_w),
                    keypoint_names=kp_names,
                )
                for k in kps_in_crop:
                    x_px = x1_full + k["x"]
                    y_px = y1_full + k["y"]
                    entry["keypoints"].append({
                        "name": k["name"],
                        "x": x_px,
                        "y": y_px,
                        "conf": k["conf"],
                    })

            equipment_list.append(entry)

        result["equipment"] = _apply_sdl_max_det(equipment_list)

        if return_annotated:
            from .visualize import draw_annotations
            result["annotated_image"] = draw_annotations(rgb, result)

        return result
