"""
Pure-numpy / onnxruntime YOLO inference — boxes AND pose (1-keypoint) variants.

The wire-tracer uses four YOLO models:
  * pole_detection          — 1-class box (shared from calibration_sdk)
  * ruler_detection         — 1-class box
  * wire_detection          — 2-class pose, 1 keypoint  (attachment)
  * wire_attachment_hw      — 8-class pose, 1 keypoint  (attachment)

Ultralytics pose export output layout (NMS-free, opset 17):
  box detector : (1, 4+nc, A)
  pose detector: (1, 4+nc+3*nkpt, A)   # box(4) + class scores(nc) + per-kpt (x,y,conf)
Coordinates are in letterboxed-input space; we undo letterbox to original pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in original image px
    conf: float
    cls: int
    keypoint: Optional[Tuple[float, float, float]] = None  # (x_px, y_px, conf) original frame


def _letterbox(
    img_rgb: np.ndarray,
    new_size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize-and-pad to (new_size, new_size). Returns padded, scale, (pad_x, pad_y)."""
    h, w = img_rgb.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = new_size - new_w
    pad_h = new_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=color,
    )
    return padded, scale, (left, top)


def _nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Pure-numpy NMS. Returns kept indices, score desc."""
    if boxes_xyxy.size == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def _nms_class_aware(
    boxes_xyxy: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray,
    iou_threshold: float, max_detections: int,
) -> List[int]:
    """Per-class NMS, merged and truncated by score (matches Ultralytics agnostic=False)."""
    if boxes_xyxy.size == 0:
        return []
    keep_all: List[int] = []
    for cls in np.unique(cls_ids):
        indices = np.where(cls_ids == cls)[0]
        if indices.size == 0:
            continue
        sub_keep = _nms_numpy(boxes_xyxy[indices], scores[indices], iou_threshold)
        keep_all.extend(int(indices[i]) for i in sub_keep)
    keep_all.sort(key=lambda i: -scores[i])
    return keep_all[:max_detections]


class YoloOnnxDetector:
    """YOLO detector (box or pose) through onnxruntime."""

    def __init__(
        self,
        onnx_path: str | Path,
        input_size: int,
        conf_threshold: float,
        num_keypoints: int = 0,
        iou_threshold: float = 0.7,
        max_detections: int = 1,
        class_aware_nms: bool = False,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.num_keypoints = int(num_keypoints)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.class_aware_nms = bool(class_aware_nms)
        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, img_rgb: np.ndarray) -> List[Detection]:
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {img_rgb.shape}")
        h0, w0 = img_rgb.shape[:2]
        padded, scale, (pad_x, pad_y) = _letterbox(img_rgb, self.input_size)

        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        blob = np.ascontiguousarray(blob)

        out = self.session.run(None, {self.input_name: blob})[0]
        out = np.squeeze(out, axis=0)          # (4+nc+3*nkpt, A)
        if out.shape[0] < 5:
            return []

        nkpt = self.num_keypoints
        n_box_cls = out.shape[0] - 3 * nkpt    # 4 + nc
        cxcywh = out[:4, :].T
        cls_scores = out[4:n_box_cls, :]
        kpt_block = out[n_box_cls:, :] if nkpt else None

        if cls_scores.shape[0] == 1:
            scores = cls_scores[0]
            cls_ids = np.zeros_like(scores, dtype=np.int64)
        else:
            cls_ids = np.argmax(cls_scores, axis=0)
            scores = cls_scores[cls_ids, np.arange(cls_scores.shape[1])]

        keep_mask = scores >= self.conf_threshold
        if not np.any(keep_mask):
            return []

        cxcywh = cxcywh[keep_mask]
        scores = scores[keep_mask]
        cls_ids = cls_ids[keep_mask]
        if kpt_block is not None:
            kpt_block = kpt_block[:, keep_mask]   # (3*nkpt, M)

        cx, cy, ww, hh = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]
        xyxy = np.stack([cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2], axis=1)

        # undo letterbox for boxes
        xyxy[:, [0, 2]] -= pad_x
        xyxy[:, [1, 3]] -= pad_y
        xyxy /= scale
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, w0 - 1)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, h0 - 1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 1, w0)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 1, h0)

        # undo letterbox for keypoint(s): we only use the first keypoint downstream.
        kps_orig = None
        if kpt_block is not None and nkpt > 0:
            kx = (kpt_block[0, :] - pad_x) / scale
            ky = (kpt_block[1, :] - pad_y) / scale
            kc = kpt_block[2, :]
            kps_orig = np.stack([kx, ky, kc], axis=1)   # (M, 3)

        if self.class_aware_nms:
            keep = _nms_class_aware(
                xyxy, scores, cls_ids, self.iou_threshold, self.max_detections,
            )
        else:
            keep = _nms_numpy(xyxy, scores, self.iou_threshold)[: self.max_detections]

        results: List[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
            x1 = max(0, min(x1, w0 - 1))
            y1 = max(0, min(y1, h0 - 1))
            x2 = max(x1 + 1, min(x2, w0))
            y2 = max(y1 + 1, min(y2, h0))
            kp = None
            if kps_orig is not None:
                kp = (float(kps_orig[i, 0]), float(kps_orig[i, 1]), float(kps_orig[i, 2]))
            results.append(Detection((x1, y1, x2, y2), float(scores[i]), int(cls_ids[i]), kp))
        return results
