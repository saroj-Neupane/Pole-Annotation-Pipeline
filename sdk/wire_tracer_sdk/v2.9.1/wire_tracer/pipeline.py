"""
WireTracerPipeline — public entry point for the desktop app (V2).

Reconstructs one pole-mid-pole span from three photos, mirroring the PRODUCTION operating
point of src/wire_tracer (build_default_tracer defaults: pole_source='unified' +
midspan_source='strip' + the LEARNED edge-cost matcher):

    pole-A photo ─┐
                  ├─ pole_detection ─ upper70% 2:5 crop ─ unified_pole_detection (ONE joint-class
    pole-B photo ─┘                   model: hardware x cable_type x crossarm-K)
    midspan photo(s) ── ruler_detection ─ column strip ─ wire_strip HRNet ─ 1-D wire peaks
                  └────────── A<->B-coupled LEARNED-cost monotonic matcher ──────────┘

Output: pole-A insulators, pole-B insulators, midspan wire crossings, and A<->B traces.
`wire_type` (primary/secondary/neutral/comm) is NOT inferred — the user assigns it. V2 surfaces
two NEW non-authoritative hints the unified model predicts: `cable_type_hint` (coarse electrical
class) and `crossarm_k` (model-predicted wire-count). All x/y are in original-image percent (0-100).

See ../README.md "How v2 differs from v1" for the full v1->v2 delta.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .constants import (
    CROSSARM_HW,
    DEFAULT_EDGE_MODEL_PATH,
    DEFAULT_POLE_WEIGHTS_PATH,
    DEFAULT_RULER_WEIGHTS_PATH,
    DEFAULT_UNIFIED_CONF_JSON,
    INSULATOR_DISPLAY,
    NMS_IOU_THRESHOLD,
    POLE_CONF_THRESHOLD,
    POLE_DEDUP_Y_PCT,
    POLE_DEDUP_Y_PCT_DETECT,
    POLE_INPUT_SIZE,
    POLE_MAX_DETECTIONS,
    RULER_CONF_THRESHOLD,
    RULER_INPUT_SIZE,
    RULER_MAX_DETECTIONS,
    STRIP_ADAPTIVE,
    STRIP_RELAX_LADDER,
    STRIP_WIDTH_EXPAND,
    TOKEN_TO_SPEC,
    DOWN_GUY_CONF_GATE,
    DOWN_GUY_DEDUP_INCH,
    SUBGATE_FLOOR,
    SUBGATE_PEN,
    UNIFIED_CONF_FLOOR,
    UNIFIED_CONF_PER_CLASS,
    UNIFIED_INPUT_SIZE,
    UNIFIED_MAX_DETECTIONS,
    UNIFIED_POLE_DETECTION_CLASS_NAMES,
    WEIGHTS_DIR,
    WIRE_HW_TO_TIER,
)
from .crop import compute_pole_upper70_2x5_crop
from .edge_model import NumpyEdgeCostModel
from .matcher import MatchConfig, match_span
from .strip_onnx import WireStripOnnx
from .unified import unified_point
from .yolo_onnx import YoloOnnxDetector

ImageInput = Union[str, Path, np.ndarray]


def _load_image_bgr(image: ImageInput) -> np.ndarray:
    """Coerce path / ndarray into an HxWx3 BGR uint8 array (OpenCV convention)."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return bgr
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3/4 image array, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return image
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _to_matcher_side(detected: List[Dict], is_pole: bool,
                     mult: Optional[Dict[int, int]] = None) -> List[Dict]:
    """Mirror src/wire_tracing_e2e.to_matcher_side (carrying wire_class + conf for the learned cost)."""
    side = []
    for i, d in enumerate(detected):
        if is_pole:
            spec = TOKEN_TO_SPEC.get(d.get("hw_token"))
            traces = [{"insulator_spec": spec, "cable_type": None}]
            kind = d["kind"]
        else:
            traces = [{"insulator_spec": None, "cable_type": None}]
            kind = "wire"
        k = (mult or {}).get(i, d.get("pred_mult", 1))
        side.append({"x": d["x"], "y": d["y"], "kind": kind, "multiplicity": max(1, k),
                     "traces": traces, "i": i, "wire_class": d.get("wire_class"),
                     "conf": d.get("conf"),
                     "tier3": d.get("tier3")})   # v2.9: midspan patch tier / pole fine-class tier
    return side


def _dedup_pole_points(points: List[Dict], y_tol: float) -> List[Dict]:
    """Height-band dedup, KIND-AWARE (mirror src/wire_tracer._dedup_pole_points).

    Conductors merge by height only; guying nodes partition by token so a guy and a co-located
    down_guy stay DISTINCT (different labels). An unread guying node (token None) is compatible
    with either, preserving the token promotion below."""
    def dedup_group(d):
        if d.get("kind") == "guying" or d.get("hw_token") in ("guy", "down_guy"):
            return d.get("hw_token")  # "guy", "down_guy", or None (unread guying)
        return "cond"

    def compatible(a, b):
        # v2.5: down_guy NEVER merges here (mirror src/wire_tracing_e2e same_group) — its
        # dedicated dedup+anchor step (_select_down_guys) owns down_guy duplicate handling.
        if a.get("hw_token") == "down_guy" or b.get("hw_token") == "down_guy":
            return False
        ga, gb = dedup_group(a), dedup_group(b)
        if ga == "cond" or gb == "cond":
            return ga == gb
        return ga == gb or ga is None or gb is None
    kept: List[Dict] = []
    for p in sorted(points, key=lambda d: -float(d.get("conf", 0.0))):
        near = next((k for k in kept if abs(k["y"] - p["y"]) <= y_tol and compatible(k, p)), None)
        if near is None:
            kept.append(dict(p))
        elif near.get("hw_token") is None and p.get("hw_token") is not None:
            near["hw_token"] = p["hw_token"]
            near["kind"] = p.get("kind", near.get("kind"))
    return kept


def _select_down_guys(points: List[Dict], k_expected: Optional[int]) -> List[Dict]:
    """v2.5 down_guy dedup + anchor-count guidance (constants.DOWN_GUY_*).

    Mirrors src/wire_tracing_e2e.dedup_pole_points_for_photo's down_guy path (val-tuned,
    test kp-F1@6in 0.660 -> 0.717): height-dedup within DOWN_GUY_DEDUP_INCH (inches via the
    detection's own 1 ft bbox height), anchor-K guard re-admits genuine same-height twins,
    then gate at DOWN_GUY_CONF_GATE with anchor-RELAX down to the floor until K is met.
    k_expected = down-guy count from the job JSON anchor inventory (None = no anchor data)."""
    dg = sorted((p for p in points if p.get("hw_token") == "down_guy"),
                key=lambda p: -float(p.get("conf", 0.0)))
    if not dg:
        return points
    rest = [p for p in points if p.get("hw_token") != "down_guy"]

    def _close(a, b) -> bool:
        ha, hb = a.get("box_h_pct"), b.get("box_h_pct")
        if ha and hb:  # bbox = 1 ft -> inches
            return abs(a["y"] - b["y"]) / ((ha + hb) / 2.0) * 12.0 <= DOWN_GUY_DEDUP_INCH
        return abs(a["y"] - b["y"]) <= POLE_DEDUP_Y_PCT_DETECT  # scale-less fallback

    kept: List[Dict] = []
    dropped: List[Dict] = []
    for p in dg:
        (dropped if any(_close(p, o) for o in kept) else kept).append(p)
    if k_expected is not None and len(kept) < k_expected and dropped:
        kept += dropped[: k_expected - len(kept)]      # anchor guard: genuine twins
        kept.sort(key=lambda p: -float(p.get("conf", 0.0)))
    keep = [p for p in kept if float(p.get("conf", 0.0)) >= DOWN_GUY_CONF_GATE]
    if k_expected is not None and len(keep) < k_expected:
        keep += [p for p in kept if float(p.get("conf", 0.0)) < DOWN_GUY_CONF_GATE][
            : k_expected - len(keep)]                  # anchor RELAX below the gate
    return rest + keep


def _normalize_guy_kind(points: List[Dict]) -> None:
    for p in points:
        if p.get("hw_token") in ("guy", "down_guy"):
            p["kind"] = "guying"


class WireTracerPipeline:
    """Wire-tracer ONNX pipeline (V2: unified node source + learned matcher). Models load lazily."""

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        pole_weights_path: Optional[Path] = None,
        ruler_weights_path: Optional[Path] = None,
        providers: Optional[List[str]] = None,
        edge_model_path: Optional[Path] = None,
        pole_dedup_y: float = POLE_DEDUP_Y_PCT,
        pole_dedup_y_detect: float = POLE_DEDUP_Y_PCT_DETECT,
    ) -> None:
        self.weights_dir = Path(weights_dir) if weights_dir else WEIGHTS_DIR
        self.pole_weights_path = Path(pole_weights_path) if pole_weights_path else DEFAULT_POLE_WEIGHTS_PATH
        # v2.5: ruler_detection is shared with calibration_sdk too (not bundled here)
        self.ruler_weights_path = Path(ruler_weights_path) if ruler_weights_path else DEFAULT_RULER_WEIGHTS_PATH
        self._providers = providers or ["CPUExecutionProvider"]
        self.pole_dedup_y = float(pole_dedup_y)             # tracer 1.5% band
        self.pole_dedup_y_detect = float(pole_dedup_y_detect)  # detector 0.6% kind-aware band

        # learned edge-cost matcher (the V2 lever). Loaded from weights/edge_matcher_unified_v2.json.
        edge_path = Path(edge_model_path) if edge_model_path else (self.weights_dir / DEFAULT_EDGE_MODEL_PATH.name)
        if not edge_path.exists():
            raise FileNotFoundError(
                f"Missing learned edge model: {edge_path}. Run tools/export_onnx.py."
            )
        self._cfg = MatchConfig(edge_model=NumpyEdgeCostModel.load(edge_path))

        # v2.5: the constants map IS the operating point (flat 0.20 + arm floor 0.10 + guy
        # 0.2852 + down_guy 0.05-floor-then-gated). A bundled
        # unified_perclass_conf.json (the dropped F1 gate) would override it — do NOT ship one.
        conf_json = self.weights_dir / DEFAULT_UNIFIED_CONF_JSON.name
        if conf_json.exists():
            import json
            self.unified_conf_per_class = json.loads(conf_json.read_text())
        else:
            self.unified_conf_per_class = dict(UNIFIED_CONF_PER_CLASS)

        self._pole: Optional[YoloOnnxDetector] = None
        self._unified: Optional[YoloOnnxDetector] = None
        self._ruler: Optional[YoloOnnxDetector] = None
        self._strip: Optional[WireStripOnnx] = None
        self._tier = None            # v2.9 midspan tier classifier (lazy; None until loaded)
        self._tier_missing = False   # True after a failed load (don't retry every span)

    # ----------------------------------------------------------------- weights
    def _w(self, name: str) -> Path:
        p = self.weights_dir / name
        if not p.exists():
            raise FileNotFoundError(
                f"Missing ONNX weight: {p}. Run sdk/wire_tracer_sdk/v2/tools/export_onnx.py."
            )
        return p

    def _get_pole(self) -> YoloOnnxDetector:
        if self._pole is None:
            if not self.pole_weights_path.exists():
                raise FileNotFoundError(
                    f"Missing pole ONNX (shared with calibration_sdk): {self.pole_weights_path}"
                )
            self._pole = YoloOnnxDetector(
                self.pole_weights_path, input_size=POLE_INPUT_SIZE,
                conf_threshold=POLE_CONF_THRESHOLD, num_keypoints=0,
                iou_threshold=NMS_IOU_THRESHOLD, max_detections=POLE_MAX_DETECTIONS,
                class_aware_nms=False, providers=self._providers,
            )
        return self._pole

    def _get_unified(self) -> YoloOnnxDetector:
        if self._unified is None:
            self._unified = YoloOnnxDetector(
                self._w("unified_pole_detection.onnx"), input_size=UNIFIED_INPUT_SIZE,
                conf_threshold=UNIFIED_CONF_FLOOR, num_keypoints=1, iou_threshold=NMS_IOU_THRESHOLD,
                max_detections=UNIFIED_MAX_DETECTIONS, class_aware_nms=True, providers=self._providers,
            )
        return self._unified

    def _get_ruler(self) -> YoloOnnxDetector:
        if self._ruler is None:
            if not self.ruler_weights_path.exists():
                raise FileNotFoundError(
                    f"Missing ruler ONNX (shared with calibration_sdk): {self.ruler_weights_path}"
                )
            self._ruler = YoloOnnxDetector(
                self.ruler_weights_path, input_size=RULER_INPUT_SIZE,
                conf_threshold=RULER_CONF_THRESHOLD, num_keypoints=0,
                iou_threshold=NMS_IOU_THRESHOLD, max_detections=RULER_MAX_DETECTIONS,
                class_aware_nms=False, providers=self._providers,
            )
        return self._ruler

    def _get_strip(self) -> WireStripOnnx:
        if self._strip is None:
            self._strip = WireStripOnnx(
                self._w("midspan_wire_strip_detection.onnx"), providers=self._providers,
            )
        return self._strip

    def _get_tier(self):
        """v2.9 midspan tier classifier — OPTIONAL: a bundle without the tier ONNX still
        traces (tier3=None everywhere, the matcher bonus is a no-op)."""
        if self._tier is None and not self._tier_missing:
            from .constants import TIER_ONNX_NAME
            p = self.weights_dir / TIER_ONNX_NAME
            if p.exists():
                from .tier_onnx import MidspanTierOnnx
                self._tier = MidspanTierOnnx(p, providers=self._providers)
            else:
                self._tier_missing = True
        return self._tier

    def warmup(self) -> None:
        """Pre-load all ONNX sessions."""
        self._get_pole()
        self._get_unified()
        self._get_ruler()
        self._get_strip()
        self._get_tier()

    # ----------------------------------------------------------------- detectors
    def _detect_pole_points(self, photo: ImageInput) -> List[Dict]:
        """Unified-model pole points on the upper-70% 2:5 crop.

        Mirrors src/wire_tracing_e2e.detect_pole_points (pole_source='unified') + its 0.6%
        kind-aware dedup. Returns [{x, y, kind, hw_token, conf, wire_class, pred_mult, display}].
        """
        img = _load_image_bgr(photo)
        H, W = img.shape[:2]
        rgb_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pole_dets = self._get_pole()(rgb_full)
        if not pole_dets:
            return []
        x1, y1, x2, y2 = pole_dets[0].bbox
        crop_res = compute_pole_upper70_2x5_crop(img, (x1, y1, x2, y2), W, H)
        if crop_res is None:
            return []
        crop_bgr, cx1, cy1, _cx2, _cy2, _cw, _ch = crop_res
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        def to_pct(kx, ky):
            return (100.0 * (cx1 + kx) / W, 100.0 * (cy1 + ky) / H)

        points: List[Dict] = []
        for det in self._get_unified()(crop_rgb):
            if det.cls >= len(UNIFIED_POLE_DETECTION_CLASS_NAMES):
                continue
            name = UNIFIED_POLE_DETECTION_CLASS_NAMES[det.cls]
            below_gate = det.conf < self.unified_conf_per_class.get(name, UNIFIED_CONF_FLOOR)
            # v2.9.1 SUB-GATE retention: keep [SUBGATE_FLOOR, gate) conductor dets as flagged
            # candidates for the tier-corroborated second pass (held out of pass-1).
            is_subgate = below_gate and det.conf >= SUBGATE_FLOOR
            if below_gate and not is_subgate:
                continue
            if det.keypoint is not None and (det.keypoint[0] != 0 or det.keypoint[1] != 0):
                kx, ky = det.keypoint[0], det.keypoint[1]
            else:  # box-center fallback (mirror the e2e unified path)
                kx, ky = (det.bbox[0] + det.bbox[2]) / 2.0, (det.bbox[1] + det.bbox[3]) / 2.0
            xp, yp = to_pct(kx, ky)
            box_h_pct = 100.0 * (det.bbox[3] - det.bbox[1]) / H   # 1 ft label box, % of image H
            pt = unified_point(name, xp, yp, det.conf, box_h_pct=box_h_pct)
            if pt is not None:
                if is_subgate:
                    if pt.get("kind") == "guying":
                        continue          # guys are never span endpoints
                    pt["_subgate"] = True
                points.append(pt)
        # 0.6% kind-aware detector dedup (the +2.27pp lever), before the tracer's coarser 1.5%.
        # Sub-gate candidates bypass dedup — they are not pass-1 points.
        sub = [p for p in points if p.get("_subgate")]
        points = [p for p in points if not p.get("_subgate")]
        if self.pole_dedup_y_detect and self.pole_dedup_y_detect > 0:
            points = _dedup_pole_points(points, self.pole_dedup_y_detect)
        return points + sub

    def _detect_midspan_points(self, photos: List[ImageInput],
                               min_peaks: Optional[int] = None,
                               ticks: Optional[List[Optional[List[Tuple[float, float, float]]]]] = None,
                               ) -> tuple[List[Dict], Optional[Any]]:
        """Strip-based midspan wire detection across burst frames (most detections wins).

        v2.6: frames WITH calibration ticks use the RULER-LINE crop (the geometry the strip
        ONNX was trained on): axis = tick line, 3 ft rectified width, ground-line bottom;
        every emitted x is projected onto the tick line (the annotation convention — wire
        markers sit ON the ruler axis). Frames without ticks fall back to the legacy
        ruler-ONNX column crop. min_peaks enables the v2.3 count-guided adaptive extraction
        (see run()); None = fixed 0.40 gate. Returns ([{x, y, conf}], frame_used).
        """
        from .ruler_line import extract_ruler_line_strip
        best_pts: List[Dict] = []
        best_frame = photos[0] if photos else None
        best_fi = 0
        for fi, photo in enumerate(photos):
            img = _load_image_bgr(photo)
            H, W = img.shape[:2]
            ftk = ticks[fi] if ticks and fi < len(ticks) else None
            pts: Optional[List[Dict]] = None
            if ftk:
                out = extract_ruler_line_strip(img, ftk)
                if out is not None:
                    strip, lmeta = out
                    wires = self._get_strip().infer(
                        cv2.cvtColor(strip, cv2.COLOR_BGR2RGB),
                        min_peaks=min_peaks, relax_heights=STRIP_RELAX_LADDER)
                    gy = lmeta["ground_y_px"]
                    pts = []
                    for w in wires:
                        y_px = w["y_norm"] * gy                       # strip bottom = ground line
                        x_px = lmeta["line_m"] * y_px + lmeta["line_c"]
                        pts.append({"x": 100.0 * x_px / W, "y": 100.0 * y_px / H, "conf": w["conf"]})
            if pts is None:
                # legacy column fallback (no ticks): ruler ONNX -> 3x-widened column crop.
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rdets = self._get_ruler()(rgb)
                if not rdets:
                    continue
                rx1f, _ry1, rx2f, _ry2 = rdets[0].bbox
                cx = 0.5 * (rx1f + rx2f)
                half = 0.5 * (rx2f - rx1f) * STRIP_WIDTH_EXPAND
                rx1, rx2 = int(max(0, cx - half)), int(min(W, cx + half))
                if rx2 - rx1 < 4:
                    continue
                strip_rgb = cv2.cvtColor(img[:, rx1:rx2], cv2.COLOR_BGR2RGB)
                wires = self._get_strip().infer(strip_rgb, min_peaks=min_peaks,
                                                relax_heights=STRIP_RELAX_LADDER)
                xc = 100.0 * cx / W
                pts = [{"x": xc, "y": 100.0 * w["y_norm"], "conf": w["conf"]} for w in wires]
            if len(pts) > len(best_pts):
                best_pts, best_frame, best_fi = pts, photo, fi
        # v2.9 MIDSPAN TIER stage: classify each crossing bare/multiplex/comm from a
        # PPI-normalized photo patch (PPI from the winning frame's calibration ticks).
        # No ticks / no classifier ONNX -> tier3 stays None (matcher term no-op).
        tier = self._get_tier()
        if tier is not None and best_pts:
            ftk = ticks[best_fi] if ticks and best_fi < len(ticks) else None
            if ftk:
                img = _load_image_bgr(best_frame)
                from .tier_onnx import ppi_from_ticks
                tier.classify_points(img, best_pts, ppi_from_ticks(img.shape[0], ftk))
        return best_pts, best_frame

    # ----------------------------------------------------------------- reconstruction
    def _build_pole_attachments(self, det_side: List[Dict],
                                pred_for_side: List[Optional[int]], side: str) -> List[Dict]:
        wires_by_point: Dict[int, List[int]] = defaultdict(list)
        for m, pi in enumerate(pred_for_side):
            if pi is not None:
                wires_by_point[pi].append(m)
        out = []
        for i, d in enumerate(det_side):
            token = d.get("hw_token")
            is_guy = (d.get("kind") == "guying") or token in ("guy", "down_guy")
            traced = wires_by_point.get(i, [])
            traced_count = len(traced)
            if is_guy:
                wire_count, role = 0, "guying"
                name = "Guy" if token == "guy" else "Down Guy"
            else:
                wire_count = max(traced_count, 1)
                role = "crossarm" if traced_count > 1 else "single"
                name = INSULATOR_DISPLAY.get(token, INSULATOR_DISPLAY[None])
            out.append({
                "id": f"{side}{i}",
                "hardware": token,
                "insulator_name": name,
                "x": round(d["x"], 2),
                "y": round(d["y"], 2),
                "conf": round(float(d.get("conf", 0.0)), 3),
                "tier_hint": WIRE_HW_TO_TIER.get(token),
                # V2 additive hints from the unified model (non-authoritative; wire_type still user-set):
                "cable_type_hint": d.get("wire_class"),     # primary/secondary/neutral/comm | None
                "cable_type_fine": d.get("cable_fine"),     # keeps catv/telco/fiber distinct | None
                "crossarm_k": int(d.get("pred_mult", 1) or 1),  # model-predicted wire-count
                "role": role,
                "wire_count": wire_count,
                "traced_midspan_count": traced_count,
                "traced_midspan_ids": [f"M{m}" for m in traced],
                "wire_type": None,
            })
        return out

    def _build_traces(self, predA, predB, poleA, poleB, detM) -> List[Dict]:
        traces = []
        for m in range(len(detM)):
            ai = predA[m] if m < len(predA) else None
            bi = predB[m] if m < len(predB) else None
            traces.append({
                "midspan_id": f"M{m}",
                "midspan_y": round(detM[m]["y"], 2),
                "pole_a_attachment": f"A{ai}" if ai is not None else None,
                "pole_a_insulator": poleA[ai]["insulator_name"] if ai is not None else None,
                "pole_b_attachment": f"B{bi}" if bi is not None else None,
                "pole_b_insulator": poleB[bi]["insulator_name"] if bi is not None else None,
                "wire_type": None,
            })
        return traces

    def run(
        self,
        pole_a_image: ImageInput,
        midspan_images: Union[ImageInput, List[ImageInput]],
        pole_b_image: ImageInput,
        *,
        mult_cap: Optional[int] = None,
        return_annotated: bool = False,
        down_guy_expected_a: Optional[int] = None,
        down_guy_expected_b: Optional[int] = None,
        midspan_ticks: Optional[List[Optional[List[Tuple[float, float, float]]]]] = None,
    ) -> Dict[str, Any]:
        """Trace one span. midspan_images may be a single image or a list of burst frames.

        midspan_ticks (v2.6): CALIBRATION ruler tick anchors per midspan frame, aligned with
        midspan_images — each entry a list of (height_ft, percent_x, percent_y) triples (the
        2.5/6.5/10.5/14.5/16.5 ft ticks the calibration step already produced; from the job
        JSON: photofirst_data.anchor_calibration[*].height + .pixel_selection[0].percentX/Y —
        helper in INTEGRATION.md). Pass a single list to apply to every frame. Enables the
        RULER-LINE strip crop (the promoted geometry, +3.7pp e2e): axis = tick line, 3ft
        rectified, ground-line bottom; the model was TRAINED on this crop, so supplying ticks
        is strongly recommended. None (or a frame without ticks) falls back to the legacy
        ruler-ONNX column crop.

        down_guy_expected_a/b (v2.5): down-guy count for each pole from the job JSON anchor
        inventory — sum the comma-separated `sizes_of_attached_dn_guys` entries over anchors
        CONNECTED to the pole node (skip `node_type == 'new anchor'`); pass 0 when the pole
        verifiably has no anchors, None when unknown. Guides the down_guy dedup/gate
        (constants.DOWN_GUY_*): duplicates are merged, genuine same-height twins are kept,
        and sub-gate candidates are admitted only up to this count.

        Returns a dict mirroring src/wire_tracer.trace_span: poles {A, B}, midspan, traces,
        midspan_wire_count, config. Each pole attachment additionally carries the V2
        cable_type_hint + crossarm_k hints.
        """
        if not isinstance(midspan_images, (list, tuple)):
            midspan_images = [midspan_images]
        if midspan_ticks:
            first = midspan_ticks[0]
            # a bare tick list [(ft,px,py), ...] (single-frame form) -> apply to every frame
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], (int, float)):
                midspan_ticks = [list(midspan_ticks)] * len(midspan_images)
            elif len(midspan_ticks) == 1 and len(midspan_images) > 1:
                midspan_ticks = list(midspan_ticks) * len(midspan_images)

        detA = self._detect_pole_points(pole_a_image)
        detB = self._detect_pole_points(pole_b_image)
        # v2.9.1: sub-gate candidates held out of ALL pass-1 processing
        subA = [d for d in detA if d.get("_subgate")]
        subB = [d for d in detB if d.get("_subgate")]
        detA = [d for d in detA if not d.get("_subgate")]
        detB = [d for d in detB if not d.get("_subgate")]
        if self.pole_dedup_y and self.pole_dedup_y > 0:
            detA = _dedup_pole_points(detA, self.pole_dedup_y)
            detB = _dedup_pole_points(detB, self.pole_dedup_y)
        detA = _select_down_guys(detA, down_guy_expected_a)
        detB = _select_down_guys(detB, down_guy_expected_b)
        _normalize_guy_kind(detA)
        _normalize_guy_kind(detB)
        # v2.3 COUNT-GUIDED ADAPTIVE midspan: nearly every span wire reaches both poles, so the
        # strip should find at least min(#A, #B) conductor wires (model-predicted crossarm K,
        # guys excluded — they never cross a span). Fewer peaks => the strip MISSED wires
        # (unrecoverable); the extractor then relaxes its height gate for THIS span only
        # (STRIP_RELAX_LADDER) and the matcher dustbin absorbs any false extra.
        min_peaks = None
        if STRIP_ADAPTIVE and detA and detB:
            def _cond_count(side):
                return sum(max(1, d.get("pred_mult") or 1)
                           for d in side if d.get("kind") != "guying")
            min_peaks = min(_cond_count(detA), _cond_count(detB)) or None
        detM, _frame = self._detect_midspan_points(list(midspan_images), min_peaks=min_peaks, ticks=midspan_ticks)

        nM = len(detM)
        def _mult(side):
            # MODEL-PREDICTED multiplicity (2026-07-30 fix, mirror src/wire_tracer): the
            # unified model's arm classes carry crossarm-K (pred_mult) — a class-'pin'
            # point is ONE insulator; the old unbounded midspan cap invented "Crossarm xK"
            # hardware that was never detected. mult_cap only LOWERS the model K.
            def k_of(d):
                if d.get("hw_token") not in CROSSARM_HW:
                    return 1
                k = max(1, d.get("pred_mult", 1) or 1)
                return k if mult_cap is None else min(k, max(1, mult_cap))
            return {i: k_of(d) for i, d in enumerate(side)}

        det_span = {"sides": {
            "A": _to_matcher_side(detA, True, _mult(detA)),
            "M": _to_matcher_side(detM, False),
            "B": _to_matcher_side(detB, True, _mult(detB)),
        }}
        preds = match_span(det_span, self._cfg)

        # v2.9.1 TIER-CORROBORATED SUB-GATE ADMISSION (EXP-0007, +0.52pp): a pass-1
        # DUSTBINNED midspan wire with a tier3 admits held-out sub-gate pole dets whose
        # class-tier AGREES (edge penalty SUBGATE_PEN); the span is then re-matched.
        # Never fires without the tier stage (no midspan tier3 -> no trigger).
        if (subA or subB) and any(d.get("tier3") for d in detM):
            def _admit(base, sub, pred):
                need = {detM[m].get("tier3") for m, pi in enumerate(pred)
                        if pi is None and detM[m].get("tier3")}
                if not need:
                    return base, set()
                seen = {(round(d["y"], 3), d.get("hw_token")) for d in base}
                out, marks = list(base), set()
                for d in sub:
                    if d.get("tier3") in need and (round(d["y"], 3), d.get("hw_token")) not in seen:
                        marks.add(len(out))
                        out.append(d)
                return out, marks
            detA2, mkA = _admit(detA, subA, preds["A"])
            detB2, mkB = _admit(detB, subB, preds["B"])
            if mkA or mkB:
                detA, detB = detA2, detB2
                det_span = {"sides": {
                    "A": _to_matcher_side(detA, True, _mult(detA)),
                    "M": _to_matcher_side(detM, False),
                    "B": _to_matcher_side(detB, True, _mult(detB)),
                }}
                preds = match_span(det_span, self._cfg,
                                   extra={"A": {i: SUBGATE_PEN for i in mkA},
                                          "B": {i: SUBGATE_PEN for i in mkB}})
        predA, predB = preds["A"], preds["B"]

        poleA = self._build_pole_attachments(detA, predA, "A")
        poleB = self._build_pole_attachments(detB, predB, "B")
        traces = self._build_traces(predA, predB, poleA, poleB, detM)

        result = {
            "midspan_wire_count": nM,
            "midspan": [
                {"id": f"M{m}", "x": round(d["x"], 2), "y": round(d["y"], 2),
                 "conf": round(float(d.get("conf", 0.0)), 3),
                 "tier3": d.get("tier3")}   # v2.9 patch-classifier tier (non-authoritative hint)
                for m, d in enumerate(detM)
            ],
            "poles": {"A": poleA, "B": poleB},
            "traces": traces,
            "config": {
                "version": "v2.9.1",
                "pole_source": "unified",
                "midspan_source": "strip",
                "midspan_tier": self._tier is not None,   # v2.9 tier stage active for this run
                "matcher": "learned edge-cost + A<->B-coupled monotonic (w_x=0, comm_iso off) "
                           "+ mid-tier3 agreement bonus 0.6",
                "pole_dedup_y": self.pole_dedup_y,
                "pole_dedup_y_detect": self.pole_dedup_y_detect,
                "mult_cap": mult_cap,
            },
        }
        if return_annotated:
            from .visualize import draw_span_grid
            result["annotated_image"] = draw_span_grid(
                _load_image_bgr(pole_a_image),
                _load_image_bgr(_frame) if _frame is not None else None,
                _load_image_bgr(pole_b_image),
                result,
            )
        return result
