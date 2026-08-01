"""Build the demo span-trace payload: wire tracer + per-photo calibration heights.

Produces the pole-mid[-mid]-pole view model the demo page renders: every photo in
the span gets its own annotation set (pole attachments with insulator/cable hints,
midspan wire crossings), plus a height label per point derived from the ruable
calibration ticks via a 1-D projective fit (the same 5 tick anchors Katapult
annotators calibrate against).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Fit sanity gate: reject a tick fit whose round-trip error exceeds this (feet).
_MAX_FIT_ERR_FT = 1.0


def _fit_projective_height(ticks: Sequence[Tuple[float, float]]):
    """Fit height_ft = ((a + b*x) / (1 + c*x)) / 12 with x = y_pct/100.

    The camera projection of the ruler makes percentY -> height a rational
    function, not a line. This is the same formulation as Katapult's own
    calibration (validated in MR-QC's ruler_height_model to ~0.5 inch against
    Katapult's _measured_height): linearize inches = a + b*x - c*(x*inches)
    and least-squares solve. ticks are (height_ft, y_pct) anchor pairs.

    Returns y_pct -> height_ft callable, or None when the fit is unusable.
    """
    if len({round(y, 4) for _, y in ticks}) < 3:
        return None
    x = np.array([y for _, y in ticks], dtype=np.float64) / 100.0
    h_in = np.array([h * 12.0 for h, _ in ticks], dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x, -x * h_in])
    try:
        (a, b, c), *_ = np.linalg.lstsq(design, h_in, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not all(np.isfinite(v) for v in (a, b, c)):
        return None

    def y_to_h(y_pct: float) -> Optional[float]:
        xv = float(y_pct) / 100.0
        den = 1.0 + c * xv
        if den <= 1e-9:
            return None
        val = (a + b * xv) / den
        return val / 12.0 if val > 0 else None

    for hv, yv in ticks:
        fit = y_to_h(yv)
        if fit is None or abs(fit - hv) > _MAX_FIT_ERR_FT:
            return None
    return y_to_h


def format_feet_inches(height_ft: Optional[float]) -> Optional[str]:
    if height_ft is None or not np.isfinite(height_ft) or height_ft <= 0 or height_ft > 200:
        return None
    ft = int(height_ft)
    inch = int(round((height_ft - ft) * 12))
    if inch >= 12:
        ft, inch = ft + 1, 0
    return f"{ft}'-{inch}\""


def _calibrate_photo(calibrator, img_rgb: np.ndarray, is_pole: bool) -> Dict[str, Any]:
    """Run calibration on one photo; return ruler ticks (percent), pole top, height fn."""
    h_px, w_px = img_rgb.shape[:2]
    out: Dict[str, Any] = {"ruler_ticks": [], "pole_top": None, "_y_to_h": None}
    try:
        result = calibrator.run(img_rgb, detect_pole=is_pole)
    except Exception:
        logger.exception("Calibration failed for a span photo; heights unavailable.")
        return out

    anchors: List[Tuple[float, float]] = []
    for kp in result.get("ruler_keypoints") or []:
        try:
            height_ft = float(kp.get("name"))
        except (TypeError, ValueError):
            continue
        x_pct = 100.0 * float(kp["x"]) / w_px
        y_pct = 100.0 * float(kp["y"]) / h_px
        anchors.append((height_ft, y_pct))
        out["ruler_ticks"].append({
            "height_ft": height_ft,
            "height_label": format_feet_inches(height_ft),
            "x": round(x_pct, 2),
            "y": round(y_pct, 2),
            "conf": round(float(kp.get("conf", 0.0)), 3),
        })

    y_to_h = _fit_projective_height(anchors)
    out["_y_to_h"] = y_to_h

    pt = result.get("pole_top")
    if is_pole and pt:
        y_pct = 100.0 * float(pt["y"]) / h_px
        entry = {
            "x": round(100.0 * float(pt["x"]) / w_px, 2),
            "y": round(y_pct, 2),
            "conf": round(float(pt.get("conf", 0.0)), 3),
            "height_ft": None,
            "height_label": None,
        }
        if y_to_h is not None:
            hf = y_to_h(y_pct)
            entry["height_ft"] = round(hf, 2) if hf is not None else None
            entry["height_label"] = format_feet_inches(hf)
        out["pole_top"] = entry
    return out


def _stamp_height(entry: Dict[str, Any], y_to_h) -> None:
    hf = y_to_h(float(entry["y"])) if y_to_h is not None else None
    entry["height_ft"] = round(hf, 2) if hf is not None else None
    entry["height_label"] = format_feet_inches(hf)


# Max |frame height - traced height| (ft) for a sibling-frame detection to bind
# to a trace (MR-QC section_match GATE_FT): absorbs ruler-model + detection noise
# while staying under adjacent-conductor spacing.
_LINK_GATE_FT = 3.0


def _link_traces_by_height(frame_wires: List[Dict], trace_heights: List[Optional[float]]) -> None:
    """Bind a sibling midspan frame's wires to traces by ABSOLUTE height.

    Both photos see the same physical wires at (nearly) the same calibrated
    height, so a non-crossing minimum-|dh| alignment with a 3 ft gate lets extra
    detections and missed wires drop out without shifting the rest (the failure
    mode of rank-zip linking). Wires and traces are matched in top-to-bottom
    order; unmatched wires keep trace_index None (dustbin).
    """
    order = sorted(range(len(frame_wires)), key=lambda i: frame_wires[i]["y"])
    heights = [frame_wires[i].get("height_ft") for i in order]
    tr = sorted(range(len(trace_heights)), key=lambda i: (trace_heights[i] is None,
                                                          -(trace_heights[i] or 0.0)))
    n, m = len(order), len(tr)
    if not n or not m:
        return
    NEG = (-(10 ** 9), 0.0)

    # dp[i][j] = best (matches, -cost) aligning wires[:i] with traces[:j]
    dp = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = max(dp[i - 1][j], dp[i][j - 1])
            hw, ht = heights[i - 1], trace_heights[tr[j - 1]]
            if hw is not None and ht is not None and abs(hw - ht) <= _LINK_GATE_FT:
                cand = (dp[i - 1][j - 1][0] + 1, dp[i - 1][j - 1][1] - abs(hw - ht))
                best = max(best, cand)
            dp[i][j] = best if best != NEG else (0, 0.0)

    i, j = n, m
    while i > 0 and j > 0:
        hw, ht = heights[i - 1], trace_heights[tr[j - 1]]
        matched = (hw is not None and ht is not None and abs(hw - ht) <= _LINK_GATE_FT
                   and dp[i][j] == (dp[i - 1][j - 1][0] + 1, dp[i - 1][j - 1][1] - abs(hw - ht)))
        if matched:
            frame_wires[order[i - 1]]["trace_index"] = tr[j - 1]
            i, j = i - 1, j - 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1


def midspan_wire_display(tracer, calibrator, img_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """Single-photo midspan annotation: strip wire crossings + calibration heights."""
    try:
        det, _ = tracer._detect_midspan_points([img_rgb])
    except Exception:
        logger.exception("Midspan wire detection failed")
        det = []
    calib = _calibrate_photo(calibrator, img_rgb, is_pole=False)
    wires = []
    for wi, w in enumerate(sorted(det, key=lambda d: d["y"])):
        entry = {
            "name": f"Wire {wi + 1}",
            "x": round(float(w["x"]), 2),
            "y": round(float(w["y"]), 2),
            "conf": round(float(w.get("conf", 0.0)), 3),
        }
        _stamp_height(entry, calib["_y_to_h"])
        wires.append(entry)
    return wires


def build_span_payload(
    tracer,
    calibrator,
    pole_a_rgb: np.ndarray,
    mid_rgbs: List[np.ndarray],
    pole_b_rgb: np.ndarray,
) -> Dict[str, Any]:
    """Run the wire tracer over one span and assemble the per-photo view model."""
    result = tracer.run(pole_a_rgb, list(mid_rgbs), pole_b_rgb)
    traces = result.get("traces") or []
    trace_by_mid = {t["midspan_id"]: idx for idx, t in enumerate(traces)}

    att_index = {}
    for side in ("A", "B"):
        for att in result["poles"][side]:
            att_index[att["id"]] = att
    for idx, tr in enumerate(traces):
        for key in ("pole_a_attachment", "pole_b_attachment"):
            att = att_index.get(tr.get(key))
            if att is not None:
                att.setdefault("trace_indices", []).append(idx)

    photos: List[Dict[str, Any]] = []

    for role, img in (("pole_a", pole_a_rgb), ("pole_b", pole_b_rgb)):
        calib = _calibrate_photo(calibrator, img, is_pole=True)
        side = "A" if role == "pole_a" else "B"
        attachments = result["poles"][side]
        for att in attachments:
            _stamp_height(att, calib["_y_to_h"])
            att.setdefault("trace_indices", [])
        photos.append({
            "role": role,
            "side": side,
            "attachments": attachments,
            "ruler_ticks": calib["ruler_ticks"],
            "pole_top": calib["pole_top"],
        })

    # The tracer picks ONE frame for matching; annotate every frame with its own
    # strip detections and link wires to traces (identity on the used frame,
    # y-rank order on siblings with matching counts).
    used_wires = result.get("midspan") or []
    used_ys = [w["y"] for w in used_wires]
    used_frame_idx: Optional[int] = None
    # Mirror run()'s count-guided adaptive extraction so per-frame detections
    # reproduce the traced frame's wire set (min expected conductors per side).
    def _cond_count(side: str) -> int:
        return sum(max(1, int(att.get("crossarm_k") or 1))
                   for att in result["poles"][side] if att.get("role") != "guying")

    min_peaks = min(_cond_count("A"), _cond_count("B")) or None
    frames: List[Dict[str, Any]] = []
    for fi, mid in enumerate(mid_rgbs):
        try:
            det, _ = tracer._detect_midspan_points([mid], min_peaks=min_peaks)
        except Exception:
            logger.exception("Midspan detection failed on frame %d", fi)
            det = []
        wires = [{
            "id": f"F{fi}W{wi}",
            "x": round(float(w["x"]), 2),
            "y": round(float(w["y"]), 2),
            "conf": round(float(w.get("conf", 0.0)), 3),
            "trace_index": None,
        } for wi, w in enumerate(det)]
        if used_frame_idx is None and len(wires) == len(used_ys) and all(
            abs(w["y"] - uy) < 0.75 for w, uy in zip(sorted(wires, key=lambda w: w["y"]),
                                                     sorted(used_ys))
        ):
            used_frame_idx = fi
            for w, mid_entry in zip(sorted(wires, key=lambda w: w["y"]),
                                    sorted(used_wires, key=lambda m: m["y"])):
                w["trace_index"] = trace_by_mid.get(mid_entry["id"])
        frames.append({"role": "midspan", "index": fi, "wires": wires})

    for fi, (frame, img) in enumerate(zip(frames, mid_rgbs)):
        calib = _calibrate_photo(calibrator, img, is_pole=False)
        for w in frame["wires"]:
            _stamp_height(w, calib["_y_to_h"])
        frame["ruler_ticks"] = calib["ruler_ticks"]
        frame["used_for_trace"] = (fi == used_frame_idx)

    # Sibling frames (not the traced one) bind to traces by absolute calibrated
    # height — the matcher-faithful linking; unmatched detections stay dustbinned
    # (trace_index None) and the UI hides them.
    trace_heights: List[Optional[float]] = [None] * len(traces)
    if used_frame_idx is not None:
        for w in frames[used_frame_idx]["wires"]:
            ti = w.get("trace_index")
            if ti is not None:
                trace_heights[ti] = w.get("height_ft")
    for fi, frame in enumerate(frames):
        if fi != used_frame_idx and any(h is not None for h in trace_heights):
            _link_traces_by_height(frame["wires"], trace_heights)

    # Interleave: pole_a, mid..., pole_b
    ordered = [photos[0]] + frames + [photos[1]]

    # The tracer never infers a midspan wire's cable type, so each trace
    # inherits it from the pole attachment it lands on: pole A wins, pole B
    # fills the gap, and an explicit type beats "unspecified" on either side.
    for idx, tr in enumerate(traces):
        tr["trace_index"] = idx
        cands = []
        for key in ("pole_a_attachment", "pole_b_attachment"):
            att = att_index.get(tr.get(key)) or {}
            # prefer the fine class (catv/telco/fiber) over the coarse hint ('comm')
            cands.append(att.get("cable_type_fine") or att.get("cable_type_hint"))
        guying = any(
            (att_index.get(tr.get(key)) or {}).get("role") == "guying"
            for key in ("pole_a_attachment", "pole_b_attachment")
        )
        named = [c for c in cands if c and c != "unspecified"]
        tr["cable_type"] = (
            "guy" if guying else (named[0] if named else (cands[0] or cands[1] or None))
        )

    return {
        "photos": ordered,
        "traces": traces,
        "midspan_wire_count": result.get("midspan_wire_count"),
        "config": result.get("config"),
    }
