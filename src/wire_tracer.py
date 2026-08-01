#!/usr/bin/env python3
"""
wire_tracer — inference pipeline that reconstructs non-MI-job-like annotation data
from photos using only the trained detectors.

Per pole-mid-pole group (pole A photo, midspan section photo(s), pole B photo):
  pole_detection -> upper-70% 2:5 crop -> unified_pole_detection
      (joint-class pose: insulator/hardware NAME + cable class + crossarm-K)     [pole A & B]
  midspan_wire_strip_detection (HRNet ruler-column heatmap, 1-D wire peaks)      [midspan]
  A<->B-coupled Hungarian matcher (src.wire_tracing_match)                       [tie together]

It then emits the non-MI structure the Katapult jobs carry, e.g.
  three_bolt  -> 1 wire (comm)
  spool       -> 1 wire (secondary)
  crossarm    -> pin insulator x3 -> wire x3

CROSSARM HANDLING — the physical fact, from the user:
  A crossarm collapses to ONE pole keypoint, but its real wire count only separates
  into distinct keypoints at MIDSPAN. So a pole point's wire COUNT is recovered from
  the matcher: count = number of midspan wires traced through that pole point. To let a
  single arm keypoint absorb K midspan wires (the default matcher caps each pole point at
  one), pole points are given generous `multiplicity` here. A point that absorbs >1 midspan
  wire is reported as a crossarm of that many insulators/wires.

wire_type (primary / secondary / neutral / comm) is INTENTIONALLY NOT inferred — the user
assigns it. The hardware-derived coarse tier (WIRE_HW_TO_TIER) is surfaced only as a
non-authoritative `tier_hint` to help that assignment.

Source of truth for the underlying detector/matcher logic: src/wire_tracing_e2e.py and
src/wire_tracing_match.py — this module is a thin reconstruction/serialization layer on top.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import WIRE_HW_TO_TIER
from src.wire_tracing_match import MatchConfig, match_span, match_span_multi, compose_multi_chains
from src.wire_tracing_e2e import (
    Detectors,
    dedup_pole_points_by_height,
    detect_pole_points,
    detect_midspan_points_strip,
    load_detectors,
    resolve_span_photos,
    section_disk_photos,
    to_matcher_side,
)

# Friendly insulator/hardware names for the reconstructed annotation (the "insulator name"
# the user asked to see). Mirrors the Katapult insulator_spec vocabulary the non-MI jobs use.
INSULATOR_DISPLAY = {
    "spool": "Spool",
    "three_bolt": "Three-Bolt",
    "pin": "Pin Insulator",
    "post": "Post Insulator",
    "deadend": "Deadend",
    "davit": "Davit Arm",
    "guy": "Guy",
    "down_guy": "Down Guy",
    None: "Unread HW",
}

# Only POWER-tier hardware forms a multi-insulator CROSSARM (the Katapult arm markers carry
# pin/post/davit children, and deadend arms terminate power). spool (secondary) and three_bolt
# (comm) are single-wire attachments — a "spool x3" / "three_bolt x3" crossarm is a false
# positive from over-eager midspan multiplicity, so those tokens are capped at 1 wire.
CROSSARM_HW = ("pin", "post", "davit", "deadend")


def _detect_midspan_with_frame(photos: List[str], det: Detectors):
    """Detect midspan wires AND report which burst frame was used (needed for visualization).

    Mirrors the e2e detectors' 'keep the frame with the most detections' rule by probing one
    frame at a time. Returns (points, frame_path). Most sections here are single-frame."""
    fn = detect_midspan_points_strip
    best_pts: List[Dict] = []
    best_frame = photos[0] if photos else None
    for ph in photos:
        pts = fn([ph], det)
        if len(pts) > len(best_pts):
            best_pts, best_frame = pts, ph
    # MIDSPAN TIER stage (EXP-0001): classify each detected crossing bare/multiplex/comm
    # from a PPI-normalized photo patch; feeds the matcher's w_mid_tier3_bonus term.
    if best_pts and best_frame and getattr(det, "mid_tier", None) is not None:
        det.mid_tier.classify_points(best_frame, best_pts)
    return best_pts, best_frame


def _dedup_pole_points(points: List[Dict], y_tol: float = 1.5) -> List[Dict]:
    """Legacy percent-band dedup when inch dedup is off — delegates to shared e2e logic.

    Hybrid: conductors merge by height (class-blind); ``down_guy`` never deduped;
    conductor never merges with ``guy``."""
    return dedup_pole_points_by_height(points, y_tol)


def _normalize_guy_kind(points: List[Dict]) -> None:
    """In-place: force kind='guying' wherever the hardware token is a guy class.

    The pole-point `kind` comes from the WIRE detector (wire vs down_guy) while the token comes
    from the HARDWARE detector; they can disagree (wire detector says 'wire', hw says 'down_guy').
    The matcher gates matchability on `kind`, so a guy/down_guy must be marked guying or it would
    wrongly be matched as a span endpoint — guys/down-guys never cross a span."""
    for p in points:
        if p.get("hw_token") in ("guy", "down_guy"):
            p["kind"] = "guying"


def _build_pole_attachments(det_side: List[Dict], pred_for_side: List[Optional[int]],
                            side: str) -> List[Dict]:
    """Reconstruct one pole's attachments from detected points + the matcher's midspan->pole
    assignment for this side. wire_count = midspan wires traced through the point (>=1 for any
    real insulator; >1 => crossarm); guying carries no span wire."""
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
            name = "Guy" if token == "guy" else "Down Guy"   # wire-detector down-guy carries no hw token
        else:
            # the hardware itself implies >=1 wire even if midspan missed it; >1 => crossarm
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
            "tier_hint": WIRE_HW_TO_TIER.get(token),     # hardware-derived hint, NOT authoritative
            "role": role,
            "wire_count": wire_count,
            "traced_midspan_count": traced_count,
            "traced_midspan_ids": [f"M{m}" for m in traced],
            "wire_type": None,                            # user-assigned
        })
    return out


def _build_traces(predA: List[Optional[int]], predB: List[Optional[int]],
                  poleA: List[Dict], poleB: List[Dict], detM: List[Dict]) -> List[Dict]:
    """One trace per detected midspan wire: which pole-A insulator <-> which pole-B insulator
    it connects (the Katapult shared-_trace correspondence, here detected not GT)."""
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
            "wire_type": None,                            # user-assigned
        })
    return traces


def trace_span(span: Dict, det: Detectors, cfg: MatchConfig,
               mult_cap: Optional[int] = None, pole_dedup_y: float = 1.5) -> Dict:
    """Run the full wire_tracer on one pole-mid-pole group and return the reconstruction.

    mult_cap: max midspan wires a single pole point may absorb (None => unbounded = #midspan
    detected). Bounds spurious crossarm inflation if a single insulator sits near many wires.
    pole_dedup_y: height band (% of image) for collapsing duplicate pole detections (0 = off).

    Multi-section spans (sides.M_sections, pole-A → M1 → … → Mk → pole-B) dispatch to
    _trace_span_multi, which detects each section's photo(s) independently and threads the wire
    through all of them (match_span_multi). Single-section spans keep the legacy path below
    byte-for-byte.
    """
    secs = (span.get("sides") or {}).get("M_sections")
    if secs and len(secs) > 1:
        return _trace_span_multi(span, det, cfg, mult_cap, pole_dedup_y)

    photos = span.get("_photos") or resolve_span_photos(span)
    A_photo, B_photo, M_photos = photos.get("A"), photos.get("B"), photos.get("M") or []

    detA = detect_pole_points(A_photo, det) if A_photo else []
    detB = detect_pole_points(B_photo, det) if B_photo else []
    # SUB-GATE candidates (EXP-0007): below-gate conductor dets retained by the detector,
    # held OUT of pass-1; the tier-corroborated second pass below may admit them.
    subA = [d for d in detA if d.get("_subgate")]
    subB = [d for d in detB if d.get("_subgate")]
    detA = [d for d in detA if not d.get("_subgate")]
    detB = [d for d in detB if not d.get("_subgate")]
    # detect_pole_points already dedups via Detectors. When the DEPLOYED inch dedup is on
    # (pole_dedup_inch, projective ruler model), skip this coarser percent pass — a looser
    # 1.5% band would re-merge the ~1ft-spaced stacked-rack nodes the inch dedup preserved.
    if not getattr(det, "pole_dedup_inch", None) and pole_dedup_y and pole_dedup_y > 0:
        detA = _dedup_pole_points(detA, pole_dedup_y)
        detB = _dedup_pole_points(detB, pole_dedup_y)
    _normalize_guy_kind(detA)
    _normalize_guy_kind(detB)
    # COUNT-GUIDED ADAPTIVE midspan: nearly every span wire reaches both poles, so the
    # midspan strip should find at least min(#A, #B) conductor wires (pred_mult-weighted
    # crossarm K, guys excluded — they never cross a span). Fewer peaks = the strip MISSED
    # wires (unrecoverable: the chain can never be traced); the strip extractor then relaxes
    # its height gate for THIS span only (Detectors.strip_relax_ladder), and the matcher
    # dustbin absorbs any false extra. Only meaningful when both pole photos resolved.
    if getattr(det, "strip_adaptive", False) and detA and detB:
        def _cond_count(side):
            return sum(max(1, d.get("pred_mult") or 1)
                       for d in side if d.get("kind") != "guying")
        det.strip_min_peaks = min(_cond_count(detA), _cond_count(detB)) or None
    try:
        detM, M_used = _detect_midspan_with_frame(M_photos, det)
    finally:
        det.strip_min_peaks = None

    nM = len(detM)

    def _mult(side):
        # MODEL-PREDICTED multiplicity (2026-07-30, user-reported fix): the unified model
        # explicitly predicts crossarm-K via its arm2/arm3/arm4plus classes (pred_mult;
        # K-acc 0.816) — a class-'pin' point is ONE insulator and must not absorb every
        # midspan wire (the old unbounded cap turned a single pin into "Crossarm ×3").
        # This is also what the balanced harness (all validated e2e numbers) always used.
        # mult_cap now only LOWERS the model K; non-power hardware stays at 1.
        def k_of(d):
            if d.get("hw_token") not in CROSSARM_HW:
                return 1
            k = max(1, d.get("pred_mult", 1) or 1)
            return k if mult_cap is None else min(k, max(1, mult_cap))
        return {i: k_of(d) for i, d in enumerate(side)}

    multA, multB = _mult(detA), _mult(detB)

    det_span = {"sides": {
        "A": to_matcher_side(detA, True, multA),
        "M": to_matcher_side(detM, False),
        "B": to_matcher_side(detB, True, multB),
    }}
    preds = match_span(det_span, cfg)

    # TIER-CORROBORATED SUB-GATE ADMISSION (EXP-0007, +0.52pp balanced e2e): a pass-1
    # DUSTBINNED midspan wire with a tier3 opens admission of held-out sub-gate pole dets
    # whose class-tier AGREES; admitted dets carry an extra edge penalty (they must beat
    # the dustbin on corroboration, not on their own), then the span is re-matched.
    pen = getattr(det, "subgate_pen", 0.6)
    if (subA or subB) and any(p.get("tier3") for p in detM):
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
            multA, multB = _mult(detA), _mult(detB)
            nR = len(detM)
            extra = {"A": [{i: pen for i in mkA}] * nR, "B": [{i: pen for i in mkB}] * nR}
            preds = match_span({"sides": {
                "A": to_matcher_side(detA, True, multA),
                "M": to_matcher_side(detM, False),
                "B": to_matcher_side(detB, True, multB),
            }}, cfg, extra=extra)
    predA, predB = preds["A"], preds["B"]

    poleA = _build_pole_attachments(detA, predA, "A")
    poleB = _build_pole_attachments(detB, predB, "B")
    traces = _build_traces(predA, predB, poleA, poleB, detM)

    return {
        "job": span["job"],
        "pole_a_scid": span["pole_a"]["scid"],
        "pole_b_scid": span["pole_b"]["scid"],
        "photos": {
            "A": A_photo,
            "B": B_photo,
            "M": M_photos,
            "M_used": M_used,        # burst frame the midspan detections came from (for viz)
        },
        "config": {
            "pole_source": getattr(det, "pole_source", "unified"),
            "midspan_source": getattr(det, "midspan_source", "strip"),
            "pole_imgsz": getattr(det, "pole_crop_imgsz", 960),
            "matcher": cfg.label(),
            "mult_cap": mult_cap,
            "pole_dedup_y": pole_dedup_y,
        },
        "midspan_wire_count": nM,
        "midspan": [
            {"id": f"M{m}", "x": round(d["x"], 2), "y": round(d["y"], 2),
             "conf": round(float(d.get("conf", 0.0)), 3),
             "tier3": d.get("tier3")}   # patch-classifier tier (non-authoritative hint)
            for m, d in enumerate(detM)
        ],
        "poles": {"A": poleA, "B": poleB},
        "traces": traces,
    }


def _detect_poles_for_span(span: Dict, det: Detectors, pole_dedup_y: float):
    """Shared pole-side detection prelude (A & B): detect, kind-aware dedup, guy-normalize.

    Returns (detA, detB, subA, subB, A_photo, B_photo) — sub* are the held-out sub-gate
    admission candidates (EXP-0007), empty unless Detectors.subgate_floor is set."""
    photos = span.get("_photos") or resolve_span_photos(span)
    A_photo, B_photo = photos.get("A"), photos.get("B")
    detA = detect_pole_points(A_photo, det) if A_photo else []
    detB = detect_pole_points(B_photo, det) if B_photo else []
    subA = [d for d in detA if d.get("_subgate")]
    subB = [d for d in detB if d.get("_subgate")]
    detA = [d for d in detA if not d.get("_subgate")]
    detB = [d for d in detB if not d.get("_subgate")]
    if not getattr(det, "pole_dedup_inch", None) and pole_dedup_y and pole_dedup_y > 0:
        detA = _dedup_pole_points(detA, pole_dedup_y)
        detB = _dedup_pole_points(detB, pole_dedup_y)
    _normalize_guy_kind(detA)
    _normalize_guy_kind(detB)
    return detA, detB, subA, subB, A_photo, B_photo


def _build_multi_traces(chains: List[Dict], poleA: List[Dict], poleB: List[Dict],
                        det_sections: List[List[Dict]]) -> List[Dict]:
    """One trace per composed wire: pole-A insulator → its crossing in each section → pole-B
    insulator. ``chains`` come from compose_multi_chains (spine-anchored), so M_path[s] is the
    detected midspan-point index in section s (or None where the wire was missed there)."""
    out = []
    for ci, c in enumerate(chains):
        ai, bi = c["A"], c["B"]
        # the wire's best OBSERVED y (spine-first) — the display anchor for inferred
        # pass-throughs in sections where the detector found nothing (user ruling 3b,
        # 2026-07-30: show the threaded wire but mark it INFERRED, never as a detection)
        obs_y = next((det_sections[s][mi]["y"] for s, mi in enumerate(c["M_path"])
                      if mi is not None and mi < len(det_sections[s])), None)
        path = []
        for s, mi in enumerate(c["M_path"]):
            if mi is not None and mi < len(det_sections[s]):
                d = det_sections[s][mi]
                path.append({"section_index": s, "wire_id": f"S{s}_M{mi}",
                             "x": round(d["x"], 2), "y": round(d["y"], 2)})
            elif obs_y is not None:
                path.append({"section_index": s, "wire_id": None, "inferred": True,
                             "y": round(obs_y, 2)})   # approximate position, NOT a detection
        out.append({
            "chain_id": f"T{ci}",
            "pole_a_attachment": f"A{ai}" if ai is not None else None,
            "pole_a_insulator": poleA[ai]["insulator_name"] if ai is not None and ai < len(poleA) else None,
            "pole_b_attachment": f"B{bi}" if bi is not None else None,
            "pole_b_insulator": poleB[bi]["insulator_name"] if bi is not None and bi < len(poleB) else None,
            "n_sections_observed": sum(1 for p in path if not p.get("inferred")),
            "midspan_path": path,
            "wire_type": None,                                # user-assigned
        })
    return out


def _trace_span_multi(span: Dict, det: Detectors, cfg: MatchConfig,
                      mult_cap: Optional[int], pole_dedup_y: float) -> Dict:
    """Reconstruct a multi-section span: detect each midspan section independently and thread
    every wire through the full pole-A → M1 → … → Mk → pole-B path (match_span_multi).

    Each section keeps its OWN photo, frame and detected crossings — the per-section output the
    production annotation format carries. Falls back to the single-section path when the section↔
    photo mapping can't be resolved (photo_id index unavailable)."""
    secs_meta = span["sides"]["M_sections"]
    grouped, _leftover = section_disk_photos(span)
    if grouped is None:                                       # no photo_id index → degrade
        return trace_span({**span, "sides": {**span["sides"], "M_sections": None}},
                          det, cfg, mult_cap, pole_dedup_y)

    detA, detB, subA, subB, A_photo, B_photo = _detect_poles_for_span(span, det, pole_dedup_y)

    # COUNT-GUIDED ADAPTIVE midspan (same rule as single-section): aim for at least the detected
    # pole conductor count per section. Set once from the pole counts, applied to every section.
    if getattr(det, "strip_adaptive", False) and detA and detB:
        def _cond_count(side):
            return sum(max(1, d.get("pred_mult") or 1)
                       for d in side if d.get("kind") != "guying")
        det.strip_min_peaks = min(_cond_count(detA), _cond_count(detB)) or None

    det_sections: List[List[Dict]] = []
    frames_used: List[Optional[str]] = []
    try:
        for s_ph in grouped:
            pts, frame = _detect_midspan_with_frame(s_ph, det) if s_ph else ([], None)
            det_sections.append(pts)
            frames_used.append(frame)
    finally:
        det.strip_min_peaks = None

    matcher_sections = []
    for i, pts in enumerate(det_sections):
        matcher_sections.append({"section_id": secs_meta[i].get("section_id"),
                                 "lat": secs_meta[i].get("lat"), "lon": secs_meta[i].get("lon"),
                                 "dist_a_m": secs_meta[i].get("dist_a_m"),
                                 "points": to_matcher_side(pts, False), "i": i})
    spine_idx = max(range(len(matcher_sections)),
                    key=lambda i: (len(matcher_sections[i]["points"]), -i)) if matcher_sections else 0
    spine_points = matcher_sections[spine_idx]["points"] if matcher_sections else []

    nM = len(spine_points)                                    # distinct wire count = spine wires

    def _mult(side):
        # MODEL-PREDICTED multiplicity (same fix as trace_span): report what the pole
        # detector detected — pred_mult from the arm classes, never midspan-inflated.
        def k_of(d):
            if d.get("hw_token") not in CROSSARM_HW:
                return 1
            k = max(1, d.get("pred_mult", 1) or 1)
            return k if mult_cap is None else min(k, max(1, mult_cap))
        return {i: k_of(d) for i, d in enumerate(side)}

    det_span = {"sides": {
        "A": to_matcher_side(detA, True, _mult(detA)),
        "B": to_matcher_side(detB, True, _mult(detB)),
        "M": spine_points,
        "M_sections": matcher_sections,
    }}
    preds = match_span_multi(det_span, cfg)

    # TIER-CORROBORATED SUB-GATE ADMISSION (EXP-0007, multi-section parity 2026-07-30):
    # same second pass as trace_span, keyed on DUSTBINNED spine wires with a tier3.
    pen = getattr(det, "subgate_pen", 0.6)
    if (subA or subB) and any(p.get("tier3") for p in spine_points):
        def _admit(base, sub, pred):
            need = {spine_points[m].get("tier3") for m, pi in enumerate(pred)
                    if pi is None and spine_points[m].get("tier3")}
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
            nR = len(spine_points)
            det_span["sides"]["A"] = to_matcher_side(detA, True, _mult(detA))
            det_span["sides"]["B"] = to_matcher_side(detB, True, _mult(detB))
            preds = match_span_multi(det_span, cfg,
                                     extra={"A": [{i: pen for i in mkA}] * nR,
                                            "B": [{i: pen for i in mkB}] * nR})
    chains = compose_multi_chains(preds)

    poleA = _build_pole_attachments(detA, preds["A"], "A")
    poleB = _build_pole_attachments(detB, preds["B"], "B")
    traces = _build_multi_traces(chains, poleA, poleB, det_sections)

    return {
        "job": span["job"],
        "pole_a_scid": span["pole_a"]["scid"],
        "pole_b_scid": span["pole_b"]["scid"],
        "n_sections": len(secs_meta),
        "spine_section": preds["spine"],
        "photos": {
            "A": A_photo, "B": B_photo,
            "sections": [{"section_index": i, "section_id": secs_meta[i].get("section_id"),
                          "lat": secs_meta[i].get("lat"), "lon": secs_meta[i].get("lon"),
                          "photos": grouped[i], "frame_used": frames_used[i]}
                         for i in range(len(secs_meta))],
        },
        "config": {
            "pole_source": getattr(det, "pole_source", "unified"),
            "midspan_source": getattr(det, "midspan_source", "strip"),
            "pole_imgsz": getattr(det, "pole_crop_imgsz", 960),
            "matcher": cfg.label(), "mult_cap": mult_cap, "pole_dedup_y": pole_dedup_y,
        },
        "midspan_sections": [
            {"section_index": i, "wire_count": len(det_sections[i]),
             "wires": [{"id": f"S{i}_M{j}", "x": round(d["x"], 2), "y": round(d["y"], 2),
                        "conf": round(float(d.get("conf", 0.0)), 3)}
                       for j, d in enumerate(det_sections[i])]}
            for i in range(len(det_sections))],
        "poles": {"A": poleA, "B": poleB},
        "traces": traces,
    }


def _fmt_attachments(atts: List[Dict]) -> List[str]:
    lines = []
    if not atts:
        return ["      (no attachments detected)"]
    for a in atts:
        if a["role"] == "guying":
            desc = "guying (pole-only, no span wire)"
        elif a["role"] == "crossarm":
            desc = f"CROSSARM  {a['insulator_name']} x{a['wire_count']}  ->  wire x{a['wire_count']}"
        else:
            desc = f"{a['insulator_name']}  ->  1 wire"
        tier = f"  [hw=>{a['tier_hint']}]" if a["tier_hint"] else ""
        lines.append(
            f"      - {a['insulator_name']:<13} y={a['y']:>5.1f}%  conf {a['conf']:.2f}{tier}   => {desc}"
        )
    return lines


def format_trace_report(r: Dict, idx: Optional[int] = None) -> str:
    """Human-readable per-group reconstruction. Mentions every insulator name."""
    L = []
    head = f"GROUP {idx}  " if idx is not None else ""
    L.append(f"{head}{r['job']}    span: pole (scid {r['pole_a_scid']})  ->  pole (scid {r['pole_b_scid']})")
    pa = Path(r["photos"]["A"]).name if r["photos"]["A"] else "-"
    pb = Path(r["photos"]["B"]).name if r["photos"]["B"] else "-"
    L.append(f"   photos:  A={pa}")
    L.append(f"            B={pb}")
    L.append(f"            M={len(r['photos']['M'])} burst frame(s)")
    L.append(f"   --- Pole A (scid {r['pole_a_scid']}) : {len(r['poles']['A'])} attachment(s) ---")
    L.extend(_fmt_attachments(r["poles"]["A"]))
    L.append(f"   --- Midspan : {r['midspan_wire_count']} wire crossing(s) detected ---")
    L.append(f"   --- Pole B (scid {r['pole_b_scid']}) : {len(r['poles']['B'])} attachment(s) ---")
    L.extend(_fmt_attachments(r["poles"]["B"]))
    L.append("   --- Traces  (midspan wire  ->  pole-A insulator  <->  pole-B insulator) ---")
    if not r["traces"]:
        L.append("      (no midspan wires detected)")
    for t in r["traces"]:
        a = t["pole_a_insulator"] or "(unmatched)"
        b = t["pole_b_insulator"] or "(unmatched)"
        L.append(f"      {t['midspan_id']} (y={t['midspan_y']:>5.1f}%):  {a} [A]  <->  {b} [B]")
    L.append("   wire_type left blank on every wire — to be assigned by the user.")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Shared setup (one source of truth for the runner AND the visualizer)
# --------------------------------------------------------------------------- #

# Defaults resolve through the model registry (models/registry.json — the single source of
# truth): models/production/<name>/production is a symlink to the current production version.
# Never point these at runs/ — promote a run first (scripts/deploy_ops/promote_model.py).
# Current production unified pole = v1.3.0 (yolo11m on honest+mined site-disjoint; balanced
# e2e 0.5496 / annotation micro-F1 0.717; provenance in its metadata.json).
DEFAULT_UNIFIED_WEIGHTS = "models/production/unified_pole_detection/production/model.pt"
DEFAULT_UNIFIED_CONF_JSON = "models/production/unified_pole_detection/production/perclass_conf.json"
DEFAULT_EDGE_MODEL = "models/edge_matcher_unified_v2.json"
# Midspan strip: RULER-LINE geometry checkpoint (PROMOTED 2026-07-04, user decision). Strip
# axis = the straight line through the CALIBRATION ruler tick anchors (the label-faithful
# reading axis — wires are annotated ON the tick line, median dev 0.085%), 3ft rectified width,
# ground-line bottom, 1740x96 input. Trained on clean + MN mined + multi-section + 2,883 NEOM
# mined strips. Balanced e2e (ft2 pole, production combo): ALL 0.4796 vs mined-column 0.4426
# (+3.7pp; NEOM 0.5680 vs 0.5333). Requires strip_mode='ruler-line' + strip_resize_hw=(1740,96)
# + tick anchors available (label store / job JSON); photos with no ticks fall back to the
# legacy column crop, whose width_expand stays 3.0. best_f1.pth = deployed-op-point F1
# selection (ep~17; later epochs trade recall for precision = e2e-negative).
# Production strip = v1.2.0 (ruler-line 1740x96, best_f1 checkpoint).
DEFAULT_STRIP_WEIGHTS = "models/production/midspan_wire_strip_detection/production/model.pth"
DEFAULT_STRIP_WIDTH_EXPAND = 3.0
DEFAULT_STRIP_MODE = "ruler-line"
DEFAULT_STRIP_RESIZE_HW = (1740, 96)


def build_default_tracer(device: str = "cuda",
                         pole_imgsz: int = 1024,
                         unified_weights: str = DEFAULT_UNIFIED_WEIGHTS,
                         unified_conf_json: Optional[str] = None,
                         unified_imgsz: int = 960,
                         strip_weights: str = DEFAULT_STRIP_WEIGHTS,
                         strip_width_expand: float = DEFAULT_STRIP_WIDTH_EXPAND,
                         strip_height: float = 0.40, strip_prom: float = 0.02,
                         strip_adaptive: bool = True,
                         strip_mode: str = DEFAULT_STRIP_MODE,
                         strip_resize_hw: Optional[Tuple[int, int]] = DEFAULT_STRIP_RESIZE_HW,
                         unified_arm_floor: Optional[float] = 0.10,
                         w_couple_class: float = 0.20,
                         edge_model: Optional[str] = "auto", edge_dust: float = 1.0,
                         down_guy_dedup_inch: Optional[float] = 4.0,
                         down_guy_conf: Optional[float] = 0.20,
                         mid_tier_weights: Optional[str] = "auto",
                         mid_tier_gates: Optional[Tuple[float, float, float]] = None,
                         w_mid_tier3_bonus: float = 0.6,
                         subgate_floor: Optional[float] = 0.10,
                         subgate_pen: float = 0.6):
    """Detectors + matcher config for the PRODUCTION wire_tracer operating point.

    DEFAULT = the validated winning config (e2e chain acc 0.3804 -> 0.6027, 2119 spans /
    9751 chains; <=0.5855 live-eval-verified, the 2026-06-11 lever stack verified on the
    cached-detection harness that reproduces the live eval to +-0.05pp):
      * pole_source='unified' — ONE joint-class model (hardware x cable_type
        x crossarm-K) as the node source; gives the per-pole hardware naming AND the wire_class
        for class coupling. FLAT-0.20 conf op-point (unified_conf_json=None; +2.4pp over the
        per-class F1 gate) with a LOWER 0.10 floor for arm2/3/4plus only (unified_arm_floor —
        repairs the MI-data crossarm conf sag, +0.6pp).
      * strip midspan = w3sharp checkpoint (3x ruler width + sharp sigma) at strip_width_expand=3.0
        and the e2e-optimal peak op-point (height 0.40 / prom 0.02) — +1.99pp localization win.
        Shares the ruler x so matching is height-only (w_x=0). midspan recall ~0.94.
        COUNT-GUIDED ADAPTIVE extraction (strip_adaptive, +0.9pp, crossarm +6.9pp): relax the
        height gate per span when peaks < min(#A,#B) detected pole conductors.
      * LEARNED edge-cost matcher (models/edge_matcher_unified_v2.json; retrain on the new
        distribution probed and REFUTED, v2 transfers) + fine cable A<->B coupling
        (w_couple_class=0.2, re-priced for the better MI-trained cable classes) + raised
        dustbin (edge_dust=1.0).

    edge_model='auto' loads DEFAULT_EDGE_MODEL (the model is distribution-specific — trained on
    the unified node source); pass an explicit path to override, or None to disable.
    comm_isolation is turned OFF with the learned cost (the model already weights tier softly;
    hard isolation hurt)."""
    weights = {"unified": unified_weights}
    if strip_weights:
        weights["strip"] = strip_weights
    det = load_detectors(device=device, weights=weights, midspan_source="strip",
                         strip_resize_hw=strip_resize_hw)
    det.pole_source = "unified"
    det.pole_crop_imgsz = pole_imgsz
    det.strip_peak_height = strip_height
    det.strip_peak_prom = strip_prom
    det.strip_width_expand = strip_width_expand   # column-crop fallback stays 3x ruler width
    # RULER-LINE strip geometry (PROMOTED 2026-07-04): crop along the calibration tick
    # line, ground->top, 3ft wide; ticks from label store / job JSON, NO ruler inference.
    det.strip_mode = strip_mode
    # COUNT-GUIDED ADAPTIVE peak extraction (+0.9pp e2e, crossarm +6.9pp): trace_span
    # relaxes the height gate per span when the strip finds fewer wires than the pole
    # conductor counts say must cross. Detector-robust (validated mi_clean + armboost).
    det.strip_adaptive = strip_adaptive
    det.unified_imgsz = unified_imgsz
    if unified_conf_json:
        det.unified_conf_per_class = json.loads(Path(unified_conf_json).read_text())
        det.unified_conf = 0.01        # low floor; per-class map gates
    elif unified_arm_floor is not None and unified_arm_floor < 0.20:
        # FLAT-0.20 op-point + LOWER floor for crossarm classes only (+~0.2-1.1pp e2e,
        # biggest on MI-diluted retrains whose arm conf calibration sags): run the
        # detector at the arm floor, per-class map keeps everything else at 0.20.
        from src.config import UNIFIED_POLE_DETECTION_CLASS_NAMES as _ucls
        arms = ("arm2", "arm3", "arm4plus")
        pcm = {c: (unified_arm_floor if c in arms else 0.20) for c in _ucls}
        # guy/down_guy are AUTO-DUSTBINNED in tracing (zero e2e effect), so they are
        # annotation-only: use their F1-optimal conf (from the model's sibling perclass_conf.json)
        # instead of the e2e flat-0.20. Tracing classes keep the e2e op-point.
        _pc_path = Path(unified_weights).parent.parent / "perclass_conf.json"
        if _pc_path.exists():
            _f1 = json.loads(_pc_path.read_text())
            for _g in ("guy", "down_guy"):
                if _g in _f1:
                    pcm[_g] = float(_f1[_g])
        if down_guy_dedup_inch:
            # down_guy dedup+anchor pipeline: detect at the 0.05 floor; the dedup step
            # (dedup_pole_points_for_photo) gates at down_guy_conf and relaxes sub-gate
            # candidates back in only up to the anchor-inventory K.
            pcm["down_guy"] = 0.05
        det.unified_conf_per_class = pcm
        det.unified_conf = min(unified_arm_floor, *(pcm[g] for g in ("guy", "down_guy") if g in pcm))
    # else: FLAT-0.20 op-point (Detectors.unified_conf=0.20, per_class stays None) —
    # e2e-validated +2.4pp vs the per-class F1 gate (the gate was tuned for per-pole
    # fidelity, not e2e).
    if down_guy_dedup_inch:
        # OPT-OUT down_guy dedup + anchor-count guidance (annotation-quality lever,
        # e2e-neutral: down_guy is auto-dustbinned in tracing). Val-tuned on armboost:
        # test kp-F1@6" 0.660 -> 0.717. Anchor K guards genuine same-height twins AND
        # bounds the sub-gate relax.
        det.down_guy_dedup_inch = down_guy_dedup_inch
        if down_guy_conf is not None:
            det.down_guy_conf_gate = down_guy_conf
        try:
            from src.pole_anchor_down_guy import build_photo_expectations
            det.down_guy_expected, _ = build_photo_expectations()
        except Exception:
            det.down_guy_expected = None   # no job JSONs on this machine — plain dedup
    # MIDSPAN TIER lever (EXP-0001, PROMOTED 2026-07-30 — default ON): patch-classifier
    # tier3 on each midspan crossing + matcher bonus on tier-agreeing edges. Winning
    # config = 4-class 'none'-veto resnet18 (production midspan_tier_classifier v1.0.0)
    # + protect-bare gates (0,.7,.7) + bonus 0.6 (+1.2pp balanced e2e, 0.5496 -> 0.5615,
    # a floor given incomplete GT). mid_tier_weights='auto' = production path when it
    # exists (None to disable, or an explicit path).
    det.mid_tier = None
    if mid_tier_weights == "auto":
        from src.midspan_tier import DEFAULT_TIER_WEIGHTS as _dtw
        mid_tier_weights = _dtw if Path(_dtw).exists() else None
    if mid_tier_weights:
        from src.midspan_tier import DEFAULT_TIER_GATES, MidspanTierClassifier
        det.mid_tier = MidspanTierClassifier(
            mid_tier_weights, device=getattr(det, "device", device),
            gates=mid_tier_gates or DEFAULT_TIER_GATES)
    cfg = MatchConfig(w_couple_tier=0.2, w_couple_chain=0.25, w_deadend=0.06, w_couple_class=w_couple_class,
                      class_signal="none", w_x=0.0,   # strip midspan shares the ruler x
                      monotonic=True, comm_isolation=True)
    if det.mid_tier is not None:
        # Bonus (subtract from tier3-agreeing edges) — the rescue-from-dustbin mechanism.
        cfg.w_mid_tier3_bonus = w_mid_tier3_bonus
        # TIER-CORROBORATED SUB-GATE ADMISSION (EXP-0007, +0.52pp): retain below-gate
        # conductor dets down to subgate_floor; trace_span's second pass admits the ones a
        # dustbinned tier-agreeing midspan wire corroborates (edge penalty subgate_pen).
        # Requires the tier stage — without it the trigger never fires, so it is only
        # enabled alongside mid_tier.
        if subgate_floor is not None:
            det.subgate_floor = subgate_floor
            det.subgate_pen = subgate_pen
    # LEARNED edge cost (trained on the unified node source).
    if edge_model == "auto":
        edge_model = DEFAULT_EDGE_MODEL if Path(DEFAULT_EDGE_MODEL).exists() else None
    if edge_model:
        from src.wire_tracing_match import NumpyEdgeCostModel
        cfg.edge_model = NumpyEdgeCostModel.load(edge_model)
        cfg.comm_isolation = False
        if edge_dust is not None:
            cfg.dust = edge_dust
    return det, cfg


def pick_groups(spans: List[Dict], n: int):
    """Resolvable spans, round-robin across distinct jobs for variety. Shared selection so the
    runner and the visualizer render the SAME groups. Returns (picked, n_resolvable)."""
    from collections import OrderedDict
    for s in spans:
        if "_photos" not in s:
            s["_photos"] = resolve_span_photos(s)
    resolvable = [s for s in spans if s["_photos"]["resolvable"]]
    by_job: "OrderedDict[str, list]" = OrderedDict()
    for s in resolvable:
        by_job.setdefault(s["job"], []).append(s)
    picked: List[Dict] = []
    while len(picked) < n and any(by_job.values()):
        for job in list(by_job):
            if by_job[job]:
                picked.append(by_job[job].pop(0))
                if len(picked) >= n:
                    break
    return picked, len(resolvable)
