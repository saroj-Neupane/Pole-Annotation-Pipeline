#!/usr/bin/env python3
"""
Stage-0 wire-tracing extractor.

Reads raw Katapult job JSONs from the authoritative deduped set (``data/jobs/*.json``,
``config.WIRE_TRACING_JOB_SOURCE_DIR``; legacy fallback ``data/data_midspan/*.json``) — each
self-contained, holding pole attachment markers AND midspan wire markers linked by a
shared ``_trace`` id — and emits one record per in-scope span (pole A ↔ midspan ↔
pole B) with ground-truth correspondence. This is the supervised dataset the per-span
graph matcher trains/evaluates against.

Data model (important):
  * A pole-side detection is a POINT carrying a SET of trace ids. Top-level insulators
    are usually 1 trace : 1 point. A crossarm is ONE point (the ``arm`` marker pixel) with
    2-6 coincident traces — its child insulators have empty ``pixel_selection``. Within-
    crossarm phase identity is therefore NOT geometrically recoverable; such midspan
    traces are flagged ``group_ambiguous`` so evaluation can score them at group level.
  * A midspan detection is a POINT carrying exactly ONE trace id (``wire`` markers are
    per-trace and spatially distinct).

GT match rule: a midspan detection (trace t) matches the pole-side POINT whose trace-set
contains t. This naturally expresses crossarm many-to-one structure.

Scope / filtering (see config):
  * MI-regime jobs (``insulator_count <= WIRE_TRACING_MI_MAX_INSULATORS``) excluded — a
    lossy, contradictory annotation regime. Detected by content; filename is secondary.
  * In-scope spans: connection_type in WIRE_TRACING_IN_SCOPE_CONNECTION_TYPES AND both
    endpoints node_type in WIRE_TRACING_POLE_NODE_TYPES (drops service-drop / tap spans).
  * ``proposed`` traces (future installs) are dropped.

Nothing is filtered silently: every drop is counted and surfaced in the stats report.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import (
    BASE_DIR_MIDSPAN,
    WIRE_TRACING_JOB_SOURCE_DIR,
    WIRE_TRACING_DATASET_DIR,
    WIRE_TRACING_MI_MAX_INSULATORS,
    WIRE_TRACING_IN_SCOPE_CONNECTION_TYPES,
    WIRE_TRACING_POLE_NODE_TYPES,
    WIRE_TRACING_SINGLE_SECTION_ONLY,
    WIRE_TRACING_MULTI_SECTION,
    MIDSPAN_WIRE_EXCLUDED_JOB_PREFIXES,
)


def sections_with_photos(conn: Dict) -> int:
    """Number of midspan sections on a connection that actually carry photos. >1 means a
    multi-midspan connection (pole-mid-mid-...-pole) whose SCID-pair photos are ambiguous."""
    return sum(1 for sv in (conn.get("sections") or {}).values() if (sv or {}).get("photos"))

# --------------------------------------------------------------------------- #
# Katapult attribute / geometry helpers
# --------------------------------------------------------------------------- #

def _attr_value(attrs: Optional[Dict], name: str) -> Optional[str]:
    """Katapult wraps attribute values as {name: {<varying-subkey>: value}}.

    Return the single inner value regardless of the subkey (``imported``,
    ``button_added``, ``auto_button``, or a random push id), or None.
    """
    a = (attrs or {}).get(name)
    if a is None:
        return None
    if isinstance(a, dict):
        for v in a.values():
            return v
        return None
    return a


def _first_pixel(marker: Dict) -> Optional[Tuple[float, float]]:
    """Return (percentX, percentY) of a marker's first pixel_selection, or None."""
    sel = marker.get("pixel_selection") or []
    if not sel:
        return None
    px = sel[0]
    x, y = px.get("percentX"), px.get("percentY")
    if x is None or y is None:
        return None
    return float(x), float(y)


def _exists(marker: Dict) -> bool:
    """Markers may be soft-deleted via _exists=False; default True."""
    return marker.get("_exists", True) is not False


def _haversine_m(lat1, lon1, lat2, lon2) -> Optional[float]:
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _bearing_deg(lat1, lon1, lat2, lon2) -> Optional[float]:
    if None in (lat1, lon1, lat2, lon2):
        return None
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


# --------------------------------------------------------------------------- #
# Job-level regime detection
# --------------------------------------------------------------------------- #

def count_insulators(job: Dict) -> int:
    """Total insulator markers (top-level + arm children) across a job's photos."""
    n = 0
    for p in (job.get("photos") or {}).values():
        pf = p.get("photofirst_data") or {}
        n += len(pf.get("insulator") or {})
        for a in (pf.get("arm") or {}).values():
            n += len((a.get("_children") or {}).get("insulator") or {})
    return n


def job_is_mi_regime(job: Dict) -> bool:
    """Content-based MI detection: a per-wire job has many insulators, MI has ~none."""
    return count_insulators(job) <= WIRE_TRACING_MI_MAX_INSULATORS


def filename_says_mi(stem: str) -> bool:
    return any(stem.upper().startswith(p.upper()) for p in MIDSPAN_WIRE_EXCLUDED_JOB_PREFIXES)


def proposed_trace_ids(job: Dict) -> set:
    out = set()
    for tid, t in (job.get("traces", {}).get("trace_data", {}) or {}).items():
        if t.get("proposed") is True or t.get("proposed") == "true":
            out.add(tid)
    return out


def trace_meta(job: Dict, trace_id: str) -> Dict:
    t = (job.get("traces", {}).get("trace_data", {}) or {}).get(trace_id, {}) or {}
    return {
        "trace_type": t.get("_trace_type"),
        "cable_type": (t.get("cable_type") or None),
        "company": t.get("company"),
    }


# --------------------------------------------------------------------------- #
# Marker extraction
# --------------------------------------------------------------------------- #

def _wire_child_traces(node_with_children: Dict) -> List[str]:
    """Trace ids of the wire children of an insulator (insulator -> _children.wire)."""
    out = []
    for w in (((node_with_children.get("_children") or {}).get("wire")) or {}).values():
        if _exists(w) and w.get("_trace"):
            out.append(w["_trace"])
    return out


def pole_marker_instances(photo: Dict, photo_id: str) -> List[Dict]:
    """Pole-side attachment marker instances from one photo.

    Each instance: {pixel:(x,y)|None, kind, arm_id, traces:[(trace_id, spec)], photo_id}.
    Pole attachments live as: top-level insulator, arm -> child insulators, or guying.
    """
    pf = photo.get("photofirst_data") or {}
    out: List[Dict] = []

    # Top-level insulators (pole-mounted; own pixel)
    for iid, ins in (pf.get("insulator") or {}).items():
        if not _exists(ins):
            continue
        traces = [(t, ins.get("insulator_spec")) for t in _wire_child_traces(ins)]
        if traces:
            out.append({"pixel": _first_pixel(ins), "kind": "insulator", "arm_id": None,
                        "traces": traces, "photo_id": photo_id})

    # Crossarms: ONE arm pixel shared by all child insulators (children have empty pixel)
    for aid, arm in (pf.get("arm") or {}).items():
        if not _exists(arm):
            continue
        children = ((arm.get("_children") or {}).get("insulator")) or {}
        traces = []
        for iid, ins in children.items():
            for t in _wire_child_traces(ins):
                traces.append((t, ins.get("insulator_spec")))
        if traces:
            out.append({"pixel": _first_pixel(arm), "kind": "arm", "arm_id": aid,
                        "traces": traces, "photo_id": photo_id})

    # Down guys / guying (pole-only; never cross the span -> dustbin signal)
    for gid, g in (pf.get("guying") or {}).items():
        if not _exists(g) or not g.get("_trace"):
            continue
        out.append({"pixel": _first_pixel(g), "kind": "guying", "arm_id": None,
                    "traces": [(g["_trace"], None)], "photo_id": photo_id})

    return out


def midspan_marker_instances(photo: Dict, photo_id: str) -> List[Dict]:
    """Midspan wire markers from one section photo (per-trace, spatially distinct)."""
    pf = photo.get("photofirst_data") or {}
    out = []
    for mid, m in (pf.get("wire") or {}).items():
        if not _exists(m) or not m.get("_trace"):
            continue
        px = _first_pixel(m)
        out.append({"pixel": px, "trace_id": m["_trace"], "photo_id": photo_id,
                    "wire_spec": m.get("wire_spec") or None})
    return out


# --------------------------------------------------------------------------- #
# Side assembly (dedup across the many photos of one pole / section)
# --------------------------------------------------------------------------- #

def collect_pole_points(job: Dict, node_id: str, drop_traces: set) -> List[Dict]:
    """Build deduped pole-side POINTS for a node.

    A physical attachment is seen across several photos of the same pole. We union
    marker instances that SHARE a trace id (a trace attaches at exactly one point per
    pole), which correctly merges the same crossarm seen from multiple photos and keeps
    distinct attachments separate. Proposed traces are dropped.
    """
    photos = job.get("photos") or {}
    node = (job.get("nodes") or {}).get(node_id) or {}
    instances: List[Dict] = []
    for pid in (node.get("photos") or []):
        photo = photos.get(pid)
        if not photo:
            continue
        for inst in pole_marker_instances(photo, pid):
            # drop proposed traces from the instance
            inst = dict(inst)
            inst["traces"] = [(t, s) for (t, s) in inst["traces"] if t not in drop_traces]
            if inst["traces"]:
                instances.append(inst)

    # union-find over instances by shared trace id
    parent = list(range(len(instances)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    trace_to_inst: Dict[str, int] = {}
    for idx, inst in enumerate(instances):
        for (t, _s) in inst["traces"]:
            if t in trace_to_inst:
                union(idx, trace_to_inst[t])
            else:
                trace_to_inst[t] = idx

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(instances)):
        groups[find(idx)].append(idx)

    points: List[Dict] = []
    for members in groups.values():
        traces: Dict[str, Optional[str]] = {}
        pixel = None
        is_arm = False
        guying = False
        for idx in members:
            inst = instances[idx]
            if inst["pixel"] and pixel is None:
                pixel = inst["pixel"]
            if inst["kind"] == "arm":
                is_arm = True
            if inst["kind"] == "guying":
                guying = True
            for (t, s) in inst["traces"]:
                if t not in traces or (traces[t] is None and s is not None):
                    traces[t] = s
        kind = "arm" if is_arm else ("guying" if guying else "insulator")
        tlist = [{"trace_id": t, "insulator_spec": s, **trace_meta(job, t)}
                 for t, s in traces.items()]
        points.append({
            "x": pixel[0] if pixel else None,
            "y": pixel[1] if pixel else None,
            "kind": kind,
            "is_crossarm": is_arm and len(tlist) > 1,
            "multiplicity": len(tlist),
            "traces": tlist,
        })
    # stable order: top of pole first (smaller percentY), Nones last
    points.sort(key=lambda p: (p["y"] is None, p["y"] if p["y"] is not None else 0.0))
    for i, p in enumerate(points):
        p["i"] = i
    return points


def collect_midspan_points(job: Dict, section_photo_ids: List[str], drop_traces: set) -> List[Dict]:
    """Per-trace midspan POINTS, deduped across the section's photos (keep first seen)."""
    photos = job.get("photos") or {}
    seen: Dict[str, Dict] = {}
    for pid in section_photo_ids:
        photo = photos.get(pid)
        if not photo:
            continue
        for m in midspan_marker_instances(photo, pid):
            t = m["trace_id"]
            if t in drop_traces or t in seen:
                continue
            px = m["pixel"]
            seen[t] = {"x": px[0] if px else None, "y": px[1] if px else None,
                       "trace_id": t, "wire_spec": m["wire_spec"], **trace_meta(job, t)}
    points = list(seen.values())
    points.sort(key=lambda p: (p["y"] is None, p["y"] if p["y"] is not None else 0.0))
    for i, p in enumerate(points):
        p["i"] = i
    return points


# --------------------------------------------------------------------------- #
# Span sample
# --------------------------------------------------------------------------- #

def _node_info(job: Dict, node_id: str) -> Dict:
    n = (job.get("nodes") or {}).get(node_id) or {}
    return {
        "node_id": node_id,
        "scid": _attr_value(n.get("attributes"), "scid"),
        "node_type": _attr_value(n.get("attributes"), "node_type"),
        "lat": n.get("latitude"),
        "lon": n.get("longitude"),
    }


def collect_sections(job: Dict, conn: Dict, pole_a: Dict, drop_traces: set) -> List[Dict]:
    """Ordered A->B list of the connection's photo-bearing midspan sections.

    A multi-midspan connection is physically pole-A -> M1 -> ... -> Mk -> pole-B; each Mi is a
    distinct waypoint with its OWN photo(s), lat/lon and wire (sag) heights. Sections are ordered
    by great-circle distance from pole-A so M1 is nearest A and Mk nearest B (geometry-less
    sections, if any, sort last). Each entry carries its own deduped per-trace midspan points.
    """
    out: List[Dict] = []
    for sid, sv in (conn.get("sections") or {}).items():
        sv = sv or {}
        photo_ids = list((sv.get("photos") or {}).keys())
        if not photo_ids:
            continue
        lat, lon = sv.get("latitude"), sv.get("longitude")
        out.append({
            "section_id": sid,
            "lat": lat, "lon": lon,
            "dist_a_m": _haversine_m(pole_a["lat"], pole_a["lon"], lat, lon),
            "photo_ids": photo_ids,
            "points": collect_midspan_points(job, photo_ids, drop_traces),
        })
    out.sort(key=lambda s: (s["dist_a_m"] is None, s["dist_a_m"] if s["dist_a_m"] is not None else 0.0))
    for i, s in enumerate(out):
        s["i"] = i
    return out


def _spine_section(sections: List[Dict]) -> Optional[Dict]:
    """The section whose points back the legacy single-M fields (backward compat). Prefer the
    canonical ``midpoint_section`` (what the legacy single-section reader used) so single-section
    output stays byte-identical; else fall back to the section nearest pole-A."""
    for s in sections:
        if s["section_id"] == "midpoint_section":
            return s
    return sections[0] if sections else None


def build_span_sample(job: Dict, job_stem: str, conn_id: str, conn: Dict,
                      drop_traces: set, multi_section: bool = False) -> Optional[Dict]:
    """Build one span record, or None if the span has no midspan markers.

    When ``multi_section`` is True the record additionally carries the ordered per-section
    structure ``sides.M_sections`` (pole-A -> M1 -> ... -> Mk -> pole-B) and full-path GT
    ``gt.chains_multi`` (one midspan-point index per section, None where the trace is unseen in
    that section), and multi-midspan connections are no longer dropped. The legacy single-M fields
    (``sides.M``, ``gt.chains``, ``gt.dustbin``) are still emitted against a SPINE section so
    existing single-M consumers keep working unchanged; for a genuine single-section span the
    spine IS ``midpoint_section`` and every legacy field is byte-identical to the flag-off build.
    """
    a = _node_info(job, conn["node_id_1"])
    b = _node_info(job, conn["node_id_2"])

    A = collect_pole_points(job, a["node_id"], drop_traces)
    B = collect_pole_points(job, b["node_id"], drop_traces)

    sections = None
    if multi_section:
        sections = collect_sections(job, conn, a, drop_traces)
        if not any(s["points"] for s in sections):
            return None
        spine = _spine_section(sections)
        M = spine["points"] if spine else []
    else:
        sec = (conn.get("sections") or {}).get("midpoint_section") or {}
        sec_photo_ids = list((sec.get("photos") or {}).keys())
        M = collect_midspan_points(job, sec_photo_ids, drop_traces)
        if not M:
            return None

    # index: trace_id -> pole point index (per side)
    def trace_index(points):
        idx = {}
        for p in points:
            for t in p["traces"]:
                idx[t["trace_id"]] = p["i"]
        return idx

    aidx, bidx = trace_index(A), trace_index(B)

    # --- legacy single-M chains / dustbin (against the spine section) ---------- #
    chains, dustbin_m = [], []
    used_a, used_b = set(), set()
    for m in M:
        t = m["trace_id"]
        ia = aidx.get(t)
        ib = bidx.get(t)
        if ia is None and ib is None:
            dustbin_m.append(m["i"])
            continue
        amb = (ia is not None and A[ia]["multiplicity"] > 1) or \
              (ib is not None and B[ib]["multiplicity"] > 1)
        if ia is not None:
            used_a.add(ia)
        if ib is not None:
            used_b.add(ib)
        chains.append({"trace_id": t, "cable_type": m.get("cable_type"),
                       "trace_type": m.get("trace_type"), "M": m["i"],
                       "A": ia, "B": ib, "group_ambiguous": amb})

    # traces present at BOTH poles but with no midspan marker (midspan miss).
    # sorted() makes the order reproducible across runs (set iteration is PYTHONHASHSEED-dependent).
    pole_only = []
    a_traces = sorted(set(aidx) - {c["trace_id"] for c in chains})
    for t in a_traces:
        if t in bidx:
            pole_only.append({"trace_id": t, "A": aidx[t], "B": bidx[t],
                              **trace_meta(job, t)})
            used_a.add(aidx[t])
            used_b.add(bidx[t])

    dustbin_a = [p["i"] for p in A if p["i"] not in used_a]
    dustbin_b = [p["i"] for p in B if p["i"] not in used_b]

    sample = {
        "job": job_stem,
        "connection_id": conn_id,
        "connection_type": _attr_value(conn.get("attributes"), "connection_type"),
        "pole_a": a,
        "pole_b": b,
        "geometry": {
            "bearing_deg": _bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"]),
            "length_m": _haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]),
        },
        "sides": {"A": A, "M": M, "B": B},
        "gt": {"chains": chains, "pole_only_chains": pole_only,
               "dustbin": {"A": dustbin_a, "M": dustbin_m, "B": dustbin_b}},
    }

    if not multi_section:
        return sample

    # --- full ordered-path multi-section GT (additive) ------------------------- #
    sec_maps = [{p["trace_id"]: p["i"] for p in s["points"]} for s in sections]
    all_traces = set(aidx) | set(bidx)
    for m in sec_maps:
        all_traces |= set(m)

    chains_multi = []
    used_sec = [set() for _ in sections]
    for t in sorted(all_traces):                 # deterministic iteration (set order is unstable)
        ia, ib = aidx.get(t), bidx.get(t)
        if ia is None and ib is None:
            continue  # midspan-only trace -> per-section dustbin below
        m_path = [m.get(t) for m in sec_maps]
        amb = (ia is not None and A[ia]["multiplicity"] > 1) or \
              (ib is not None and B[ib]["multiplicity"] > 1)
        for si, mi in enumerate(m_path):
            if mi is not None:
                used_sec[si].add(mi)
        chains_multi.append({"trace_id": t, "A": ia, "B": ib, "M_path": m_path,
                             "group_ambiguous": amb, **trace_meta(job, t)})

    def _ckey(c):
        for si, mi in enumerate(c["M_path"]):
            if mi is not None:
                y = sections[si]["points"][mi]["y"]
                if y is not None:
                    return (0, y)
        if c["A"] is not None and A[c["A"]]["y"] is not None:
            return (1, A[c["A"]]["y"])
        if c["B"] is not None and B[c["B"]]["y"] is not None:
            return (2, B[c["B"]]["y"])
        return (3, 0.0)
    chains_multi.sort(key=lambda c: (_ckey(c), c["trace_id"]))   # trace_id breaks height ties

    dustbin_sections = [[p["i"] for p in s["points"] if p["i"] not in used_sec[si]]
                        for si, s in enumerate(sections)]

    sample["n_sections"] = len(sections)
    sample["sides"]["M_sections"] = [
        {k: s[k] for k in ("section_id", "lat", "lon", "dist_a_m", "photo_ids", "points", "i")}
        for s in sections
    ]
    sample["gt"]["chains_multi"] = chains_multi
    sample["gt"]["dustbin_sections"] = dustbin_sections
    return sample


# --------------------------------------------------------------------------- #
# Order concordance (clean, multiplicity==1 chains only)
# --------------------------------------------------------------------------- #

def _inversions(seq: List[float]) -> int:
    return sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j])


def span_order_stats(sample: Dict) -> Tuple[int, int, bool]:
    """(inversions, comparable_pairs, perfectly_ordered) for pole-A vs midspan,
    using only cleanly-resolvable (non-group-ambiguous) chains with valid y."""
    A = sample["sides"]["A"]
    M = sample["sides"]["M"]
    pairs = []
    for c in sample["gt"]["chains"]:
        if c["group_ambiguous"] or c["A"] is None:
            continue
        ay, my = A[c["A"]]["y"], M[c["M"]]["y"]
        if ay is not None and my is not None:
            pairs.append((my, ay))
    if len(pairs) < 2:
        return 0, 0, len(pairs) >= 0
    pairs.sort(key=lambda p: p[0])
    inv = _inversions([ay for _my, ay in pairs])
    n = len(pairs) * (len(pairs) - 1) // 2
    return inv, n, inv == 0


# --------------------------------------------------------------------------- #
# Dataset build
# --------------------------------------------------------------------------- #

def build_dataset(midspan_dir: Path = WIRE_TRACING_JOB_SOURCE_DIR,
                  out_dir: Path = WIRE_TRACING_DATASET_DIR,
                  verbose: bool = False,
                  multi_section: Optional[bool] = None) -> Dict:
    """Walk all job JSONs, emit spans.jsonl + stats.json, return the stats dict.

    ``multi_section`` (default: config WIRE_TRACING_MULTI_SECTION) keeps multi-midspan spans and
    emits the ordered per-section structure; when False the output is byte-identical to the
    legacy single-section build.
    """
    if multi_section is None:
        multi_section = WIRE_TRACING_MULTI_SECTION
    out_dir.mkdir(parents=True, exist_ok=True)
    spans_path = out_dir / "spans.jsonl"
    stats_path = out_dir / "stats.json"

    job_files = sorted(p for p in midspan_dir.glob("*.json"))
    S = {
        "jobs_total": len(job_files), "jobs_kept": 0,
        "jobs_excluded_mi": [], "regime_name_content_disagree": [],
        "connections_total": 0, "connection_types": Counter(),
        "aerial_total": 0, "spans_kept": 0,
        "spans_dropped_no_midspan": 0, "spans_dropped_multi_section": 0,
        "spans_dropped_endpoint": 0, "dropped_endpoint_pairs": Counter(),
        "proposed_traces_dropped": 0,
        "pole_points_A": 0, "pole_points_B": 0, "midspan_points": 0,
        "pole_points_crossarm": 0, "pole_points_guying": 0, "pole_points_no_pixel": 0,
        "midspan_chained_both": 0, "midspan_chained_one": 0, "midspan_dustbin": 0,
        "group_ambiguous_chains": 0, "pole_only_chains": 0,
        "dustbin_pole_points": 0,
        "multi_section_enabled": bool(multi_section),
        "spans_multi_section": 0, "sections_total": 0, "section_count_dist": Counter(),
        "midspan_points_all_sections": 0, "chains_multi_total": 0,
        "order_inversions": 0, "order_pairs": 0, "spans_perfect_order": 0,
        "spans_with_order_metric": 0,
        "cable_types_midspan": Counter(), "arm_child_counts": Counter(),
    }

    n_written = 0
    with open(spans_path, "w") as fout:
        for jf in job_files:
            stem = jf.name.split(" - ")[0].split(".")[0]
            try:
                job = json.load(open(jf))
            except Exception as e:
                if verbose:
                    print(f"  ! skip unreadable {jf.name}: {e}")
                continue

            is_mi = job_is_mi_regime(job)
            if is_mi != filename_says_mi(stem):
                S["regime_name_content_disagree"].append(
                    {"job": stem, "filename_mi": filename_says_mi(stem),
                     "content_mi": is_mi, "insulators": count_insulators(job)})
            if is_mi:
                S["jobs_excluded_mi"].append(stem)
                continue
            S["jobs_kept"] += 1

            drop_traces = proposed_trace_ids(job)
            S["proposed_traces_dropped"] += len(drop_traces)

            conns = job.get("connections") or {}
            S["connections_total"] += len(conns)
            for cid, c in conns.items():
                ctype = _attr_value(c.get("attributes"), "connection_type")
                S["connection_types"][ctype] += 1
                if ctype not in WIRE_TRACING_IN_SCOPE_CONNECTION_TYPES:
                    continue
                S["aerial_total"] += 1
                ta = _attr_value((job.get("nodes", {}).get(c.get("node_id_1"), {}) or {}).get("attributes"), "node_type")
                tb = _attr_value((job.get("nodes", {}).get(c.get("node_id_2"), {}) or {}).get("attributes"), "node_type")
                if ta not in WIRE_TRACING_POLE_NODE_TYPES or tb not in WIRE_TRACING_POLE_NODE_TYPES:
                    S["spans_dropped_endpoint"] += 1
                    S["dropped_endpoint_pairs"][tuple(sorted((str(ta), str(tb))))] += 1
                    continue

                # multi-midspan connection: when multi_section is off, the SCID-pair photo naming
                # can't say which section a photo belongs to -> drop (legacy). When on, the
                # ruler-keypoint re-keying resolves it, so keep the full ordered path.
                if not multi_section and sections_with_photos(c) > 1:
                    S["spans_dropped_multi_section"] += 1
                    continue

                sample = build_span_sample(job, stem, cid, c, drop_traces,
                                           multi_section=multi_section)
                if sample is None:
                    S["spans_dropped_no_midspan"] += 1
                    continue

                # accumulate stats
                S["spans_kept"] += 1
                A, M, B = sample["sides"]["A"], sample["sides"]["M"], sample["sides"]["B"]
                S["pole_points_A"] += len(A)
                S["pole_points_B"] += len(B)
                S["midspan_points"] += len(M)
                for side in (A, B):
                    for p in side:
                        if p["is_crossarm"]:
                            S["pole_points_crossarm"] += 1
                        if p["kind"] == "guying":
                            S["pole_points_guying"] += 1
                        if p["x"] is None:
                            S["pole_points_no_pixel"] += 1
                        if p["kind"] == "arm":
                            S["arm_child_counts"][min(p["multiplicity"], 6)] += 1
                for m in M:
                    S["cable_types_midspan"][m.get("cable_type") or "(none)"] += 1
                for c2 in sample["gt"]["chains"]:
                    if c2["A"] is not None and c2["B"] is not None:
                        S["midspan_chained_both"] += 1
                    else:
                        S["midspan_chained_one"] += 1
                    if c2["group_ambiguous"]:
                        S["group_ambiguous_chains"] += 1
                S["midspan_dustbin"] += len(sample["gt"]["dustbin"]["M"])
                S["pole_only_chains"] += len(sample["gt"]["pole_only_chains"])
                S["dustbin_pole_points"] += len(sample["gt"]["dustbin"]["A"]) + len(sample["gt"]["dustbin"]["B"])

                if multi_section:
                    secs = sample["sides"].get("M_sections") or []
                    nsec = sample.get("n_sections", len(secs))
                    S["sections_total"] += nsec
                    S["section_count_dist"][min(nsec, 6)] += 1
                    if nsec > 1:
                        S["spans_multi_section"] += 1
                    S["midspan_points_all_sections"] += sum(len(s["points"]) for s in secs)
                    S["chains_multi_total"] += len(sample["gt"].get("chains_multi") or [])

                inv, npairs, perfect = span_order_stats(sample)
                S["order_inversions"] += inv
                S["order_pairs"] += npairs
                if npairs > 0:
                    S["spans_with_order_metric"] += 1
                    if perfect:
                        S["spans_perfect_order"] += 1

                fout.write(json.dumps(sample) + "\n")
                n_written += 1

    # derived rates
    mp = max(S["midspan_points"], 1)
    S["rates"] = {
        "chain_both_pct": round(100 * S["midspan_chained_both"] / mp, 1),
        "midspan_dustbin_pct": round(100 * S["midspan_dustbin"] / mp, 1),
        "group_ambiguous_pct": round(100 * S["group_ambiguous_chains"] / mp, 1),
        "order_concordant_pct": round(100 * (1 - S["order_inversions"] / max(S["order_pairs"], 1)), 1),
        "spans_perfect_order_pct": round(100 * S["spans_perfect_order"] / max(S["spans_with_order_metric"], 1), 1),
    }
    S["spans_written"] = n_written

    # Counters -> plain dicts for JSON
    for k in ("connection_types", "dropped_endpoint_pairs", "cable_types_midspan",
              "arm_child_counts", "section_count_dist"):
        S[k] = {str(kk): vv for kk, vv in sorted(S[k].items(), key=lambda x: -x[1])}

    with open(stats_path, "w") as f:
        json.dump(S, f, indent=2)
    return S


def format_report(S: Dict) -> str:
    """Human-readable sanity report."""
    L = []
    L.append("=" * 64)
    L.append("WIRE TRACING — Stage-0 extraction report")
    L.append("=" * 64)
    L.append(f"jobs: {S['jobs_kept']} kept / {S['jobs_total']} total  "
             f"({len(S['jobs_excluded_mi'])} MI-excluded)")
    if S["jobs_excluded_mi"]:
        L.append(f"  MI-excluded (insulators==0): {', '.join(S['jobs_excluded_mi'])}")
    if S["regime_name_content_disagree"]:
        L.append(f"  ⚠ name/content regime disagreement: {S['regime_name_content_disagree']}")
    L.append("")
    L.append(f"connections: {S['connections_total']} total  | by type: {S['connection_types']}")
    L.append(f"aerial in-scope: {S['aerial_total']}  -> spans kept: {S['spans_kept']}  "
             f"(written {S['spans_written']})")
    L.append(f"  dropped — non-pole endpoint: {S['spans_dropped_endpoint']}  "
             f"{S['dropped_endpoint_pairs']}")
    L.append(f"  dropped — no midspan markers: {S['spans_dropped_no_midspan']}")
    L.append(f"  dropped — multi-midspan connection (ambiguous): {S['spans_dropped_multi_section']}")
    if S.get("multi_section_enabled"):
        L.append(f"  multi-section: ON — {S['spans_multi_section']} multi-section spans, "
                 f"{S['sections_total']} sections total, dist(1..6+)={S['section_count_dist']}")
        L.append(f"    all-section midspan points: {S['midspan_points_all_sections']}  "
                 f"full-path chains: {S['chains_multi_total']}")
    L.append(f"  proposed traces dropped: {S['proposed_traces_dropped']}")
    L.append("")
    L.append(f"detections: poleA={S['pole_points_A']}  midspan={S['midspan_points']}  "
             f"poleB={S['pole_points_B']}")
    L.append(f"  crossarm pole points: {S['pole_points_crossarm']}  "
             f"guying: {S['pole_points_guying']}  no-pixel: {S['pole_points_no_pixel']}")
    L.append(f"  arm child-count dist (1..6+): {S['arm_child_counts']}")
    L.append("")
    R = S["rates"]
    L.append(f"chain rate (midspan reaching BOTH poles): {S['midspan_chained_both']}/{S['midspan_points']} "
             f"= {R['chain_both_pct']}%")
    L.append(f"midspan reaching only ONE pole: {S['midspan_chained_one']}")
    L.append(f"midspan dustbin (neither pole): {S['midspan_dustbin']} = {R['midspan_dustbin_pct']}%")
    L.append(f"group-ambiguous chains (crossarm, phase unresolvable): {S['group_ambiguous_chains']} "
             f"= {R['group_ambiguous_pct']}%")
    L.append(f"pole-only chains (both poles, midspan missed): {S['pole_only_chains']}")
    L.append(f"dustbin pole points (no continuation): {S['dustbin_pole_points']}")
    L.append("")
    L.append(f"order concordance (clean chains, A vs midspan): {R['order_concordant_pct']}% "
             f"({S['order_inversions']} inv / {S['order_pairs']} pairs)")
    L.append(f"spans perfectly ordered: {S['spans_perfect_order']}/{S['spans_with_order_metric']} "
             f"= {R['spans_perfect_order_pct']}%")
    L.append("")
    L.append(f"midspan cable_type dist: {S['cable_types_midspan']}")
    L.append("=" * 64)
    return "\n".join(L)
