"""Ground-truth overlay for the demo span viewer (LOCAL ONLY).

Resolves a demo span (job + pole SCIDs from ``inference/spans/manifest.json``)
to the annotated ground truth in the local label store:

    data/photo_lookup.json      job -> scid -> photo_id (pole),
                                job -> "scidA|scidB" -> [photo_id] (midspan)
    data/labels/<job>.json      photo_id -> attachments / wires / anchors

None of that data ships with the public deployment, so every entry point
degrades to "GT unavailable" when the store is absent — the HF Space build
stays inference-only by construction.

Heights are stamped with the same projective ruler fit the predictions use,
but fitted on the GT anchor ticks (label-faithful, not model-dependent).
"""
from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from deploy.shared import PROJECT_ROOT
from deploy.span_trace import _fit_projective_height, format_feet_inches

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
_LOOKUP_PATH = DATA_DIR / "photo_lookup.json"
_LABELS_DIR = DATA_DIR / "labels"
_PHOTOS_DIR = DATA_DIR / "Photos"
_JOBS_DIR = DATA_DIR / "jobs"

# Katapult trace cable_type -> the canonical keys the frontend palette uses.
_CABLE_TYPE_MAP = [
    ("open sec", "open_secondary"),
    ("primary", "primary"),
    ("neutral", "neutral"),
    ("secondary", "secondary"),
    ("catv", "catv"),
    ("telco", "telco"),
    ("telephone", "telco"),
    ("fiber", "fiber"),
    ("down guy", "down_guy"),
    ("guy", "guy"),
    ("com", "comm"),
]


def _canon_cable_type(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    low = str(raw).lower()
    for needle, canon in _CABLE_TYPE_MAP:
        if needle in low:
            return canon
    return None


@lru_cache(maxsize=4)
def _job_wire_markers(job_key: str) -> Dict[str, List[Dict[str, Any]]]:
    """pid -> [{x, y, cable_type}] from the raw Katapult job JSON wire markers
    (the label store keeps only positions; cable type lives on the trace)."""
    path = _JOBS_DIR / f"{job_key}.json"
    try:
        job = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning("raw job JSON unavailable for GT cable types: %s", path)
        return {}
    trace_data = (job.get("traces") or {}).get("trace_data") or {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for pid, photo in (job.get("photos") or {}).items():
        markers = []
        wire = ((photo.get("photofirst_data") or {}).get("wire")) or {}
        for m in wire.values():
            if not isinstance(m, dict):
                continue
            sel = (m.get("pixel_selection") or [{}])[0]
            x, y = sel.get("percentX"), sel.get("percentY")
            if x is None or y is None:
                continue
            tr = trace_data.get(m.get("_trace")) or {}
            markers.append({
                "x": float(x), "y": float(y),
                "cable_type": _canon_cable_type(tr.get("cable_type")),
            })
        if markers:
            out[pid] = markers
    return out


def gt_available() -> bool:
    return _LOOKUP_PATH.exists() and _LABELS_DIR.is_dir()


@lru_cache(maxsize=1)
def _lookup() -> Dict[str, Any]:
    try:
        return json.loads(_LOOKUP_PATH.read_text())
    except (OSError, ValueError):
        logger.exception("photo_lookup.json unreadable")
        return {}


@lru_cache(maxsize=8)
def _labels(job_key: str) -> Dict[str, Any]:
    path = _LABELS_DIR / f"{job_key}.json"
    try:
        return json.loads(path.read_text()).get("photos", {})
    except (OSError, ValueError):
        logger.exception("label store unreadable: %s", path)
        return {}


def _resolve_job_key(job: Optional[str]) -> Optional[str]:
    """Manifest jobs are short ('COAR-FR01'); lookup keys may carry suffixes
    ('COAR-FR01 - 3'). Exact match wins, else a unique prefix match."""
    if not job:
        return None
    keys = set(_lookup().get("pole", {})) | set(_lookup().get("midspan", {}))
    if job in keys:
        return job
    matches = sorted(k for k in keys if k.startswith(job))
    if len(matches) == 1:
        return matches[0]
    if matches:
        logger.warning("Ambiguous GT job for %r: %s — using %s", job, matches, matches[0])
        return matches[0]
    return None


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


_WIRE_CT_GATE_PCT = 1.0   # max marker<->label distance (percent) to bind a cable type


def _photo_gt(entry: Dict[str, Any], wire_markers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """One photo's GT view-model: percent coords + GT-calibrated height labels."""
    anchors = entry.get("anchors") or []
    ticks = [(float(a[0]), float(a[2])) for a in anchors if len(a) >= 3]
    y_to_h = _fit_projective_height(ticks) if len(ticks) >= 3 else None

    def _pt(x: float, y: float, name: str) -> Dict[str, Any]:
        h = y_to_h(float(y)) if y_to_h is not None else None
        h = float(h) if h is not None else None
        return {
            "name": name,
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "height_ft": round(h, 2) if h is not None else None,
            "height_label": format_feet_inches(h),
        }

    out: Dict[str, Any] = {
        "role": entry.get("role"),
        "anchor_ticks": [
            {"height_ft": float(a[0]), "x": round(float(a[1]), 2), "y": round(float(a[2]), 2)}
            for a in anchors if len(a) >= 3
        ],
        "attachments": [
            _pt(att["x"], att["y"], att.get("name") or "attachment")
            for att in entry.get("attachments") or []
        ],
        "wires": [
            _pt(w["x"], w["y"], w.get("name") or "wire")
            for w in entry.get("wires") or []
        ],
        "pole_top": None,
    }
    # Midspan wires: bind each label wire to the nearest raw Katapult wire
    # marker (same annotation, richer fields) to carry the trace cable_type.
    for w in out["wires"]:
        best, best_d = None, _WIRE_CT_GATE_PCT
        for m in wire_markers or []:
            d = max(abs(m["x"] - w["x"]), abs(m["y"] - w["y"]))
            if d < best_d:
                best, best_d = m, d
        w["cable_type"] = best["cable_type"] if best else None
    pt = entry.get("pole_top")
    if pt and len(pt) >= 2:
        out["pole_top"] = _pt(pt[0], pt[1], "pole_top")
    return out


def load_span_gt(manifest_entry: Dict[str, Any], span_dir: Path) -> Dict[str, Any]:
    """GT payload for one demo span: {'available': bool, 'photos': {...}}.

    photos: 'pole_a' / 'pole_b' -> GT dict or None, 'midspans' -> list aligned
    with the manifest's midspan files (None where no GT photo matches).
    """
    if not gt_available():
        return {"available": False}

    job_key = _resolve_job_key(manifest_entry.get("job"))
    if job_key is None:
        return {"available": False}
    labels = _labels(job_key)
    if not labels:
        return {"available": False}
    lk = _lookup()

    photos: Dict[str, Any] = {"pole_a": None, "pole_b": None, "midspans": []}
    pole_map = lk.get("pole", {}).get(job_key, {})
    for side in ("a", "b"):
        pid = pole_map.get(str(manifest_entry.get(f"pole_{side}_scid")))
        entry = labels.get(pid) if pid else None
        if entry:
            photos[f"pole_{side}"] = _photo_gt(entry)

    scids = [str(manifest_entry.get("pole_a_scid")), str(manifest_entry.get("pole_b_scid"))]
    conn_key = "|".join(sorted(scids))
    mid_pids = [p for p in lk.get("midspan", {}).get(job_key, {}).get(conn_key, [])
                if p in labels]
    mid_files: List[str] = (manifest_entry.get("files") or {}).get("midspans", [])

    # Bind each demo frame to its GT photo by content hash (frames were copied
    # from data/Photos verbatim); fall back to index order when hashes miss.
    pid_by_sha = {}
    for pid in mid_pids:
        sha = _sha256(_PHOTOS_DIR / f"{pid}.jpg")
        if sha:
            pid_by_sha[sha] = pid
    assigned: List[Optional[str]] = []
    unmatched = list(mid_pids)
    for fname in mid_files:
        sha = _sha256(span_dir / fname)
        pid = pid_by_sha.get(sha) if sha else None
        if pid in unmatched:
            unmatched.remove(pid)
        assigned.append(pid)
    for i, pid in enumerate(assigned):
        if pid is None and unmatched:
            assigned[i] = unmatched.pop(0)
    markers_by_pid = _job_wire_markers(job_key)
    photos["midspans"] = [
        _photo_gt(labels[pid], markers_by_pid.get(pid)) if pid and pid in labels else None
        for pid in assigned
    ]

    any_gt = photos["pole_a"] or photos["pole_b"] or any(photos["midspans"])
    return {"available": bool(any_gt), "photos": photos, "job_key": job_key}
