"""photo_id layout read-path (Phase 2 cutover, opt-in via USE_PHOTO_ID_LAYOUT=1).

When enabled, photos live at ``data/Photos/<photo_id>.jpg`` and labels at
``data/labels/<job>.json`` keyed by photo_id. This module is the single adapter the
e2e/tracer read-path delegates to; the legacy glob + ``<stem>_location.txt`` path is
unchanged when disabled (default), so the two can be A/B-compared.

Faithfulness contract (proven by the e2e parity gate):
  * resolvers return the SAME photo as the legacy glob (pole proven byte-identical),
  * heights come from the SAME projective fit over the SAME 5 ruler ticks,
  * the eval-only inch-tolerance PPI is RECOMPUTED from the stored anchors (identical to
    the dropped ``# PPI=`` line) -- projection is the height model; PPI is not stored.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
PHOTOS = DATA / "Photos"
LABELS = DATA / "labels"
# Transient gitignored parser-compat cache: reconstructed *_location.txt text so the legacy
# text parsers consume the JSON store unchanged. NOT a source of truth (rebuildable from
# data/labels); retired once parsers go JSON-native. One file per pid -> thread-safe.
TXT_CACHE = DATA / ".labels_txt_cache"

# Default ON (2026-06-21 cutover): the pipeline reads the photo_id layout (data/Photos +
# data/labels) unless explicitly disabled with USE_PHOTO_ID_LAYOUT=0. Validated byte-identical to
# the legacy layout (parity_layout_smoke + e2e ON==OFF). Set USE_PHOTO_ID_LAYOUT=0 for the legacy
# data/data_pole + data/data_midspan read path.
ENABLED = os.environ.get("USE_PHOTO_ID_LAYOUT", "1") not in ("0", "false", "False")

# Lazy module-level indexes (built once per process on first use).
_LABEL_BY_PID: Optional[Dict[str, dict]] = None
_POLE_PID: Optional[Dict] = None            # (real_job, scid) -> pid
_MID_PID: Optional[Dict] = None             # (real_job, frozenset{sa,sb}) -> [pid]
_REAL_JOBS: Optional[List[str]] = None      # real job stems, longest first
_DISK_TO_PID: Optional[Dict[str, str]] = None   # legacy disk stem (src_disk minus .jpg) -> pid


def _scid_keys(scid):
    s = str(scid)
    out = {s}
    if s.isdigit():
        out.add(str(int(s)))
        out.add(s.zfill(3))
    return out


def _build():
    """Resolver index from the authoritative data/photo_lookup.json (covers ALL on-disk
    photos, labeled or not); labels loaded separately as a (possibly sparse) overlay."""
    global _LABEL_BY_PID, _POLE_PID, _MID_PID, _REAL_JOBS, _DISK_TO_PID
    _LABEL_BY_PID, _POLE_PID, _MID_PID, _DISK_TO_PID = {}, {}, {}, {}
    lut = json.loads((DATA / "photo_lookup.json").read_text())
    jobs = set()
    for job, scid_pid in lut.get("pole", {}).items():
        jobs.add(job)
        for scid, pid in scid_pid.items():
            for sk in _scid_keys(scid):
                _POLE_PID[(job, sk)] = pid
    for job, key_pids in lut.get("midspan", {}).items():
        jobs.add(job)
        for key, pids in key_pids.items():
            _MID_PID[(job, frozenset(key.split("|")))] = pids
    _REAL_JOBS = sorted(jobs, key=len, reverse=True)
    # label overlay (annotations/heights); photos without a label simply have no entry
    for jf in sorted(LABELS.glob("*.json")):
        for pid, r in json.loads(jf.read_text())["photos"].items():
            _LABEL_BY_PID[pid] = r
            sd = r.get("src_disk")
            if sd:
                _DISK_TO_PID[sd[:-4] if sd.endswith(".jpg") else sd] = pid


def _ensure():
    if _LABEL_BY_PID is None:
        _build()


def _real_job(span_job: str) -> Optional[str]:
    """Spans store a normalized job ('COAR-FR01'); labels use the real stem ('COAR-FR01 - 3')."""
    _ensure()
    for j in _REAL_JOBS:
        if j == span_job or j.startswith(span_job):
            return j
    return None


# ---- public read-path ----
def label_for(pid: str) -> Optional[dict]:
    _ensure()
    return _LABEL_BY_PID.get(pid)


def photo_path(pid: str) -> str:
    return str(PHOTOS / f"{pid}.jpg")


def pole_photo(job: str, scid) -> Optional[str]:
    if scid is None:
        return None
    _ensure()
    rj = _real_job(job)
    if rj is None:
        return None
    for sk in _scid_keys(scid):
        pid = _POLE_PID.get((rj, sk))
        if pid:
            p = photo_path(pid)
            return p if os.path.exists(p) else None
    return None


def midspan_photos(job: str, scid_a, scid_b) -> List[str]:
    if scid_a is None or scid_b is None:
        return []
    _ensure()
    rj = _real_job(job)
    if rj is None:
        return []
    pids = _MID_PID.get((rj, frozenset((str(scid_a), str(scid_b)))), [])
    return [photo_path(p) for p in pids if os.path.exists(photo_path(p))]


def _anchors_to_ruler(rec) -> list:
    return [(a[2], a[0] * 12.0) for a in rec.get("anchors", []) if a[0] in (2.5, 6.5, 10.5, 14.5, 16.5)]


def ruler_fit(pid: str):
    rec = label_for(pid)
    if rec is None:
        return None
    from src.ruler_height_model import fit_photo_height
    ruler = _anchors_to_ruler(rec)
    return fit_photo_height(ruler) if ruler else None


def wire_ys(pid: str) -> Optional[List[float]]:
    rec = label_for(pid)
    if rec is None or rec.get("role") != "midspan":
        return None
    ys = [w["y"] for w in rec.get("wires", []) if w.get("y") is not None]
    return sorted(set(round(y, 1) for y in ys)) if ys else None


def iter_photos(role: Optional[str] = None):
    """Yield (pid, photo_path) for every on-disk photo, optionally filtered to a role
    ('pole'|'midspan'). Iteration source = photo_lookup.json (ALL photos, labeled or not),
    mirroring the legacy ``photos_dir.glob('*.jpg')`` set. Role for unlabeled photos is
    inferred from which lookup table (pole vs midspan) the pid came from."""
    _ensure()
    pole_pids, mid_pids = set(), set()
    for (j, sk), pid in _POLE_PID.items():
        pole_pids.add(pid)
    for (j, key), pids in _MID_PID.items():
        mid_pids.update(pids)
    if role in (None, "pole"):
        for pid in sorted(pole_pids):
            p = photo_path(pid)
            if os.path.exists(p):
                yield pid, p
    if role in (None, "midspan"):
        for pid in sorted(mid_pids - pole_pids):
            p = photo_path(pid)
            if os.path.exists(p):
                yield pid, p


def _fmt(v) -> str:
    """Shortest round-trip string for a stored float (float(_fmt(v)) == v)."""
    return repr(v) if isinstance(v, float) else str(v)


def location_lines(pid: str) -> Optional[List[str]]:
    """Reconstruct the faithful ``*_location.txt`` body from the JSON record, so the legacy
    text parsers (parse_attachments_with_keypoints, load_*_from_location_file, …) consume the
    new store unchanged. Reproduces every line a parser reads; the only originals omitted are
    the ``17.0`` ruler tick (height-neutral, no parser reads it) and ``# PPI=`` (recomputed
    below from the stored anchors -> identical value). Stored floats round-trip exactly, so
    re-parsed numeric values are byte-identical to the original file."""
    rec = label_for(pid)
    if rec is None:
        return None
    lines: List[str] = []
    for a in rec.get("anchors", []):
        lines.append(f"{_fmt(a[0])},{_fmt(a[1])},{_fmt(a[2])}")
    pt = rec.get("pole_top")
    if pt is not None:
        lines.append(f"pole_top,{_fmt(pt[0])},{_fmt(pt[1])}")
    for m in rec.get("attachments", []) + rec.get("wires", []):
        lines.append(f"{m['name']},{_fmt(m['x'])},{_fmt(m['y'])}")
    for b in rec.get("bboxes", []):
        lines.append(",".join([b["name"]] + [_fmt(c) for c in b["coords"]]))
    pbb = rec.get("pole_bbox_raw")
    if pbb:
        lines.append("# Pole bounding box (percentage coordinates)")
        lines.append("# " + ",".join(_fmt(c) for c in pbb))
    rbb = rec.get("ruler_bbox_raw")
    if rbb:
        lines.append("# Ruler bounding box (percentage coordinates)")
        lines.append("# " + ",".join(_fmt(c) for c in rbb))
    for prefix, tok in rec.get("hw", {}).items():
        lines.append(f"{prefix}_hw,{tok}")
    for prefix, tok in rec.get("ct", {}).items():
        lines.append(f"{prefix}_ct,{tok}")
    for prefix, k in rec.get("arm", {}).items():
        lines.append(f"{prefix}_arm,{_fmt(k)}")
    p = ppi(pid)
    if p is not None:
        lines.append(f"# PPI={_fmt(p)}")
    return lines


def loc_path(labels_dir, stem: str):
    """Shared label-path gate for ALL consumers (prep, eval, viz, review). Legacy:
    ``<labels_dir>/<stem>_location.txt``. When USE_PHOTO_ID_LAYOUT is on: the JSON-store
    reconstruction (transient cache) or None if the photo has no label. labels_dir is ignored
    when ENABLED (the stem resolves to its pid regardless of pole/midspan dir)."""
    from pathlib import Path
    if ENABLED:
        return label_path_for_stem(stem)
    return Path(labels_dir) / f"{stem}_location.txt"


def pid_for_disk_stem(disk_stem: str) -> Optional[str]:
    """Map a legacy disk filename stem ('<job>_<scid>_1_Main' or midspan stem) to its pid."""
    _ensure()
    return _DISK_TO_PID.get(disk_stem)


def materialize_label(pid: str):
    """Write the reconstructed location.txt for a pid into the transient cache; return its Path
    (None if the photo has no label). One file per pid -> safe under worker threads."""
    lines = location_lines(pid)
    if lines is None:
        return None
    TXT_CACHE.mkdir(parents=True, exist_ok=True)
    p = TXT_CACHE / f"{pid}_location.txt"
    if not p.exists():
        p.write_text("\n".join(lines))
    return p


def label_path_for_stem(stem: str):
    """Materialize the reconstructed location.txt for a photo stem and return its Path (None if the
    photo has no label). The stem may be EITHER a pid (a data/Photos/<pid>.jpg image, the new image
    source) OR a legacy disk stem (data/data_*/Photos/<disk>.jpg) — try pid first, then the
    disk->pid reverse index. One file per pid -> safe under prep's worker threads."""
    _ensure()
    if stem in _LABEL_BY_PID:                       # stem is already a pid
        return materialize_label(stem)
    pid = pid_for_disk_stem(stem)
    return materialize_label(pid) if pid is not None else None


def iter_location_files(role: Optional[str] = None):
    """Yield (location_txt Path, photo Path) for every LABELED photo (optionally a role), sourcing
    both from the photo_id store (materialized location.txt + data/Photos/<pid>.jpg). The layout-ON
    replacement for ``labels_dir.glob('*_location.txt')`` + ``photos_dir/<stem>.jpg`` in the viz
    helpers, so they read the new store instead of the legacy Labels dirs."""
    _ensure()
    for pid, r in sorted(_LABEL_BY_PID.items()):
        if role and r.get("role") != role:
            continue
        lp = materialize_label(pid)
        ph = photo_path(pid)
        if lp is not None and os.path.exists(ph):
            yield lp, Path(ph)


def ppi(pid: str, image_height_px: float = 3840.0) -> Optional[float]:
    """Recompute the eval-only inch-tolerance PPI from the stored anchors (identical to the
    dropped ``# PPI=`` line). Projection is the height model; this is scoring-only."""
    rec = label_for(pid)
    if rec is None:
        return None
    from src.height_calculations import calculate_ppi_from_measurements
    hm = {a[0]: {"percentX": a[1], "percentY": a[2]} for a in rec.get("anchors", [])}
    return calculate_ppi_from_measurements(hm, image_height_px)
