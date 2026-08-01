"""Pole-level down-guy counts from Katapult anchor inventory (eval-only).

For each pole, sum comma-separated ``sizes_of_attached_dn_guys`` on connected anchors
(``connections`` with ``button == "anchor"``), resolving pole vs anchor by ``node_type``.

Excluded anchors (out of scope):
  * ``node_type`` is ``new anchor`` (proposed install)
  * anchor linked to a TelecomCo ``down_guy`` trace on the pole main photo
    (``guying[].anchor_id`` + ``traces.trace_data[_trace].company``)

Jobs with no anchor metadata are omitted from the anchor-eval index entirely.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Set, Tuple

from src.config import WIRE_TRACING_JOB_SOURCE_DIR
from src.wire_tracing import _attr_value


@dataclass(frozen=True)
class PoleDownGuyExpectation:
    """Per pole main-photo down-guy eval target."""

    mode: str          # "anchor_count" | "zero" | "label_fallback"
    count: int         # anchor_count / zero only
    job: str
    pole_node_id: str
    anchor_count: int  # non-excluded anchors considered


def _node_type(nodes: Dict, nid: str) -> str:
    return str(_attr_value((nodes.get(nid) or {}).get("attributes"), "node_type") or "").strip().lower()


def node_role(nodes: Dict, nid: str) -> str:
    """``pole`` | ``anchor`` | ``other`` from Katapult node_type."""
    nt = _node_type(nodes, nid)
    if "anchor" in nt:
        return "anchor"
    if "pole" in nt:
        return "pole"
    return "other"


def parse_dn_guy_sizes(raw: Optional[str]) -> Optional[int]:
    """Comma-separated guy sizes -> count. ``None`` if field missing/blank."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return len(parts) if parts else None


def pole_main_photo_id(node: Dict) -> Optional[str]:
    for pid, pm in (node.get("photos") or {}).items():
        assoc = pm.get("association") if isinstance(pm, dict) else pm
        if assoc == "main":
            return str(pid)
    return None


def metro_anchor_ids_for_photo(main_photo_id: str, photos: Dict, traces: Dict) -> Set[str]:
    """Anchor ids with a TelecomCo down_guy guying marker on ``main_photo_id``."""
    out: Set[str] = set()
    pfd = (photos.get(main_photo_id) or {}).get("photofirst_data") or {}
    for g in (pfd.get("guying") or {}).values():
        aid = g.get("anchor_id")
        if not aid:
            continue
        tid = g.get("_trace")
        t = (traces or {}).get(tid) or {}
        if t.get("_trace_type") != "down_guy":
            continue
        company = str(t.get("company") or "").lower()
        if "telecomco" in company:
            out.add(str(aid))
    return out


def anchor_is_excluded(anchor_nid: str, nodes: Dict, metro_anchors: Set[str]) -> bool:
    if anchor_nid in metro_anchors:
        return True
    return _node_type(nodes, anchor_nid) == "new anchor"


def expected_down_guy_for_pole(
    pole_nid: str,
    anchor_nids: list,
    nodes: Dict,
    photos: Dict,
    traces: Dict,
) -> PoleDownGuyExpectation:
    """Compute down-guy expectation for one pole (job stem filled by caller)."""
    main_pid = pole_main_photo_id(nodes[pole_nid])
    metro = metro_anchor_ids_for_photo(main_pid, photos, traces) if main_pid else set()

    usable = []
    total = 0
    for aid in anchor_nids:
        if anchor_is_excluded(aid, nodes, metro):
            continue
        usable.append(aid)
        raw = _attr_value((nodes.get(aid) or {}).get("attributes"), "sizes_of_attached_dn_guys")
        n = parse_dn_guy_sizes(raw)
        if n is None:
            return PoleDownGuyExpectation("label_fallback", 0, "", pole_nid, len(usable))
        total += n

    if not usable:
        return PoleDownGuyExpectation("zero", 0, "", pole_nid, 0)

    return PoleDownGuyExpectation("anchor_count", total, "", pole_nid, len(usable))


def job_has_anchor_metadata(job: Dict) -> bool:
    """Job qualifies for anchor down-guy eval when it has anchor connections + size fields."""
    if not any(c.get("button") == "anchor" for c in (job.get("connections") or {}).values()):
        return False
    nodes = job.get("nodes") or {}
    for n in nodes.values():
        attrs = n.get("attributes") or {}
        nt = str(_attr_value(attrs, "node_type") or "").lower()
        if "anchor" not in nt:
            continue
        if "sizes_of_attached_dn_guys" in attrs:
            return True
    return False


def iter_pole_expectations(job: Dict, job_stem: str) -> Iterator[Tuple[str, PoleDownGuyExpectation]]:
    """Yield ``(main_photo_id, expectation)`` for every pole node in a job."""
    nodes = job.get("nodes") or {}
    photos = job.get("photos") or {}
    traces = (job.get("traces") or {}).get("trace_data") or {}
    conns = job.get("connections") or {}

    pole_anchors: Dict[str, list] = {}
    for c in conns.values():
        if c.get("button") != "anchor":
            continue
        n1, n2 = c.get("node_id_1"), c.get("node_id_2")
        k1, k2 = node_role(nodes, n1), node_role(nodes, n2)
        if k1 == "pole" and k2 == "anchor":
            pole_anchors.setdefault(n1, []).append(n2)
        elif k2 == "pole" and k1 == "anchor":
            pole_anchors.setdefault(n2, []).append(n1)

    seen_poles = set()
    for pole_nid, aids in pole_anchors.items():
        main_pid = pole_main_photo_id(nodes.get(pole_nid) or {})
        if not main_pid:
            continue
        exp = expected_down_guy_for_pole(pole_nid, aids, nodes, photos, traces)
        seen_poles.add(pole_nid)
        yield main_pid, PoleDownGuyExpectation(
            exp.mode, exp.count, job_stem, pole_nid, exp.anchor_count)

    # Poles with no anchor connections -> zero down guys
    for pole_nid, node in nodes.items():
        if node_role(nodes, pole_nid) != "pole" or pole_nid in seen_poles:
            continue
        main_pid = pole_main_photo_id(node)
        if not main_pid:
            continue
        yield main_pid, PoleDownGuyExpectation("zero", 0, job_stem, pole_nid, 0)


def build_photo_expectations(
    jobs_dir: Optional[Path] = None,
) -> Tuple[Dict[str, PoleDownGuyExpectation], Set[str]]:
    """``photo_id -> expectation`` and set of qualifying job stems."""
    jobs_dir = jobs_dir or WIRE_TRACING_JOB_SOURCE_DIR
    out: Dict[str, PoleDownGuyExpectation] = {}
    qualifying: Set[str] = set()
    for jf in sorted(Path(jobs_dir).glob("*.json")):
        if jf.stat().st_size < 100:
            continue
        try:
            job = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not job_has_anchor_metadata(job):
            continue
        qualifying.add(jf.stem)
        for pid, exp in iter_pole_expectations(job, jf.stem):
            out[pid] = exp
    return out, qualifying
