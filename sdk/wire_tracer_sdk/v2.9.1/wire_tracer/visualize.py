"""Span visualization (V2): POLE A | MIDSPAN | POLE B grid with labelled attachments + traces.

Labels show the insulator name plus the V2 hints — `xK` crossarm count and the predicted
`cable_type_hint` — e.g. "A2 Pin Insulator x2 [primary?]". The cable hint is rendered with a
trailing '?' to flag it as a non-authoritative model prediction (the user still assigns wire_type).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .constants import ATTACH_COLOR, GUY_COLOR, TRACE_COLORS


def _panel(img_bgr: Optional[np.ndarray], height: int, width: int):
    """Return (canvas, scale, x_offset, y_offset). Blank dark panel if img is None."""
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    if img_bgr is None:
        return canvas, 1.0, 0, 0
    h, w = img_bgr.shape[:2]
    scale = min(height / h, width / w)
    rim = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    yoff = (height - rim.shape[0]) // 2
    xoff = (width - rim.shape[1]) // 2
    canvas[yoff:yoff + rim.shape[0], xoff:xoff + rim.shape[1]] = rim
    return canvas, scale, xoff, yoff


def _att_label(a: Dict[str, Any]) -> str:
    """Insulator name + V2 hints: crossarm xK + predicted cable type (flagged with '?')."""
    label = f"{a['id']} {a['insulator_name']}"
    if a.get("role") == "crossarm" and a.get("wire_count", 1) > 1:
        label += f" x{a['wire_count']}"
    if a.get("cable_type_hint"):
        label += f" [{a['cable_type_hint']}?]"
    return label


def draw_span_grid(
    pole_a_bgr: np.ndarray,
    midspan_bgr: Optional[np.ndarray],
    pole_b_bgr: np.ndarray,
    result: Dict[str, Any],
    panel_h: int = 900,
    panel_w: int = 360,
) -> np.ndarray:
    """Render a 3-panel summary. Pole attachments are dotted + labelled (name + V2 hints);
    midspan wires are horizontal lines; traces are colour-matched across panels by midspan id."""
    pa, sa, xa, ya = _panel(pole_a_bgr, panel_h, panel_w)
    pm, sm, xm, ym = _panel(midspan_bgr, panel_h, panel_w)
    pb, sb, xb, yb = _panel(pole_b_bgr, panel_h, panel_w)

    def put(panel, pad_x, pad_y, scale, src_bgr, x_pct, y_pct, color, label=None):
        if src_bgr is None:
            return None
        h, w = src_bgr.shape[:2]
        px = int(pad_x + (x_pct / 100.0 * w) * scale)
        py = int(pad_y + (y_pct / 100.0 * h) * scale)
        cv2.circle(panel, (px, py), 5, color, -1)
        if label:
            cv2.putText(panel, label, (px + 7, py + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)
        return px, py

    for a in result["poles"]["A"]:
        c = GUY_COLOR if a["role"] == "guying" else ATTACH_COLOR
        put(pa, xa, ya, sa, pole_a_bgr, a["x"], a["y"], c, _att_label(a))
    for b in result["poles"]["B"]:
        c = GUY_COLOR if b["role"] == "guying" else ATTACH_COLOR
        put(pb, xb, yb, sb, pole_b_bgr, b["x"], b["y"], c, _att_label(b))
    for i, m in enumerate(result["midspan"]):
        if midspan_bgr is None:
            continue
        py = int(ym + (m["y"] / 100.0 * midspan_bgr.shape[0]) * sm)
        col = TRACE_COLORS[i % len(TRACE_COLORS)]
        cv2.line(pm, (0, py), (panel_w, py), col, 2)
        cv2.putText(pm, m["id"], (4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    grid = np.concatenate([pa, pm, pb], axis=1)
    titles = [("POLE A", 0), ("MIDSPAN", panel_w), ("POLE B", 2 * panel_w)]
    for txt, x0 in titles:
        cv2.putText(grid, txt, (x0 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
