"""
Unified joint-class decode — pure-numpy port of src/wire_tracing_e2e._unified_point
(+ src/config.decode_unified_class).

The single `unified_pole_detection` class carries every signal the tracer wants from the
pole side, so V2 needs only ONE detector pass (vs V1's wire ∪ wire-hw union):

  * hw_token   -> matcher tier (TOKEN_TO_SPEC -> hardware_tier_for_spec) + deadend prior.
                  crossarm ('arm') and hardware-unread 'primary' map to a power-tier proxy ('pin').
  * wire_class -> the finer w_couple_class A<->B coupling + the `cable_type_hint` output
                  (primary / secondary / neutral / comm).
  * crossarm_k -> the wire-count K the model predicts directly (arm2/arm3/arm4plus) — surfaced
                  as the non-authoritative `crossarm_k` hint (the matcher still recovers the
                  actual count from midspan multiplicity, mirroring the product).
"""

from __future__ import annotations

from typing import Dict, Optional

from .constants import UNIFIED_CLASS_TO_TIER3, UNIFIED_POLE_DECODE, UNIFIED_WIRE_CLASS


def decode_unified_class(name: str):
    """(hw_token, cable_type, crossarm_k, display) for a joint class name, or None."""
    return UNIFIED_POLE_DECODE.get(name)


def unified_point(name: str, xp: float, yp: float, conf: float,
                  box_h_pct: Optional[float] = None) -> Optional[Dict]:
    """Decode a unified class into a matcher pole-point dict, or None for an unknown class.

    Returns {x, y, kind, hw_token, conf, wire_class, pred_mult, display}, where:
      * kind='guying' for guy/down_guy (the matcher excludes them as span endpoints), else 'insulator'.
      * hw_token is the coarse hardware token used for the tier/deadend lookups.
      * wire_class is the coarse electrical class (primary/secondary/neutral/comm) for coupling + hint.
      * pred_mult is the model-predicted crossarm K (>=1); surfaced as the crossarm_k hint.
    """
    dec = decode_unified_class(name)
    if dec is None:
        return None
    hw, ct, k, display = dec
    if name == "down_guy":
        # box_h_pct (the synthetic 1 ft label box height, % of image H) scales the v2.5
        # down_guy dedup band to inches without needing a ruler fit (pipeline._select_down_guys).
        return {"x": xp, "y": yp, "kind": "guying", "hw_token": "down_guy", "conf": conf,
                "wire_class": None, "pred_mult": 1, "display": display, "box_h_pct": box_h_pct}
    if name == "guy":
        return {"x": xp, "y": yp, "kind": "guying", "hw_token": "guy", "conf": conf,
                "wire_class": None, "pred_mult": 1, "display": display}
    if hw in ("pin", "post", "davit", "deadend", "spool", "three_bolt"):
        token = hw
    elif hw == "arm" or ct == "primary":
        token = "pin"            # power-tier proxy (crossarm bundle / hardware-unread primary)
    else:
        token = None             # 'unspecified': recognized conductor, tier unknown
    return {"x": xp, "y": yp, "kind": "insulator", "hw_token": token, "conf": conf,
            "wire_class": UNIFIED_WIRE_CLASS.get(ct), "pred_mult": max(1, k or 1),
            # v2.9: 3-class midspan tier from the FINE class name (open_secondary = bare)
            "tier3": UNIFIED_CLASS_TO_TIER3.get(name),
            # fine cable class straight from the joint decode (catv/telco/fiber kept
            # distinct, unlike wire_class which merges them to 'comm') — display-only
            "cable_fine": ct,
            "display": display}
