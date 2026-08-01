"""
A<->B-coupled span matcher with a LEARNED per-edge cost — pure-numpy port of
src/wire_tracing_match.py (match_span path; the GT evaluation harness is omitted).

V2 vs V1:
  * the per-edge BASE cost is the learned NumpyEdgeCostModel evaluated on the shared 21-feature
    edge vector (compute_edge_features), NOT a hand-tuned w_y·dy + deadend term.
  * a finer cable-type A<->B coupling (w_couple_class) is added on top of the tier/chain couplings.
  * comm_isolation defaults OFF (the learned cost already weights tier softly).

Two bipartite assignments per span (midspan<->poleA, midspan<->poleB) refined by A<->B coupling
(alternating ICM), with the non-crossing monotonic DP. Uses the SDK's pure-numpy
linear_sum_assignment (no scipy). The matcher consumes the SAME point dicts the training repo
builds (to_matcher_side): {x, y, kind, multiplicity, traces:[{insulator_spec, cable_type}],
wire_class, conf, i}.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .constants import (
    EDGE_FEATURE_NAMES,
    MATCH_COMM_ISOLATION,
    MATCH_COUPLE_ITERS,
    MATCH_DUST,
    MATCH_MONOTONIC,
    MATCH_W_COUPLE_CHAIN,
    MATCH_W_COUPLE_CLASS,
    MATCH_W_COUPLE_TIER,
    MATCH_W_DEADEND,
    MATCH_W_MID_TIER3_BONUS,
    MATCH_W_X,
    MATCH_W_Y,
    WIRE_HW_DEADEND_TOKENS,
    WIRE_HW_TO_TIER,
)
from .numpy_ops import linear_sum_assignment

_BIG = 1e6


# --- tier helpers (mirror src/config.py, operating on canonical insulator_spec strings) ---
def _hardware_token_for_spec(spec: Optional[str]) -> Optional[str]:
    if not spec:
        return None
    s = str(spec).strip().lower()
    if "spool" in s:
        return "spool"
    if "deadend" in s or "dead end" in s or "dead-end" in s:
        return "deadend"
    if "three bolt" in s or "three-bolt" in s or "3 bolt" in s:
        return "three_bolt"
    if "single bolt" in s or "single-bolt" in s or "1 bolt" in s:
        return "three_bolt"   # single_bolt folds into three_bolt
    if "pin" in s:
        return "pin"
    if "post" in s:
        return "post"
    if "davit" in s:
        return "davit"
    return None


def hardware_tier_for_spec(spec: Optional[str]) -> Optional[str]:
    return WIRE_HW_TO_TIER.get(_hardware_token_for_spec(spec))


def spec_is_deadend(spec: Optional[str]) -> bool:
    return _hardware_token_for_spec(spec) in WIRE_HW_DEADEND_TOKENS


@dataclass
class MatchConfig:
    w_y: float = MATCH_W_Y
    w_x: float = MATCH_W_X
    dust: float = MATCH_DUST
    w_deadend: float = MATCH_W_DEADEND
    w_couple_tier: float = MATCH_W_COUPLE_TIER
    w_couple_chain: float = MATCH_W_COUPLE_CHAIN
    w_couple_class: float = MATCH_W_COUPLE_CLASS
    couple_iters: int = MATCH_COUPLE_ITERS
    monotonic: bool = MATCH_MONOTONIC
    comm_isolation: bool = MATCH_COMM_ISOLATION
    multiplicity_known: bool = True
    # v2.9 MIDSPAN<->POLE tier3 agreement BONUS (subtracted from tier-agreeing edges so
    # they beat the dustbin — the rescue mechanism). Both-tiers-known-only; 0.0 = off.
    w_mid_tier3_bonus: float = MATCH_W_MID_TIER3_BONUS
    # Learned per-edge base cost (NumpyEdgeCostModel). None => legacy geometric cost (w_y·dy + ...).
    edge_model: object = field(default=None, compare=False, repr=False)


# --------------------------------------------------------------------------- #
# Per-edge features (shared with the learned-cost trainer; column order is the contract)
# --------------------------------------------------------------------------- #
def _tier_flags(tier_set) -> tuple:
    p = 1.0 if "power" in tier_set else 0.0
    s = 1.0 if "secondary" in tier_set else 0.0
    c = 1.0 if "comm" in tier_set else 0.0
    n = 1.0 if not (p or s or c) else 0.0
    return p, s, c, n


def _neighbor_gaps(vals: np.ndarray):
    """Directional height gap from each item to its nearest higher / lower neighbor (1.0 at the
    extremes) — the isolation signal a graph-attention layer would otherwise learn."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ga = np.ones(n)
    gb = np.ones(n)
    for pos, i in enumerate(order):
        if pos + 1 < n:
            ga[i] = abs(vals[order[pos + 1]] - vals[i])
        if pos - 1 >= 0:
            gb[i] = abs(vals[i] - vals[order[pos - 1]])
    return ga, gb


def compute_edge_features(mids: List[Dict], slots: List[Dict],
                          nym: List[float], nys: List[float]) -> np.ndarray:
    """(R, C, F) edge-feature tensor for the learned cost; F == len(EDGE_FEATURE_NAMES)."""
    R, C, F = len(mids), len(slots), len(EDGE_FEATURE_NAMES)
    feats = np.zeros((R, C, F), dtype=np.float32)
    if R == 0 or C == 0:
        return feats
    nym = np.asarray(nym, dtype=float)
    nys = np.asarray(nys, dtype=float)
    rrank = np.empty(R)
    for k, r in enumerate(sorted(range(R), key=lambda i: nym[i])):
        rrank[r] = k / max(1, R - 1)
    srank = np.empty(C)
    for k, c in enumerate(sorted(range(C), key=lambda i: nys[i])):
        srank[c] = k / max(1, C - 1)
    inv_R, inv_C = 1.0 / R, 1.0 / C
    mx = [(m.get("x") or 0.0) / 100.0 for m in mids]
    mconf = [(1.0 if m.get("conf") is None else float(m["conf"])) for m in mids]
    sx = [(s.get("x") or 0.0) / 100.0 for s in slots]
    sconf = [(1.0 if s.get("conf") is None else float(s["conf"])) for s in slots]
    smult = [float(s.get("mult", 1) or 1) for s in slots]
    tfl = [_tier_flags(s.get("tier") or set()) for s in slots]
    sga, sgb = _neighbor_gaps(nys)
    mga, mgb = _neighbor_gaps(nym)
    nearest_slot = [int(np.argmin(np.abs(nys - nym[r]))) for r in range(R)]
    nearest_row = [int(np.argmin(np.abs(nym - nys[c]))) for c in range(C)]
    tier_rank = np.zeros(C)
    groups = defaultdict(list)
    for c in range(C):
        groups[tuple(sorted(slots[c].get("tier") or []))].append(c)
    for cs in groups.values():
        for k, c in enumerate(sorted(cs, key=lambda i: nys[i])):
            tier_rank[c] = k / max(1, len(cs) - 1)
    for r in range(R):
        for c in range(C):
            feats[r, c] = (abs(nym[r] - nys[c]), abs(mx[r] - sx[c]), abs(rrank[r] - srank[c]),
                           nym[r], nys[c], mconf[r], sconf[c], smult[c],
                           tfl[c][0], tfl[c][1], tfl[c][2], tfl[c][3], inv_R, inv_C,
                           sga[c], sgb[c], mga[r], mgb[r],
                           1.0 if nearest_slot[r] == c else 0.0,
                           1.0 if nearest_row[c] == r else 0.0, tier_rank[c])
    return feats


def _norm_y(ys: List[Optional[float]]) -> List[float]:
    vals = [y for y in ys if y is not None]
    if not vals:
        return [0.5] * len(ys)
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-6:
        return [0.5] * len(ys)
    return [((y - lo) / (hi - lo) if y is not None else 0.5) for y in ys]


def _build_slots(poles: List[Dict], cfg: MatchConfig) -> List[Dict]:
    """Matchable pole slots (exclude guying; expand each point by multiplicity). Each slot carries
    the cable-type set, hardware coarse-tier set, deadend flag, wire_class, confidence and source
    multiplicity — the same features the learned-cost trainer saw."""
    slots: List[Dict] = []
    for p in poles:
        if p["kind"] == "guying" or p["x"] is None or p["y"] is None:
            continue
        mult = p["multiplicity"] if cfg.multiplicity_known else 1
        cable_set = {t.get("cable_type") for t in p["traces"]}
        tier_set = {hardware_tier_for_spec(t.get("insulator_spec")) for t in p["traces"]}
        tier_set.discard(None)
        is_deadend = any(spec_is_deadend(t.get("insulator_spec")) for t in p["traces"])
        for _ in range(mult):
            slots.append({"i": p["i"], "cable": cable_set, "tier": tier_set, "deadend": is_deadend,
                          "wclass": p.get("wire_class"), "conf": p.get("conf"), "mult": mult,
                          "y": p["y"], "x": p["x"],
                          "tier3": p.get("tier3")})   # v2.9: fine-class 3-tier (bare/multiplex/comm)
    return slots


def _monotonic_assign(cost, slots, nym, nys, R, C):
    """Bottom-up NON-CROSSING min-cost matching (mirror of src _monotonic_assign)."""
    if R == 0:
        return []
    NEG = _BIG / 2
    rorder = sorted(range(R), key=lambda r: nym[r])
    sorder = sorted(range(C), key=lambda c: nys[c]) if C else []
    dp = np.zeros((R + 1, C + 1), dtype=float)
    bk = [[0] * (C + 1) for _ in range(R + 1)]
    for i in range(1, R + 1):
        dp[i][0] = dp[i - 1][0] + cost[rorder[i - 1], C + rorder[i - 1]]
    for i in range(1, R + 1):
        r = rorder[i - 1]
        dustc = cost[r, C + r]
        for j in range(1, C + 1):
            best = dp[i - 1][j] + dustc
            b = 0
            if dp[i][j - 1] < best:
                best, b = dp[i][j - 1], 1
            mc = cost[r, sorder[j - 1]]
            if mc < NEG and dp[i - 1][j - 1] + mc < best:
                best, b = dp[i - 1][j - 1] + mc, 2
            dp[i][j] = best
            bk[i][j] = b
    pred: List[Optional[int]] = [None] * R
    i, j = R, C
    while i > 0 and j > 0:
        b = bk[i][j]
        if b == 2:
            pred[rorder[i - 1]] = slots[sorder[j - 1]]["i"]
            i -= 1
            j -= 1
        elif b == 0:
            i -= 1
        else:
            j -= 1
    return pred


def _match_side(mids: List[Dict], poles: List[Dict], cfg: MatchConfig,
                couple: Optional[List[Dict]] = None,
                extra: Optional[Dict[int, float]] = None) -> List[Optional[int]]:
    R = len(mids)
    if R == 0:
        return []
    slots = _build_slots(poles, cfg)
    C = len(slots)

    nym = _norm_y([m.get("y") for m in mids])
    nys = _norm_y([s["y"] for s in slots]) if C else []

    # learned per-edge base cost (replaces the geometric+intrinsic term when cfg.edge_model set)
    learned = None
    if cfg.edge_model is not None and R and C:
        feats = compute_edge_features(mids, slots, nym, nys).reshape(R * C, len(EDGE_FEATURE_NAMES))
        learned = np.asarray(cfg.edge_model.edge_costs(feats), dtype=float).reshape(R, C)

    cost = np.full((R, C + R), _BIG, dtype=float)
    for r in range(R):
        dust_cost = cfg.dust
        if couple is not None and cfg.w_couple_chain and couple[r]["matched"]:
            dust_cost += cfg.w_couple_chain
        cost[r, C + r] = dust_cost
        if mids[r]["y"] is None:
            continue
        mx = (mids[r]["x"] or 0.0) / 100.0
        m_tier3 = mids[r].get("tier3")   # v2.9 midspan patch-classifier tier (None = no signal)
        o_tier = couple[r]["tier"] if couple is not None else None
        o_wclass = couple[r].get("wclass") if couple is not None else None
        for c in range(C):
            s = slots[c]
            dy = abs(nym[r] - nys[c])
            dx = abs(mx - (s["x"] or 0.0) / 100.0)
            dd = cfg.w_deadend if s["deadend"] else 0.0
            # A<->B tier coupling: penalise slots whose tier is disjoint from the tier the OTHER
            # pole assigned to this midspan (both tiers from hardware, no midspan class).
            ct = (cfg.w_couple_tier
                  if (cfg.w_couple_tier and o_tier and s["tier"] and o_tier.isdisjoint(s["tier"]))
                  else 0.0)
            # finer wire-class A<->B coupling: penalise a slot whose wire_class disagrees with the
            # class the OTHER pole assigned to this midspan (primary/secondary/neutral/comm).
            cw = (cfg.w_couple_class
                  if (cfg.w_couple_class and o_wclass and s["wclass"] and o_wclass != s["wclass"])
                  else 0.0)
            # v2.9 MIDSPAN<->POLE tier3 agreement bonus (mirror src/wire_tracing_match):
            # a tier-agreeing edge beats the dustbin. Both-known-only.
            mt3 = 0.0
            if cfg.w_mid_tier3_bonus and m_tier3 and s.get("tier3") and m_tier3 == s["tier3"]:
                mt3 = -cfg.w_mid_tier3_bonus
            if learned is not None:
                # learned base cost; deadend + geometry are folded into the features, the dynamic
                # A<->B coupling stays additive (it depends on the other side's current state).
                cval = learned[r, c] + ct + cw + mt3
            else:
                cval = cfg.w_y * dy + cfg.w_x * dx + dd + ct + cw + mt3
            # HARD three_bolt(comm) isolation (default OFF with the learned cost).
            if cfg.comm_isolation and o_tier and s["tier"] and \
                    (("comm" in s["tier"]) != ("comm" in o_tier)):
                cval = _BIG
            # v2.9.1: externally supplied per-pole-point additive cost (sub-gate penalty)
            if extra is not None:
                cval += extra.get(s["i"], 0.0)
            cost[r, c] = cval

    if cfg.monotonic:
        return _monotonic_assign(cost, slots, nym, nys, R, C)

    rows, cols = linear_sum_assignment(cost)
    pred: List[Optional[int]] = [None] * R
    for r, c in zip(rows, cols):
        pred[r] = slots[c]["i"] if c < C else None
    return pred


def _coupling_from(pred: List[Optional[int]], poles: List[Dict]) -> List[Dict]:
    """Per-midspan-row view of one side's assignment for the other side to couple against."""
    by_idx = {p["i"]: p for p in poles}
    out = []
    for pi in pred:
        if pi is None:
            out.append({"matched": False, "tier": set(), "wclass": None})
        else:
            p = by_idx[pi]
            tset = {hardware_tier_for_spec(t.get("insulator_spec")) for t in p["traces"]}
            tset.discard(None)
            out.append({"matched": True, "tier": tset, "wclass": p.get("wire_class")})
    return out


def match_span(sample: Dict, cfg: MatchConfig,
               extra: Optional[Dict[str, Dict[int, float]]] = None) -> Dict[str, List[Optional[int]]]:
    """Return {"A": [pole_idx|None per midspan], "B": [...]} (mirror of src.match_span).

    extra: optional {"A": {pole_i: cost}, "B": {...}} additive per-pole-point edge cost
    (v2.9.1 sub-gate admission penalty). None = byte-identical legacy."""
    M, A, B = sample["sides"]["M"], sample["sides"]["A"], sample["sides"]["B"]
    exA = extra.get("A") if extra else None
    exB = extra.get("B") if extra else None
    predA = _match_side(M, A, cfg, extra=exA)
    predB = _match_side(M, B, cfg, extra=exB)
    if cfg.w_couple_tier <= 0 and cfg.w_couple_chain <= 0 and cfg.w_couple_class <= 0 \
            and not cfg.comm_isolation:
        return {"A": predA, "B": predB}
    for _ in range(cfg.couple_iters):
        newA = _match_side(M, A, cfg, couple=_coupling_from(predB, B), extra=exA)
        newB = _match_side(M, B, cfg, couple=_coupling_from(newA, A), extra=exB)
        if newA == predA and newB == predB:
            predA, predB = newA, newB
            break
        predA, predB = newA, newB
    return {"A": predA, "B": predB}
