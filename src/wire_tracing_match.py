#!/usr/bin/env python3
"""
Stage-1b: Hungarian baseline matcher + evaluation harness for wire tracing.

Consumes the Stage-0 dataset (``datasets/wire_tracing_dataset/spans.jsonl``) and, for
each span, predicts which midspan wire connects to which pole-A and pole-B attachment
POINT — using only features a detector could plausibly produce (position, structural
kind/multiplicity, and optionally a class). It never reads ``trace_id`` or the
``group_ambiguous`` flag while matching; those are used only by the scorer.

Method (anchored on midspan, as designed):
  * Two bipartite assignments per span: midspan↔poleA and midspan↔poleB. They are solved
    independently UNLESS A↔B coupling is on (see below), in which case they are refined
    against each other.
  * Each pole POINT is expanded into ``multiplicity`` slots so a crossarm (one coincident
    point, K traces) can absorb up to K midspan wires.
  * Guying (down-guy) pole points are never matchable (auto-dustbin).
  * Each midspan row also gets a dustbin column (threshold cost), so orphan midspan
    markers and one-sided wires can go unmatched.
  * Cost = w_y·|Δy_norm| + w_x·|Δx| (+ w_class·class-disagreement) (+ w_deadend on
    deadend pole slots) (+ A↔B coupling terms). y is min-max normalized per side to
    cancel the systematic pole↔midspan perspective offset.

Hardware features (from the Stage-1a hardware head) enter as:
  * class_signal="hw_tier" — the pole point's hardware-derived coarse tier
    (insulator_spec → token → WIRE_HW_TO_TIER) is compared to the midspan marker's
    cable_type tier. This is what the hardware DETECTOR can deliver on the pole side
    (vs exact cable_type, which it cannot). Midspan tier still comes from GT cable_type,
    so hw-tier is a semi-oracle upper bound on the coarse-tier lever.
  * w_deadend>0 — a per-pole-point dustbin prior: deadend hardware is power-TERMINATING
    (61% pole-only), so matching a midspan wire onto a deadend slot is penalised, pushing
    deadended attachments toward the pole-dustbin. Needs NO midspan class → usable today
    on top of pure geometry.
  * down-guy — guying pole points are already auto-dustbined; the hardware head's
    down_guy class is the realizable source of that "never crosses a span" signal.

A↔B coupling (FULLY realizable end-to-end — both pole tiers from the hardware head, NO
midspan class needed; this is the realizable replacement for hw_tier's semi-oracle
midspan side). The A-attachment and B-attachment of one midspan wire are the same trace:
  * w_couple_tier>0 — penalise an A/B point pair (assigned to the same midspan) whose
    hardware tiers disagree. Transfers a confident tier read at one pole to disambiguate
    a geometrically-ambiguous wire at the other.
  * w_couple_chain>0 — penalise one-sided matches (matched at one pole, dustbin at the
    other). ~99% of midspan wires reach BOTH poles, so this fixes split decisions and
    sharpens orphan precision.
The two sides are then solved by alternating (ICM) refinement until they stabilise.

Three class regimes are reported:
  * geometry-only       — no class (generic detector).
  * +hw-tier            — hardware coarse tier (the realizable lever).
  * +oracle-cable_type  — exact GT cable_type agreement; the absolute UPPER BOUND.

Scoring is POINT-level and group-aware:
  * strict per-trace accuracy = accuracy on CLEAN (non-crossarm) chains, where one pole
    point == one wire.
  * crossarm-group accuracy   = accuracy on group-ambiguous chains, where "correct" means
    mapping to the right arm group (phase within the arm is not recoverable from images).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import hardware_tier_for_spec, spec_is_deadend, tier_for_cable_type


@dataclass
class MatchConfig:
    w_y: float = 1.0
    w_x: float = 0.2
    w_class: float = 0.5         # clean cable_type can afford >dust; noisy hw_tier wants <dust (~0.15)
    w_deadend: float = 0.0       # penalty on matching a deadend pole slot (hw dustbin prior)
    dust: float = 0.18           # dustbin threshold in normalized-cost units
    # CONF-WEIGHTED DUSTBIN (opt-in; gamma=0 => uniform, byte-identical legacy): scale a midspan
    # row's dustbin threshold by its detection-peak conf -> faint peaks are cheaper to discard.
    # dust_row = dust * (m_conf ** dust_conf_gamma). conf in ~[0.6,1].
    # REFUTED (2026-06-22, honest NEOM caches, sweep_conf_dustbin.py): best (dust=1.2,gamma=0.5)
    # = +0.13pp (3 chains, noise); gamma>0 monotonically HURTS. m_conf is already an edge-model
    # feature (redundant) and gamma>0 discards real low-conf wires (strip op-point P0.88/R0.87).
    dust_conf_gamma: float = 0.0
    norm: str = "minmax"         # "minmax" | "raw"
    # Vertical coordinate the matcher compares across sides:
    #   "minmax_image"   — percentY, min-max normalized PER SIDE (default; legacy behavior).
    #   "metric_height"  — feet above local ground (ruler perspective curve), COMMON-normalized
    #                      across A∪M∪B (fixes perspective + per-side anchoring; still has the
    #                      pole-vs-midspan ground offset).
    #   "abs_elevation"  — USGS-datum absolute elevation (ground+height), COMMON-normalized
    #                      (also removes the ground offset). Needs spans_metric.jsonl fields.
    coord: str = "minmax_image"
    sag_ft: float = 0.0          # catenary sag added to midspan height before normalizing (metric/abs)
    coord_norm: str = "common"   # metric/abs normalization: "common" (shared A∪M∪B datum) | "perside"
                                 # ("perside" = rank within each side, a diagnostic: isolates the
                                 #  metric transform from the absolute-datum decision)
    class_signal: str = "none"   # "none" | "cable_type" (oracle) | "hw_tier" (hardware proxy)
    multiplicity_known: bool = True   # assume arms are detected + insulators counted
    # A↔B coupling (fully realizable — both ends from the pole hardware head, NO midspan
    # class): the A-attachment and B-attachment of one midspan wire are the same physical
    # trace, so they should (1) share a hardware tier and (2) both match or both dustbin.
    w_couple_tier: float = 0.0   # penalty when the A/B points of one midspan disagree on tier
    w_couple_chain: float = 0.0  # penalty for a one-sided match (matched at one pole, dustbin at the other)
    # FINER A↔B coupling on the predicted WIRE CLASS (primary/secondary/neutral/comm from the
    # attachment_detection model) — splits power into primary vs neutral vs secondary, which the
    # coarse hw tier (w_couple_tier) collapses. A wire keeps its electrical class along the span,
    # so the A-attachment and B-attachment of one midspan should share a wire_class. Soft penalty
    # (a misread class should bias, not forbid — comm_isolation showed hard class constraints hurt).
    w_couple_class: float = 0.0  # penalty when the A/B points of one midspan disagree on wire_class
    # MIDSPAN<->POLE tier coupling (Experiment 2): the midspan strip predicts a 3-class tier
    # (bare/multiplex/comm) that the pole side lacks — bare (primary/neutral single conductor) vs
    # multiplex (secondary bundle) is visually clear at midspan but ambiguous at the pole (neutral &
    # secondary share the spool). Penalize matching a midspan wire to a pole slot whose tier3
    # (from wire_class) disagrees with the wire's predicted tier3. Soft, both-known-only. Applies in
    # BOTH the learned and hand-tuned cost paths. 0.0 => off (byte-identical legacy).
    w_mid_tier3: float = 0.0
    mid_tier3_hard: bool = False  # HARD variant: forbid cross-tier3 midspan<->pole matches entirely
    # BONUS variant: SUBTRACT from tier3-AGREEING edges (both known). Unlike the penalty — which
    # makes the dustbin RELATIVELY cheaper — a bonus lowers a correct edge BELOW the dustbin, the
    # mechanism that can rescue wires the matcher currently leaves unmatched. 0.0 => off.
    w_mid_tier3_bonus: float = 0.0
    # CATENARY-SAG plausibility (a coupled A+M+B physical constraint, needs elev_ft on every point —
    # absolute elevation = USGS ground + PPI height). A real span wire hangs a few feet BELOW the
    # straight chord between its two attachments, so sag = (elev_A + elev_B)/2 - elev_M sits in a
    # tight physical band (~[-3, 12] ft); a wrong pairing gives implausible sag (midspan above the
    # chord, or sag >> a real catenary). Penalize the EXCESS outside [sag_lo, sag_hi] on the coupled
    # other-side elevation. Distinct from the refuted absolute-elevation COORDINATE: this is a
    # 3-point consistency check, not a per-side height axis. w_sag=0 => off (no behavior change).
    w_sag: float = 0.0           # penalty per foot of sag outside the plausible band (0 = off)
    sag_lo: float = -3.0         # plausible catenary sag band (ft): (elev_A+elev_B)/2 - elev_M
    sag_hi: float = 12.0
    couple_iters: int = 4        # max alternating (ICM) passes when coupling is on
    # Domain rules (matcher constraints):
    #   monotonic — NON-CROSSING / order-preserving matching: span wires don't cross, so the k-th
    #     midspan wire from the bottom maps to the k-th matchable pole point from the bottom. Solved
    #     by a bottom-up monotonic min-cost DP (replaces the per-side Hungarian).
    #   comm_isolation — HARD three_bolt(comm) isolation: a wire on a three_bolt at one pole may
    #     only connect (via the A↔B coupling) to a three_bolt at the other; three_bolt↔non-three_bolt
    #     is forbidden. Needs coupling on (uses the other side's assigned tier).
    monotonic: bool = False
    comm_isolation: bool = False
    # FREED-SLOT RECOVERY (monotonic only): after the non-crossing DP, bind a dusted midspan row
    # to an empty same-point slot when that match is strictly cheaper than the row's dust (the
    # pole-top-pin-over-crossarm case the non-crossing constraint forbids). Port of the sdk v3
    # matcher._fill_residual. Opt-in; no effect without monotonic. None/False ⇒ legacy.
    fill_residual: bool = False
    # LEARNED edge cost (probe / GNN lever). When set, the per-edge geometric+intrinsic cost
    # (w_y·dy + w_x·dx + class + deadend) is REPLACED by edge_model.edge_costs(feats) — a model
    # that scores each (midspan, pole-slot) edge from the shared EDGE_FEATURE_NAMES vector. The
    # A↔B coupling terms (w_couple_*) and the dustbin stay additive on top, and the solver
    # (monotonic DP / Hungarian) is unchanged. None ⇒ legacy hand-tuned cost (no behavior change).
    edge_model: object = field(default=None, compare=False, repr=False)

    def label(self) -> str:
        name = {"none": "geometry-only", "cable_type": "+oracle-cable_type",
                "hw_tier": "+hw-tier"}.get(self.class_signal, self.class_signal)
        if self.w_couple_tier or self.w_couple_chain:
            name = "A↔B-coupled" if self.class_signal == "none" else name + " +A↔B"
        extra = f"dust={self.dust}, norm={self.norm}"
        if self.class_signal != "none":
            extra += f", w_class={self.w_class}"
        if self.w_deadend:
            extra += f", w_dead={self.w_deadend}"
        if self.w_couple_tier or self.w_couple_chain:
            extra += f", couple(t={self.w_couple_tier},c={self.w_couple_chain})"
        if not self.multiplicity_known:
            extra += ", mult=1"
        if self.monotonic:
            extra += ", monotonic"
        if self.comm_isolation:
            extra += ", comm-iso"
        if self.w_sag:
            extra += f", w_sag={self.w_sag}[{self.sag_lo},{self.sag_hi}]"
        return f"{name} ({extra})"


_BIG = 1e6


def _norm_y(ys: List[Optional[float]], method: str) -> List[float]:
    vals = [y for y in ys if y is not None]
    if not vals:
        return [0.5] * len(ys)
    if method == "raw":
        return [(y / 100.0 if y is not None else 0.5) for y in ys]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-6:
        return [0.5] * len(ys)
    return [((y - lo) / (hi - lo) if y is not None else 0.5) for y in ys]


def _zvec(items: List[Dict], cfg: MatchConfig) -> List[float]:
    """Vertical coordinate per item: common-normalized ``_z`` if every item has one (metric/abs
    modes), else legacy per-side min-max of percentY (image mode / any uncovered item)."""
    if items and all(it.get("_z") is not None for it in items):
        return [float(it["_z"]) for it in items]
    return _norm_y([it.get("y") for it in items], cfg.norm)


def assign_coords(sample: Dict, cfg: MatchConfig) -> bool:
    """Set ``_z`` on every A/M/B point per cfg.coord. Returns True if the metric/abs coordinate
    was applied to all matchable points (so the run is genuinely on the metric coordinate).

    metric/abs use a SINGLE min-max over A∪M∪B (a shared datum axis), unlike image mode's
    independent per-side normalization — that common scale is what lets a midspan wire's height
    line up with its pole attachment's height instead of being re-stretched per side.
    """
    M, A, B = sample["sides"]["M"], sample["sides"]["A"], sample["sides"]["B"]
    if cfg.coord == "minmax_image":
        for p in M + A + B:
            p["_z"] = None
        return True
    field = "h_ft" if cfg.coord == "metric_height" else "elev_ft"

    def _mn(p):
        v = p.get(field)
        return None if v is None else v + cfg.sag_ft     # midspan lift onto attachment axis

    if cfg.coord_norm == "perside":
        # diagnostic: rank within each side independently (like image min-max, but in metric space)
        for side in (M, A, B):
            is_m = side is M
            vals = [( _mn(p) if is_m else p.get(field)) for p in side]
            fin = [v for v in vals if v is not None]
            lo, hi = (min(fin), max(fin)) if fin else (0.0, 1.0)
            rng = (hi - lo) if (hi - lo) > 1e-9 else 1.0
            for p, v in zip(side, vals):
                p["_z"] = None if v is None else (v - lo) / rng
    else:
        raw = [_mn(p) for p in M] + [p.get(field) for p in A + B]
        vals = [v for v in raw if v is not None]
        if not vals:
            for p in M + A + B:
                p["_z"] = None
            return False
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) if (hi - lo) > 1e-9 else 1.0
        for p in M:
            v = _mn(p)
            p["_z"] = None if v is None else (v - lo) / rng
        for p in A + B:
            v = p.get(field)
            p["_z"] = None if v is None else (v - lo) / rng
    # matchable points = all midspan + non-guying poles; covered iff all have _z
    ok = all(p["_z"] is not None for p in M) and \
        all(p["_z"] is not None for p in A + B if p["kind"] != "guying")
    return ok


def _monotonic_assign(cost, slots, nym, nys, R, C):
    """Bottom-up NON-CROSSING min-cost matching (replaces the Hungarian when cfg.monotonic).

    Span wires don't cross, so a higher midspan wire must map to a pole point at or above a lower
    wire's. Sort midspan rows and pole slots by height and run a sequence-alignment DP with
    both-side skips: a midspan row may dustbin (cost[r, C+r]) and a slot may go unused (free). The
    result preserves vertical order — matching proceeds bottom-to-top. Coincident crossarm slots
    (same height) are adjacent, so a bundle of coincident midspan wires fills them in order.
    Returns pred[r] = pole index (or None)."""
    if R == 0:
        return []
    NEG = _BIG / 2
    rorder = sorted(range(R), key=lambda r: nym[r])
    sorder = sorted(range(C), key=lambda c: nys[c]) if C else []
    dp = np.zeros((R + 1, C + 1), dtype=float)
    bk = [[0] * (C + 1) for _ in range(R + 1)]   # 0=skip row(dustbin), 1=skip slot, 2=match
    for i in range(1, R + 1):
        dp[i][0] = dp[i - 1][0] + cost[rorder[i - 1], C + rorder[i - 1]]
    for i in range(1, R + 1):
        r = rorder[i - 1]
        dustc = cost[r, C + r]
        for j in range(1, C + 1):
            best = dp[i - 1][j] + dustc; b = 0
            if dp[i][j - 1] < best:
                best, b = dp[i][j - 1], 1
            mc = cost[r, sorder[j - 1]]
            if mc < NEG and dp[i - 1][j - 1] + mc < best:
                best, b = dp[i - 1][j - 1] + mc, 2
            dp[i][j] = best; bk[i][j] = b
    pred: List[Optional[int]] = [None] * R
    i, j = R, C
    while i > 0 and j > 0:
        b = bk[i][j]
        if b == 2:
            pred[rorder[i - 1]] = slots[sorder[j - 1]]["i"]; i -= 1; j -= 1
        elif b == 0:
            i -= 1
        else:
            j -= 1
    return pred


def _fill_residual(pred: List[Optional[int]], cost, slots: List[Dict],
                   dust_by_row: List[float], C: int) -> None:
    """Repair non-crossing leftovers IN PLACE (port of sdk v3 matcher._fill_residual).

    The non-crossing monotonic DP can strand a pole slot empty while a midspan row it could
    cheaply hold is dusted — classically a pole-top pin sitting just ABOVE a 2-wire crossarm:
    the crossarm absorbs two of three primaries and the highest midspan wire can't reach the
    higher pin without CROSSING the crossarm assignment, so the third primary drops to dust and
    the pin is left bare (detected, never traced). ``cost(row, pin) < dust`` there, so matching
    them is a strict improvement the monotonic constraint forbids.

    Relaxes ONLY the non-crossing constraint, ONLY on items both sides already left unmatched,
    ONLY when it lowers cost (cost < that row's dust); never displaces an existing match and
    respects each point's multiplicity capacity. Greedy by ascending cost (optimal for this
    small bipartite cleanup)."""
    from collections import defaultdict
    point_cap: Dict[int, int] = {}
    used: Dict[int, int] = defaultdict(int)
    for s in slots:
        point_cap[s["i"]] = int(s.get("mult", 1) or 1)
    for pi in pred:
        if pi is not None:
            used[pi] += 1
    point_of_col = [s["i"] for s in slots]
    cands: List[tuple] = []
    for r, pi in enumerate(pred):
        if pi is not None:
            continue
        for c in range(C):
            if cost[r, c] < dust_by_row[r]:
                cands.append((float(cost[r, c]), r, point_of_col[c]))
    cands.sort(key=lambda t: t[0])
    for _cst, r, i in cands:
        if pred[r] is not None or used[i] >= point_cap.get(i, 1):
            continue
        pred[r] = i
        used[i] += 1


# --------------------------------------------------------------------------- #
# Learned edge cost (cfg.edge_model)
# --------------------------------------------------------------------------- #
# Per-edge feature schema shared by matcher inference (_match_side) and the learned-cost
# trainer (scripts/tracer/probe_learned_matcher.py). Every feature is realizable from DETECTED
# nodes alone — height (per-side min-max + rank), horizontal offset, detection confidence,
# hardware tier (one-hot), crossarm multiplicity, and span density. KEEP IN SYNC with the
# trainer: column order is the contract.
EDGE_FEATURE_NAMES = [
    # --- per-edge (core 14) ---
    "dy", "dx", "rankdiff", "m_y", "s_y", "m_conf", "s_conf",
    "s_mult", "s_power", "s_secondary", "s_comm", "s_notier", "inv_R", "inv_C",
    # --- neighborhood/context (7): what graph-attention would otherwise learn ---
    "gap_above_s", "gap_below_s", "gap_above_m", "gap_below_m",
    "is_nearest_slot", "is_nearest_row", "tier_rank_s",
]
N_CORE_FEATURES = 14


def _tier_flags(tier_set) -> tuple:
    p = 1.0 if "power" in tier_set else 0.0
    s = 1.0 if "secondary" in tier_set else 0.0
    c = 1.0 if "comm" in tier_set else 0.0
    n = 1.0 if not (p or s or c) else 0.0
    return p, s, c, n


class NumpyEdgeCostModel:
    """Pure-numpy MLP edge-cost model: cost = 1 - sigmoid(MLP(standardized features[cols])).

    Frozen from a trained sklearn MLPClassifier (scripts/train/train_edge_matcher.py) into plain numpy
    arrays so it loads with NO sklearn/torch — usable by both the training-repo matcher and the
    pure-numpy wire_tracer_sdk. Set on MatchConfig.edge_model to replace the hand-tuned edge cost.
    """

    def __init__(self, Ws, bs, mean, std, cols, feature_names=None):
        self.Ws = [np.asarray(w, np.float64) for w in Ws]
        self.bs = [np.asarray(b, np.float64) for b in bs]
        self.mean = np.asarray(mean, np.float64)
        self.std = np.asarray(std, np.float64)
        self.cols = list(cols)
        self.feature_names = feature_names

    def edge_costs(self, feats: np.ndarray) -> np.ndarray:
        a = (np.asarray(feats, np.float64)[:, self.cols] - self.mean) / self.std
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            a = np.maximum(0.0, a @ W + b)        # ReLU hidden layers (sklearn default)
        z = a @ self.Ws[-1] + self.bs[-1]         # logit (binary output)
        p = 1.0 / (1.0 + np.exp(-z))
        return (1.0 - p).reshape(-1)

    def save(self, path):
        import json as _json
        d = {"n_layers": len(self.Ws), "mean": self.mean.tolist(), "std": self.std.tolist(),
             "cols": self.cols, "feature_names": self.feature_names}
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            d[f"W{i}"] = W.tolist(); d[f"b{i}"] = b.tolist()
        Path(path).write_text(_json.dumps(d))

    @classmethod
    def load(cls, path):
        import json as _json
        d = _json.loads(Path(path).read_text())
        n = d["n_layers"]
        return cls([d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)],
                   d["mean"], d["std"], d["cols"], d.get("feature_names"))

    @classmethod
    def from_sklearn_mlp(cls, clf, mean, std, cols, feature_names=None):
        return cls(clf.coefs_, clf.intercepts_, mean, std, cols, feature_names)


def _build_slots(poles: List[Dict], cfg: MatchConfig) -> List[Dict]:
    """Matchable pole slots (exclude guying / down-guy; expand each point by multiplicity).
    Each slot carries the features a detector could produce: cable_type set (oracle), hardware
    coarse-tier set, deadend flag, detection confidence, and the source multiplicity. Shared by
    the matcher and the learned-cost trainer so their edge features are identical."""
    slots: List[Dict] = []
    for p in poles:
        if p["kind"] == "guying" or p["x"] is None or p["y"] is None:
            continue
        mult = p["multiplicity"] if cfg.multiplicity_known else 1
        cable_set = {t.get("cable_type") for t in p["traces"]}
        tier_set = {hardware_tier_for_spec(t.get("insulator_spec")) for t in p["traces"]}
        tier_set.discard(None)
        is_deadend = any(spec_is_deadend(t.get("insulator_spec")) for t in p["traces"])
        tier3 = p.get("tier3")   # from _unified_point (fine class name; open_secondary=bare)
        for _ in range(mult):
            slots.append({"i": p["i"], "cable": cable_set, "tier": tier_set, "deadend": is_deadend,
                          "wclass": p.get("wire_class"), "tier3": tier3, "conf": p.get("conf"),
                          "mult": mult, "y": p["y"], "x": p["x"], "_z": p.get("_z"),
                          "elev": p.get("elev_ft")})
    return slots


def _neighbor_gaps(vals: np.ndarray):
    """Directional height gap from each item to its nearest higher / lower neighbor (1.0 at the
    extremes). Captures isolation — an isolated wire is an unambiguous match; a tightly-packed one
    is contested. The kind of neighborhood signal a graph-attention layer would otherwise learn."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ga = np.ones(n); gb = np.ones(n)
    for pos, i in enumerate(order):
        if pos + 1 < n:
            ga[i] = abs(vals[order[pos + 1]] - vals[i])
        if pos - 1 >= 0:
            gb[i] = abs(vals[i] - vals[order[pos - 1]])
    return ga, gb


def compute_edge_features(mids: List[Dict], slots: List[Dict],
                          nym: List[float], nys: List[float]) -> np.ndarray:
    """(R, C, F) edge-feature tensor for the learned cost; F == len(EDGE_FEATURE_NAMES)."""
    from collections import defaultdict
    R, C, F = len(mids), len(slots), len(EDGE_FEATURE_NAMES)
    feats = np.zeros((R, C, F), dtype=np.float32)
    if R == 0 or C == 0:
        return feats
    nym = np.asarray(nym, dtype=float); nys = np.asarray(nys, dtype=float)
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
    # context: neighbor gaps, greedy-nearest structural flags, within-tier height rank
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


def _match_side(mids: List[Dict], poles: List[Dict], cfg: MatchConfig,
                couple: Optional[List[Dict]] = None, return_margin: bool = False,
                extra: Optional[List[Dict[int, float]]] = None):
    """Bipartite Hungarian: return pred pole-point index (or None) for each midspan point.

    ``couple`` (optional, length R) carries the OTHER side's current assignment for each
    midspan row — ``{"matched": bool, "tier": set}`` — so this side can be pulled toward
    A↔B consistency: a tier penalty on slots whose tier is disjoint from the other side's,
    and a chain penalty on the dustbin option when the other side matched the row.

    return_margin=True also returns a per-row assignment MARGIN (next-best feasible cost −
    chosen cost): large ⇒ the row has a clear best pole (confident); small/negative ⇒
    ambiguous or forced. Used as the confidence signal for accuracy-vs-coverage calibration.
    """
    R = len(mids)
    if R == 0:
        return ([], []) if return_margin else []

    slots = _build_slots(poles, cfg)
    C = len(slots)

    # vertical coordinate: precomputed common-normalized _z (metric/abs modes, set by
    # assign_coords) when every point carries it, else legacy per-side min-max of percentY.
    nym = _zvec(mids, cfg)
    nys = _zvec(slots, cfg) if C else []

    # learned per-edge base cost (replaces the geometric+intrinsic term when cfg.edge_model set)
    learned = None
    if cfg.edge_model is not None and R and C:
        feats = compute_edge_features(mids, slots, nym, nys).reshape(R * C, len(EDGE_FEATURE_NAMES))
        learned = np.asarray(cfg.edge_model.edge_costs(feats), dtype=float).reshape(R, C)

    cost = np.full((R, C + R), _BIG, dtype=float)
    for r in range(R):
        # this row's own dustbin option; a one-sided match is penalised when the other
        # side already matched this midspan (chain-coherence coupling).
        dust_cost = cfg.dust
        if cfg.dust_conf_gamma:
            dust_cost = dust_cost * (max(0.0, float(mids[r].get("conf", 1.0))) ** cfg.dust_conf_gamma)
        if couple is not None and cfg.w_couple_chain and couple[r]["matched"]:
            dust_cost += cfg.w_couple_chain
        cost[r, C + r] = dust_cost
        if mids[r]["y"] is None:
            continue
        mx = (mids[r]["x"] or 0.0) / 100.0
        m_cable = mids[r].get("cable_type")
        m_tier = tier_for_cable_type(m_cable)
        m_tier3 = mids[r].get("tier3")   # midspan-predicted 3-class tier (Experiment 2)
        o_tier = couple[r]["tier"] if couple is not None else None
        o_wclass = couple[r].get("wclass") if couple is not None else None
        o_elev = couple[r].get("elev") if couple is not None else None
        m_elev = mids[r].get("elev_ft")
        for c in range(C):
            s = slots[c]
            dy = abs(nym[r] - nys[c])
            dx = abs(mx - (s["x"] or 0.0) / 100.0)
            if cfg.class_signal == "cable_type":
                cl = cfg.w_class if (m_cable is not None and m_cable not in s["cable"]) else 0.0
            elif cfg.class_signal == "hw_tier":
                # penalise only when BOTH sides have a tier and they disagree; a pole point
                # with no hardware (spec None) gives no signal → fall back to geometry.
                cl = cfg.w_class if (m_tier is not None and s["tier"] and m_tier not in s["tier"]) else 0.0
            else:
                cl = 0.0
            dd = cfg.w_deadend if s["deadend"] else 0.0
            # A↔B tier coupling: penalise slots whose tier is disjoint from the tier the
            # OTHER pole assigned to this midspan (both tiers from hardware, no midspan class).
            ct = (cfg.w_couple_tier
                  if (cfg.w_couple_tier and o_tier and s["tier"] and o_tier.isdisjoint(s["tier"]))
                  else 0.0)
            # finer wire-class A↔B coupling (predicted primary/secondary/neutral/comm): penalise a
            # slot whose wire_class disagrees with the class the OTHER pole assigned to this midspan.
            cw = (cfg.w_couple_class
                  if (cfg.w_couple_class and o_wclass and s["wclass"] and o_wclass != s["wclass"])
                  else 0.0)
            # MIDSPAN<->POLE tier3 coupling (Experiment 2): the wire's own predicted tier vs this
            # pole slot's tier3. Both-known-only; soft. This is a midspan-vs-pole term (unlike the
            # A<->B couple terms above), so it does NOT need the ICM loop / couple state.
            mt3_bad = bool(m_tier3 and s.get("tier3") and m_tier3 != s["tier3"])
            mt3 = cfg.w_mid_tier3 if (cfg.w_mid_tier3 and mt3_bad) else 0.0
            if cfg.w_mid_tier3_bonus and m_tier3 and s.get("tier3") and m_tier3 == s["tier3"]:
                mt3 -= cfg.w_mid_tier3_bonus     # tier-agreeing edge beats the dustbin
            # CATENARY-SAG plausibility (coupled A+M+B): the midspan should hang a few feet below the
            # chord between this slot and the OTHER side's assigned point. Penalise sag outside the
            # physical band, proportional to the excess (ft). Needs all three elevations.
            sg = 0.0
            if cfg.w_sag and o_elev is not None and s["elev"] is not None and m_elev is not None:
                sag = (s["elev"] + o_elev) / 2.0 - m_elev
                excess = max(cfg.sag_lo - sag, sag - cfg.sag_hi, 0.0)
                sg = cfg.w_sag * excess
            if learned is not None:
                # learned base cost; deadend + geometry are folded into the features, the
                # dynamic A↔B coupling stays additive (it depends on the other side's state).
                cval = learned[r, c] + ct + cw + sg + mt3
            else:
                cval = cfg.w_y * dy + cfg.w_x * dx + cl + dd + ct + cw + sg + mt3
            # HARD three_bolt(comm) isolation: a comm slot may only take a wire whose OTHER pole
            # is also comm, and a non-comm slot may not take a wire whose other pole is comm.
            # No constraint when either tier is unknown (empty). Uses the coupled other-side tier.
            if cfg.comm_isolation and o_tier and s["tier"] and \
                    (("comm" in s["tier"]) != ("comm" in o_tier)):
                cval = _BIG
            # HARD midspan<->pole tier3 isolation (Experiment 2): forbid a cross-tier3 edge.
            if cfg.mid_tier3_hard and mt3_bad:
                cval = _BIG
            # opt-in externally computed per-(row, pole-point) additive cost (e.g. the
            # sag-offset coherence probe). None (default) = byte-identical legacy.
            if extra is not None:
                cval += extra[r].get(s["i"], 0.0)
            cost[r, c] = cval

    if cfg.monotonic:
        pred = _monotonic_assign(cost, slots, nym, nys, R, C)
        if cfg.fill_residual and C:
            _fill_residual(pred, cost, slots, [cost[r, C + r] for r in range(R)], C)
        return (pred, [0.0] * R) if return_margin else pred

    rows, cols = linear_sum_assignment(cost)
    pred: List[Optional[int]] = [None] * R
    chosen: List[Optional[int]] = [None] * R
    for r, c in zip(rows, cols):
        pred[r] = slots[c]["i"] if c < C else None
        chosen[r] = c
    if not return_margin:
        return pred
    margins: List[float] = [0.0] * R
    for r in range(R):
        c = chosen[r]
        if c is None:
            continue
        feasible = [cost[r, cc] for cc in list(range(C)) + [C + r]
                    if cc != c and cost[r, cc] < _BIG / 2]
        margins[r] = (min(feasible) - cost[r, c]) if feasible else float("inf")
    return pred, margins


def _coupling_from(pred: List[Optional[int]], poles: List[Dict]) -> List[Dict]:
    """Per-midspan-row view of one side's assignment for the other side to couple against."""
    by_idx = {p["i"]: p for p in poles}
    out = []
    for pi in pred:
        if pi is None:
            out.append({"matched": False, "tier": set(), "wclass": None, "elev": None})
        else:
            p = by_idx[pi]
            tset = {hardware_tier_for_spec(t.get("insulator_spec")) for t in p["traces"]}
            tset.discard(None)
            out.append({"matched": True, "tier": tset, "wclass": p.get("wire_class"),
                        "elev": p.get("elev_ft")})
    return out


def match_span(sample: Dict, cfg: MatchConfig, return_conf: bool = False,
               extra: Optional[Dict[str, List[Dict[int, float]]]] = None) -> Dict[str, List[Optional[int]]]:
    assign_coords(sample, cfg)
    M, A, B = sample["sides"]["M"], sample["sides"]["A"], sample["sides"]["B"]
    exA = extra.get("A") if extra else None
    exB = extra.get("B") if extra else None
    predA = _match_side(M, A, cfg, extra=exA)
    predB = _match_side(M, B, cfg, extra=exB)
    # comm_isolation + w_sag are coupling-based (need the other side's assignment), so they also
    # need the ICM loop below.
    if cfg.w_couple_tier <= 0 and cfg.w_couple_chain <= 0 and cfg.w_couple_class <= 0 \
            and cfg.w_sag <= 0 and not cfg.comm_isolation:
        if not return_conf:
            return {"A": predA, "B": predB}
        predA, cA = _match_side(M, A, cfg, return_margin=True, extra=exA)
        predB, cB = _match_side(M, B, cfg, return_margin=True, extra=exB)
        return {"A": predA, "B": predB, "A_conf": cA, "B_conf": cB}

    # A↔B coupling on: alternating (ICM) refinement. Re-solve each side against the other
    # side's current assignment until both stabilise (or couple_iters reached). Span sizes
    # are tiny, so this converges in 1-2 passes.
    for _ in range(cfg.couple_iters):
        newA = _match_side(M, A, cfg, couple=_coupling_from(predB, B), extra=exA)
        newB = _match_side(M, B, cfg, couple=_coupling_from(newA, A), extra=exB)
        if newA == predA and newB == predB:
            predA, predB = newA, newB
            break
        predA, predB = newA, newB
    if not return_conf:
        return {"A": predA, "B": predB}
    # one more consistent ICM half-step to read margins paired with the final assignment
    predA, cA = _match_side(M, A, cfg, couple=_coupling_from(predB, B), return_margin=True, extra=exA)
    predB, cB = _match_side(M, B, cfg, couple=_coupling_from(predA, A), return_margin=True, extra=exB)
    return {"A": predA, "B": predB, "A_conf": cA, "B_conf": cB}


# --------------------------------------------------------------------------- #
# Multi-section spans (pole-A -> M1 -> ... -> Mk -> pole-B)
# --------------------------------------------------------------------------- #

def _align_to_spine(section_pts: List[Dict], spine_pts: List[Dict], cfg: MatchConfig) -> Dict[int, int]:
    """Match one midspan section's rows to the SPINE section's rows by height.

    Both sides are midspan wires (no multiplicity, both dustbinnable). Span wires keep their
    vertical order across sections (order preserved end-to-end), so this is the same monotonic /
    Hungarian min-cost assignment the pole legs use, on |Δy| alone (sections are different photos
    → no shared x). Returns ``{spine_point_i: section_point_i}`` for the matched pairs."""
    R, C = len(section_pts), len(spine_pts)
    if R == 0 or C == 0:
        return {}
    nym = _zvec(section_pts, cfg)
    nys = _zvec(spine_pts, cfg)
    cost = np.full((R, C + R), _BIG, dtype=float)
    for r in range(R):
        dr = cfg.dust
        if cfg.dust_conf_gamma:
            dr = dr * (max(0.0, float(section_pts[r].get("conf", 1.0))) ** cfg.dust_conf_gamma)
        cost[r, C + r] = dr
        if section_pts[r]["y"] is None:
            continue
        for c in range(C):
            if spine_pts[c]["y"] is None:
                continue
            cost[r, c] = cfg.w_y * abs(nym[r] - nys[c])
    if cfg.monotonic:
        pred = _monotonic_assign(cost, spine_pts, nym, nys, R, C)
    else:
        rows, cols = linear_sum_assignment(cost)
        pred = [None] * R
        for r, c in zip(rows, cols):
            pred[r] = spine_pts[c]["i"] if c < C else None
    out: Dict[int, int] = {}
    for sr, si in enumerate(pred):
        if si is not None:
            out[si] = section_pts[sr]["i"]
    return out


def match_span_multi(sample: Dict, cfg: MatchConfig, return_conf: bool = False,
                     extra: Optional[Dict[str, List[Dict[int, float]]]] = None) -> Dict:
    """Multi-section matcher: predict the full pole-A -> M1 -> ... -> Mk -> pole-B path.

    A wire is one continuous trace observed at several midspan waypoints, so the chain is anchored
    on a SPINE section (the one with the most detected wires = best A↔B linkage + widest coverage)
    via the unchanged coupled matcher, and every other section's rows are monotonically attached to
    the spine. On a single-section span (no ``M_sections``) this is byte-identical to
    :func:`match_span` (the section IS the spine, attach is the identity), so it is a drop-in
    superset. Returns the legacy ``{"A","B"[,"_conf"]}`` PLUS:
      * ``spine``    — index (into M_sections) of the section used as the anchor.
      * ``sections`` — one ``{spine_row: section_row}`` map per section (ordered A→B); compose with
                       :func:`compose_multi_chains` into per-wire ``M_path`` lists.
    """
    secs = sample["sides"].get("M_sections")
    if not secs:                                   # legacy single-section span → identity attach
        res = dict(match_span(sample, cfg, return_conf=return_conf, extra=extra))
        res["spine"] = 0
        res["sections"] = [{i: i for i in range(len(sample["sides"]["M"]))}]
        return res

    spine_idx = max(range(len(secs)), key=lambda i: (len(secs[i]["points"]), -i))
    spine_pts = secs[spine_idx]["points"]
    sub = {"sides": {"A": sample["sides"]["A"], "M": spine_pts, "B": sample["sides"]["B"]}}
    res = dict(match_span(sub, cfg, return_conf=return_conf, extra=extra))

    sec_maps: List[Dict[int, int]] = []
    for i, s in enumerate(secs):
        if i == spine_idx:
            sec_maps.append({j: j for j in range(len(spine_pts))})
        else:
            sec_maps.append(_align_to_spine(s["points"], spine_pts, cfg))
    res["spine"] = spine_idx
    res["sections"] = sec_maps
    return res


def compose_multi_chains(res: Dict) -> List[Dict]:
    """Compose :func:`match_span_multi` output into per-wire predicted chains.

    Each chain is keyed by a spine row and carries ``A``/``B`` (pole indices, or None) and
    ``M_path`` (the row index in each section, or None where unmatched/missed). Only spine rows
    that reach a pole or appear in some section are emitted."""
    predA, predB = res["A"], res["B"]
    sections = res["sections"]
    n_rows = len(predA)
    chains = []
    for r in range(n_rows):
        m_path = [sec.get(r) for sec in sections]
        ia = predA[r] if r < len(predA) else None
        ib = predB[r] if r < len(predB) else None
        if ia is None and ib is None and not any(m is not None for m in m_path):
            continue
        chains.append({"spine_row": r, "A": ia, "B": ib, "M_path": m_path})
    return chains


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _acc(d):
    return {
        "A_acc": round(d["A_ok"] / d["n"], 4) if d["n"] else None,
        "B_acc": round(d["B_ok"] / d["n"], 4) if d["n"] else None,
        "chain_acc": round(d["chain_ok"] / d["n"], 4) if d["n"] else None,
        "n": d["n"],
    }


def evaluate(spans: List[Dict], cfg: MatchConfig, exclude_above_pole_top: bool = False) -> Dict:
    clean = {"A_ok": 0, "B_ok": 0, "chain_ok": 0, "n": 0}
    ambig = {"A_ok": 0, "B_ok": 0, "chain_ok": 0, "n": 0}

    orphan_gt = orphan_pred = orphan_hit = 0
    pdust_gt = pdust_pred = pdust_hit = 0
    n_excluded = 0

    for s in spans:
        preds = match_span(s, cfg)
        M = s["sides"]["M"]
        A, B = s["sides"]["A"], s["sides"]["B"]

        for c in s["gt"]["chains"]:
            # edge case: a wire passing OVER the pole top is above the upper-70% crop, so the
            # detector structurally cannot see it — exclude it from the matcher denominator.
            if exclude_above_pole_top and (
                    (c["A"] is not None and A[c["A"]].get("above_pole_top")) or
                    (c["B"] is not None and B[c["B"]].get("above_pole_top"))):
                n_excluded += 1
                continue
            m = c["M"]
            a_ok = preds["A"][m] == c["A"]
            b_ok = preds["B"][m] == c["B"]
            bucket = ambig if c["group_ambiguous"] else clean
            bucket["n"] += 1
            bucket["A_ok"] += int(a_ok)
            bucket["B_ok"] += int(b_ok)
            bucket["chain_ok"] += int(a_ok and b_ok)

        # midspan orphan detection (GT both-None) — did we predict both None?
        gt_orphan = set(s["gt"]["dustbin"]["M"])
        pred_orphan = {m for m in range(len(M))
                       if preds["A"][m] is None and preds["B"][m] is None}
        orphan_gt += len(gt_orphan)
        orphan_pred += len(pred_orphan)
        orphan_hit += len(gt_orphan & pred_orphan)

        # pole-point dustbin: points an M-anchored matcher SHOULD leave unmatched
        for side in ("A", "B"):
            pts = s["sides"][side]
            matched = {p for p in preds[side] if p is not None}
            pred_unmatched = set(range(len(pts))) - matched
            gt_unmatched = set(s["gt"]["dustbin"][side]) | \
                {c[side] for c in s["gt"]["pole_only_chains"]}
            pdust_gt += len(gt_unmatched)
            pdust_pred += len(pred_unmatched)
            pdust_hit += len(gt_unmatched & pred_unmatched)

    overall = {k: clean[k] + ambig[k] for k in clean}

    def pr(hit, gt, pred):
        return {"recall": round(hit / gt, 4) if gt else None,
                "precision": round(hit / pred, 4) if pred else None,
                "n_gt": gt, "n_pred": pred}

    return {
        "config": asdict(cfg),
        "label": cfg.label(),
        "n_spans": len(spans),
        "strict_per_trace_clean": _acc(clean),
        "crossarm_group_ambiguous": _acc(ambig),
        "overall": _acc(overall),
        "midspan_orphan": pr(orphan_hit, orphan_gt, orphan_pred),
        "pole_dustbin": pr(pdust_hit, pdust_gt, pdust_pred),
        "n_excluded_above_pole_top": n_excluded,
    }


def load_spans(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_eval(m: Dict) -> str:
    L = []
    L.append(f"── {m['label']}  ({m['n_spans']} spans) ──")
    c, a, o = m["strict_per_trace_clean"], m["crossarm_group_ambiguous"], m["overall"]
    L.append(f"{'bucket':<26}{'n':>7}{'A pt':>9}{'B pt':>9}{'chain':>9}")
    L.append(f"{'strict per-trace (clean)':<26}{c['n']:>7}{c['A_acc']:>9}{c['B_acc']:>9}{c['chain_acc']:>9}")
    L.append(f"{'crossarm-group (ambig)':<26}{a['n']:>7}{a['A_acc']:>9}{a['B_acc']:>9}{a['chain_acc']:>9}")
    L.append(f"{'overall':<26}{o['n']:>7}{o['A_acc']:>9}{o['B_acc']:>9}{o['chain_acc']:>9}")
    mo, pd = m["midspan_orphan"], m["pole_dustbin"]
    L.append(f"midspan-orphan   recall={mo['recall']} precision={mo['precision']} "
             f"(gt={mo['n_gt']}, pred={mo['n_pred']})")
    L.append(f"pole-dustbin     recall={pd['recall']} precision={pd['precision']} "
             f"(gt={pd['n_gt']}, pred={pd['n_pred']})")
    return "\n".join(L)
