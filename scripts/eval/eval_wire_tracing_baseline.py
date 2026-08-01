#!/usr/bin/env python3
"""
Stage-1b: evaluate the Hungarian baseline matcher on the wire-tracing dataset.

Reports the number the learned matcher must beat — split into strict per-trace (clean
chains) vs crossarm-group (ambiguous) accuracy — under three class regimes
(geometry-only, +hw-tier, +oracle-cable_type) plus the hardware deadend dustbin prior,
with sensitivity sweeps.

The hardware head feeds the matcher two signals (see src/wire_tracing_match.py):
  * +hw-tier   — pole coarse tier from predicted hardware vs midspan cable_type tier;
                 the realizable slice of the class lever (oracle-cable_type is the ceiling).
  * w_deadend  — deadend = power-terminating → push deadended attachments to pole-dustbin;
                 pure pole-side signal, needs no midspan class.

Usage:
    python scripts/eval_wire_tracing_baseline.py
    python scripts/eval_wire_tracing_baseline.py --spans <spans.jsonl> --dust 0.18
    python scripts/eval_wire_tracing_baseline.py --w-deadend 0.12 --no-sweep
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import WIRE_TRACING_DATASET_DIR
from src.wire_tracing_match import MatchConfig, evaluate, load_spans, format_eval


def main():
    ap = argparse.ArgumentParser(description="Evaluate wire-tracing Hungarian baseline")
    ap.add_argument("--spans", type=Path, default=WIRE_TRACING_DATASET_DIR / "spans.jsonl")
    ap.add_argument("--dust", type=float, default=0.18)
    ap.add_argument("--norm", choices=["minmax", "raw"], default="minmax")
    ap.add_argument("--w-deadend", type=float, default=0.06,
                    help="deadend pole-slot penalty for the combined hw config (default 0.06)")
    ap.add_argument("--w-couple-tier", type=float, default=0.2,
                    help="A↔B tier-agreement coupling weight (default 0.2)")
    ap.add_argument("--w-couple-chain", type=float, default=0.25,
                    help="A↔B match-both-or-neither coupling weight (default 0.25)")
    ap.add_argument("--no-sweep", action="store_true", help="Skip the sensitivity sweeps")
    args = ap.parse_args()

    if not args.spans.exists():
        print(f"❌ spans file not found: {args.spans}\n   run scripts/build_wire_tracing_dataset.py first",
              file=sys.stderr)
        sys.exit(1)

    spans = load_spans(args.spans)
    print("=" * 64)
    print(f"WIRE TRACING — Stage-1b Hungarian baseline   ({len(spans)} spans)")
    print("=" * 64)

    # Regimes in order of realizability. Weights:
    #   hw-tier is a NOISY coarse proxy -> w_class < dust (a lone tier mismatch must not
    #     beat the dustbin). oracle cable_type is clean -> larger weight.
    #   A↔B coupling is FULLY realizable (both pole tiers from the hardware head, no midspan
    #     class): w_couple_chain exploits "~99% of wires reach both poles" (match both or
    #     neither); w_couple_tier transfers a confident tier read across the span.
    W_HW, W_ORACLE = 0.15, 0.5
    WT, WC = args.w_couple_tier, args.w_couple_chain
    regimes = [
        ("", MatchConfig(dust=args.dust, norm=args.norm, class_signal="none")),
        ("(reference: hw-tier — needs a midspan tier, here semi-oracle from cable_type)",
         MatchConfig(dust=args.dust, norm=args.norm, class_signal="hw_tier", w_class=W_HW)),
        ("(A↔B coupling — FULLY realizable: both pole tiers from hardware, NO midspan class)",
         MatchConfig(dust=args.dust, norm=args.norm, class_signal="none",
                     w_couple_tier=WT, w_couple_chain=WC)),
        ("(full realizable stack: A↔B coupling + deadend prior)",
         MatchConfig(dust=args.dust, norm=args.norm, class_signal="none",
                     w_couple_tier=WT, w_couple_chain=WC, w_deadend=args.w_deadend)),
        ("(ceiling: exact GT cable_type + A↔B coupling — the true upper bound)",
         MatchConfig(dust=args.dust, norm=args.norm, class_signal="cable_type",
                     w_class=W_ORACLE, w_couple_tier=WT, w_couple_chain=WC)),
    ]
    for header, cfg in regimes:
        if header:
            print(header)
        print(format_eval(evaluate(spans, cfg)))
        print()

    if not args.no_sweep:
        print("A↔B chain-coupling sweep (tier-couple=%.2f): chain / clean / ambig / poledust_P" % WT)
        print(f"{'w_chain':>8}{'chain':>9}{'clean':>9}{'ambig':>9}{'pdust_P':>10}")
        for wc in (0.0, 0.1, 0.18, 0.25, 0.35):
            m = evaluate(spans, MatchConfig(dust=args.dust, norm=args.norm, class_signal="none",
                                            w_couple_tier=WT, w_couple_chain=wc))
            o, c, a, pd = m["overall"], m["strict_per_trace_clean"], m["crossarm_group_ambiguous"], m["pole_dustbin"]
            print(f"{wc:>8}{o['chain_acc']:>9}{c['chain_acc']:>9}{a['chain_acc']:>9}{pd['precision']:>10}")
        print()

        print("hw-tier w_class knee (dust=%.2f): chain_acc / clean / orphan R,P / poledust_R" % args.dust)
        print(f"{'w_class':>8}{'chain':>9}{'clean':>9}{'orph_R':>9}{'orph_P':>9}{'pdust_R':>10}")
        for wc in (0.0, 0.1, 0.15, 0.2, 0.5):
            sig = "none" if wc == 0.0 else "hw_tier"
            m = evaluate(spans, MatchConfig(dust=args.dust, norm=args.norm, class_signal=sig, w_class=wc))
            o, c, mo, pd = m["overall"], m["strict_per_trace_clean"], m["midspan_orphan"], m["pole_dustbin"]
            print(f"{wc:>8}{o['chain_acc']:>9}{c['chain_acc']:>9}{mo['recall']:>9}{mo['precision']:>9}{pd['recall']:>10}")


if __name__ == "__main__":
    main()
