#!/usr/bin/env python3
"""
Accuracy-vs-coverage (confidence calibration) for the wire-tracing e2e.

The headline chain accuracy (~0.38) hides what's deployable. In an assisted workflow the
system auto-confirms the chains it's confident about and sends the rest to a human, so the
production-relevant question is: *what fraction of chains can be auto-confirmed at ~95%
precision?* This thresholds the matcher's per-assignment MARGIN (next-best cost − chosen
cost; large ⇒ a clear best pole) and plots precision vs coverage over all GT units
(clean chains + crossarm bundles).

Uses cached detections (run scripts/eval_wire_tracing_e2e.py first to populate the cache),
so it's CPU-only and fast.

Usage:
    python scripts/eval_wire_tracing_calibration.py --cache <e2e_cache.json>
    # defaults to the wire-gated + strip config; override with the same flags as the e2e eval
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import WIRE_TRACING_DATASET_DIR
from src.wire_tracing_match import MatchConfig
from src.wire_tracer import DEFAULT_UNIFIED_WEIGHTS
from src.wire_tracing_e2e import (
    load_detectors, resolve_span_photos, detect_pole_points,
    detect_midspan_points_strip,
    score_span_e2e, ruler_fit_for_photo,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", type=Path, default=WIRE_TRACING_DATASET_DIR / "spans.jsonl")
    ap.add_argument("--cache", type=Path, default=Path(".e2e_det_cache.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--unified-weights", default=DEFAULT_UNIFIED_WEIGHTS)
    ap.add_argument("--pole-imgsz", type=int, default=None)
    ap.add_argument("--wt", type=float, default=0.2)
    ap.add_argument("--wc", type=float, default=0.25)
    ap.add_argument("--wdead", type=float, default=0.06)
    ap.add_argument("--assoc-tol", type=float, default=6.0)
    ap.add_argument("--pole-tol-inch", type=float, default=3.0, help="pole assoc tol in INCHES (PPI-converted); 0 for percent")
    ap.add_argument("--mid-tol-inch", type=float, default=6.0, help="midspan (clean) assoc tol in INCHES (PPI-converted); 0 for percent")
    ap.add_argument("--mid-crossarm-tol-inch", type=float, default=36.0, help="looser midspan tol (inches) for crossarm bundles")
    ap.add_argument("--pole-crossarm-tol-inch", type=float, default=36.0, help="looser pole tol (inches) for crossarm arm points")
    ap.add_argument("--out", type=Path, default=Path("wire_tracing_calibration.png"))
    args = ap.parse_args()

    spans = [json.loads(l) for l in open(args.spans) if l.strip()]
    for s in spans:
        s["_photos"] = resolve_span_photos(s)
    resolvable = [s for s in spans if s["_photos"]["resolvable"]]
    print(f"resolvable spans: {len(resolvable)}")

    weights = {}
    if args.unified_weights: weights["unified"] = args.unified_weights
    det = load_detectors(device=args.device, weights=weights or None, midspan_source="strip")
    if args.pole_imgsz is not None: det.pole_crop_imgsz = args.pole_imgsz
    mid_axis = "y"
    wx = 0.0

    pole_sig = f"P|{det.pole_source}|isz{det.pole_crop_imgsz}|pc{det.pole_conf}|" \
               f"uni{args.unified_weights or 'def'}|uc{det.unified_conf}::"
    mid_sig = f"M|{det.midspan_source}|def|def::"
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    print(f"cache entries: {len(cache)}")
    mid_fn = detect_midspan_points_strip

    def pole_cached(p):
        k = pole_sig + p
        if k not in cache:
            cache[k] = detect_pole_points(p, det)
        return cache[k]

    def mid_cached(ps):
        k = mid_sig + "||".join(ps)
        if k not in cache:
            cache[k] = mid_fn(ps, det)
        return cache[k]

    cfg = MatchConfig(class_signal="none", w_couple_tier=args.wt, w_couple_chain=args.wc,
                      w_deadend=args.wdead, w_x=wx)

    records = []
    for s in resolvable:
        ph = s["_photos"]
        try:
            dA, dB, dM = pole_cached(ph["A"]), pole_cached(ph["B"]), mid_cached(ph["M"])
        except Exception as e:
            print(f"  ! {s['job']}: {e}")
            continue
        tA = args.pole_tol_inch if args.pole_tol_inch else None
        tB = args.pole_tol_inch if args.pole_tol_inch else None
        tM = args.mid_tol_inch if (args.mid_tol_inch and ph["M"]) else None
        tMc = args.mid_crossarm_tol_inch if (args.mid_crossarm_tol_inch and ph["M"]) else None
        tAc = args.pole_crossarm_tol_inch if args.pole_crossarm_tol_inch else None
        tBc = args.pole_crossarm_tol_inch if args.pole_crossarm_tol_inch else None
        fitA = ruler_fit_for_photo(ph["A"], "pole")
        fitB = ruler_fit_for_photo(ph["B"], "pole")
        fitM = ruler_fit_for_photo(ph["M"][0], "midspan") if ph["M"] else None
        r = score_span_e2e(s, dA, dM, dB, cfg, assoc_tol_pct=args.assoc_tol,
                           mid_assoc_axis=mid_axis, bundle_crossarm=True, collect_records=True,
                           tol_A=tA, tol_B=tB, tol_M=tM, tol_M_crossarm=tMc,
                           tol_A_crossarm=tAc, tol_B_crossarm=tBc,
                           fit_A=fitA, fit_B=fitB, fit_M=fitM)
        records.extend(r["records"])

    _report(records, args.out)


def _curve(proposed, total, key):
    """Cumulative precision as confidence (key) descends. Returns [(coverage, precision)]."""
    p = sorted(proposed, key=key, reverse=True)
    cc, curve = 0, []
    for k, r in enumerate(p, 1):
        cc += int(r["correct"])
        curve.append((k / total, cc / k))
    return curve


def _cov_at(curve, P):
    best = 0.0
    for cov, prec in curve:
        if prec >= P:
            best = cov
    return best


def _report(records, out):
    import numpy as np
    total = len(records)
    proposed = [r for r in records if r["proposed"] and r["conf"] is not None]
    overall_correct = sum(r["correct"] for r in records)
    print("\n" + "=" * 64)
    print(f"GT units (clean chains + crossarm bundles): {total}")
    print(f"proposed (midspan detected): {len(proposed)} ({100*len(proposed)/total:.0f}%)")
    print(f"overall accuracy (correct / all units):     {overall_correct/total:.3f}")
    print(f"precision of all proposals:                 "
          f"{sum(r['correct'] for r in proposed)/max(len(proposed),1):.3f}")

    # combined rank signal: require BOTH high matcher-margin AND detection-confidence
    def _pct(vals):
        v = np.asarray(vals, float)
        return np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    pm = _pct([r["conf"] for r in proposed])
    pd = _pct([r["det_conf"] if r["det_conf"] is not None else 0.0 for r in proposed])
    for i, r in enumerate(proposed):
        r["_combo"] = float(min(pm[i], pd[i]))

    signals = {
        "matcher margin": lambda r: r["conf"],
        "detection conf": lambda r: (r["det_conf"] if r["det_conf"] is not None else 0.0),
        "combined rank":  lambda r: r["_combo"],
    }
    curves = {name: _curve(proposed, total, key) for name, key in signals.items()}

    print("\n coverage @ target precision (fraction of ALL GT units auto-confirmable):")
    print(f"  {'signal':<16}" + "".join(f"{p:>9.0%}" for p in (0.95, 0.90, 0.85, 0.80)))
    for name, cv in curves.items():
        print(f"  {name:<16}" + "".join(f"{_cov_at(cv, P):>9.1%}" for P in (0.95, 0.90, 0.85, 0.80)))

    print("\n per-bucket proposal precision:")
    for b in ("clean", "crossarm"):
        bp = [r for r in proposed if r["bucket"] == b]
        if bp:
            print(f"  {b:<10} n={len(bp):>5}  precision={sum(r['correct'] for r in bp)/len(bp):.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.5, 5))
        for name, cv in curves.items():
            plt.plot([c[0] for c in cv], [c[1] for c in cv], "-", lw=1.8, label=name)
        for P in (0.95, 0.90, 0.80):
            plt.axhline(P, color="grey", ls=":", lw=0.7)
        plt.xlabel("coverage (fraction of all GT chains auto-confirmed)")
        plt.ylabel("precision of auto-confirmed subset")
        plt.title("Wire tracing — accuracy vs coverage")
        plt.ylim(0, 1.02); plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(out, dpi=130)
        print(f"\nplot saved: {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
