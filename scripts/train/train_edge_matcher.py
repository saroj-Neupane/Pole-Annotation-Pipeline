#!/usr/bin/env python3
"""
Train + freeze the LEARNED edge-cost matcher (the validated wire-tracing lever; see
scripts/probe_learned_matcher.py).  MLP(32,16) on the 21 shared edge features -> frozen to a
pure-numpy NumpyEdgeCostModel (no sklearn/torch at inference; SDK-portable).

Reports the honest held-out test number, then retrains on ALL resolvable spans and saves the
deployable artifact + its recommended dustbin operating point.

    python scripts/train_edge_matcher.py --source unified --out models/edge_matcher_unified.json
    # use it:  python scripts/eval_wire_tracing_e2e.py ... --edge-model models/edge_matcher_unified.json --edge-dust 0.6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.wire_tracing_match import (EDGE_FEATURE_NAMES, N_CORE_FEATURES, NumpyEdgeCostModel,
                                    compute_edge_features, _build_slots, _zvec)
from src.wire_tracing_e2e import resolve_span_photos
import scripts.tracer.probe_learned_matcher as P


def pick_dust(model, val, cache, pole_sig, args):
    best_d, best_v = P.DUSTS[0], -1.0
    for dust in P.DUSTS:
        c = P.base_cfg(args); c.edge_model = model; c.dust = dust
        v = P.eval_spans(val, c, cache, pole_sig)["overall"]
        if v > best_v:
            best_v, best_d = v, dust
    return best_d, best_v


def fit_numpy(spans, cache, pole_sig, args, cols):
    """Train sklearn MLP on `spans`, freeze to NumpyEdgeCostModel; verify numpy==sklearn costs."""
    X, y = P.gather_edges(spans, cache, pole_sig, P.base_cfg(args))
    skmodel, clf = P.train_model(X, y, "mlp_raw", cols)           # sklearn wrapper
    npmodel = NumpyEdgeCostModel.from_sklearn_mlp(clf, skmodel.mean, skmodel.std, cols,
                                                  [EDGE_FEATURE_NAMES[i] for i in cols])
    # parity check on a feature sample
    probe = X[:512] if len(X) else np.zeros((1, len(EDGE_FEATURE_NAMES)))
    diff = float(np.max(np.abs(skmodel.edge_costs(probe) - npmodel.edge_costs(probe)))) if len(X) else 0.0
    return npmodel, len(y), int(y.sum()), diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="unified", choices=list(POLE_SIG) if "POLE_SIG" in dir() else None)
    ap.add_argument("--spans", default="datasets/wire_tracing_dataset/spans.jsonl")
    ap.add_argument("--cache", default=".e2e_det_cache.json")
    ap.add_argument("--out", default=None, help="output model JSON (default models/edge_matcher_<source>.json)")
    ap.add_argument("--wt", type=float, default=0.2)
    ap.add_argument("--wc", type=float, default=0.25)
    ap.add_argument("--wdead", type=float, default=0.06)
    ap.add_argument("--features", default="all", choices=["core", "all"])
    args = ap.parse_args()
    out = Path(args.out or f"models/edge_matcher_{args.source}.json")
    cols = list(range(N_CORE_FEATURES if args.features == "core" else len(EDGE_FEATURE_NAMES)))

    pole_sig = P.POLE_SIG[args.source]
    cache = json.loads(Path(args.cache).read_text())
    spans = [json.loads(l) for l in open(args.spans) if l.strip()]
    for s in spans:
        s["_photos"] = resolve_span_photos(s)
    resolvable = [s for s in spans if s["_photos"]["resolvable"]]
    print(f"source={args.source} features={args.features} ({len(cols)}d) | {len(resolvable)} resolvable spans")

    # ---- honest held-out: train on train, dust on val, report test (vs hand-tuned) ----
    train, test = P.split(resolvable, 0.4, salt="A|")
    tr, val = P.split(train, 0.25, salt="val|")
    model, n_edge, n_pos, diff = fit_numpy(tr, cache, pole_sig, args, cols)
    print(f"  frozen-numpy vs sklearn max |Δcost| = {diff:.2e}  (parity)")
    dust, vbest = pick_dust(model, val, cache, pole_sig, args)
    hand = P.eval_spans(test, P.base_cfg(args), cache, pole_sig)
    c = P.base_cfg(args); c.edge_model = model; c.dust = dust
    learned = P.eval_spans(test, c, cache, pole_sig)
    print(f"  HELD-OUT test (n={hand['n']}):  hand={hand['overall']:.4f}  "
          f"learned={learned['overall']:.4f}  Δ={learned['overall']-hand['overall']:+.4f}  (dust*={dust})")
    print(f"     learned breakdown: clean={learned['clean']:.4f} crossarm={learned['crossarm']:.4f} "
          f"cond-on-mid={learned['cond_on_mid']:.4f}")

    # ---- deployable: retrain on ALL resolvable, dust on a val carve, save ----
    fit_all, val_all = P.split(resolvable, 0.18, salt="deploy_val|")   # ~18% held for dust pick
    model, n_edge, n_pos, diff = fit_numpy(fit_all, cache, pole_sig, args, cols)
    dust, vbest = pick_dust(model, val_all, cache, pole_sig, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    # full-set number with the deployable model (trained on 82%, so mildly optimistic on the val fold)
    c = P.base_cfg(args); c.edge_model = model; c.dust = dust
    full = P.eval_spans(resolvable, c, cache, pole_sig)
    hand_full = P.eval_spans(resolvable, P.base_cfg(args), cache, pole_sig)
    print(f"\nDEPLOYABLE model (trained on {len(fit_all)} spans, {n_edge} edges {100*n_pos/max(n_edge,1):.1f}% pos):")
    print(f"  saved -> {out}   recommended --edge-dust {dust}")
    print(f"  full-set ({full['n']} chains):  hand={hand_full['overall']:.4f}  "
          f"learned={full['overall']:.4f}  Δ={full['overall']-hand_full['overall']:+.4f}")
    print(f"  (full-set is mildly optimistic — the held-out test Δ above is the honest number)")


if __name__ == "__main__":
    main()
