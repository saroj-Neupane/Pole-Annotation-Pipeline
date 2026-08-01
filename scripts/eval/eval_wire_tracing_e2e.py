#!/usr/bin/env python3
"""
Stage-1b END-TO-END eval: real trained detectors on real span photos -> A↔B-coupled
matcher -> score vs Stage-0 GT chains.

Caches per-photo detections to disk (poles are shared by adjacent spans), so re-runs are
fast. Reports end-to-end chain accuracy next to the GT-points ceiling, and the midspan
detection rate (the dominant loss channel).

Usage:
    python scripts/eval_wire_tracing_e2e.py                  # full run (GPU)
    python scripts/eval_wire_tracing_e2e.py --limit 60       # quick subset
    python scripts/eval_wire_tracing_e2e.py --diagnose 4     # detected-vs-GT alignment
    python scripts/eval_wire_tracing_e2e.py --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import WIRE_TRACING_DATASET_DIR
from src.wire_tracing_match import MatchConfig, NumpyEdgeCostModel
from src.wire_tracer import DEFAULT_UNIFIED_WEIGHTS
from src.wire_tracing_e2e import (
    Detectors, load_detectors, resolve_span_photos, detect_pole_points,
    _detect_pole_points_raw, dedup_pole_points_for_photo, ruler_fit_for_photo,
    detect_midspan_points_strip,
    score_span_e2e, inch_tol_pct, resolve_gt_frame, load_span_metric_meta,
)


def _acc(d):
    n = d["n"]
    return (round(d["chain"] / n, 4) if n else None,
            round(d["A"] / n, 4) if n else None,
            round(d["B"] / n, 4) if n else None, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", type=Path, default=WIRE_TRACING_DATASET_DIR / "spans.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap number of resolvable spans")
    ap.add_argument("--diagnose", type=int, default=0, help="print detected-vs-GT alignment for N spans and exit")
    ap.add_argument("--cache", type=Path, default=Path(".e2e_det_cache.json"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--assoc-tol", type=float, default=6.0, help="detected->GT association tol (%% of image)")
    ap.add_argument("--pole-tol-inch", type=float, default=3.0, help="pole association tol in INCHES (PPI-converted per span); 0 to use --assoc-tol percent")
    ap.add_argument("--mid-tol-inch", type=float, default=6.0, help="midspan (clean) association tol in INCHES (PPI-converted per span); 0 for percent")
    ap.add_argument("--mid-crossarm-tol-inch", type=float, default=36.0, help="looser midspan tol (inches) for crossarm bundles (horizontally-parallel wires)")
    ap.add_argument("--pole-crossarm-tol-inch", type=float, default=36.0, help="looser pole tol (inches) for crossarm arm points (collapsed arm + localization)")
    ap.add_argument("--wt", type=float, default=0.2)
    ap.add_argument("--wc", type=float, default=0.25)
    ap.add_argument("--wdead", type=float, default=0.06)
    ap.add_argument("--dust", type=float, default=None,
                    help="matcher dustbin threshold (normalized-cost units). Default 0.18. "
                         "RAISING it (~0.5) force-matches more midspan wires onto pole nodes — the "
                         "matcher over-dustbins detected wires; ~0.5 gives +1.7pp chain acc (trades a "
                         "little true-orphan precision, a tiny population). See docs/reports.")
    ap.add_argument("--wcouple-class", type=float, default=0.0,
                    help="finer A<->B coupling on predicted wire_class (unified pole source)")
    ap.add_argument("--pole-conf", type=float, default=None)
    ap.add_argument("--pole-imgsz", type=int, default=None, help="detector imgsz on pole crop")
    ap.add_argument("--unified-weights", default=DEFAULT_UNIFIED_WEIGHTS,
                    help="unified_pole_detection weights (the pole node source)")
    ap.add_argument("--unified-conf-json", default=None,
                    help="per-class conf thresholds JSON (tuned operating point); runs at a 0.01 floor + per-class gate")
    ap.add_argument("--unified-conf", type=float, default=None,
                    help="flat unified conf op-point (default: Detectors default 0.20)")
    ap.add_argument("--unified-imgsz", type=int, default=960, help="unified model imgsz on the pole crop")
    ap.add_argument("--pole-dedup-y", type=float, default=0.6,
                    help="kind-aware height-band (%%) to collapse duplicate pole detections (0=off); "
                         "0.6 is +2.27pp e2e (see matcher_and_detection_levers_report.md)")
    ap.add_argument("--pole-dedup-inch", type=float, default=4.0,
                    help="PHYSICAL dedup band in INCHES via the projective ruler model (DEPLOYED "
                         "DEFAULT 4.0: +0.40pp e2e over 0.6%% percent; overrides --pole-dedup-y where a "
                         "ruler fits, falls back to it otherwise). Pass 0 for the legacy percent band.")
    ap.add_argument("--sweep-dedup-inch", default=None,
                    help="comma list of inch dedup bands to sweep (e.g. '0,1,2,3,4,6,8'); reuses one "
                         "raw-detection cache and prints chain/A/B per value. 0 entry = percent --pole-dedup-y.")
    ap.add_argument("--strip-weights", default=None, help="override midspan strip (HRNet) weights")
    ap.add_argument("--wx", type=float, default=None, help="matcher w_x (set 0 for strip midspan)")
    ap.add_argument("--class-signal", default="none", choices=["none", "hw_tier", "cable_type"],
                    help="matcher class cost source (hw_tier uses the detected hardware tier directly)")
    ap.add_argument("--wclass", type=float, default=0.15, help="matcher w_class (used when class-signal != none)")
    ap.add_argument("--no-gt-frame", action="store_true",
                    help="disable GT-frame alignment for midspan (legacy: detect on all burst frames)")
    ap.add_argument("--frame-tol", type=float, default=2.0,
                    help="max GT-to-frame match dist (%% height) before excluding a frame-mismatch span")
    ap.add_argument("--strip-height", type=float, default=None,
                    help="strip peak height gate (default 0.6); lower recovers faint top wires")
    ap.add_argument("--strip-prom", type=float, default=None, help="strip peak prominence (default 0.05)")
    ap.add_argument("--strip-width-expand", type=float, default=1.0,
                    help="widen ruler-column strip crop about its centre (1.0=legacy). MUST match the "
                         "width the --strip-weights were trained at (prepare_dataset --strip-width-expand).")
    ap.add_argument("--oracle-crossarm-mult", action="store_true",
                    help="stamp GT crossarm multiplicity onto detections (ceiling of an arm wire-count head)")
    ap.add_argument("--per-chain-crossarm", action="store_true",
                    help="score crossarms per-chain (legacy) instead of bundle-level")
    ap.add_argument("--fill-residual", action="store_true",
                    help="monotonic freed-slot recovery: bind a dusted midspan to an empty same-point "
                         "slot when cheaper than dust (pole-top-pin-over-crossarm). Port of sdk v3.")
    ap.add_argument("--sag-gate", action="store_true",
                    help="two-sided chord-sag gate (projective ruler feet): post-match, drop endpoints "
                         "with sag (=(h_a+h_b)/2-h_m) outside [--sag-min,--sag-max]. Fly-over + outlier.")
    ap.add_argument("--sag-min", type=float, default=-4.0, help="min plausible sag (ft); below = fly-over")
    ap.add_argument("--sag-max", type=float, default=20.0, help="max plausible sag (ft); above = outlier")
    ap.add_argument("--sag-tol", type=float, default=4.0, help="fly-over overshoot tolerance (ft)")
    ap.add_argument("--sag-elev", action="store_true",
                    help="use USGS ground elevations (spans_metric.jsonl _meta) for the sag chord — "
                         "elevation-corrected absolute heights, like v3. Flat-earth (e=0) otherwise.")
    ap.add_argument("--sag-len-frac", type=float, default=0.0,
                    help="span-length-dependent max sag: sag_max_eff = max(--sag-max, frac*span_ft). "
                         "0 = flat --sag-max. ~0.05-0.1 is a generous physical cap (longer span sags more).")
    ap.add_argument("--sag-covered-only", action="store_true",
                    help="restrict the eval to spans that HAVE USGS elevation meta (clean test of the "
                         "elevation-corrected sag gate, undiluted by flat-earth-fallback spans).")
    ap.add_argument("--monotonic", action="store_true",
                    help="non-crossing (order-preserving) bottom-up matching (replaces Hungarian)")
    ap.add_argument("--comm-isolation", action="store_true",
                    help="hard three_bolt(comm) isolation: forbid three_bolt<->non-three_bolt across the span")
    ap.add_argument("--edge-model", default=None,
                    help="LEARNED edge-cost model JSON (scripts/train_edge_matcher.py); replaces the "
                         "hand-tuned geometric cost. +~3-4pp chain acc. Keep coupling on; set --edge-dust.")
    ap.add_argument("--edge-dust", type=float, default=None,
                    help="dustbin threshold for the learned cost (trainer prints the recommended value)")
    ap.add_argument("--heldout-manifest", default=None,
                    help="v2 site-split manifest JSON (datasets/split_manifest_v2.json); restricts the "
                         "eval to spans whose job belongs to --heldout-split sites — the HONEST e2e "
                         "number (no span shares a site with detector training photos)")
    ap.add_argument("--heldout-split", default="test", choices=["train", "val", "test"],
                    help="which v2 split's sites to evaluate on (default test)")
    args = ap.parse_args()

    spans = [json.loads(l) for l in open(args.spans) if l.strip()]
    if args.heldout_manifest:
        v2 = json.loads(Path(args.heldout_manifest).read_text())
        allowed = set(v2["heldout_span_jobs"][args.heldout_split])
        before = len(spans)
        spans = [s for s in spans if s.get("job") in allowed]
        print(f"held-out filter ({args.heldout_split} sites): {before} -> {len(spans)} spans "
              f"({len(allowed)} eligible jobs)")
    for s in spans:
        s["_photos"] = resolve_span_photos(s)
    resolvable = [s for s in spans if s["_photos"]["resolvable"]]
    print(f"spans: {len(spans)} total | {len(resolvable)} photo-resolvable "
          f"({100*len(resolvable)/len(spans):.0f}%)")
    if args.limit:
        resolvable = resolvable[:args.limit]
        print(f"  limited to {len(resolvable)}")

    weights = {}
    if args.strip_weights: weights["strip"] = args.strip_weights
    if args.unified_weights: weights["unified"] = args.unified_weights
    det = load_detectors(device=args.device, weights=weights or None, midspan_source="strip")
    det.unified_imgsz = args.unified_imgsz
    det.pole_dedup_y = args.pole_dedup_y
    det.pole_dedup_inch = args.pole_dedup_inch
    if args.unified_conf_json:
        det.unified_conf_per_class = json.loads(Path(args.unified_conf_json).read_text())
        det.unified_conf = 0.01     # low floor; per-class gate decides what's kept
    elif args.unified_conf is not None:
        det.unified_conf = args.unified_conf
    if args.pole_conf is not None: det.pole_conf = args.pole_conf
    if args.pole_imgsz is not None: det.pole_crop_imgsz = args.pole_imgsz
    if args.strip_height is not None: det.strip_peak_height = args.strip_height
    if args.strip_prom is not None: det.strip_peak_prom = args.strip_prom
    det.strip_width_expand = args.strip_width_expand
    # strip midspan points share the ruler x -> height-only matching + scoring
    mid_axis = "y"
    if args.wx is None:
        args.wx = 0.0
    print(f"pole_source={det.pole_source} midspan_source={det.midspan_source} mid_axis={mid_axis} "
          f"wx={args.wx} | unified_conf={det.unified_conf} "
          f"unified_weights={args.unified_weights or 'default'}")

    # ---- detection cache (keyed by photo path + detector signature) ----
    # The signature invalidates the cache when anything that changes detection output
    # changes (source, confs, weights), so multiple configs can share one cache file.
    # NOTE: dedup is intentionally NOT in the signature — the cache stores RAW (pre-dedup)
    # pole points and dedup is applied post-cache (pole_dets), so one cache serves every
    # --pole-dedup-* value and the inch sweep is CPU-cheap (no re-detection).
    pole_sig = f"Praw|{det.pole_source}|isz{det.pole_crop_imgsz}|pc{det.pole_conf}|" \
               f"uni{args.unified_weights or 'none'}|uc{det.unified_conf}|uisz{det.unified_imgsz}|" \
               f"ucj{args.unified_conf_json or 'none'}::"
    mid_sig = f"M|{det.midspan_source}|{args.strip_weights or 'def'}|" \
              f"sh{det.strip_peak_height}|sp{det.strip_peak_prom}|swe{det.strip_width_expand}::"
    cache = {}
    if not args.no_cache and args.cache.exists():
        cache = json.loads(args.cache.read_text())
        print(f"loaded {len(cache)} cached detections from {args.cache}")

    def pole_cached(photo):
        """RAW (pre-dedup) pole detections, cached (shared across all dedup values)."""
        key = pole_sig + photo
        if key not in cache:
            cache[key] = _detect_pole_points_raw(photo, det)
        return cache[key]

    def pole_dets(photo):
        """Cached raw points with the CURRENT dedup config applied (cheap, post-cache)."""
        return dedup_pole_points_for_photo(pole_cached(photo), photo, det)

    def mid_cached(photos):
        key = mid_sig + "||".join(photos)
        if key not in cache:
            cache[key] = detect_midspan_points_strip(photos, det)
        return cache[key]

    # ---- diagnostic: detected vs GT alignment ----
    if args.diagnose:
        for s in resolvable[:args.diagnose]:
            ph = s["_photos"]
            dA = pole_dets(ph["A"]); dM = mid_cached(ph["M"])
            gA = [(round(p["y"], 1), round(p["x"], 1)) for p in s["sides"]["A"] if p["x"] is not None]
            pA = [(round(p["y"], 1), round(p["x"], 1)) for p in dA]
            gM = [(round(p["y"], 1), round(p["x"], 1)) for p in s["sides"]["M"] if p["x"] is not None]
            pM = [(round(p["y"], 1), round(p["x"], 1)) for p in dM]
            print(f"\n=== {s['job']} A.scid={s['pole_a']['scid']} ===")
            print(f"  poleA GT  (y,x)% sorted: {sorted(gA)}")
            print(f"  poleA DET (y,x)% sorted: {sorted(pA)}")
            print(f"  midsp GT  (y,x)% sorted: {sorted(gM)}")
            print(f"  midsp DET (y,x)% sorted: {sorted(pM)}")
        return

    cfg_kw = dict(class_signal=args.class_signal, w_class=args.wclass,
                  w_couple_tier=args.wt, w_couple_chain=args.wc, w_deadend=args.wdead,
                  w_couple_class=args.wcouple_class,
                  monotonic=args.monotonic, comm_isolation=args.comm_isolation,
                  fill_residual=args.fill_residual)
    if args.wx is not None:
        cfg_kw["w_x"] = args.wx
    cfg = MatchConfig(**cfg_kw)
    if args.dust is not None:
        cfg.dust = args.dust
    if args.edge_model:
        cfg.edge_model = NumpyEdgeCostModel.load(args.edge_model)
        if args.edge_dust is not None:
            cfg.dust = args.edge_dust
        print(f"LEARNED edge cost: {args.edge_model}  (dust={cfg.dust})")
    if args.pole_tol_inch or args.mid_tol_inch:
        print(f"association tol (PPI-converted/span): pole {args.pole_tol_inch}\" clean / "
              f"{args.pole_crossarm_tol_inch}\" crossarm | midspan {args.mid_tol_inch}\" clean / "
              f"{args.mid_crossarm_tol_inch}\" crossarm")
    else:
        print(f"association tol: {args.assoc_tol}% of image (legacy percent gate)")
    metric_meta = {}
    if (args.sag_gate and (args.sag_elev or args.sag_len_frac)) or args.sag_covered_only:
        metric_meta = load_span_metric_meta()
        n_cov = sum(1 for s in resolvable
                    if (s.get("job"), s.get("connection_id")) in metric_meta
                    and metric_meta[(s.get("job"), s.get("connection_id"))].get("e_a") is not None)
        print(f"sag elevations/length: {n_cov}/{len(resolvable)} spans have metric meta "
              f"(elev={args.sag_elev}, len_frac={args.sag_len_frac})")
        if args.sag_covered_only:
            before = len(resolvable)
            resolvable = [s for s in resolvable
                          if metric_meta.get((s.get("job"), s.get("connection_id")), {}).get("e_a") is not None]
            print(f"restricted to {len(resolvable)}/{before} elevation-covered spans")

    def run_pass(verbose=True):
        """One full scoring pass over the resolvable spans using the CURRENT det dedup
        config. Reuses the shared raw-detection + midspan caches, so re-runs across dedup
        values only re-do CPU dedup+match+score. Returns (agg, n_frame_excluded)."""
        agg = {"clean": {"n": 0, "A": 0, "B": 0, "chain": 0},
               "ambig": {"n": 0, "A": 0, "B": 0, "chain": 0},
               "midspan_detected": 0, "midspan_total": 0, "sag_dropped": 0,
               "prec": {"prop_A": 0, "corr_A": 0, "prop_B": 0, "corr_B": 0}}
        n_frame_excluded = 0
        for idx, s in enumerate(resolvable):
            ph = s["_photos"]
            # align midspan detection to the GT-annotation burst frame (camera moved between
            # frames); exclude spans whose GT matches no available frame.
            mid_photos = ph["M"]
            if not args.no_gt_frame:
                gt_frame, _d = resolve_gt_frame(s, ph["M"], tol_pct=args.frame_tol)
                if gt_frame is None:
                    n_frame_excluded += 1
                    continue
                mid_photos = [gt_frame]
            try:
                dA = pole_dets(ph["A"]); dB = pole_dets(ph["B"]); dM = mid_cached(mid_photos)
            except Exception as e:
                print(f"  ! {s['job']} span {idx}: detection failed: {e}")
                continue
            # PROJECTION-model association (no PPI): pass raw INCH tolerances + per-photo fits.
            tA = args.pole_tol_inch or None
            tB = args.pole_tol_inch or None
            tM = (args.mid_tol_inch if mid_photos else None) or None
            tMc = (args.mid_crossarm_tol_inch if mid_photos else None) or None
            tAc = args.pole_crossarm_tol_inch or None
            tBc = args.pole_crossarm_tol_inch or None
            fitA = ruler_fit_for_photo(ph["A"], "pole")
            fitB = ruler_fit_for_photo(ph["B"], "pole")
            fitM = ruler_fit_for_photo(mid_photos[0], "midspan") if mid_photos else None
            sag_fits = None
            sag_elev = (0.0, 0.0, 0.0)
            sag_max_eff = args.sag_max
            if args.sag_gate:
                sag_fits = (ruler_fit_for_photo(ph["A"], "pole"),
                            ruler_fit_for_photo(ph["B"], "pole"),
                            ruler_fit_for_photo(mid_photos[0], "midspan") if mid_photos else None)
                if args.sag_elev or args.sag_len_frac:
                    m = metric_meta.get((s.get("job"), s.get("connection_id")), {})
                    if args.sag_elev:
                        sag_elev = (m.get("e_a") or 0.0, m.get("e_b") or 0.0, m.get("e_mid") or 0.0)
                    if args.sag_len_frac and m.get("length_ft"):
                        sag_max_eff = max(args.sag_max, args.sag_len_frac * m["length_ft"])
            r = score_span_e2e(s, dA, dM, dB, cfg, assoc_tol_pct=args.assoc_tol, mid_assoc_axis=mid_axis,
                               bundle_crossarm=not args.per_chain_crossarm, tol_A=tA, tol_B=tB, tol_M=tM,
                               tol_M_crossarm=tMc, tol_A_crossarm=tAc, tol_B_crossarm=tBc,
                               oracle_crossarm_mult=args.oracle_crossarm_mult,
                               sag_fits=sag_fits, sag_min=args.sag_min, sag_max=sag_max_eff,
                               sag_tol=args.sag_tol, sag_elev=sag_elev,
                               fit_A=fitA, fit_B=fitB, fit_M=fitM)
            for b in ("clean", "ambig"):
                for k in ("n", "A", "B", "chain"):
                    agg[b][k] += r[b][k]
            agg["midspan_detected"] += r["midspan_detected"]
            agg["midspan_total"] += r["midspan_total"]
            agg["sag_dropped"] += r.get("sag_dropped", 0)
            for pk in ("prop_A", "corr_A", "prop_B", "corr_B"):
                agg["prec"][pk] += r.get("prec", {}).get(pk, 0)
            if verbose and (idx + 1) % 50 == 0:
                print(f"  ...{idx+1}/{len(resolvable)} spans")
        return agg, n_frame_excluded

    # ---- SWEEP MODE: vary the pole dedup band over the shared cache ----
    if args.sweep_dedup_inch:
        vals = [float(v) for v in args.sweep_dedup_inch.split(",") if v.strip() != ""]
        print(f"\nSWEEP pole dedup over {vals} (0 = percent --pole-dedup-y {args.pole_dedup_y}); "
              f"clean A/B pt = secondary-rack node-recall proxy\n")
        print(f"{'dedup':>8}{'overall ch':>12}{'A pt':>8}{'B pt':>8}{'clean ch':>10}"
              f"{'clean A':>9}{'clean B':>9}{'mid%':>7}")
        rows = []
        for v in vals:
            # v == 0 -> legacy percent dedup; v > 0 -> inch dedup (ruler model)
            det.pole_dedup_inch = v if v > 0 else None
            agg, _nfx = run_pass(verbose=False)
            ov = {k: agg["clean"][k] + agg["ambig"][k] for k in ("n", "A", "B", "chain")}
            n = max(ov["n"], 1)
            cn = max(agg["clean"]["n"], 1)
            mr = 100 * agg["midspan_detected"] / max(agg["midspan_total"], 1)
            row = (v, ov["chain"]/n, ov["A"]/n, ov["B"]/n,
                   agg["clean"]["chain"]/cn, agg["clean"]["A"]/cn, agg["clean"]["B"]/cn, mr)
            rows.append(row)
            tag = "%" if v == 0 else "in"
            print(f"{v:>6.1f}{tag:>2}{row[1]:>12.4f}{row[2]:>8.4f}{row[3]:>8.4f}"
                  f"{row[4]:>10.4f}{row[5]:>9.4f}{row[6]:>9.4f}{row[7]:>7.1f}")
        best = max(rows, key=lambda r: r[1])
        print(f"\nbest overall chain acc: {best[1]:.4f} at dedup={best[0]:.1f}"
              f"{'%' if best[0] == 0 else ' in'}")
        if not args.no_cache:
            args.cache.write_text(json.dumps(cache))
        return

    agg, n_frame_excluded = run_pass()

    if not args.no_cache:
        args.cache.write_text(json.dumps(cache))

    overall = {k: agg["clean"][k] + agg["ambig"][k] for k in ("n", "A", "B", "chain")}
    print("\n" + "=" * 64)
    print(f"END-TO-END (real detectors)   {len(resolvable)-n_frame_excluded} spans, {overall['n']} GT chains")
    print("=" * 64)
    if not args.no_gt_frame:
        print(f"GT-frame alignment ON: midspan detected on the GT burst frame; "
              f"{n_frame_excluded} spans excluded (no frame matches GT within {args.frame_tol}%)")
    dedup_desc = (f"{args.pole_dedup_inch}in (ruler projective)" if args.pole_dedup_inch
                  else f"{args.pole_dedup_y}% (image-height)")
    print(f"pole dedup band: {dedup_desc}")
    if args.sag_gate:
        print(f"sag gate ON [{args.sag_min},{args.sag_max}]ft tol={args.sag_tol}: "
              f"{agg['sag_dropped']} endpoints dropped")
    pa, ca = agg["prec"]["prop_A"], agg["prec"]["corr_A"]
    pb, cb = agg["prec"]["prop_B"], agg["prec"]["corr_B"]
    prop, corr = pa + pb, ca + cb
    print(f"endpoint marker PRECISION (correct/proposed): A {ca}/{pa}={ca/max(pa,1):.4f}  "
          f"B {cb}/{pb}={cb/max(pb,1):.4f}  overall {corr}/{prop}={corr/max(prop,1):.4f}")
    print(f"midspan-wire detection rate: {agg['midspan_detected']}/{agg['midspan_total']} "
          f"= {100*agg['midspan_detected']/max(agg['midspan_total'],1):.1f}%  "
          f"(upper bound on recoverable chains)")
    print(f"{'bucket':<22}{'n':>7}{'A pt':>9}{'B pt':>9}{'chain':>9}")
    for name, b in (("clean (per-trace)", agg["clean"]), ("crossarm-group", agg["ambig"]), ("overall", overall)):
        ch, A, B, n = _acc(b)
        print(f"{name:<22}{n:>7}{A:>9}{B:>9}{ch:>9}")
    # chain acc conditioned on the midspan being detected (isolates matcher from midspan recall)
    det_rate = agg["midspan_detected"] / max(agg["midspan_total"], 1)
    ch = overall["chain"] / max(overall["n"], 1)
    print(f"\noverall chain acc: {ch:.4f}  |  conditioned on midspan detected: "
          f"{ch/det_rate if det_rate else 0:.4f}")


if __name__ == "__main__":
    main()
