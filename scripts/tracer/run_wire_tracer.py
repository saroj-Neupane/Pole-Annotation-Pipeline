#!/usr/bin/env python3
"""
Run the `wire_tracer` inference pipeline on N pole-mid-pole groups and emit, per group, a
non-MI-job-like reconstruction (pole insulators + implied/crossarm wires + midspan crossings
+ A<->B traces) as a readable report plus a per-group JSON sidecar.

Groups are drawn from datasets/wire_tracing_dataset/spans.jsonl, which is non-MI by
construction (MI-regime jobs are excluded at build time), and spread across distinct jobs.

Models used (the production set): pole_detection, unified_pole_detection,
midspan_wire_strip_detection (+ ruler_detection for the strip-column fallback).

Usage:
    python scripts/tracer/run_wire_tracer.py --n 10
    python scripts/tracer/run_wire_tracer.py --n 10 --device cpu --out wire_tracer_out
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.wire_tracer import (
    build_default_tracer, pick_groups, trace_span, format_trace_report,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", type=Path,
                    default=config.WIRE_TRACING_DATASET_DIR / "spans.jsonl")
    ap.add_argument("--n", type=int, default=10, help="number of pole-mid-pole groups")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("wire_tracer_out"))
    # detector operating points (production config: unified pole source + strip midspan)
    ap.add_argument("--edge-model", default="auto",
                    help="learned edge-cost matcher path; 'auto' enables the unified model, "
                         "'none' to disable")
    ap.add_argument("--edge-dust", type=float, default=1.0, help="dustbin threshold for the learned cost")
    ap.add_argument("--pole-imgsz", type=int, default=1024)
    ap.add_argument("--mult-cap", type=int, default=0,
                    help="max midspan wires one pole point may absorb (0 = unbounded)")
    ap.add_argument("--pole-dedup-y", type=float, default=1.5,
                    help="height band (%% of image) to merge duplicate pole detections (0 = off)")
    args = ap.parse_args()

    spans = [json.loads(l) for l in open(args.spans) if l.strip()]
    picked, n_resolvable = pick_groups(spans, args.n)
    print(f"spans: {len(spans)} total | {n_resolvable} photo-resolvable | picked {len(picked)} "
          f"across {len({s['job'] for s in picked})} job(s)")

    edge_model = None if str(args.edge_model).lower() == "none" else args.edge_model
    det, cfg = build_default_tracer(
        device=args.device, pole_imgsz=args.pole_imgsz,
        edge_model=edge_model, edge_dust=args.edge_dust)
    mult_cap = args.mult_cap if args.mult_cap > 0 else None

    print(f"pole_source={det.pole_source} midspan_source={det.midspan_source} | "
          f"unified_conf={det.unified_conf} pole_imgsz={det.pole_crop_imgsz} | "
          f"matcher: {cfg.label()} | mult_cap={mult_cap}")

    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for i, span in enumerate(picked, 1):
        try:
            r = trace_span(span, det, cfg, mult_cap=mult_cap, pole_dedup_y=args.pole_dedup_y)
        except Exception as e:
            print(f"  ! group {i} ({span['job']}): {e}")
            continue
        rep = format_trace_report(r, idx=i)
        reports.append(rep)
        stem = f"group_{i:02d}_{r['job']}_{r['pole_a_scid']}_to_{r['pole_b_scid']}"
        (args.out / f"{stem}.json").write_text(json.dumps(r, indent=2))
        print(f"  [{i}/{len(picked)}] {r['job']} ({r['pole_a_scid']}->{r['pole_b_scid']}): "
              f"A={len(r['poles']['A'])} att, M={r['midspan_wire_count']} wires, B={len(r['poles']['B'])} att")

    report_txt = ("=" * 88 + "\n"
                  "WIRE TRACER — reconstructed non-MI annotation data from photos\n"
                  + "=" * 88 + "\n\n" + "\n\n".join(reports) + "\n")
    report_path = args.out / "wire_tracer_report.txt"
    report_path.write_text(report_txt)
    print(f"\nwrote {len(reports)} group JSONs + {report_path}")
    print("\n" + report_txt)


if __name__ == "__main__":
    main()
