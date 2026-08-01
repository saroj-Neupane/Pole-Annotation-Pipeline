"""
Parity check: SDK (numpy + ONNX) vs the training repo (torch + scipy) on a real span — V2.

Runs both pipelines on the same pole-A / midspan / pole-B photos and compares:
  * the numpy Hungarian + find_peaks ports against scipy (unit-level, synthetic);
  * the strip wire y-positions, unified pole-point detections, and final A<->B traces
    (end-to-end, on real photos), including the learned-cost matcher.

The reference side is src/wire_tracer.build_default_tracer() + trace_span() — whose DEFAULTS are
already the V2 config (pole_source='unified' + learned edge matcher), so this verifies the numpy/
ONNX port reproduces the production torch pipeline.

Run from the project root (needs torch, ultralytics, scipy for the reference side):

    python sdk/wire_tracer_sdk/v2.9/tools/parity_check.py                  # first resolvable span
    python sdk/wire_tracer_sdk/v2.9/tools/parity_check.py --job MNRV-FR02 --scid-a 095 --scid-b 094
    python sdk/wire_tracer_sdk/v2.9/tools/parity_check.py --skip-e2e       # numpy-ops only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SDK_PKG = Path(__file__).resolve().parents[1]   # SELF-RELATIVE (kills the v2.8 copy-forward bug)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SDK_PKG))


def check_numpy_ops() -> bool:
    """Unit-parity: SDK linear_sum_assignment / find_peaks vs scipy on random inputs."""
    from scipy.optimize import linear_sum_assignment as scipy_lsa
    from scipy.signal import find_peaks as scipy_fp
    from wire_tracer.numpy_ops import linear_sum_assignment as sdk_lsa
    from wire_tracer.numpy_ops import find_peaks as sdk_fp

    rng = np.random.default_rng(0)
    ok = True
    n_bad = 0
    for _ in range(200):
        R = rng.integers(1, 7)
        C = rng.integers(1, 9)
        cost = rng.random((R, C)) * 10
        sr, sc = scipy_lsa(cost)
        dr, dc = sdk_lsa(cost)
        if abs(cost[sr, sc].sum() - cost[dr, dc].sum()) > 1e-6:
            n_bad += 1
    print(f"  linear_sum_assignment: {200 - n_bad}/200 random matrices match optimum cost")
    ok &= (n_bad == 0)

    n_bad = 0
    for _ in range(200):
        x = rng.random(400)
        sp, _ = scipy_fp(x, height=0.40, distance=12, prominence=0.02)
        dp, _ = sdk_fp(x, height=0.40, distance=12, prominence=0.02)
        if not np.array_equal(sp, dp):
            n_bad += 1
    print(f"  find_peaks:            {200 - n_bad}/200 random signals match exactly")
    ok &= (n_bad == 0)
    return ok


def _resolve_span(job, scid_a, scid_b):
    """Return (span, photos) for a resolvable span. Defaults to the first resolvable in spans.jsonl."""
    from src.config import WIRE_TRACING_DATASET_DIR
    from src.wire_tracing_match import load_spans
    from src.wire_tracing_e2e import resolve_span_photos
    spans = load_spans(WIRE_TRACING_DATASET_DIR / "spans.jsonl")
    if job:
        spans = [s for s in spans if s["job"] == job
                 and (scid_a is None or str(s["pole_a"]["scid"]) == str(scid_a))
                 and (scid_b is None or str(s["pole_b"]["scid"]) == str(scid_b))]
    for s in spans:
        photos = resolve_span_photos(s)
        if photos["resolvable"]:
            return s, photos
    return None, None


def _ref_trace(span, photos):
    """Reference output via src/wire_tracer (torch + scipy), production defaults (= V2 config)."""
    from src.wire_tracer import build_default_tracer, trace_span
    det, cfg = build_default_tracer(device="cpu")
    span = dict(span)
    span["_photos"] = photos
    return trace_span(span, det, cfg)


def _sdk_trace(pole_a, midspan, pole_b):
    from wire_tracer import WireTracerPipeline
    pipe = WireTracerPipeline()
    return pipe.run(pole_a, midspan, pole_b, return_annotated=True)


def _summ(traces):
    return [(t["midspan_id"], t["pole_a_insulator"], t["pole_b_insulator"]) for t in traces]


def check_e2e(span, photos, save_dir: Path) -> bool:
    print(f"\n[end-to-end on real photos]  {span['job']} {span['pole_a']['scid']}->{span['pole_b']['scid']}")
    ref = _ref_trace(span, photos)
    sdk = _sdk_trace(photos["A"], photos["M"], photos["B"])

    rm = sorted(round(m["y"], 1) for m in ref["midspan"])
    sm = sorted(round(m["y"], 1) for m in sdk["midspan"])
    print(f"  midspan wires:  ref={ref['midspan_wire_count']}  sdk={sdk['midspan_wire_count']}")
    print(f"    ref y%: {rm}")
    print(f"    sdk y%: {sm}")
    print(f"  pole A att:  ref={len(ref['poles']['A'])}  sdk={len(sdk['poles']['A'])}")
    print(f"  pole B att:  ref={len(ref['poles']['B'])}  sdk={len(sdk['poles']['B'])}")
    rt, st = _summ(ref["traces"]), _summ(sdk["traces"])
    print(f"  traces ref: {rt}")
    print(f"  traces sdk: {st}")

    y_ok = len(rm) == len(sm) and all(abs(a - b) <= 0.5 for a, b in zip(rm, sm))
    trace_ok = rt == st
    print(f"  => midspan match: {y_ok} | trace match: {trace_ok}")

    if save_dir is not None:
        import json
        import cv2
        save_dir.mkdir(parents=True, exist_ok=True)
        ann = sdk.pop("annotated_image", None)
        stem = f"span_{span['pole_a']['scid']}_to_{span['pole_b']['scid']}"
        (save_dir / f"{stem}.json").write_text(json.dumps(sdk, indent=2))
        if ann is not None:
            cv2.imwrite(str(save_dir / f"{stem}.png"), cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
        print(f"  saved test_results/{stem}.json + .png")
    return y_ok and trace_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=None)
    ap.add_argument("--scid-a", default=None)
    ap.add_argument("--scid-b", default=None)
    ap.add_argument("--skip-e2e", action="store_true", help="only run the numpy-ops unit parity")
    ap.add_argument("--save-dir", default=str(SDK_PKG / "test_results"))
    args = ap.parse_args(argv)

    print("[numpy-ops unit parity vs scipy]")
    ops_ok = check_numpy_ops()

    e2e_ok = True
    if not args.skip_e2e:
        span, photos = _resolve_span(args.job, args.scid_a, args.scid_b)
        if span is None:
            print("\n[end-to-end] SKIPPED — no resolvable span found")
        else:
            e2e_ok = check_e2e(span, photos, Path(args.save_dir))

    print(f"\nPARITY: numpy-ops={'OK' if ops_ok else 'FAIL'}  e2e={'OK' if e2e_ok else 'FAIL'}")
    return 0 if (ops_ok and e2e_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
