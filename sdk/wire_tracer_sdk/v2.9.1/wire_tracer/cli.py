"""
CLI smoke-test for the wire-tracer SDK (V2).

    PYTHONPATH=wire_tracer python -m wire_tracer.cli \
        --pole-a A.jpg --pole-b B.jpg --midspan M0.jpg M1.jpg \
        --json out.json --annotated grid.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .pipeline import WireTracerPipeline


def _fmt_att(a: dict) -> str:
    xk = f" x{a['wire_count']}" if a.get("role") == "crossarm" and a.get("wire_count", 1) > 1 else ""
    cab = f"  cable_hint={a['cable_type_hint']}?" if a.get("cable_type_hint") else ""
    return (f"  {a.get('id','')} {a['insulator_name']:<14}{xk:<4} y={a.get('y',0):>5.1f}%  "
            f"role={a.get('role','')}  tier_hint={a.get('tier_hint')}{cab}")


def _fmt_report(r: dict) -> str:
    L = [f"midspan wires detected: {r['midspan_wire_count']}", "--- Pole A attachments ---"]
    L += [_fmt_att(a) for a in r["poles"]["A"]] or ["  (none)"]
    L.append("--- Pole B attachments ---")
    L += [_fmt_att(b) for b in r["poles"]["B"]] or ["  (none)"]
    L.append("--- Traces (midspan -> A insulator <-> B insulator) ---")
    for t in r["traces"]:
        a = t["pole_a_insulator"] or "(unmatched)"
        b = t["pole_b_insulator"] or "(unmatched)"
        L.append(f"  {t['midspan_id']} (y={t['midspan_y']:>5.1f}%): {a} [A] <-> {b} [B]")
    L.append("cable_type_hint / crossarm_k are NON-authoritative model hints; "
             "wire_type stays blank for the user to assign.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Wire-tracer SDK v2 smoke test")
    ap.add_argument("--pole-a", required=True)
    ap.add_argument("--pole-b", required=True)
    ap.add_argument("--midspan", required=True, nargs="+", help="one or more burst frames")
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--pole-weights", default=None)
    ap.add_argument("--json", default=None, help="write result JSON here")
    ap.add_argument("--annotated", default=None, help="write the 3-panel PNG here")
    args = ap.parse_args(argv)

    pipe = WireTracerPipeline(weights_dir=args.weights_dir, pole_weights_path=args.pole_weights)
    r = pipe.run(args.pole_a, args.midspan, args.pole_b, return_annotated=bool(args.annotated))

    annotated = r.pop("annotated_image", None)
    print(_fmt_report(r))
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2))
        print(f"\nwrote {args.json}")
    if annotated is not None and args.annotated:
        cv2.imwrite(args.annotated, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"wrote {args.annotated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
