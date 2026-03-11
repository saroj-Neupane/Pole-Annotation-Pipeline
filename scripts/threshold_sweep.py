#!/usr/bin/env python3
"""
Run threshold sweep on YOLO detection models' validation data to find optimal
confidence thresholds (F1-maximizing) per class.

Uses shared logic from src.threshold_utils.
Saves results to runs/threshold_sweep_results.json and optionally updates config.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.threshold_utils import (
    YOLO_DETECTION_MODELS,
    run_threshold_sweep,
    update_config_all,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run threshold sweep on YOLO models to find optimal confidence thresholds."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(YOLO_DETECTION_MODELS),
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "threshold_sweep_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update src/config.py with optimal thresholds",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose val output")
    args = parser.parse_args()

    results = {}
    for model_name in args.models:
        if model_name not in YOLO_DETECTION_MODELS:
            print(f"Unknown model: {model_name}, skipping")
            continue
        wpath, dpath = YOLO_DETECTION_MODELS[model_name]
        weights = PROJECT_ROOT / wpath
        data = PROJECT_ROOT / dpath
        print(f"\nRunning threshold sweep on {model_name}...")
        r = run_threshold_sweep(model_name, weights, data, verbose=args.verbose)
        results[model_name] = r
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Overall optimal: conf={r['optimal_overall']['confidence']}, F1={r['optimal_overall']['f1']}")
            for cls, v in r["optimal_per_class"].items():
                print(f"    {cls}: conf={v['confidence']}, F1={v['f1']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    if args.update_config:
        update_config_all(results)


if __name__ == "__main__":
    main()
