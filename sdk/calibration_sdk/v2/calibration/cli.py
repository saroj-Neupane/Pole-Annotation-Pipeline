"""
Command-line smoke-test for the calibration pipeline.

    python -m calibration.cli IMAGE.jpg [--annotated out.png] [--json out.json] [--tta] [--no-pole]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .pipeline import CalibrationPipeline


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        # Only metadata in JSON; never the annotated image.
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calibration.cli")
    parser.add_argument("image", type=Path, help="Path to a single image file")
    parser.add_argument("--annotated", type=Path, default=None, help="Optional output PNG with annotations drawn")
    parser.add_argument("--json", dest="json_path", type=Path, default=None, help="Optional output JSON file with detections")
    parser.add_argument("--tta", action="store_true", help="Enable vertical-shift TTA (slower, tighter localization)")
    parser.add_argument("--no-pole", action="store_true", help="Skip pole + pole-top detection (e.g. midspan photos)")
    parser.add_argument("--weights-dir", type=Path, default=None, help="Override the path to the ONNX weights directory")
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    pipe = CalibrationPipeline(weights_dir=args.weights_dir)
    result = pipe.run(
        args.image,
        use_tta=args.tta,
        return_annotated=args.annotated is not None,
        detect_pole=not args.no_pole,
    )

    # Stdout summary.
    print(f"image: {args.image}  shape={result['image_shape']}  tta={args.tta}")
    print(f"  pole:     {result['pole']}")
    print(f"  ruler:    {result['ruler']}")
    if result["ruler_keypoints"]:
        for kp in result["ruler_keypoints"]:
            print(f"    {kp['name']:>4} ft -> ({kp['x']:.1f}, {kp['y']:.1f}) conf={kp['conf']:.3f}")
    print(f"  pole_top: {result['pole_top']}")

    if args.annotated is not None:
        annotated = result["annotated_image"]
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        args.annotated.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.annotated), bgr)
        print(f"wrote {args.annotated}")

    if args.json_path is not None:
        payload = _to_jsonable({k: v for k, v in result.items() if k != "annotated_image"})
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
