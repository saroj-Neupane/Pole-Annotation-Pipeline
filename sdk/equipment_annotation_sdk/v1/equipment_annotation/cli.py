"""
Command-line smoke-test for the equipment annotation pipeline.

    python -m equipment_annotation.cli IMAGE.jpg [--annotated out.png] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .pipeline import EquipmentAnnotationPipeline


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="equipment_annotation.cli")
    parser.add_argument("image", type=Path, help="Path to a single image file")
    parser.add_argument("--annotated", type=Path, default=None, help="Optional output PNG with annotations")
    parser.add_argument("--json", dest="json_path", type=Path, default=None, help="Optional output JSON")
    parser.add_argument("--weights-dir", type=Path, default=None, help="ONNX weights directory")
    parser.add_argument(
        "--pole-weights",
        type=Path,
        default=None,
        help="Path to pole_detection.onnx (default: calibration_sdk shared weights)",
    )
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    pipe = EquipmentAnnotationPipeline(
        weights_dir=args.weights_dir,
        pole_weights_path=args.pole_weights,
    )
    result = pipe.run(args.image, return_annotated=args.annotated is not None)

    print(f"image: {args.image}  shape={result['image_shape']}")
    print(f"  pole:        {result['pole']}")
    print(f"  crop_bounds: {result['crop_bounds']}")
    print(f"  equipment:   {len(result['equipment'])} detection(s)")
    for det in result["equipment"]:
        print(f"    {det['cls_name']} bbox={det['bbox']} conf={det['conf']:.3f}")
        for kp in det.get("keypoints") or []:
            print(f"      {kp['name']} -> ({kp['x']:.1f}, {kp['y']:.1f}) conf={kp['conf']:.3f}")

    if args.annotated is not None:
        bgr = cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR)
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
