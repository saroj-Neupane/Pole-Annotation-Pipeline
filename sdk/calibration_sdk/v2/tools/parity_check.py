"""
Compare ONNX calibration output against the torch baseline on real images.

Run from the repo root:

    USE_PRODUCTION_MODELS=true python desktop_app/tools/parity_check.py [N]

Picks the first N images from inference/pole/images and checks:

  * pole bbox: IoU
  * ruler bbox: IoU
  * ruler keypoints: per-keypoint pixel distance
  * pole_top: pixel distance

Both pipelines are run with TTA OFF (default) for a fair comparison.

This is a dev-side script. Not shipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "desktop_app"))

os.environ.setdefault("USE_PRODUCTION_MODELS", "true")

import cv2  # noqa: E402

from calibration import CalibrationPipeline  # noqa: E402
from src.inference import load_all_models, run_end_to_end_inference_simple  # noqa: E402


def iou(b1, b2) -> float:
    if b1 is None or b2 is None:
        return float("nan")
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-9)


def main(n: int = 5) -> int:
    images = sorted((REPO_ROOT / "inference" / "pole" / "images").glob("*.jpg"))[:n]
    if not images:
        print("No images found.", file=sys.stderr)
        return 1

    print(f"Loading torch models...")
    torch_models = load_all_models()
    print(f"Loading ONNX pipeline...")
    onnx_pipe = CalibrationPipeline()
    onnx_pipe.warmup()

    rows = []
    for p in images:
        print(f"\n{p.name}")

        # Torch baseline (TTA off, no viz to keep it quiet)
        t_res = run_end_to_end_inference_simple(p, torch_models, use_tta=False, show_visualization=False)
        # ONNX
        o_res = onnx_pipe.run(p, use_tta=False)

        # Pole bbox
        t_pole = t_res.get("pole")
        o_pole = o_res["pole"]["bbox"] if o_res["pole"] else None
        pole_iou = iou(t_pole, o_pole) if t_pole is not None else float("nan")
        print(f"  pole  iou={pole_iou:.4f}  torch={t_pole}  onnx={o_pole}")

        # Ruler bbox
        t_ruler = t_res.get("ruler")
        o_ruler = o_res["ruler"]["bbox"] if o_res["ruler"] else None
        ruler_iou = iou(t_ruler, o_ruler) if t_ruler is not None else float("nan")
        print(f"  ruler iou={ruler_iou:.4f}  torch={t_ruler}  onnx={o_ruler}")

        # Ruler keypoints
        kp_max_d = float("nan")
        if t_res.get("keypoints") and o_res.get("ruler_keypoints"):
            t_by = {k["name"]: (k["x"], k["y"]) for k in t_res["keypoints"]}
            o_by = {k["name"]: (k["x"], k["y"]) for k in o_res["ruler_keypoints"]}
            ds = []
            for name in t_by.keys() & o_by.keys():
                d = np.hypot(t_by[name][0] - o_by[name][0], t_by[name][1] - o_by[name][1])
                ds.append(d)
                print(f"    kp {name:>4}ft  d={d:.3f} px  torch={t_by[name]}  onnx={o_by[name]}")
            kp_max_d = float(np.max(ds)) if ds else float("nan")

        # Pole top
        pt_d = float("nan")
        t_pt = t_res.get("pole_top"); o_pt = o_res.get("pole_top")
        if t_pt and o_pt:
            pt_d = float(np.hypot(t_pt["x"] - o_pt["x"], t_pt["y"] - o_pt["y"]))
            print(f"  pole_top d={pt_d:.3f} px  torch=({t_pt['x']:.2f},{t_pt['y']:.2f})  onnx=({o_pt['x']:.2f},{o_pt['y']:.2f})")

        rows.append((p.name, pole_iou, ruler_iou, kp_max_d, pt_d))

    # Summary
    print("\n=== Parity summary ===")
    print(f"{'image':<32}  {'pole_iou':>9}  {'ruler_iou':>10}  {'kp_max_d':>9}  {'top_d':>7}")
    for name, pi, ri, kpd, ptd in rows:
        print(f"{name:<32}  {pi:>9.4f}  {ri:>10.4f}  {kpd:>9.3f}  {ptd:>7.3f}")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sys.exit(main(n))
