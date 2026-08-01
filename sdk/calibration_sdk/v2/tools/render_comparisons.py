"""
Render side-by-side ONNX vs torch annotated comparisons on real test images.

Run from the repo root:

    USE_PRODUCTION_MODELS=true python desktop_app/tools/render_comparisons.py

Writes one PNG per test image into desktop_app/test_results/ — each PNG has
the torch baseline on the left and the ONNX pipeline on the right, with a
header showing the per-image metrics (bbox IoU, max keypoint delta, pole-top
delta, latency).

Dev-side script. Not shipped.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "desktop_app"))

os.environ.setdefault("USE_PRODUCTION_MODELS", "true")

from calibration import CalibrationPipeline  # noqa: E402
from calibration.visualize import draw_annotations  # noqa: E402
from src.inference import load_all_models, run_end_to_end_inference_simple  # noqa: E402

OUT_DIR = REPO_ROOT / "desktop_app" / "test_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick a mix of pole and midspan photos.
POLE_DIR = REPO_ROOT / "inference" / "pole" / "images"
MIDSPAN_DIR = REPO_ROOT / "inference" / "midspan" / "images"


def torch_result_to_ndict(t_res: dict) -> dict:
    """Adapt the upstream torch result dict to the ONNX-pipeline schema so we
    can reuse calibration.visualize.draw_annotations on it."""
    out = {
        "pole": None,
        "ruler": None,
        "ruler_keypoints": None,
        "pole_top": None,
    }
    pole = t_res.get("pole")
    if pole is not None:
        out["pole"] = {"bbox": tuple(int(v) for v in pole), "conf": 0.0}
    ruler = t_res.get("ruler")
    if ruler is not None:
        out["ruler"] = {"bbox": tuple(int(v) for v in ruler), "conf": 0.0}
    kps = t_res.get("keypoints")
    if kps:
        out["ruler_keypoints"] = [
            {"name": k["name"], "x": float(k["x"]), "y": float(k["y"]), "conf": float(k["conf"])}
            for k in kps
        ]
    pt = t_res.get("pole_top")
    if pt is not None:
        out["pole_top"] = {"x": float(pt["x"]), "y": float(pt["y"]), "conf": float(pt["conf"])}
    return out


def iou(b1, b2) -> float:
    if b1 is None or b2 is None:
        return float("nan")
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-9)


def fit_to_height(rgb: np.ndarray, target_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == target_h:
        return rgb
    scale = target_h / h
    return cv2.resize(rgb, (int(round(w * scale)), target_h), interpolation=cv2.INTER_AREA)


def add_panel_label(rgb: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = rgb.copy()
    pad = 14
    cv2.rectangle(out, (0, 0), (out.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(out, text, (pad, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return out


def add_header(image: np.ndarray, lines: list[str]) -> np.ndarray:
    line_h = 28
    pad = 12
    header_h = line_h * len(lines) + pad * 2
    w = image.shape[1]
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            header, line, (pad, pad + (i + 1) * line_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return np.concatenate([header, image], axis=0)


def make_comparison(
    image_path: Path,
    onnx_pipe: CalibrationPipeline,
    torch_models: dict,
    target_height: int = 1600,
    detect_pole: bool = True,
) -> tuple[np.ndarray, dict]:
    rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    # Torch baseline. For midspan photos (detect_pole=False) we mask out the
    # pole models so the upstream pipeline skips those stages — same as
    # CalibrationPipeline.run(detect_pole=False) does.
    if detect_pole:
        t_models = torch_models
    else:
        t_models = {**torch_models, "pole_detector": None, "pole_top_model": None}

    t0 = time.perf_counter()
    t_res = run_end_to_end_inference_simple(image_path, t_models, use_tta=False, show_visualization=False)
    t_lat = time.perf_counter() - t0
    t_norm = torch_result_to_ndict(t_res)

    # ONNX
    t0 = time.perf_counter()
    o_res = onnx_pipe.run(image_path, use_tta=False, detect_pole=detect_pole)
    o_lat = time.perf_counter() - t0

    # Metrics
    pole_iou = iou(t_norm["pole"]["bbox"] if t_norm["pole"] else None,
                   o_res["pole"]["bbox"] if o_res["pole"] else None)
    ruler_iou = iou(t_norm["ruler"]["bbox"] if t_norm["ruler"] else None,
                    o_res["ruler"]["bbox"] if o_res["ruler"] else None)

    kp_max_d = float("nan")
    if t_norm["ruler_keypoints"] and o_res["ruler_keypoints"]:
        t_by = {k["name"]: (k["x"], k["y"]) for k in t_norm["ruler_keypoints"]}
        o_by = {k["name"]: (k["x"], k["y"]) for k in o_res["ruler_keypoints"]}
        ds = [
            np.hypot(t_by[n][0] - o_by[n][0], t_by[n][1] - o_by[n][1])
            for n in t_by.keys() & o_by.keys()
        ]
        kp_max_d = float(np.max(ds)) if ds else float("nan")

    pt_d = float("nan")
    if t_norm["pole_top"] and o_res["pole_top"]:
        pt_d = float(np.hypot(
            t_norm["pole_top"]["x"] - o_res["pole_top"]["x"],
            t_norm["pole_top"]["y"] - o_res["pole_top"]["y"],
        ))

    # Render
    torch_img = draw_annotations(rgb, t_norm)
    onnx_img = draw_annotations(rgb, o_res)

    torch_img = fit_to_height(torch_img, target_height)
    onnx_img = fit_to_height(onnx_img, target_height)

    torch_img = add_panel_label(torch_img, "Torch baseline (.pt)")
    onnx_img = add_panel_label(onnx_img, "ONNX pipeline (desktop_app)")

    # Vertical separator.
    sep = np.full((target_height, 4, 3), 80, dtype=np.uint8)
    side_by_side = np.concatenate([torch_img, sep, onnx_img], axis=1)

    h_kp = kp_max_d if not np.isnan(kp_max_d) else float("nan")
    h_pt = pt_d if not np.isnan(pt_d) else float("nan")
    header_lines = [
        f"{image_path.name}  ({rgb.shape[1]}x{rgb.shape[0]})",
        f"pole IoU={pole_iou:.4f}    ruler IoU={ruler_iou:.4f}    "
        f"max kp delta={h_kp:.2f}px    pole_top delta={h_pt:.2f}px",
        f"latency  torch={t_lat*1000:.0f}ms    onnx={o_lat*1000:.0f}ms",
    ]
    final = add_header(side_by_side, header_lines)

    metrics = dict(
        image=image_path.name,
        pole_iou=pole_iou, ruler_iou=ruler_iou,
        kp_max_delta=h_kp, pole_top_delta=h_pt,
        torch_latency_ms=t_lat * 1000, onnx_latency_ms=o_lat * 1000,
    )
    return final, metrics


def main() -> int:
    pole_imgs = sorted(POLE_DIR.glob("*.jpg"))[:5]
    midspan_imgs = sorted(MIDSPAN_DIR.glob("*.jpg"))[:3]
    test_imgs = pole_imgs + midspan_imgs

    if not test_imgs:
        print("No test images found.", file=sys.stderr)
        return 1

    print(f"Loading torch models...")
    torch_models = load_all_models()
    print(f"Loading ONNX pipeline...")
    onnx_pipe = CalibrationPipeline()
    onnx_pipe.warmup()

    summary_rows = []
    for img in test_imgs:
        # Midspan photos have no pole in frame; skip the pole+pole-top stages
        # on both pipelines (matches recommended `detect_pole=False` usage).
        is_midspan = "midspan" in str(img.parent).lower()
        kind = "midspan" if is_midspan else "pole"
        print(f"\n>>> [{kind}] {img.name}")
        comp, metrics = make_comparison(
            img, onnx_pipe, torch_models, detect_pole=not is_midspan,
        )
        metrics["kind"] = kind
        out_path = OUT_DIR / f"compare_{img.stem}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(
            f"    pole IoU={metrics['pole_iou']:.4f}  ruler IoU={metrics['ruler_iou']:.4f}  "
            f"kp max d={metrics['kp_max_delta']:.2f}px  top d={metrics['pole_top_delta']:.2f}px"
        )
        print(f"    wrote {out_path.name}")
        summary_rows.append(metrics)

    # Write a markdown summary alongside the PNGs.
    md = ["# Side-by-side comparison: torch baseline vs ONNX pipeline", ""]
    md.append("Both pipelines run with TTA off. Midspan photos run with the pole + pole-top stages disabled (recommended `detect_pole=False` usage).")
    md.append("")
    md.append("| image | kind | pole IoU | ruler IoU | kp max Δ (px) | pole_top Δ (px) | torch (ms) | onnx (ms) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for m in summary_rows:
        pole_iou = "—" if np.isnan(m["pole_iou"]) else f"{m['pole_iou']:.4f}"
        pt_d = "—" if np.isnan(m["pole_top_delta"]) else f"{m['pole_top_delta']:.2f}"
        kp_d = "—" if np.isnan(m["kp_max_delta"]) else f"{m['kp_max_delta']:.2f}"
        md.append(
            f"| `{m['image']}` | {m['kind']} | {pole_iou} | {m['ruler_iou']:.4f} | "
            f"{kp_d} | {pt_d} | "
            f"{m['torch_latency_ms']:.0f} | {m['onnx_latency_ms']:.0f} |"
        )
    md.append("")
    md.append("Generated by `desktop_app/tools/render_comparisons.py`.")
    (OUT_DIR / "README.md").write_text("\n".join(md))
    print(f"\nWrote {len(summary_rows)} comparison PNGs + README.md to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
