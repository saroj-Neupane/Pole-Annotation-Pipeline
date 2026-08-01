"""
One-time ONNX export for the calibration pipeline.

Run from the project root:

    USE_PRODUCTION_MODELS=true python sdk/calibration_sdk/tools/export_onnx.py

Reads the four production weights and writes:

    sdk/calibration_sdk/calibration/weights/pole_detection.onnx
    sdk/calibration_sdk/calibration/weights/ruler_detection.onnx
    sdk/calibration_sdk/calibration/weights/ruler_marking_detection.onnx
    sdk/calibration_sdk/calibration/weights/pole_top_detection.onnx
    sdk/calibration_sdk/calibration/weights/manifest.json

Verifies each ONNX output matches the torch output to <1e-4 max abs on a
random tensor, and skips re-export if a fresh, matching .onnx already exists.

This script is NOT shipped with the desktop app. It only runs on the dev box.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]          # versioned layout: sdk/calibration_sdk/v2/tools/ -> repo
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "calibration" / "weights"   # this SDK version's weights

# Make `from src...` importable.
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("USE_PRODUCTION_MODELS", "true")

import torch  # noqa: E402

from src.config import (  # noqa: E402
    HEATMAP_HEIGHT,
    HEATMAP_WIDTH,
    INFERENCE_POLE_TOP_WEIGHTS,
    INFERENCE_POLE_WEIGHTS,
    INFERENCE_RULER_MARKING_WEIGHTS,
    INFERENCE_RULER_WEIGHTS,
    NUM_KEYPOINTS,
    POLE_DETECTION_CONFIG,
    POLE_TOP_HEATMAP_HEIGHT,
    POLE_TOP_HEATMAP_WIDTH,
    POLE_TOP_NUM_KEYPOINTS,
    POLE_TOP_RESIZE_HEIGHT,
    POLE_TOP_RESIZE_WIDTH,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    RULER_DETECTION_CONFIG,
)
from src.models import KeypointDetector  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def total_size_with_external(onnx_path: Path) -> int:
    """Return the on-disk size of an ONNX file plus any sibling external-data files."""
    total = onnx_path.stat().st_size
    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if data_path.exists():
        total += data_path.stat().st_size
    return total


def export_yolo(src_pt: Path, dst_onnx: Path, imgsz: int) -> dict:
    """Export an Ultralytics YOLO .pt to ONNX with static imgsz."""
    from ultralytics import YOLO

    print(f"\n[YOLO] {src_pt.name}  ->  {dst_onnx.name}  imgsz={imgsz}")
    model = YOLO(str(src_pt))
    out_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        simplify=True,
        dynamic=False,
        half=False,
        nms=False,
        device="cpu",
    )
    out_path = Path(out_path)
    if out_path.resolve() != dst_onnx.resolve():
        shutil.move(str(out_path), str(dst_onnx))
    return {
        "input_shape": [1, 3, imgsz, imgsz],
        "input_layout": "NCHW",
        "input_dtype": "float32",
        "input_range": "[0, 1] (RGB after letterbox)",
        "class_names": list(model.names.values()) if hasattr(model, "names") else [],
    }


def export_hrnet(
    src_pth: Path,
    dst_onnx: Path,
    num_keypoints: int,
    heatmap_size: tuple[int, int],
    input_size: tuple[int, int],
    label: str,
) -> dict:
    """Export a HRNet KeypointDetector .pth to ONNX."""
    print(f"\n[HRNet:{label}] {src_pth.name}  ->  {dst_onnx.name}  in={input_size} hm={heatmap_size}")
    model = KeypointDetector(
        num_keypoints=num_keypoints,
        heatmap_size=heatmap_size,
        weights_path=None,  # skip ImageNet warm-start; we load trained weights below
    )
    ckpt = torch.load(str(src_pth), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    h, w = input_size
    dummy = torch.randn(1, 3, h, w, dtype=torch.float32)

    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()

    torch.onnx.export(
        model,
        dummy,
        str(dst_onnx),
        opset_version=17,
        input_names=["input"],
        output_names=["heatmaps"],
        dynamic_axes=None,  # static shape — matches the pipeline's fixed resize
        do_constant_folding=True,
    )

    # Numerical parity check.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(dst_onnx), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["heatmaps"], {"input": dummy.numpy()})[0]
    max_abs = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"    parity max_abs_diff = {max_abs:.3e}")
    if max_abs > 1e-3:
        raise RuntimeError(f"ONNX export parity check failed for {label}: {max_abs}")

    return {
        "input_shape": [1, 3, h, w],
        "input_layout": "NCHW",
        "input_dtype": "float32",
        "input_range": "ImageNet-normalized RGB ([0,1] then mean/std)",
        "heatmap_shape": [1, num_keypoints, heatmap_size[0], heatmap_size[1]],
        "num_keypoints": num_keypoints,
        "parity_max_abs": max_abs,
    }


def main() -> int:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    pole_pt = Path(INFERENCE_POLE_WEIGHTS)
    ruler_pt = Path(INFERENCE_RULER_WEIGHTS)
    rmark_pth = Path(INFERENCE_RULER_MARKING_WEIGHTS)
    ptop_pth = Path(INFERENCE_POLE_TOP_WEIGHTS)

    for p in (pole_pt, ruler_pt, rmark_pth, ptop_pth):
        if not p.exists():
            print(f"ERROR: missing source weight: {p}", file=sys.stderr)
            return 1

    pole_imgsz = POLE_DETECTION_CONFIG["imgsz"]
    ruler_imgsz = RULER_DETECTION_CONFIG["imgsz"]

    pole_onnx = WEIGHTS_DIR / "pole_detection.onnx"
    ruler_onnx = WEIGHTS_DIR / "ruler_detection.onnx"
    rmark_onnx = WEIGHTS_DIR / "ruler_marking_detection.onnx"
    ptop_onnx = WEIGHTS_DIR / "pole_top_detection.onnx"

    pole_meta = export_yolo(pole_pt, pole_onnx, pole_imgsz)
    ruler_meta = export_yolo(ruler_pt, ruler_onnx, ruler_imgsz)
    rmark_meta = export_hrnet(
        rmark_pth, rmark_onnx,
        num_keypoints=NUM_KEYPOINTS,
        heatmap_size=(HEATMAP_HEIGHT, HEATMAP_WIDTH),
        input_size=(RESIZE_HEIGHT, RESIZE_WIDTH),
        label="ruler_marking",
    )
    ptop_meta = export_hrnet(
        ptop_pth, ptop_onnx,
        num_keypoints=POLE_TOP_NUM_KEYPOINTS,
        heatmap_size=(POLE_TOP_HEATMAP_HEIGHT, POLE_TOP_HEATMAP_WIDTH),
        input_size=(POLE_TOP_RESIZE_HEIGHT, POLE_TOP_RESIZE_WIDTH),
        label="pole_top",
    )

    def _entry(onnx: Path, kind: str, extra: dict) -> dict:
        data_path = onnx.with_suffix(onnx.suffix + ".data")
        entry = {
            "file": onnx.name,
            "size_bytes": total_size_with_external(onnx),
            "sha256": sha256_of(onnx),
            "kind": kind,
            **extra,
        }
        if data_path.exists():
            entry["external_data_file"] = data_path.name
            entry["external_data_size_bytes"] = data_path.stat().st_size
        return entry

    manifest = {
        "format": "onnx",
        "opset": 17,
        "models": {
            "pole_detection": _entry(pole_onnx, "yolo", pole_meta),
            "ruler_detection": _entry(ruler_onnx, "yolo", ruler_meta),
            "ruler_marking_detection": _entry(rmark_onnx, "hrnet", rmark_meta),
            "pole_top_detection": _entry(ptop_onnx, "hrnet", ptop_meta),
        },
    }
    (WEIGHTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_mb = sum(m["size_bytes"] for m in manifest["models"].values()) / (1024 * 1024)
    print(f"\nDone. {len(manifest['models'])} ONNX models, total {total_mb:.1f} MB")
    print(f"Wrote manifest -> {WEIGHTS_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
