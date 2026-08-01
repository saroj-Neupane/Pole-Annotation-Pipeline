"""
One-time ONNX export for the equipment annotation pipeline.

Run from the project root:

    USE_PRODUCTION_MODELS=true python sdk/equipment_annotation_sdk/tools/export_onnx.py

Writes five ONNX models to equipment_annotation/weights/ (pole is shared with
calibration_sdk — not exported here).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]   # tools/ -> v1/ -> equipment_annotation_sdk/ -> sdk/ -> repo
WEIGHTS_DIR = (
    Path(__file__).resolve().parents[1] / "equipment_annotation" / "weights"   # self-relative (v2.6 lesson)
)

sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("USE_PRODUCTION_MODELS", "true")

import torch  # noqa: E402

from src.config import (  # noqa: E402
    EQUIPMENT_DETECTION_CONFIG,
    EQUIPMENT_KEYPOINT_CONFIGS,
    INFERENCE_EQUIPMENT_WEIGHTS,
    RISER_KEYPOINT_DETECTION_CONFIG,
    RISER_NUM_KEYPOINTS,
    SECONDARY_DRIP_LOOP_KEYPOINT_DETECTION_CONFIG,
    SECONDARY_DRIP_LOOP_NUM_KEYPOINTS,
    STREET_LIGHT_KEYPOINT_DETECTION_CONFIG,
    STREET_LIGHT_NUM_KEYPOINTS,
    TRANSFORMER_KEYPOINT_DETECTION_CONFIG,
    TRANSFORMER_NUM_KEYPOINTS,
)
from src.models import KeypointDetector  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def total_size_with_external(onnx_path: Path) -> int:
    total = onnx_path.stat().st_size
    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if data_path.exists():
        total += data_path.stat().st_size
    return total


def export_yolo(src_pt: Path, dst_onnx: Path, imgsz: int) -> dict:
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
    print(f"\n[HRNet:{label}] {src_pth.name}  ->  {dst_onnx.name}  in={input_size} hm={heatmap_size}")
    model = KeypointDetector(
        num_keypoints=num_keypoints,
        heatmap_size=heatmap_size,
        weights_path=None,
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
        dynamic_axes=None,
        do_constant_folding=True,
    )

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


def _cfg_sizes(cfg: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (cfg["resize_height"], cfg["resize_width"]),
        (cfg["heatmap_height"], cfg["heatmap_width"]),
    )


def main() -> int:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    equip_pt = Path(INFERENCE_EQUIPMENT_WEIGHTS)
    if not equip_pt.exists():
        print(f"ERROR: missing source weight: {equip_pt}", file=sys.stderr)
        return 1

    hrnet_exports = [
        ("riser", "riser_keypoint_detection.onnx", RISER_NUM_KEYPOINTS, RISER_KEYPOINT_DETECTION_CONFIG),
        ("transformer", "transformer_keypoint_detection.onnx", TRANSFORMER_NUM_KEYPOINTS, TRANSFORMER_KEYPOINT_DETECTION_CONFIG),
        ("street_light", "street_light_keypoint_detection.onnx", STREET_LIGHT_NUM_KEYPOINTS, STREET_LIGHT_KEYPOINT_DETECTION_CONFIG),
        ("secondary_drip_loop", "secondary_drip_loop_keypoint_detection.onnx", SECONDARY_DRIP_LOOP_NUM_KEYPOINTS, SECONDARY_DRIP_LOOP_KEYPOINT_DETECTION_CONFIG),
    ]

    for cls_name, onnx_name, num_kp, cfg in hrnet_exports:
        _, _, weights = EQUIPMENT_KEYPOINT_CONFIGS[cls_name]
        pth = Path(weights)
        if not pth.exists():
            print(f"ERROR: missing source weight: {pth}", file=sys.stderr)
            return 1

    equip_imgsz = EQUIPMENT_DETECTION_CONFIG["imgsz"]
    equip_onnx = WEIGHTS_DIR / "equipment_detection.onnx"
    equip_meta = export_yolo(equip_pt, equip_onnx, equip_imgsz)

    hrnet_metas = {}
    for cls_name, onnx_name, num_kp, cfg in hrnet_exports:
        _, _, weights = EQUIPMENT_KEYPOINT_CONFIGS[cls_name]
        input_size, heatmap_size = _cfg_sizes(cfg)
        dst = WEIGHTS_DIR / onnx_name
        hrnet_metas[cls_name] = export_hrnet(
            Path(weights), dst, num_kp, heatmap_size, input_size, cls_name,
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
        "pole_weights": "shared with sdk/calibration_sdk/calibration/weights/pole_detection.onnx",
        "models": {
            "equipment_detection": _entry(equip_onnx, "yolo", equip_meta),
            "riser_keypoint_detection": _entry(
                WEIGHTS_DIR / "riser_keypoint_detection.onnx", "hrnet", hrnet_metas["riser"],
            ),
            "transformer_keypoint_detection": _entry(
                WEIGHTS_DIR / "transformer_keypoint_detection.onnx", "hrnet", hrnet_metas["transformer"],
            ),
            "street_light_keypoint_detection": _entry(
                WEIGHTS_DIR / "street_light_keypoint_detection.onnx", "hrnet", hrnet_metas["street_light"],
            ),
            "secondary_drip_loop_keypoint_detection": _entry(
                WEIGHTS_DIR / "secondary_drip_loop_keypoint_detection.onnx",
                "hrnet",
                hrnet_metas["secondary_drip_loop"],
            ),
        },
    }
    (WEIGHTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_mb = sum(m["size_bytes"] for m in manifest["models"].values()) / (1024 * 1024)
    print(f"\nDone. {len(manifest['models'])} ONNX models, total {total_mb:.1f} MB")
    print(f"Wrote manifest -> {WEIGHTS_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
