"""
One-time ONNX export for the wire-tracer pipeline — V2 (unified + learned matcher).

Run from the project root (needs torch, ultralytics, onnx, onnxruntime):

    python sdk/wire_tracer_sdk/v2.5/tools/export_onnx.py

Exports THREE ONNX models + bundles TWO JSON artifacts into v2/wire_tracer/weights/
(pole_detection is shared with calibration_sdk and NOT exported here):

    unified_pole_detection.onnx        YOLO pose, 17-class, 1 keypoint   (imgsz 960)
    midspan_wire_strip_detection.onnx  HRNet single-channel heatmap      (1740x96 RULER-LINE, best_f1)
    edge_matcher_unified_v2.json       learned per-edge cost model (copied verbatim)
    unified_perclass_conf.json         per-class F1-optimal conf map (copied verbatim)

V2 vs V1: drops wire_detection.onnx + wire_attachment_hw_detection.onnx (the union pole
source); adds unified_pole_detection.onnx + the two JSON artifacts. The strip checkpoint is the
deployed EPOCH-39 best.pth (F1-selected), not the old val-loss checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "wire_tracer" / "weights"   # self-relative: tools/../wire_tracer/weights
sys.path.insert(0, str(REPO_ROOT))

# Source checkpoints resolve through the model registry ONLY (models/registry.json is the
# single source of truth; models/production/<name>/production symlinks the current version).
# Never point these at runs/ — promote first (scripts/deploy_ops/promote_model.py).
SRC = {
    "unified_pole_detection": REPO_ROOT / "models/production/unified_pole_detection/production/model.pt",
    "midspan_wire_strip_detection": REPO_ROOT / "models/production/midspan_wire_strip_detection/production/model.pth",
    "midspan_tier_classifier": REPO_ROOT / "models/production/midspan_tier_classifier/production/model.pth",
}
# JSON artifacts copied verbatim into the bundle. v2.2 ships ONLY the learned matcher — the
# per-class pole conf map is DROPPED (flat-0.20 op-point, see v2.1/README + constants.py).
JSON_SRC = {
    "edge_matcher_unified_v2.json": REPO_ROOT / "models/edge_matcher_unified_v2.json",
}

UNIFIED_IMGSZ = 960     # unified runs on the pole crop at 960 (build_default_tracer unified_imgsz)
RULER_IMGSZ = 960
STRIP_HW = (1740, 96)   # v2.6 ruler-line strip resolution (WIRE_STRIP_HEATMAP_*)

import torch  # noqa: E402


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
    out_path = Path(model.export(
        format="onnx", imgsz=imgsz, opset=17, simplify=True,
        dynamic=False, half=False, nms=False, device="cpu",
    ))
    if out_path.resolve() != dst_onnx.resolve():
        shutil.move(str(out_path), str(dst_onnx))
    names = list(model.names.values()) if hasattr(model, "names") else []
    return {
        "input_shape": [1, 3, imgsz, imgsz],
        "input_layout": "NCHW", "input_dtype": "float32",
        "input_range": "[0, 1] (RGB after letterbox)",
        "class_names": names,
    }


def export_strip(src_pth: Path, dst_onnx: Path) -> dict:
    from src.models import KeypointDetector
    h, w = STRIP_HW
    print(f"\n[HRNet:strip] {src_pth.name}  ->  {dst_onnx.name}  in={STRIP_HW} (1 channel)")
    model = KeypointDetector(num_keypoints=1, heatmap_size=STRIP_HW, weights_path=None)
    ckpt = torch.load(str(src_pth), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, h, w, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()
    torch.onnx.export(
        model, dummy, str(dst_onnx), opset_version=17,
        input_names=["input"], output_names=["heatmaps"],
        dynamic_axes=None, do_constant_folding=True,
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(str(dst_onnx), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["heatmaps"], {"input": dummy.numpy()})[0]
    max_abs = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"    parity max_abs_diff = {max_abs:.3e}")
    if max_abs > 1e-3:
        raise RuntimeError(f"ONNX export parity check failed for strip: {max_abs}")
    return {
        "input_shape": [1, 3, h, w], "input_layout": "NCHW", "input_dtype": "float32",
        "input_range": "ImageNet-normalized RGB ([0,1] then mean/std)",
        "heatmap_shape": [1, 1, h, w], "num_keypoints": 1, "parity_max_abs": max_abs,
    }


def export_tier(src_pth: Path, dst_onnx: Path) -> dict:
    """v2.9: resnet18 midspan tier classifier (4-class incl 'none' veto), raw-RGB/255 input."""
    import torch.nn as nn
    from torchvision.models import resnet18
    ck = torch.load(str(src_pth), map_location="cpu", weights_only=False)
    assert ck["arch"] == "resnet18"
    n_out = len(ck.get("tiers", ("bare", "multiplex", "comm", "none")))
    print(f"\n[resnet18:tier] {src_pth.name}  ->  {dst_onnx.name}  in=(64, 256) out={n_out}")
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, n_out)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    dummy = torch.randn(2, 3, 64, 256, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()
    torch.onnx.export(
        model, dummy, str(dst_onnx), opset_version=17,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(str(dst_onnx), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"input": dummy.numpy()})[0]
    max_abs = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"    parity max_abs_diff = {max_abs:.3e}")
    if max_abs > 1e-3:
        raise RuntimeError(f"ONNX export parity check failed for tier: {max_abs}")
    return {"input_shape": [-1, 3, 64, 256], "input_layout": "NCHW", "input_dtype": "float32",
            "input_range": "RGB/255 (NO ImageNet norm)", "num_classes": n_out,
            "classes": list(ck.get("tiers", ())), "parity_max_abs": max_abs}


def _entry(onnx: Path, kind: str, extra: dict) -> dict:
    data_path = onnx.with_suffix(onnx.suffix + ".data")
    entry = {"file": onnx.name, "size_bytes": total_size_with_external(onnx),
             "sha256": sha256_of(onnx), "kind": kind, **extra}
    if data_path.exists():
        entry["external_data_file"] = data_path.name
        entry["external_data_size_bytes"] = data_path.stat().st_size
    return entry


def main() -> int:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, p in {**SRC, **JSON_SRC}.items():
        if not p.exists():
            print(f"ERROR: missing source: {p}", file=sys.stderr)
            return 1

    models = {}
    models["unified_pole_detection"] = _entry(
        WEIGHTS_DIR / "unified_pole_detection.onnx", "yolo_pose",
        export_yolo(SRC["unified_pole_detection"], WEIGHTS_DIR / "unified_pole_detection.onnx", UNIFIED_IMGSZ))
    models["midspan_wire_strip_detection"] = _entry(
        WEIGHTS_DIR / "midspan_wire_strip_detection.onnx", "hrnet",
        export_strip(SRC["midspan_wire_strip_detection"],
                     WEIGHTS_DIR / "midspan_wire_strip_detection.onnx"))
    models["midspan_tier_classifier"] = _entry(
        WEIGHTS_DIR / "midspan_tier_classifier.onnx", "resnet18",
        export_tier(SRC["midspan_tier_classifier"],
                    WEIGHTS_DIR / "midspan_tier_classifier.onnx"))

    artifacts = {}
    for dst_name, src in JSON_SRC.items():
        dst = WEIGHTS_DIR / dst_name
        shutil.copyfile(src, dst)
        artifacts[dst_name] = {"file": dst_name, "size_bytes": dst.stat().st_size,
                               "sha256": sha256_of(dst), "kind": "json", "source": str(src.relative_to(REPO_ROOT))}
        print(f"\n[copy] {src.name}  ->  {dst_name}")

    # Bundled model versions: resolve each SRC through the registry so the manifest records
    # exactly which production model versions this SDK build packages (SDK <-> model link).
    registry = json.loads((REPO_ROOT / "models/registry.json").read_text())
    bundled = {}
    for name, src in SRC.items():
        m = registry["models"][name]
        ver = m["production_version"]
        bundled[name] = {
            "version": ver,
            "source": str(src.resolve().relative_to(REPO_ROOT)),
            "source_sha256": m["versions"][ver].get("sha256") or sha256_of(src.resolve()),
        }

    manifest = {
        "format": "onnx", "opset": 17, "sdk_version": "v2.9.1",
        "pole_weights": "shared with sdk/calibration_sdk/v2/calibration/weights/pole_detection.onnx",
        "ruler_weights": "shared with sdk/calibration_sdk/v2/calibration/weights/ruler_detection.onnx",
        "bundled_model_versions": bundled,
        "models": models, "artifacts": artifacts,
    }
    (WEIGHTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_mb = sum(m["size_bytes"] for m in models.values()) / (1024 * 1024)
    print(f"\nDone. {len(models)} ONNX models + {len(artifacts)} JSON artifacts, ONNX total {total_mb:.1f} MB")
    print(f"Wrote manifest -> {WEIGHTS_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
