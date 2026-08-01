# Pole calibration pipeline — desktop integration package

> **v2 (2026-06-23) — PRODUCTION.** Calibration models retrained on the honest
> site-disjoint split (`split_manifest.json`), exported from the honest `runs/`
> weights. **Verified leak-free TIE vs v1/`_preHonest`**: pole_detection mAP50
> 0.9950 = 0.9950, ruler_detection 0.9950 ≈ 0.9950 (calibration is a saturated,
> well-generalizing task — the held-out training data doesn't move the needle, so
> the honest model is functionally an all-sites model). ONNX↔torch parity:
> HRNet ruler_marking 1.1e-05 / pole_top 5.7e-06; YOLO ultralytics-validated.
> End-to-end smoke test PASS (pole→ruler→ticks→pole-top). Re-export:
> `USE_PRODUCTION_MODELS=false python sdk/calibration_sdk/v2/tools/export_onnx.py`.
> Version chain: v1 (preHonest all-sites) → **v2 (honest, deployed)**.

Self-contained ONNX inference for the pole-photo calibration pipeline:

```
pole detection  →  ruler detection  →  ruler-marking keypoints  →  pole-top keypoint
```

Designed to be dropped into a tkinter Windows desktop app. **No torch, no
ultralytics on the destination machine** — just `onnxruntime`, `numpy`,
`opencv-python-headless`, and `Pillow`.

---

## What's in this folder

```
sdk/calibration_sdk/
├── README.md                ← you are here
├── INTEGRATION.md           ← copy-paste tkinter recipe
├── requirements.txt         ← runtime deps for the destination Windows machine
├── calibration/             ← the importable package
│   ├── __init__.py            (exports CalibrationPipeline)
│   ├── pipeline.py            (4-stage orchestration; public API)
│   ├── yolo_onnx.py           (YOLO ONNX + numpy NMS)
│   ├── hrnet_onnx.py          (HRNet ONNX + heatmap decode)
│   ├── tta.py                 (vertical-shift TTA, off by default)
│   ├── visualize.py           (OpenCV box / keypoint drawing)
│   ├── cli.py                 (`python -m calibration.cli image.jpg`)
│   ├── constants.py           (mirror of training-repo config; no deps)
│   └── weights/               (4 ONNX models + sidecar .data + manifest.json)
└── tools/
    └── export_onnx.py       ← one-time dev-side script (not shipped to consumer)
```

## Install

On the destination Windows machine (Python 3.12, CPU-only):

```cmd
pip install -r sdk\calibration_sdk\requirements.txt
```

That's it. Drop the `calibration/` folder into the consuming app
and `from calibration import CalibrationPipeline`.

## Quick smoke-test

```bash
cd sdk/calibration_sdk
PYTHONPATH=. python -m calibration.cli path/to/photo.jpg --annotated out.png
```

You should see something like:

```
image: photo.jpg  shape=(3840, 2560)  tta=False
  pole:     {'bbox': (1062, 730, 1434, 3435), 'conf': 0.879}
  ruler:    {'bbox': (1203, 2276, 1287, 3366), 'conf': 0.821}
     2.5 ft -> (1249.3, 3176.6) conf=0.982
     6.5 ft -> (1249.3, 2927.6) conf=0.981
    10.5 ft -> (1249.3, 2691.5) conf=0.963
    14.5 ft -> (1242.3, 2465.2) conf=0.981
    16.5 ft -> (1238.8, 2356.2) conf=0.981
  pole_top: {'x': 1249.9, 'y': 899.8, 'conf': 0.897}
```

Add `--tta` for higher-precision keypoint localisation (~2-3× slower).

## Result schema

`CalibrationPipeline.run(image)` returns:

```python
{
  "pole":            {"bbox": (x1, y1, x2, y2), "conf": float} | None,
  "ruler":           {"bbox": (x1, y1, x2, y2), "conf": float} | None,
  "ruler_keypoints": [
      {"name": "2.5", "x": float, "y": float, "conf": float, "weighted_conf": float},
      ...  # 5 keypoints: 2.5, 6.5, 10.5, 14.5, 16.5 ft
  ] | None,
  "pole_top":        {"x": float, "y": float, "conf": float} | None,
  "image_shape":     (height, width),
  "annotated_image": np.ndarray,   # only if return_annotated=True (RGB)
}
```

All x/y are in the **original image's pixel coordinates**. Pass these straight
to Katapult's X/Y upload.

## Public API

```python
from calibration import CalibrationPipeline

# Construct once, reuse for many images. Models are loaded lazily.
pipe = CalibrationPipeline()

# Inputs supported: file path | ndarray RGB | PIL.Image
result = pipe.run("photo.jpg")

# Optional flags:
result = pipe.run(
    "photo.jpg",
    use_tta=False,          # True for vertical-shift TTA (slower, tighter)
    return_annotated=False, # True to include "annotated_image" in result
    detect_pole=True,       # False to skip pole + pole-top (e.g. midspan)
)

pipe.warmup()  # optional: pre-load all sessions before first run()
```

## Updating weights

If you retrain any model in the parent repo:

```bash
USE_PRODUCTION_MODELS=true python sdk/calibration_sdk/tools/export_onnx.py
```

This regenerates `sdk/calibration_sdk/calibration/weights/*.onnx` and rewrites
`manifest.json`. The export does a numerical parity check (`max_abs_diff`)
between torch and ONNX outputs and aborts if it exceeds 1e-3.

This script needs the dev-side dependencies (`torch`, `ultralytics`,
`onnx`, `onnxruntime`, `onnxscript`). It is **not** required on the destination
Windows machine.

## Footprint

| Item                                          | Size    |
|-----------------------------------------------|---------|
| ONNX weights (4 models, bundled)              | ~150 MB |
| `onnxruntime` + `numpy` + `opencv-headless` + `Pillow` wheels | ~60 MB  |
| **Total**                                     | **~210 MB** |

Reference: a torch + ultralytics setup on the destination would be ~900 MB.
