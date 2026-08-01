# Pole equipment annotation — desktop integration package

Self-contained ONNX inference for the equipment annotation pipeline:

```
pole detection  →  upper 70% 2:5 crop  →  equipment detection  →  per-class keypoints
```

Designed to be dropped into a tkinter Windows desktop app. **No torch, no
ultralytics on the destination machine** — just `onnxruntime`, `numpy`,
`opencv-python-headless`, and `Pillow`.

**Pole detection** reuses `pole_detection.onnx` from
[`sdk/calibration_sdk`](../calibration_sdk) (not duplicated in this bundle).

---

## What's in this folder

```
sdk/equipment_annotation_sdk/
├── README.md
├── INTEGRATION.md
├── requirements.txt
├── equipment_annotation/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── crop.py
│   ├── constants.py
│   ├── yolo_onnx.py
│   ├── hrnet_onnx.py
│   ├── visualize.py
│   ├── cli.py
│   └── weights/          (5 ONNX models + .data sidecars + manifest.json)
└── tools/
    ├── export_onnx.py
    └── parity_check.py
```

> **Note:** The `/sdk` directory is gitignored in the parent repo. Ship this
> folder via release zip or submodule as needed.

## Install

On the destination Windows machine (Python 3.12, CPU-only):

```cmd
pip install -r sdk\equipment_annotation_sdk\requirements.txt
```

Copy `equipment_annotation/` into your app and ensure
`calibration/weights/pole_detection.onnx` is available (or pass
`pole_weights_path`).

## Quick smoke-test

```bash
cd sdk/equipment_annotation_sdk
PYTHONPATH=equipment_annotation python -m equipment_annotation.cli path/to/photo.jpg --annotated out.png
```

## Result schema

`EquipmentAnnotationPipeline.run(image)` returns:

```python
{
  "pole":         {"bbox": (x1, y1, x2, y2), "conf": float} | None,
  "crop_bounds":  (x1, y1, x2, y2) | None,
  "equipment": [
      {
          "cls_name": "riser" | "transformer" | "street_light" | "secondary_drip_loop",
          "cls_id": int,
          "bbox": (x1, y1, x2, y2),
          "conf": float,
          "keypoints": [{"name", "x", "y", "conf"}, ...],
      },
      ...
  ],
  "image_shape": (height, width),
  "annotated_image": np.ndarray,  # only if return_annotated=True
}
```

All x/y are in **original image pixel coordinates**.

## Public API

```python
from equipment_annotation import EquipmentAnnotationPipeline

pipe = EquipmentAnnotationPipeline(
    pole_weights_path="path/to/calibration/weights/pole_detection.onnx",
)
pipe.warmup()
result = pipe.run("photo.jpg", return_annotated=True)
```

## Exporting ONNX weights (dev machine)

```bash
USE_PRODUCTION_MODELS=true python sdk/equipment_annotation_sdk/tools/export_onnx.py
```

Requires torch, ultralytics, onnx, onnxruntime on the dev box. Regenerates
`equipment_annotation/weights/*.onnx` and `manifest.json`.

## Footprint

| Item | Size |
|------|------|
| Equipment ONNX weights (5 models) | ~150–200 MB |
| Pole ONNX (shared with calibration_sdk) | ~37 MB (not duplicated) |
| Runtime wheels | ~60 MB |
| **Total (equipment bundle only)** | **~210–260 MB** |
