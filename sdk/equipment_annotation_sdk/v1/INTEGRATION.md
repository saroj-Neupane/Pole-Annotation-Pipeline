# Integrating equipment annotation into a tkinter desktop app

Copy-paste recipe for the consuming Windows app. Assumes you already integrate
[`sdk/calibration_sdk`](../calibration_sdk) for calibration (shared pole model).

## 1. Drop folders in

```
your_app/
├── main.py
├── calibration/              ← from sdk/calibration_sdk/calibration/
│   └── weights/
│       └── pole_detection.onnx
└── equipment_annotation/     ← from sdk/equipment_annotation_sdk/equipment_annotation/
    └── weights/
        ├── equipment_detection.onnx
        ├── riser_keypoint_detection.onnx (+ .data)
        ├── transformer_keypoint_detection.onnx (+ .data)
        ├── street_light_keypoint_detection.onnx (+ .data)
        ├── secondary_drip_loop_keypoint_detection.onnx (+ .data)
        └── manifest.json
```

> Copy every `*.onnx.data` sidecar next to its `.onnx` graph file.

## 2. Install runtime deps

Same as calibration (one install covers both):

```
numpy>=1.26,<3
onnxruntime>=1.18,<2
opencv-python-headless>=4.9,<5
Pillow>=10,<12
```

## 3. Use from tkinter

```python
from pathlib import Path
from equipment_annotation import EquipmentAnnotationPipeline

POLE_ONNX = Path(__file__).parent / "calibration" / "weights" / "pole_detection.onnx"

pipe = EquipmentAnnotationPipeline(pole_weights_path=POLE_ONNX)
pipe.warmup()

def on_run_inference(image_path: str):
    def worker():
        result = pipe.run(image_path, return_annotated=True)
        root.after(0, lambda: show_result(result))
    threading.Thread(target=worker, daemon=True).start()
```

## 4. Map results to your upload layer

| Result key | Use |
|------------|-----|
| `equipment[].keypoints` | Equipment marker (x, y) per keypoint name |
| `equipment[].bbox` | UI overlay |
| `pole`, `crop_bounds` | Debug / overlay only |

Katapult marker formatting is **not** included in this SDK — map
`cls_name` + `keypoint name` in your app (see `deploy/model_editor_formats.py`
in the training repo for server-side reference).

## 5. Threading

`pipe.run()` releases the GIL during ONNX inference. Run on a worker thread;
update the UI with `root.after(0, ...)`.

## Pitfalls

- **Missing pole ONNX:** set `pole_weights_path` to your copied
  `calibration/weights/pole_detection.onnx`.
- **Missing .onnx.data:** HRNet models fail to load without sidecar files.
- **BGR vs RGB:** pass file paths or RGB ndarrays; the pipeline loads paths as RGB.
