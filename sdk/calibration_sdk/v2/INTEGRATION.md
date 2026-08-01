# Integrating the calibration pipeline into a tkinter desktop app

This file is the copy-paste recipe for the consuming Windows app.

## 1. Drop the folder in

Copy the `sdk/calibration_sdk/calibration/` folder into your tkinter app's source
tree, alongside (or under) the package that owns the inference UI. The folder
is self-contained — including the ONNX weights — so it works regardless of
the rest of your project layout.

```
your_app/
├── main.py                 ← your tkinter entry point
├── ...
└── calibration/            ← copied from sdk/calibration_sdk/calibration/
    ├── __init__.py
    ├── pipeline.py
    ├── yolo_onnx.py
    ├── hrnet_onnx.py
    ├── tta.py
    ├── visualize.py
    ├── cli.py
    ├── constants.py
    └── weights/
        ├── pole_detection.onnx
        ├── ruler_detection.onnx
        ├── ruler_marking_detection.onnx
        ├── ruler_marking_detection.onnx.data
        ├── pole_top_detection.onnx
        ├── pole_top_detection.onnx.data
        └── manifest.json
```

> **Important:** the two `*.onnx.data` files are the actual HRNet weights
> (external data sidecar). Always copy them alongside their `*.onnx` graph
> file — onnxruntime expects them in the same directory and will fail to
> load otherwise.

## 2. Install runtime deps

Add to your app's `requirements.txt` (or `pyproject.toml`):

```
numpy>=1.26,<3
onnxruntime>=1.18,<2
opencv-python-headless>=4.9,<5
Pillow>=10,<12
```

These are the **only** runtime dependencies the calibration pipeline needs.
No torch, no ultralytics.

## 3. Use it from tkinter

Minimal end-to-end example. Loads an image with `PIL`, runs the pipeline,
shows the annotated result in a tkinter window, and prints the (X, Y)
coordinates that your app would upload to Katapult.

```python
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from calibration import CalibrationPipeline

# Load the pipeline once, near app start. Models are lazy-loaded on first run.
pipe = CalibrationPipeline()
pipe.warmup()  # optional — pay the load cost up front

def on_open():
    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if not path:
        return

    result = pipe.run(path, return_annotated=True)

    # ----- Display -----
    annotated_rgb = result["annotated_image"]                # numpy RGB
    pil_img = Image.fromarray(annotated_rgb).resize((800, 1200))
    tk_img = ImageTk.PhotoImage(pil_img)
    canvas.config(width=tk_img.width(), height=tk_img.height())
    canvas.create_image(0, 0, image=tk_img, anchor="nw")
    canvas.image = tk_img  # prevent GC

    # ----- Coordinates to upload to Katapult -----
    if result["pole_top"]:
        pt = result["pole_top"]
        print(f"pole_top: x={pt['x']:.1f}, y={pt['y']:.1f}, conf={pt['conf']:.3f}")

    for kp in result["ruler_keypoints"] or []:
        print(f"ruler {kp['name']}ft: x={kp['x']:.1f}, y={kp['y']:.1f}, conf={kp['conf']:.3f}")

root = tk.Tk()
root.title("Pole Calibration")
tk.Button(root, text="Open image…", command=on_open).pack()
canvas = tk.Canvas(root, bg="black")
canvas.pack()
root.mainloop()
```

## 4. Mapping result keys to Katapult uploads

| Result key                     | What it is                                 | Where it goes                                  |
|--------------------------------|--------------------------------------------|------------------------------------------------|
| `result["pole"]["bbox"]`       | Pole bounding box in pixels                | Optional — UI overlay only                      |
| `result["ruler"]["bbox"]`      | Ruler bounding box in pixels               | Optional — UI overlay only                      |
| `result["ruler_keypoints"]`    | 5 keypoints @ 2.5/6.5/10.5/14.5/16.5 ft    | Upload each `(x, y)` to Katapult per keypoint   |
| `result["pole_top"]`           | Single `(x, y)` for top of pole            | Upload `(x, y)` to Katapult as pole-top marker  |

All coordinates are in **original image pixels** (top-left origin, +y down)
— the same convention tkinter / Katapult expect. No further transformation
needed.

## 5. Performance and threading

- One CPU inference on a 4K image takes ~0.6 s without TTA, ~2 s with TTA on
  a recent Intel laptop. tkinter's main loop will block during inference.
- For large batches or smoother UX, run `pipe.run(...)` on a worker thread:

```python
import threading

def run_in_background(image_path, callback):
    def work():
        result = pipe.run(image_path, return_annotated=True)
        # Marshal back to UI thread:
        canvas.after(0, lambda: callback(result))
    threading.Thread(target=work, daemon=True).start()
```

`onnxruntime` releases the GIL during inference, so a worker thread does not
block tkinter's redraws.

## 6. Updating weights without re-shipping the app

If only the weights need updating, replace these files in
`calibration/weights/` and the app will pick them up on next launch:

```
pole_detection.onnx
ruler_detection.onnx
ruler_marking_detection.onnx
ruler_marking_detection.onnx.data
pole_top_detection.onnx
pole_top_detection.onnx.data
manifest.json
```

(`manifest.json` is informational — the pipeline does not enforce checksums.
Add a check yourself if your supply chain needs it.)

If the model **architecture** changes (different input size, different
keypoint count, different ONNX opset), the consuming code in
`calibration/constants.py` and `calibration/pipeline.py` must be updated to
match — the manifest alone is not enough.

## 7. Common pitfalls

- **`FileNotFoundError: Missing ONNX weight: ...pole_detection.onnx`** —
  the `weights/` folder did not get copied. Verify it sits next to
  `pipeline.py` inside the deployed app.
- **`onnxruntime ... InvalidArgument: Missing external data file`** —
  you copied `*.onnx` without the corresponding `*.onnx.data`. They must
  be next to each other.
- **Images come out tinted** — you passed a BGR `cv2.imread()` result
  directly. The pipeline treats `np.ndarray` inputs as RGB. Either convert
  with `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)` or pass the file path / a
  PIL.Image (handled correctly).
- **Bad keypoints on a midspan photo with no pole** — pass
  `pipe.run(img, detect_pole=False)` to skip the pole and pole-top stages.
