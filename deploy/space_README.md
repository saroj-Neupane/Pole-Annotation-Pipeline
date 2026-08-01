---
title: Pole Annotation Demo
emoji: 🗼
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Utility-pole photo analysis — calibration, equipment, wire tracing
models:
  - nsaroj789/pole-annotation-models
---

# Pole Annotation — Interactive Demo

Computer-vision pipelines for utility-pole field photos, running entirely on
CPU with ONNX:

- **Calibration** — detect the pole, the height ruler, its tick keypoints
  (2.5–16.5 ft), and the pole top, enabling pixel→feet projection.
- **Equipment annotation** — YOLO equipment detection (transformer, riser,
  street light, drip loop) + per-class HRNet keypoints.
- **Wire tracing** — trace which pole attachment connects to which midspan
  wire across a span (pole A ↔ midspan ↔ pole B): YOLO-pose joint-class pole
  detection, a 1-D ruler-strip wire detector, and a learned edge-cost
  bipartite matcher with non-crossing constraints.

Upload your own photos or use the bundled samples. Weights download from
[nsaroj789/pole-annotation-models](https://huggingface.co/nsaroj789/pole-annotation-models)
at startup.

Source code: see the linked GitHub repository on the profile.
