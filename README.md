# Pole Annotation Pipeline

Computer-vision pipelines that turn utility-pole field photos into structured
annotation data: **height calibration**, **equipment annotation**, and
**wire tracing** across a span. Training runs on PyTorch; inference ships as
three pure-`numpy` + ONNX Runtime SDKs (no torch, no scipy) suitable for
CPU-only desktop deployment.

> **Weights and data are proprietary** and are not part of this repository.
> The training corpus (~24k labeled field photos across four US regions) and
> the trained checkpoints belong to the client engagement that produced them.
> Everything needed to train on your own data — dataset preparation, trainers,
> evaluation harnesses, and the full inference/SDK source — is here.
>
> **[Live demo →](https://pole-annotation-app-98267233596.us-central1.run.app/demo)**
> (weights load server-side; upload your own photos or use the bundled
> anonymized samples — first request after idle takes ~30 s to warm up)

![Span trace sample](assets/span_trace_sample.jpg)

## The three stacks

### 1. Calibration — pixel → feet
Detects the pole, the survey height ruler, its tick keypoints (2.5–16.5 ft),
and the pole top. A 1-D projective fit through the tick anchors gives a
per-photo pixel→height model (validated to ~0.5 in against survey software
output), so every downstream detection can be reported in feet and inches.

Models: 4 × YOLO11 (pole, pole top, ruler, ruler markings — the markings model
is YOLO-pose with tick keypoints).

### 2. Equipment annotation
Shared pole detection → upper-70 % 2:5 crop → YOLO equipment detection
(transformer, riser, street light, secondary drip loop) → per-class HRNet
keypoint heads for attachment-point localization, reported as calibrated
heights.

### 3. Wire tracing — which attachment connects to which wire
Reconstructs span structure from a pole-A / midspan / pole-B photo triplet:

- **Unified pole model** — one YOLO11-pose model over 17 joint classes
  (hardware × cable-tier × crossarm-count: pin/post/davit/deadend/arm2-4+/
  primary/secondary/neutral/catv/telco/fiber/guy/down-guy/…), giving per-pole
  attachment inventory and tracer nodes in a single pass.
- **Ruler-line midspan strip detector** — a 1-D heatmap CNN over a 1740×96
  rectified strip along the calibration ruler line finds wire crossings;
  a resnet18 tier classifier (bare / multiplex / comm, with a "none" veto)
  classifies each crossing from a PPI-normalized patch.
- **Learned edge-cost matcher** — bipartite assignment (pure-numpy Hungarian)
  with a frozen 21-feature MLP edge cost, A↔B coupling (match-both-or-neither),
  a non-crossing monotonic constraint (span wires don't cross), and
  tier-agreement bonuses. Crossarm bundles are matched at bundle level.

## Honest evaluation methodology

All quoted metrics come from a **site-disjoint split**: photos are grouped
into geographic sites (10 m radius) and every site lives entirely in train,
val, OR test — because utilities re-photograph the same poles across job
revisions, a naive photo-level split leaks and inflates scores by 5–10 pp.
Checkpoint selection uses the deployed operating point (F1 at the production
confidence gates), never val-loss.

| Metric (held-out, site-disjoint) | Score |
|---|---|
| Per-pole annotation micro-F1 (17 joint classes) | **0.717** |
| Crossarm-count accuracy | **0.816** |
| End-to-end span-trace chain accuracy* | **0.56** |
| Equipment detection F1 | **0.84** |
| Equipment keypoint PCK@2 in | 0.43–0.62 per class |
| Calibration height projection vs survey SW | ~0.5 in |

\* Chain accuracy = a midspan wire is traced to the correct attachment on
*both* poles; scored against annotation ground truth that is itself ~90 %
complete, so this is a floor. The tracer is deployed as an assisted-review
first pass, not an auto-annotator.

## Repository layout

```
src/                    training-time pipelines, matcher, eval logic (torch)
train.py                single entry point for all model trainings
scripts/data/           dataset preparation (site-disjoint split, label store)
scripts/train/          trainers + variant/experiment launchers
scripts/eval/           evaluation harnesses (per-model + end-to-end)
scripts/tracer/         run the full wire tracer over span groups
sdk/                    pure-numpy ONNX inference SDKs (no torch/scipy)
  calibration_sdk/        pole + ruler + ticks + pole top
  equipment_annotation_sdk/  equipment YOLO + HRNet keypoints
  wire_tracer_sdk/        unified pole + strip + tier + learned matcher
deploy/                 FastAPI demo webapp (the HF Space)
```

## Quickstart

### Inference SDKs (ONNX, CPU-only)

```
pip install numpy onnxruntime opencv-python-headless Pillow
```

```python
from calibration import CalibrationPipeline           # sdk/calibration_sdk/v*/
from equipment_annotation import EquipmentAnnotationPipeline
from wire_tracer import WireTracerPipeline

calib = CalibrationPipeline()
result = calib.run(rgb_image)          # pole bbox, ruler ticks, pole top

tracer = WireTracerPipeline()
trace = tracer.run(pole_a_rgb, [midspan_rgb], pole_b_rgb)
```

Each SDK expects its `weights/` directory next to the package (see each SDK's
`README.md` / `INTEGRATION.md`). Weights are exported from trained checkpoints
with the SDK's `tools/export_onnx.py`; parity against the torch pipeline is
checked by `tools/parity_check.py` (detection tolerance ~1e-4, matcher
byte-exact).

Notable engineering: the wire-tracer SDK reimplements
`scipy.optimize.linear_sum_assignment` and `scipy.signal.find_peaks` in pure
numpy (`numpy_ops.py`, verified 200/200 random cases each against scipy) so
the destination machine needs no scipy/torch.

### Training on your own data

1. Build the label store and site-disjoint split
   (`scripts/data/build_honest_split.py`) — see
   [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for the expected label schema.
2. `python scripts/data/prepare_dataset.py --production` builds every stack's
   dataset from the split manifest.
3. `python train.py --model unified_pole_detection` (or
   `midspan_wire_strip_detection`, `equipment_detection`, calibration models,
   …).
4. Evaluate: `scripts/eval/eval_unified_pole.py`,
   `eval_midspan_strip_f1.py`, `eval_wire_tracing_e2e.py`.
5. Export ONNX: `python sdk/<sdk>/v*/tools/export_onnx.py`, then
   `tools/parity_check.py`.

### Demo webapp

```
pip install -r requirements-deploy.txt
python -m uvicorn deploy.main:app --port 8000
# open http://localhost:8000/demo
```

Per-photo annotation views (calibration / equipment / attachments) and a
span-trace view (pole A | midspans | pole B with traced wires), with
ground-truth overlay left of the ruler line when a local label store exists.

## Sample outputs

| | |
|---|---|
| ![Calibration](assets/calibration_sample.jpg) | ![Equipment](assets/equipment_sample.jpg) |
| Ruler ticks + pole top + height model | Equipment + keypoints at calibrated heights |

## Lessons that shaped the design

- **Site-disjoint or it didn't happen.** Photo-level random splits leaked
  5–10 pp on every model; all metrics here are from site-disjoint splits.
- **Select checkpoints at the deployed operating point.** Val-loss checkpoint
  selection cost 2.1 pp end-to-end on the strip detector; the trainer now
  tracks F1 at the production peak-extraction op-point.
- **Stage independence.** Later stages never invent detections to satisfy
  earlier-stage expectations (e.g. crossarm wire multiplicity is a matcher
  outcome, not a detector inflation).
- **The matcher is information-bound.** Same-tier wires at similar heights
  carry no distinguishing signal in single photos; the residual error budget
  points at capture-time changes, not model capacity.

## License

MIT (code). Model weights and training data are not distributed.
