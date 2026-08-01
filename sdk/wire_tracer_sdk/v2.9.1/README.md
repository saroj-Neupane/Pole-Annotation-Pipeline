# wire_tracer_sdk v2.9.1 (2026-07-30, PRODUCTION)

**v2.9.1 = v2.9 + TIER-CORROBORATED SUB-GATE ADMISSION** (EXP-0007): conductor dets with
conf in [0.10, class gate) are retained but held out of pass-1; a pass-1 DUSTBINNED
midspan wire with a tier3 admits the ones whose class-tier agrees (edge penalty 0.6),
then the span re-matches. Recovers real occluded/dark attachments (crop-audited 6/7
real). Balanced e2e 0.5615 -> ~0.567 (+0.52pp). Only fires when the tier stage runs
(needs `midspan_ticks`); zero admissions -> byte-identical to v2.9. No new ONNX.
- Parity: numpy-ops 200/200+200/200; standard span (COAR-FR01 131->130) traces identical.
  NEOM110 004->003 shows trace diffs that PRE-DATE this version (verified with tier AND
  sub-gate disabled on both paths — pure torch-vs-ONNX detector conf drift, 5-vs-4 att,
  the documented class since v2.3; NOT an admission-port bug).

# wire_tracer_sdk v2.9 (2026-07-30)

**v2.9 = v2.8 + MIDSPAN TIER stage** (EXP-0001, promoted 2026-07-30):
- NEW `midspan_tier_classifier.onnx` (resnet18, 4-class bare/multiplex/comm/'none'-veto,
  production model v1.0.0): classifies every detected midspan crossing from a
  PPI-normalized 40"x10" photo patch (256x64, RGB/255 — NO ImageNet norm).
- PPI comes from `run(midspan_ticks=…)` (same ticks as the ruler-line strip). Frames
  without ticks -> `tier3=None` on every point (graceful; matcher term is a no-op).
  The tier ONNX itself is OPTIONAL — a bundle without it still traces.
- Matcher: `w_mid_tier3_bonus=0.6` SUBTRACTED from tier-AGREEING midspan<->pole edges
  (pole tier3 from the fine unified class; open_secondary/neutral = bare, only triplex
  secondary = multiplex). Gates (0, .7, .7) = protect-bare asymmetry.
- Validated: balanced e2e 0.5496 -> 0.5615 (+1.2pp, a floor given incomplete GT).
- Output: midspan entries + config carry `tier3` / `midspan_tier` (non-authoritative
  hints; `wire_type` is still user-assigned).
- Parity: numpy-ops 200/200+200/200; tier ONNX-vs-torch 1.4e-06 logits / 169/171 tier
  assignments on real frames (2 flips = gate-borderline peaks under the documented
  PPI-source difference: repo stored-PPI vs SDK tick-fit); e2e traces identical to the
  torch reference (pole att-count ±1 = documented conf drift at the 0.20 gate).
- `tools/parity_check.py` SDK_PKG is now SELF-RELATIVE (fixes the v2.8 copy-forward bug
  class for good).
- Re-export: `USE_PRODUCTION_MODELS=true python sdk/wire_tracer_sdk/v2.9/tools/export_onnx.py`

# Wire Tracer SDK — v2.8

> **v2.8 (2026-07-08):** unified-pole ONNX upgraded to the **yolo11m capacity model** (production `unified_pole_detection` v1.3.0; honest+mined site-disjoint retrain). Balanced e2e **0.5496** (+2.5pp vs v2.7 s-baseline), annotation micro-F1 **0.717** (+3.1pp); `last.pt` e2e-selected, leak-verified. Strip + learned matcher unchanged from v2.7/v2.6. Parity: numpy-ops 200/200, strip ONNX 1.18e-04, e2e traces byte-identical (pole attachment-count drift = documented ONNX-vs-torch conf-drift at the 0.20 gate, no trace impact).


> **Status: BUILT + ONNX smoke-validated (2026-07-07). PRODUCTION.** v2.6 with ONE change: the
> unified-pole ONNX is now the **HONEST site-disjoint model** (honest test micro-F1 0.686 / balanced
> honest e2e 0.4648), replacing the leaky all-sites ft2 per the honest-only training policy. Strip +
> matcher unchanged from v2.6. (Broad new-geography pole-shelf mining was tested and refuted — it
> regressed honest e2e, so it is excluded.)
>
> _v2.6 lineage below:_ v2.5 with ONE change:
>
> **RULER-LINE midspan strip @1740×96** (`midspan_wire_strip_detection.onnx`, source
> `runs/midspan_wire_strip_rulerline_neom/weights/best_f1.pth`). The strip crop is no longer a
> vertical ruler-bbox column: its axis is the straight line through the CALIBRATION ruler tick
> anchors (label-faithful — Katapult wire markers sit ON the tick line, median dev 0.085% of
> image width over 80k markers), rectified by a shear warp, 3 ft wide via the projective height
> model's local scale, bottom at the projected 0.0 ft ground line, top at the photo top
> (`ruler_line.py`). Trained on clean + MN-mined + multi-section + 2,883 NEOM-mined strips at
> 1740×96 (half the old height; peak min-distance 12→6). Balanced honest e2e, production combo
> (ft2 pole): **ALL 0.4796 / MN 0.4577 / NEOM 0.5680 vs v2.5's 0.4426 (+3.7pp)**.
>
> **API: pass the calibration ticks** — `run(..., midspan_ticks=[[(2.5, px, py), (6.5, px, py),
> ...], ...])`, one tick-list per midspan frame (or a single list for all frames); the tkinter
> app already has these from its calibration step (job JSON `anchor_calibration`; helper in
> INTEGRATION.md §3b). The model was TRAINED on the ruler-line crop, so ticks should always be
> supplied in production; a frame without ticks falls back to the legacy ruler-ONNX column crop
> (kept for robustness, geometry-mismatched with the new weights → lower quality). Every
> emitted midspan `x` is projected onto the tick line (matches the annotation convention).
>
> Parity 2026-07-04: strip torch↔ONNX 1.7e-04; 8-span e2e vs the torch tracer: **midspan-y 8/8
> byte-identical through the ruler-line path**; trace-level diffs are the long-documented
> ONNX-vs-torch pole-keypoint drift (±0.1% coords / 0.20-gate flips), unchanged from v2.5 (the
> pole ONNX is identical). NOTE: **pole_detection AND ruler_detection are both shared from
> `calibration_sdk/v2/` and are NOT in this bundle** (2 ONNX + 1 JSON ship here: unified pole,
> strip, edge matcher). Deploy `calibration_sdk/v2/` alongside — the defaults resolve
> `../../calibration_sdk/v2/calibration/weights/{pole,ruler}_detection.onnx`,
> or pass `WireTracerPipeline(pole_weights_path=..., ruler_weights_path=...)` explicitly.

---

## Inherited v2.4 docs below

> **Status: BUILT + ONNX-parity-passed (2026-07-03).** v2.3 with ONE change: the midspan strip
> ONNX is the **clean+mined** checkpoint (`runs/midspan_wire_strip_mined/weights/best_f1.pth` —
> MI-clean honest corpus + 5,124 mined MN strips, w3sharp recipe, best_f1 selection; strip
> torch↔ONNX parity 2.5e-05). Beats the old w3sharp strip on every leak-free column: balanced
> e2e MN 0.4293 vs 0.4075, NEOM 0.5333, ALL 0.4500; corpus-test tuned F1@2in 0.784 (the old
> 0.883 was train-on-test-inflated). Pole/ruler ONNX + matcher JSON + all config unchanged from
> v2.3 (docs below inherited; re-run parity_check.py against the torch tracer for a full audit).

> **Status: BUILT + parity-passed (2026-06-11).** Bundle live at `wire_tracer/weights/`
> (3 ONNX + 1 JSON, 112.6 MB). `parity_check.py`: numpy-ops 200/200 + 200/200; strip
> torch↔ONNX 4.2e-05; 10-span e2e vs the torch production tracer: midspan-y 9/10
> byte-identical, full traces 4/10 identical (diffs = ONNX-vs-torch detector conf drift
> around the new lower thresholds, NOT port bugs — see "Parity notes" below).

**Builds on v2.2. One new ONNX (the armboost pole model) + three config/logic changes
(arm conf floor, count-guided adaptive midspan, w_couple_class 0.2).**

> Version chain: `v2` (unified 17-class pole + learned numpy edge-cost matcher) →
> `v2.1` (flat-0.20 pole op-point, +2.4pp) → `v2.2` (w3sharp 3×-width strip, +2.0pp) →
> **`v2.3` (this)**. Read those READMEs for the inherited design; this file covers only
> what v2.3 changes.

---

## TL;DR — what changes from v2.2

| # | Change | What ships | e2e gain |
|---|--------|-----------|----------|
| 1 | **Pole model → ARMBOOST** (`unified_pole_detection.onnx` re-exported from `runs/unified_pole_mi_armboost`) | 1 new ONNX | +0.4pp e2e, **per-pole micro-F1 0.65 → 0.722** (arm2 0.773, arm3 0.756, fiber +12pp, guy +9pp, secondary +8pp) |
| 2 | **Per-class ARM conf floor 0.10** (arm2/arm3/arm4plus; everything else stays flat 0.20) | constants only | +0.5pp |
| 3 | **Count-guided ADAPTIVE midspan extraction** | ~20 lines (pipeline + strip_onnx) | **+0.9pp** (crossarm chains +6.9pp) |
| 4 | **`MATCH_W_COUPLE_CLASS` 0.10 → 0.20** | constants only | +0.5pp |

**Total: e2e chain accuracy 0.5881 → 0.6027 on the 2119-span / 9751-chain benchmark
(first config over 0.60; +22.2pp over the original 0.3804 baseline).** All four levers were
validated independently AND combined (additive, no job family regresses), and levers 3+4
were replicated on three different detector checkpoints (detector-robust).

## Why each change is better than v2.2

### 1. Armboost pole weights (the new ONNX)

v2.2's pole model trained on non-MI poles only (~4.0k photos). Armboost adds **4,461
clean-MI photos** (MI primary labels dropped — they're unreliable; secondary/neutral/comm/guy
kept — their hardware is deterministic) and **duplicates arm-bearing train photos 2×** to
undo the crossarm-class dilution the MI data would otherwise cause (arm share 39.6% → 18.7%
→ restored to 31.5%). Net: much better cable classes AND better crossarms — the only pole
checkpoint that wins both e2e and per-class F1.

### 2. Arm conf floor (constants)

The flat-0.20 gate (v2.1's win) stays for 14 of 17 classes. The three crossarm classes
get a 0.10 floor: their confidence calibration sits lower, and a missed arm node loses K
chains at once, while a false arm is dustbinned. `UNIFIED_CONF_PER_CLASS` is now a full
17-class map (flat 0.20, arms 0.10) and the ONNX session floor (`UNIFIED_CONF_FLOOR`)
drops to 0.10 so arm candidates survive to the gate. **Do NOT bundle
`unified_perclass_conf.json`** — a stale copy would override this map (pipeline prefers it).

### 3. Count-guided adaptive midspan (the cleverest one)

Nearly every span wire reaches both poles, so the midspan strip should find at least
`min(#A, #B)` detected pole conductors (crossarm-K-weighted, guys excluded — they never
cross a span). When the 0.40 height gate finds fewer, the extractor re-extracts peaks from
the SAME heatmap at 0.30 → 0.20 → 0.10 until the count is plausible (no extra ONNX pass —
only the find_peaks call repeats). Asymmetry that makes it safe: a missed midspan wire is
an **unrecoverable** chain; a false extra peak is absorbed by the matcher dustbin. This is
the biggest single matcher win since the learned edge cost, and it costs ~nothing at runtime.

### 4. w_couple_class 0.2

The A↔B cable-type coupling was priced (0.10) for a detector with noisy cable classes.
Armboost's cable classes are ~5pp more accurate, so class agreement across the span
deserves double the weight. (0.3 ≈ 0.2; the plateau is wide.)

---

## Deployment

### Requirements (unchanged from v2.2)

```
numpy>=1.26,<3
onnxruntime>=1.18,<2
opencv-python-headless>=4.9,<5
Pillow>=10,<12
```
No torch, no scipy, no sklearn. Hungarian + find_peaks are pure numpy (`numpy_ops.py`).

### Install / copy-in (desktop tkinter app)

1. Copy the whole `v2.3/wire_tracer/` package folder into the app (next to the existing
   v2.2 one or replacing it). The bundle must contain:
   - `weights/unified_pole_detection.onnx` ← **NEW (armboost)**
   - `weights/ruler_detection.onnx` (re-exported, functionally unchanged)
   - `weights/midspan_wire_strip_detection.onnx` + `.onnx.data` (w3sharp, unchanged)
   - `weights/edge_matcher_unified_v2.json` (unchanged — a v2.3 retrain was tested and
     REFUTED; the v2 matcher transfers)
   - `weights/manifest.json` (sha256s; `sdk_version: v2.3`)
   - **NO `unified_perclass_conf.json`** — ship without it (see §2).
2. `pole_detection.onnx` is still shared from `calibration_sdk/calibration/weights/`
   (pass `pole_weights_path` if it lives elsewhere).
3. Usage is unchanged:

```python
from wire_tracer import WireTracerPipeline
pipe = WireTracerPipeline()          # optional: pole_weights_path=...
result = pipe.run(pole_a_image, midspan_images, pole_b_image)
```

`result["config"]["version"]` now reports `"v2.3"`. The output schema is identical to
v2/v2.2 (poles {A,B} with `cable_type_hint`/`crossarm_k`, midspan, traces) — drop-in.

### Upgrading from v2.2 in place (if you patch instead of copy)

1. Replace `weights/unified_pole_detection.onnx` with the v2.3 one (armboost).
2. Replace `constants.py` (floor 0.10 + full per-class map + wclass 0.2 + ladder),
   `strip_onnx.py` (adaptive `infer(min_peaks=, relax_heights=)`), and `pipeline.py`
   (conductor-count target + plumb `min_peaks` through `_detect_midspan_points`).
3. Delete any `weights/unified_perclass_conf.json` left over from v2.0.

### Re-export from the training repo

```
python sdk/wire_tracer_sdk/v2.3/tools/export_onnx.py     # 3 ONNX + manifest
python sdk/wire_tracer_sdk/v2.3/tools/parity_check.py    # numpy-ops + 1-span e2e vs torch
```

---

## Parity notes (read before filing a "trace mismatch" bug)

- `linear_sum_assignment` / `find_peaks`: 200/200 + 200/200 exact vs scipy.
- Strip ONNX: max |Δ| 4.2e-05 vs torch on the same input.
- 10-span e2e vs the torch production tracer (`build_default_tracer`, which IS the v2.3
  config): **midspan-y 9/10 byte-identical; full traces 4/10 identical.** Every diff traces
  to sub-1e-2 detector confidence drift between onnxruntime and torch around a threshold
  (0.10/0.20 gates) or between two near-tied matcher candidates — the same documented mode
  as v2/v2.2 (which shipped at 14/20). v2.3 sits on MORE decision boundaries by design:
  the 0.10 arm floor admits borderline detections, and adaptive midspan propagates pole-side
  drift into the strip count target. The two sides' outputs are equally plausible
  reconstructions; aggregate accuracy is the benchmark number above, not per-span identity.
- Known pre-existing difference (since v2): the SDK uses the 0.6% + 1.5% percent-band pole
  dedup; the training repo's tracer uses the projective-ruler inch-4.0 dedup (needs
  per-photo ruler calibration files the desktop doesn't have).
