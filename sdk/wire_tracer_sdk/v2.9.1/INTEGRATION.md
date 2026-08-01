# Integrating the wire tracer into a tkinter desktop app (v2.5)

> **v2.5 weights layout:** `pole_detection.onnx` **and** `ruler_detection.onnx` come from
> `calibration_sdk/v2/` (single owner: the calibration stack) — this bundle ships only the
> unified pole + strip ONNX + matcher JSON. Keep `calibration_sdk/` and `wire_tracer_sdk/`
> as sibling folders (same layout as the HF repo) and both defaults resolve; otherwise pass
> `pole_weights_path=` / `ruler_weights_path=` to `WireTracerPipeline`.

Copy-paste recipe for the consuming Windows app. Assumes you already integrate
[`sdk/calibration_sdk`](../../calibration_sdk) (shared pole model).

> **Upgrading from v1?** The public API is identical and the result schema is a superset —
> see [README.md → How v2 differs from v1](README.md#how-v2-differs-from-v1). The two
> changes that touch integration: the **weights bundle is different** (step 1), and there
> are **two new non-authoritative hints** per attachment (step 4).

## 1. Drop folders in

```
your_app/
├── main.py
├── calibration/            ← from sdk/calibration_sdk/calibration/
│   └── weights/
│       └── pole_detection.onnx
└── wire_tracer/            ← from sdk/wire_tracer_sdk/v2.3/wire_tracer/
    └── weights/
        ├── unified_pole_detection.onnx
        ├── ruler_detection.onnx
        ├── midspan_wire_strip_detection.onnx (+ .data)
        ├── edge_matcher_unified_v2.json        ← learned matcher (required)
        ├── unified_perclass_conf.json          ← per-class conf map (required)
        └── manifest.json
```

> Copy every `*.onnx.data` sidecar next to its `.onnx` graph file (the strip
> model has one). The two `*.json` artifacts are required — the pipeline raises if
> `edge_matcher_unified_v2.json` is missing.
>
> **Changed from v1:** `wire_detection.onnx` and `wire_attachment_hw_detection.onnx`
> are gone; `unified_pole_detection.onnx` + the two JSON artifacts replace them.

## 2. Install runtime deps

Same set as calibration / equipment (one install covers all of them):

```
numpy>=1.26,<3
onnxruntime>=1.18,<2
opencv-python-headless>=4.9,<5
Pillow>=10,<12
```

No torch, no scipy, no ultralytics, no sklearn. (The learned matcher is a frozen
pure-numpy MLP — `wire_tracer/edge_model.py`.)

## 3. Use from tkinter

A span needs three inputs: the pole-A photo, the pole-B photo, and one or more
midspan burst frames between them.

```python
import threading
from pathlib import Path
from wire_tracer import WireTracerPipeline

POLE_ONNX = Path(__file__).parent / "calibration" / "weights" / "pole_detection.onnx"

pipe = WireTracerPipeline(pole_weights_path=POLE_ONNX)
pipe.warmup()   # load all ONNX sessions up front (off the UI thread)

def on_trace_span(pole_a: str, midspan: list[str], pole_b: str,
                  dg_count_a: int | None = None, dg_count_b: int | None = None):
    def worker():
        result = pipe.run(pole_a, midspan, pole_b, return_annotated=True,
                          down_guy_expected_a=dg_count_a,   # v2.5, see below
                          down_guy_expected_b=dg_count_b)
        root.after(0, lambda: show_result(result))
    threading.Thread(target=worker, daemon=True).start()
```

### v2.6: calibration ruler ticks (`midspan_ticks`) — REQUIRED for full quality

The strip model reads a RULER-LINE crop built from the calibration tick anchors. The tkinter
app already has these (calibration runs before tracing); pull them per midspan section photo
from the job JSON and pass them to `run()`:

```python
def ruler_ticks(job: dict, photo_id: str) -> list[tuple[float, float, float]]:
    """(height_ft, percent_x, percent_y) calibration ticks for one section main photo,
    from photofirst_data.anchor_calibration. The SDK keeps only the real anchor heights
    (2.5/6.5/10.5/14.5/16.5 ft) and needs >= 2; return [] when absent (the SDK then
    falls back to the legacy ruler-ONNX column crop)."""
    pf = (job.get("photos", {}).get(photo_id, {}) or {}).get("photofirst_data", {}) or {}
    ac = pf.get("anchor_calibration") or {}
    out = []
    for v in (ac.values() if isinstance(ac, dict) else ac):
        try:
            ft = float(v.get("height"))
            sel = (v.get("pixel_selection") or [{}])[0]
            out.append((ft, float(sel["percentX"]), float(sel["percentY"])))
        except (TypeError, KeyError, ValueError):
            continue
    return out

result = pipeline.run(pole_a, midspan_frames, pole_b,
                      midspan_ticks=[ruler_ticks(job, pid) for pid in midspan_pids],
                      down_guy_expected_a=ka, down_guy_expected_b=kb)
```

If the calibration step produced the ticks in the app's own format, any
`[(height_ft, percent_x, percent_y), ...]` list works — a single list applies to all burst
frames of the section.

### v2.5: anchor-inventory down-guy counts (`down_guy_expected_a/b`)

The job JSON records how many down-guys each pole actually has; passing that count lets the
SDK correct over/under-detection (validated down_guy keypoint-F1 0.660 → 0.717). Compute K per
pole node from the job JSON the app already loads:

```python
def down_guy_count(job: dict, pole_node_id: str) -> int | None:
    """Down-guy count K for a pole node, mirroring the training repo's
    src/pole_anchor_down_guy.py. Return None when the count can't be trusted
    (the SDK then runs unguided); 0 is a real, trusted answer ("no down guys")."""
    def attr(node, key):  # Katapult attributes are {key: {push_key: value}}
        vals = (node.get("attributes") or {}).get(key) or {}
        return next(iter(vals.values()), None) if isinstance(vals, dict) else vals

    nodes = job.get("nodes") or {}

    # anchors excluded from the count: proposed installs + TelecomCo down-guy anchors
    # (linked via a TelecomCo down_guy trace on the pole's main photo)
    main_pid = next((pid for pid, pm in (nodes.get(pole_node_id, {}).get("photos") or {}).items()
                     if (pm.get("association") if isinstance(pm, dict) else pm) == "main"), None)
    traces = (job.get("traces") or {}).get("trace_data") or {}
    metro = set()
    if main_pid:
        pfd = ((job.get("photos") or {}).get(main_pid) or {}).get("photofirst_data") or {}
        for g in (pfd.get("guying") or {}).values():
            t = traces.get(g.get("_trace")) or {}
            if g.get("anchor_id") and t.get("_trace_type") == "down_guy" \
                    and "telecomco" in str(t.get("company") or "").lower():
                metro.add(str(g["anchor_id"]))

    total, usable = 0, 0
    for conn in (job.get("connections") or {}).values():
        if conn.get("button") != "anchor":
            continue
        n1, n2 = conn.get("node_id_1"), conn.get("node_id_2")
        if pole_node_id not in (n1, n2):
            continue
        aid = n2 if n1 == pole_node_id else n1
        ntype = str(attr(nodes.get(aid, {}), "node_type") or "").strip().lower()
        if "anchor" not in ntype or ntype == "new anchor" or aid in metro:
            continue
        raw = attr(nodes.get(aid, {}), "sizes_of_attached_dn_guys")
        if raw is None or not str(raw).strip():
            return None            # anchor without size data -> count untrustworthy
        usable += 1
        total += len([t for t in str(raw).split(",") if t.strip()])
    return total if usable else 0  # pole with no (usable) anchors -> zero down guys
```

Pass `None` when unsure — the SDK degrades to plain dedup + gate (still better than v2.4).
Do NOT pass a guess: K is trusted (it re-admits sub-threshold detections up to K).

## 4. Map results to your annotation / upload layer

| Result key | Use |
|------------|-----|
| `poles["A"][].x/.y` | Pole-A attachment marker position (image %) |
| `poles["A"][].insulator_name` / `.hardware` | Label / insulator_spec for the marker |
| `poles["A"][].tier_hint` | Hardware-derived hint for the user's `wire_type` choice (not authoritative) |
| `poles["A"][].cable_type_hint` | **NEW (v2)** model-predicted cable class (primary/secondary/neutral/comm) — a stronger pre-fill hint, still not authoritative |
| `poles["A"][].crossarm_k` | **NEW (v2)** model-predicted wire-count for a crossarm point |
| `poles["A"][].role` / `.wire_count` | `crossarm` ⇒ K coincident wires on one arm point (`wire_count` = midspan-recovered count) |
| `midspan[].y` | Midspan wire crossing height (image %) |
| `traces[]` | One per midspan wire: links pole-A insulator ↔ pole-B insulator (the shared-trace correspondence) |

Every `wire_type` field is left `None` — the user assigns
primary/secondary/neutral/comm. `cable_type_hint` is a good default to pre-select in
the UI, but mark it as a suggestion (the model can be wrong, e.g. open_secondary↔neutral
or catv↔telco confusion). A trace endpoint may be `None` when the matcher finds no
attachment on that pole (a one-sided wire); surface it for review rather than forcing a match.

`crossarm_k` (the model's predicted K) and `wire_count` (the midspan-recovered count) can
disagree — prefer `wire_count` for the trace structure and treat `crossarm_k` as a
sanity-check hint.

## 5. Notes / tuning

- **Per-class confidence** is always on for the unified model (the precision-leaning
  F1-optimal map ships as `weights/unified_perclass_conf.json`). To experiment, edit that
  JSON or pass a different `weights_dir`.
- **Crossarm multiplicity** is restricted to power-tier hardware
  (pin/post/davit/deadend); spool/three_bolt are capped at one wire.
- **Midspan is height-only** (every strip wire shares the ruler-column x), so the matcher
  runs with `w_x = 0` and matching is purely by height; the learned edge cost provides the
  per-edge base cost.
- **Pole dedup** runs twice: a tight 0.6% kind-aware band inside the detector, then the
  tracer's 1.5% band. Both are exposed (`pole_dedup_y_detect`, `pole_dedup_y`).
- **CPU is the target.** For GPU on the dev box pass
  `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`.
