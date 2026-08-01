# Data layout and label format

The pipelines read a photo-id-keyed store under `data/`. To train on your own
corpus, produce these four artifacts; everything downstream
(`scripts/data/prepare_dataset.py`, the eval harnesses, the demo GT overlay)
is derived from them.

```
data/
  Photos/<photo_id>.jpg        all photos (pole + midspan), content-addressed id
  photo_manifest.json          photo_id index (role, job, dimensions)
  labels/<job>.json            photo_id-keyed annotations (schema below)
  jobs/<job>.json              raw source job JSON (optional; wire tracing GT)
```

`photo_id` is any stable unique key per image (a content hash works well —
it deduplicates photos shared across adjacent jobs).

## `labels/<job>.json`

One JSON object per job; keys are photo ids.

```jsonc
{
  "<photo_id>": {
    "role": "pole",                  // or "midspan"
    "has_height": true,              // false if no usable height anchors
    "anchors": [                     // calibration ruler ticks
      {"ft": 2.5,  "percent_x": 46.3, "percent_y": 81.4},
      {"ft": 6.5,  "percent_x": 46.0, "percent_y": 69.9}
      // ... typically 2.5 / 6.5 / 10.5 / 14.5 (+ 16.5) ft
    ],
    "pole_top": {"percent_x": 45.1, "percent_y": 8.2},

    // pole photos: attachment points on the pole
    "attachments": [
      {
        "percent_x": 48.1, "percent_y": 22.4,
        "hw": "pin",                 // pin|post|davit|deadend|spool|three_bolt|guy|down_guy
        "ct": "primary",             // primary|secondary|open_secondary|neutral|catv|telco|fiber|unspecified
        "arm": 0                     // crossarm wire count; 0 = single insulator
      }
    ],

    // midspan photos: wire crossings on the ruler line
    "wires": [
      {"percent_x": 50.2, "percent_y": 31.0, "cable_type": "Primary"}
    ],

    "bboxes": [                      // optional detection boxes (equipment etc.)
      {"class": "transformer", "x1": 0.41, "y1": 0.18, "x2": 0.55, "y2": 0.31}
    ]
  }
}
```

Coordinates are percentages of image width/height (0–100). Heights are never
stored — they are recomputed from the tick anchors by a 1-D projective fit
(`src/height_calculations.py`), which is robust to camera tilt and distance.

## Split manifest

`datasets/split_manifest.json` assigns every photo id to `train`/`val`/`test`.
Build it with `scripts/data/build_honest_split.py`, which groups photos into
geographic sites and keeps each site entirely within one split — do not use
photo-level random splits (see README, "Honest evaluation methodology").

## Wire-tracing ground truth

Span-level GT (which pole attachment connects to which midspan wire) is
extracted from source job JSONs that carry shared trace ids across photos:
`scripts/data/build_wire_tracing_dataset.py` →
`datasets/wire_tracing_dataset/spans.jsonl`. If your source system has no
trace ids, the tracer still runs — you just can't score chain accuracy.
