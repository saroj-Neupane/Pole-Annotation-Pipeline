"""
Constants mirrored from the training repo's src/config.py + src/wire_tracer.py for
numerical parity — V2.5 (ft2 pole weights + down_guy dedup/anchor pipeline; otherwise = v2.4).

V2 differs from V1 (see ../README.md "How v2 differs from v1"):
  * pole node source is ONE joint-class model (`unified_pole_detection`, 17 classes =
    hardware x cable_type x crossarm-K), NOT the wire ∪ wire-hw union of two detectors.
  * the matcher's per-edge cost is a LEARNED pure-numpy MLP (edge_matcher_unified_v2.json),
    not a hand-tuned geometric cost; comm_isolation is OFF and a finer cable A<->B coupling
    (w_couple_class) is ON.
  * the midspan strip runs at the e2e-optimal peak op-point (height 0.40 / prominence 0.02).

Keep in sync with src/config.py and src/wire_tracer.build_default_tracer when upstream
values change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

_SDK_ROOT = Path(__file__).resolve().parent.parent           # .../wire_tracer_sdk/v2
# Pole detection ONNX is shared with calibration_sdk (not duplicated in this bundle).
# calibration_sdk is not versioned into v1/v2, so it sits one level above the v2 root.
# calibration_sdk is now versioned (v1/v2); v2.5 pairs with calibration v2's detectors.
# v2.5: BOTH pole_detection AND ruler_detection are shared from the calibration stack —
# neither ships in this bundle (same weights, exported once; deploy calibration_sdk/v2
# alongside, as INTEGRATION.md step 1 already requires).
_CALIB_WEIGHTS = _SDK_ROOT.parent.parent / "calibration_sdk" / "v2" / "calibration" / "weights"
DEFAULT_POLE_WEIGHTS_PATH = _CALIB_WEIGHTS / "pole_detection.onnx"
DEFAULT_RULER_WEIGHTS_PATH = _CALIB_WEIGHTS / "ruler_detection.onnx"
DEFAULT_EDGE_MODEL_PATH = WEIGHTS_DIR / "edge_matcher_unified_v2.json"
DEFAULT_UNIFIED_CONF_JSON = WEIGHTS_DIR / "unified_perclass_conf.json"

# -----------------------------------------------------------------------------
# Normalization (ImageNet) — shared by the HRNet strip input
# -----------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Pole detection (shared ONNX from calibration_sdk)
# src/config.py: INFERENCE_POLE_CONF_THRESHOLD=0.01, pole bbox detector runs at imgsz 960
# -----------------------------------------------------------------------------
POLE_INPUT_SIZE = 960
POLE_CONF_THRESHOLD = 0.01
POLE_MAX_DETECTIONS = 1
NMS_IOU_THRESHOLD = 0.7

# -----------------------------------------------------------------------------
# Pole crop geometry (src/data_utils.py::_compute_pole_upper70_2x5_crop)
# -----------------------------------------------------------------------------
POLE_CROP_HEIGHT_FRACTION = 0.70
POLE_CROP_ASPECT_W_OVER_H = 2 / 5

# -----------------------------------------------------------------------------
# Unified pole detection (V2 node source) — ONE YOLO11-pose model, 17 joint classes.
# Runs on the upper-70% 2:5 crop at imgsz 960 (src/wire_tracer.build_default_tracer
# unified_imgsz=960). Each detection's class decodes to (hw_token, cable_type, crossarm-K)
# via UNIFIED_POLE_DECODE; see unified.py.
# -----------------------------------------------------------------------------
UNIFIED_INPUT_SIZE = 960
UNIFIED_MAX_DETECTIONS = 60
# v2.3: the YOLO session runs at the ARM floor (0.10) and the per-class map below gates
# everything else at the flat 0.20 op-point. v2.1's flat-0.20 rationale still holds (a low-conf
# real pole node is a lost chain, a false one is dustbinned — node sourcing wants recall); v2.3
# additionally drops the floor to 0.10 for the crossarm classes ONLY (arm2/arm3/arm4plus), whose
# conf calibration sags on the MI-augmented armboost weights. +0.5pp e2e over flat-0.20.
# v2.5: session floor drops to 0.05 — down_guy is now DETECTED at 0.05 (its gate is applied
# later by the dedup+anchor pipeline, see DOWN_GUY_* below); everything else still gated by
# the per-class map (flat 0.20 / arms 0.10), so only down_guy candidates pass the lower floor.
UNIFIED_CONF_FLOOR = 0.05
UNIFIED_CONF_FLAT = 0.20

UNIFIED_POLE_DETECTION_CLASS_NAMES: Tuple[str, ...] = (
    # power tier (cable_type = Primary; class = hardware sub-type / arm wire-count)
    "pin", "post", "davit", "deadend", "arm2", "arm3", "arm4plus", "primary",
    # secondary tier (hardware = spool; class = cable_type)
    "secondary", "open_secondary", "neutral",
    # comm tier (hardware = three_bolt; class = cable_type)
    "catv", "telco", "fiber",
    # guys (no insulator)
    "guy", "down_guy",
    # recognized conductor, tier unknown (pole-top recovery)
    "unspecified",
)

# v2.3: full 17-class gate map — flat 0.20 everywhere EXCEPT the crossarm classes at 0.10
# (mirrors src/wire_tracer.build_default_tracer(unified_arm_floor=0.10)). This replaces v2.1/v2.2's
# empty map + 0.20 session floor; the per-class F1 gate (unified_perclass_conf.json) stays DROPPED
# — it was tuned for per-pole fidelity and costs −2.4pp at e2e. Ship without that JSON.
UNIFIED_CONF_PER_CLASS: Dict[str, float] = {
    name: (0.10 if name in ("arm2", "arm3", "arm4plus") else 0.20)
    for name in UNIFIED_POLE_DETECTION_CLASS_NAMES
}
# v2.5 annotation-only gates (guy/down_guy never cross a span — zero e2e effect):
#   guy      -> ft2's val-tuned F1-optimal conf (runs/unified_pole_armboost_ft2/perclass_conf.json)
#   down_guy -> 0.05 FLOOR only; the real gate (DOWN_GUY_CONF_GATE) is applied by the
#               dedup+anchor pipeline in pipeline._select_down_guys.
UNIFIED_CONF_PER_CLASS["guy"] = 0.2852
UNIFIED_CONF_PER_CLASS["down_guy"] = 0.05

# -----------------------------------------------------------------------------
# v2.5 down_guy dedup + anchor-count guidance (mirrors src/wire_tracing_e2e
# dedup_pole_points_for_photo down_guy path; val-tuned on the honest split, test
# kp-F1@6in 0.660 -> 0.717 on armboost / 0.683 -> 0.714 on ft2):
#   1. detect down_guy at the 0.05 floor;
#   2. height-dedup down_guy vs down_guy within DOWN_GUY_DEDUP_INCH (inches via the
#      detection's own bbox height = the synthetic 1 ft label box — self-contained,
#      no ruler fit needed), keeping the highest-conf per cluster;
#      GUARD: if the caller-supplied anchor-inventory count K (job JSON
#      sizes_of_attached_dn_guys on connected anchors) exceeds the survivors,
#      merged-away ones are re-admitted (conf order) up to K — a genuine pair of
#      same-height down_guys is indistinguishable from a duplicate EXCEPT by K;
#   3. gate at DOWN_GUY_CONF_GATE, RELAXED back down to the floor until K is met
#      (a sub-gate candidate is admitted only when the inventory proves one is missing).
# K=None (caller has no anchor data) => plain dedup + gate, still +3pp F1.
# -----------------------------------------------------------------------------
DOWN_GUY_DEDUP_INCH = 4.0
DOWN_GUY_CONF_GATE = 0.20

# unified joint-class cable_type -> coarse electrical wire_class for the matcher's
# w_couple_class A<->B coupling + the cable_type_hint output (src/wire_tracing_e2e._UNIFIED_WIRE_CLASS).
UNIFIED_WIRE_CLASS: Dict[str, str] = {
    "primary": "primary", "secondary": "secondary", "open_secondary": "secondary",
    "neutral": "neutral", "catv": "comm", "telco": "comm", "fiber": "comm",
}

# joint class -> (hw_token, cable_type, crossarm_k, display) (src/config.UNIFIED_POLE_DECODE)
UNIFIED_POLE_DECODE: Dict[str, Tuple] = {
    "pin":            ("pin", "primary", 1, "Pin Insulator"),
    "post":           ("post", "primary", 1, "Post Insulator"),
    "davit":          ("davit", "primary", 1, "Davit Arm"),
    "deadend":        ("deadend", "primary", 1, "Deadend"),
    "arm2":           ("arm", "primary", 2, "Crossarm x2"),
    "arm3":           ("arm", "primary", 3, "Crossarm x3"),
    "arm4plus":       ("arm", "primary", 4, "Crossarm x4+"),
    "primary":        (None, "primary", 1, "Primary (hardware unread)"),
    "secondary":      ("spool", "secondary", 1, "Spool (Secondary)"),
    "open_secondary": ("spool", "open_secondary", 1, "Spool (Open Secondary)"),
    "neutral":        ("spool", "neutral", 1, "Spool (Neutral)"),
    "catv":           ("three_bolt", "catv", 1, "Three-Bolt (CATV)"),
    "telco":          ("three_bolt", "telco", 1, "Three-Bolt (Telco)"),
    "fiber":          ("three_bolt", "fiber", 1, "Three-Bolt (Fiber)"),
    "guy":            (None, "guy", 1, "Guy"),
    "down_guy":       (None, "down_guy", 1, "Down Guy"),
    "unspecified":    (None, None, 1, "Unspecified Wire"),
}

# -----------------------------------------------------------------------------
# Hardware token -> coarse tier (src/config.py WIRE_HW_TO_TIER)
# -----------------------------------------------------------------------------
WIRE_HW_TO_TIER: Dict[str, str] = {
    "spool": "secondary", "three_bolt": "comm",
    "pin": "power", "post": "power", "deadend": "power", "davit": "power",
    "guy": "guy", "down_guy": "guy",
}
WIRE_HW_DEADEND_TOKENS = ("deadend",)

# Friendly insulator names (src/wire_tracer.py INSULATOR_DISPLAY) — used when the joint
# class decodes to a bare hardware token (the per-class display string is also available
# from UNIFIED_POLE_DECODE).
INSULATOR_DISPLAY: Dict[object, str] = {
    "spool": "Spool",
    "three_bolt": "Three-Bolt",
    "pin": "Pin Insulator",
    "post": "Post Insulator",
    "deadend": "Deadend",
    "davit": "Davit Arm",
    "guy": "Guy",
    "down_guy": "Down Guy",
    None: "Unread HW",
}

# Only POWER-tier hardware forms a multi-insulator crossarm (src/wire_tracer.py CROSSARM_HW).
# A crossarm's K is recovered from MIDSPAN multiplicity (how many wires trace through the
# point), matching the product; the model's own predicted K is surfaced as the `crossarm_k` hint.
CROSSARM_HW = ("pin", "post", "davit", "deadend")

# Canonical insulator_spec strings so a detected token round-trips through the tier helpers
# (src/wire_tracing_e2e.py TOKEN_TO_SPEC). Only the tier/deadend lookups use these.
TOKEN_TO_SPEC: Dict[str, object] = {
    "spool": 'Spool 3"', "three_bolt": "Three Bolt", "pin": "Pin Insulator",
    "post": "Post Insulator", "deadend": "Deadend", "davit": "Davit",
    "guy": None, "down_guy": None,
}

# -----------------------------------------------------------------------------
# Ruler detection (for the midspan strip column) — src/config.py RULER_DETECTION_CONFIG
# -----------------------------------------------------------------------------
RULER_INPUT_SIZE = 960
RULER_CONF_THRESHOLD = 0.01
RULER_MAX_DETECTIONS = 5

# -----------------------------------------------------------------------------
# Midspan wire strip HRNet (src/config.py WIRE_STRIP_* + inference_utils strip path).
# v2.6: RULER-LINE strip geometry @1740x96 (repo promotion 2026-07-04, balanced e2e
# production combo 0.4796 vs 0.4426). The strip axis is the straight line through the
# CALIBRATION ruler tick anchors (label-faithful: Katapult wire markers sit ON the tick
# line, median dev 0.085% of width), rectified by a shear warp; width = 3 ft via the
# projective height model's local scale; bottom = projected 0.0 ft ground line; top =
# photo top. Ticks come from the CALLER (the tkinter calibration step / job JSON
# anchor_calibration) — see pipeline.run(midspan_ticks=). Photos with no ticks fall
# back to the legacy ruler-ONNX column crop below (STRIP_WIDTH_EXPAND matched pair).
# The strip ONNX input is (1740, 96); peak min-distance scales 12 -> 6 with the height.
# -----------------------------------------------------------------------------
STRIP_WIDTH_EXPAND = 3.0             # column-FALLBACK crop width (no-ticks photos only)
WIRE_STRIP_RESIZE_HEIGHT, WIRE_STRIP_RESIZE_WIDTH = 1740, 96
WIRE_STRIP_HEATMAP_HEIGHT, WIRE_STRIP_HEATMAP_WIDTH = 1740, 96
WIRE_STRIP_PEAK_MIN_DISTANCE = 6     # 12 @3480 scaled to the 1740 heatmap
RULER_ANCHOR_FEET = (2.5, 6.5, 10.5, 14.5, 16.5)   # the real calibration tick heights
WIRE_STRIP_LINE_WIDTH_FT = 3.0       # rectified strip physical width
WIRE_STRIP_PEAK_HEIGHT = 0.40        # V2 e2e-optimal (V1 shipped the F1-balanced 0.6)
WIRE_STRIP_PEAK_PROMINENCE = 0.02    # V2 e2e-optimal (V1 shipped 0.05)
WIRE_STRIP_PROFILE_BAND = 16
# v2.3: COUNT-GUIDED ADAPTIVE peak extraction (+0.9pp e2e, crossarm +6.9pp). Nearly every span
# wire reaches both poles, so the strip should find >= min(#A, #B) detected pole conductors
# (crossarm-K-weighted, guys excluded). When it finds fewer, the height gate is relaxed FOR THAT
# SPAN ONLY down this ladder (same heatmap — peaks are re-extracted, no extra model pass): a
# missed midspan wire is an unrecoverable chain, a false extra peak is absorbed by the matcher
# dustbin. Mirrors src/inference_utils.infer_wires_on_strip(min_peaks=, relax_heights=).
STRIP_ADAPTIVE = True
STRIP_RELAX_LADDER: Tuple[float, ...] = (0.30, 0.20, 0.10)

# -----------------------------------------------------------------------------
# Learned edge-cost matcher (src/wire_tracer.build_default_tracer -> MatchConfig + edge_model).
# unified pole source + strip midspan => height-only matching (w_x = 0). The learned MLP
# replaces the geometric per-edge cost; the A<->B couplings stay additive on top.
# -----------------------------------------------------------------------------
MATCH_W_Y = 1.0                # only used as a fallback if no edge_model is loaded
MATCH_W_X = 0.0                # strip midspan shares the ruler x -> height-only
MATCH_DUST = 1.0               # edge_dust: raised dustbin tuned with the learned cost
MATCH_W_COUPLE_TIER = 0.2
MATCH_W_COUPLE_CHAIN = 0.25
# v2.3: 0.10 -> 0.20 (+0.5pp e2e). The MI-augmented detector's cable classes are ~5pp more
# accurate than the model the 0.10 weight was tuned on, so the A<->B coupling was under-priced.
MATCH_W_COUPLE_CLASS = 0.20    # finer cable-type (primary/secondary/neutral/comm) A<->B coupling
MATCH_W_DEADEND = 0.06
MATCH_COUPLE_ITERS = 4
MATCH_MONOTONIC = True         # non-crossing order-preserving assignment
MATCH_COMM_ISOLATION = False   # OFF with the learned cost (it weights tier softly; hard iso hurt)

# Per-edge feature schema for the learned cost — column order is the contract; KEEP IN SYNC
# with src/wire_tracing_match.EDGE_FEATURE_NAMES and the trainer (the edge model's `cols`
# index into this 21-vector). 14 core + 7 neighborhood/context features.
EDGE_FEATURE_NAMES = (
    # --- per-edge (core 14) ---
    "dy", "dx", "rankdiff", "m_y", "s_y", "m_conf", "s_conf",
    "s_mult", "s_power", "s_secondary", "s_comm", "s_notier", "inv_R", "inv_C",
    # --- neighborhood/context (7) ---
    "gap_above_s", "gap_below_s", "gap_above_m", "gap_below_m",
    "is_nearest_slot", "is_nearest_row", "tier_rank_s",
)
N_CORE_FEATURES = 14

# Pole-point dedup (src/wire_tracer.trace_span pole_dedup_y=1.5 + the unified detect path's
# 0.6% kind-aware dedup). The detector dedup (0.6%) collapses the tightest duplicates;
# the tracer's 1.5% band then collapses a real arm's same-height insulators to one point
# (the crossarm K is recovered from the midspan side).
POLE_DEDUP_Y_PCT = 1.5
POLE_DEDUP_Y_PCT_DETECT = 0.6

# -----------------------------------------------------------------------------
# Visualization colors (RGB)
# -----------------------------------------------------------------------------
POLE_BOX_COLOR = (0, 200, 0)
CROP_BOX_COLOR = (128, 128, 128)
ATTACH_COLOR = (255, 165, 0)
GUY_COLOR = (200, 0, 200)
MIDSPAN_COLOR = (0, 200, 255)
TRACE_COLORS = (
    (255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 220, 60),
    (255, 80, 255), (60, 230, 230), (255, 150, 60), (170, 120, 255),
)

# -----------------------------------------------------------------------------
# v2.9 MIDSPAN TIER classifier (EXP-0001, promoted 2026-07-30; production
# midspan_tier_classifier v1.0.0). 4-class resnet18 (bare/multiplex/comm/none) on
# a PPI-normalized 40"x10" patch centred on each detected midspan crossing;
# 'none' = veto (false-peak absorber). tier feeds the matcher's agreement BONUS
# (subtracted from tier-agreeing midspan<->pole edges so they beat the dustbin).
# Winning validated config: bonus 0.6 + protect-bare gates (0, .7, .7)
# -> balanced e2e 0.5496 -> 0.5615 (+1.2pp, a floor given incomplete GT).
# -----------------------------------------------------------------------------
TIER_ONNX_NAME = "midspan_tier_classifier.onnx"
TIER3_CLASSES = ("bare", "multiplex", "comm")          # index 3 = 'none' veto
TIER_GATES = (0.0, 0.7, 0.7)                            # protect-bare asymmetry
TIER_PATCH_IN_W, TIER_PATCH_IN_H = 40.0, 10.0           # inches
TIER_PATCH_W, TIER_PATCH_H = 256, 64                    # classifier input (w, h)
MATCH_W_MID_TIER3_BONUS = 0.6

# Fine unified class name -> 3-class midspan tier (mirror src/config.UNIFIED_CLASS_TO_TIER3;
# open_secondary/neutral are BARE — only triplex 'secondary' is multiplex).
UNIFIED_CLASS_TO_TIER3 = {
    "pin": "bare", "post": "bare", "davit": "bare", "deadend": "bare",
    "arm2": "bare", "arm3": "bare", "arm4plus": "bare", "primary": "bare",
    "open_secondary": "bare", "neutral": "bare",
    "secondary": "multiplex",
    "catv": "comm", "telco": "comm", "fiber": "comm", "comm": "comm",
}

# -----------------------------------------------------------------------------
# v2.9.1 TIER-CORROBORATED SUB-GATE ADMISSION (EXP-0007, promoted 2026-07-30).
# Conductor dets with conf in [SUBGATE_FLOOR, class gate) are RETAINED but held out
# of pass-1 matching; a pass-1 DUSTBINNED midspan wire with a tier3 admits the ones
# whose class-tier AGREES (edge penalty SUBGATE_PEN), then the span re-matches.
# Balanced e2e 0.5615 -> ~0.567 (+0.52pp); only fires when the tier stage runs.
# -----------------------------------------------------------------------------
SUBGATE_FLOOR = 0.10
SUBGATE_PEN = 0.6
