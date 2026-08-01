#!/usr/bin/env python3
"""
Stage-1b END-TO-END: run the real trained detectors on the real span photos, feed the
detected points into the A↔B-coupled matcher, and score against the Stage-0 GT chains.

Photo resolution (the calibration-pipeline convention):
  * pole node SCID  -> ``data/data_pole/Photos/<job>*_<scid>_*Main.jpg`` (one canonical
    _1_Main shot per pole).
  * midspan section -> ``data/data_midspan/Photos/<job>*(<scidA>)-to-(<scidB>)_*.jpg``.

Per pole photo: pole_detection (max_det=1) -> upper-70% 2:5 crop -> unified_pole_detection
(ONE joint-class pose model: hardware x cable_type x crossarm-K). Per midspan photo: the
HRNet ruler-column STRIP detector (1-D wire peaks). Keypoints are mapped
crop->full-image->percent so they live in the same frame as the GT.

Scoring is detection-aware and group-aware: each detected pole point is associated to its
nearest GT pole point (within a tolerance; many detected -> one GT crossarm is allowed). A
GT chain is recovered iff its midspan point was detected AND the matcher maps it to detected
pole points that associate back to the correct GT pole points. Detector misses, false
positives, localization error and hardware-class noise therefore all flow into the number.
"""

from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (RULER_DETECTION_CONFIG,
                        UNIFIED_POLE_DETECTION_CLASS_NAMES, decode_unified_class,
                        unified_class_to_tier3)
from src.wire_tracing_match import MatchConfig, match_span, match_span_multi

# Canonical insulator_spec strings so a detected hardware token round-trips through
# config.hardware_tier_for_spec / spec_is_deadend (which substring-match on the raw spec).
TOKEN_TO_SPEC = {
    'spool': 'Spool 3"', 'three_bolt': 'Three Bolt', 'pin': 'Pin Insulator',
    'post': 'Post Insulator', 'deadend': 'Deadend', 'davit': 'Davit',
    'guy': None, 'down_guy': None,
}

# unified joint-class -> coarse electrical wire_class for the matcher's w_couple_class A<->B
# coupling (same vocab as attachment_detection: primary/secondary/neutral/comm).
_UNIFIED_WIRE_CLASS = {
    'primary': 'primary', 'secondary': 'secondary', 'open_secondary': 'secondary',
    'neutral': 'neutral', 'catv': 'comm', 'telco': 'comm', 'fiber': 'comm',
    'comm': 'comm',   # 14-class merge variant emits 'comm' directly
}


def _unified_point(name: str, xp: float, yp: float, conf: float) -> Optional[Dict]:
    """Decode a unified_pole_detection class into a matcher pole-point dict (or None).

    The single joint class carries every signal the tracer wants from the pole side:
      * hw_token -> matcher tier (TOKEN_TO_SPEC -> hardware_tier_for_spec) + deadend prior;
        crossarm ('arm') and hardware-unread 'primary' map to a power-tier proxy ('pin').
      * wire_class -> the finer w_couple_class A<->B coupling (primary/secondary/neutral/comm).
      * pred_mult  -> the crossarm wire-count K the model predicts directly (arm2/arm3/arm4plus),
        so a crossarm point can absorb its K coincident midspan wires (vs the wire/hw source,
        which finds 1 point and leans on bundle-level scoring).
    """
    dec = decode_unified_class(name)
    if dec is None:
        return None
    hw, ct, k, _disp = dec
    if name == 'down_guy':
        return {"x": xp, "y": yp, "kind": "guying", "hw_token": "down_guy", "conf": conf,
                "wire_class": None, "pred_mult": 1, "tier3": None}
    if name == 'guy':
        return {"x": xp, "y": yp, "kind": "guying", "hw_token": "guy", "conf": conf,
                "wire_class": None, "pred_mult": 1, "tier3": None}
    if hw in ('pin', 'post', 'davit', 'deadend', 'spool', 'three_bolt'):
        token = hw
    elif hw == 'arm' or ct == 'primary':
        token = 'pin'           # power-tier proxy (crossarm bundle / hardware-unread primary)
    else:
        token = None            # 'unspecified': recognized conductor, tier unknown
    return {"x": xp, "y": yp, "kind": "insulator", "hw_token": token, "conf": conf,
            "wire_class": _UNIFIED_WIRE_CLASS.get(ct), "pred_mult": max(1, k or 1),
            # tier3 from the FINE class name (Experiment 2) — distinguishes open_secondary (bare)
            # from secondary (multiplex), which wire_class cannot.
            "tier3": unified_class_to_tier3(name)}


POLE_PHOTOS = "data/data_pole/Photos"
MID_PHOTOS = "data/data_midspan/Photos"

# Survey camera images are 3840x2560 (verified for both pole _1_Main and midspan photos);
# the ruler-calibrated PPI per photo lets us express an inch tolerance in image-percent.
SURVEY_IMAGE_HEIGHT_PX = 3840
_PPI_CACHE: Dict[str, Optional[float]] = {}


def ppi_for_photo(photo: str, kind: str) -> Optional[float]:
    """LEGACY scalar PPI for a pole/midspan photo. DEPRECATED — use the projection model
    (``ruler_fit_for_photo`` + ``height_in_at``) for height/tolerance; a single PPI scalar
    cannot capture the within-photo perspective nonlinearity. Kept only for old diag probes."""
    if photo in _PPI_CACHE:
        return _PPI_CACHE[photo]
    stem = Path(photo).stem
    from src import photo_id_layout as _pil
    if _pil.ENABLED:
        ppi = _pil.ppi(stem)
        _PPI_CACHE[photo] = ppi
        return ppi
    lbl_dir = "data/data_pole/Labels" if kind == "pole" else "data/data_midspan/Labels"
    lbl = Path(lbl_dir) / f"{stem}_location.txt"
    ppi = None
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            if line.startswith("# PPI="):
                try:
                    ppi = float(line.split("=", 1)[1])
                except ValueError:
                    ppi = None
                break
    _PPI_CACHE[photo] = ppi
    return ppi


def inch_tol_pct(photo: str, kind: str, inches: float) -> Optional[float]:
    """LEGACY: inch tolerance -> image-percent via scalar PPI. DEPRECATED — prefer the
    projection-native path (`score_span_e2e(..., fit_A/fit_B/fit_M=ruler_fit_for_photo(...))`
    with raw inch tolerances), which is perspective-aware. None if no PPI."""
    ppi = ppi_for_photo(photo, kind)
    if not ppi:
        return None
    return inches * ppi / SURVEY_IMAGE_HEIGHT_PX * 100.0


# The canonical ruler anchors + parsing now live in src.height_calculations
# (RULER_ANCHOR_FEET / fit_height_from_location_file); this cache just memoizes the
# photo->fit lookup on the tracer's hot path.
_RULER_FIT_CACHE: Dict[str, object] = {}


def ruler_fit_for_photo(photo: str, kind: str):
    """Projective ``percentY -> inches`` HeightFit for a photo, or None.

    Parses the 5 real ruler anchors from ``<stem>_location.txt`` and fits the
    canonical projective model (``src.ruler_height_model.fit_photo_height``), so a
    detected point's percentY can be read as a physically-consistent height in
    inches — the within-photo nonlinearity a single PPI scalar (inch_tol_pct)
    cannot capture. Cached per photo. Returns a HeightFit or None when the photo
    has no fittable ruler (caller falls back to a percent band).
    """
    if photo in _RULER_FIT_CACHE:
        return _RULER_FIT_CACHE[photo]
    stem = Path(photo).stem
    from src import photo_id_layout as _pil
    if _pil.ENABLED:
        fit = _pil.ruler_fit(stem)
        _RULER_FIT_CACHE[photo] = fit
        return fit
    # Delegate anchor parsing + projective fit to the central source of truth
    # (src.height_calculations.fit_height_from_location_file); cache the photo->fit
    # mapping here for the tracer's hot path.
    from src.height_calculations import fit_height_from_location_file
    lbl_dir = "data/data_pole/Labels" if kind == "pole" else "data/data_midspan/Labels"
    lbl = Path(lbl_dir) / f"{stem}_location.txt"
    fit = fit_height_from_location_file(lbl)
    _RULER_FIT_CACHE[photo] = fit
    return fit


# --------------------------------------------------------------------------- #
# Photo resolution
# --------------------------------------------------------------------------- #

def _scid_variants(scid) -> List[str]:
    s = str(scid)
    out = [s]
    if s.isdigit() and len(s) < 3:
        out.append(s.zfill(3))
    return out


def resolve_pole_photo(job: str, scid, pole_dir: str = POLE_PHOTOS) -> Optional[str]:
    if scid is None:
        return None
    from src import photo_id_layout as _pil
    if _pil.ENABLED:
        return _pil.pole_photo(job, scid)
    for sc in _scid_variants(scid):
        g = sorted(glob.glob(f"{pole_dir}/{glob.escape(job)}*_{sc}_*Main.jpg"))
        if g:
            return g[0]
    return None


def resolve_midspan_photos(job: str, scid_a, scid_b, mid_dir: str = MID_PHOTOS) -> List[str]:
    if scid_a is None or scid_b is None:
        return []
    from src import photo_id_layout as _pil
    if _pil.ENABLED:
        return _pil.midspan_photos(job, scid_a, scid_b)
    for x, y in [(scid_a, scid_b), (scid_b, scid_a)]:
        for xx in _scid_variants(x):
            for yy in _scid_variants(y):
                g = sorted(glob.glob(f"{mid_dir}/{glob.escape(job)}*({xx})-to-({yy})*"))
                if g:
                    return g
    return []


def _loc_wire_ys(stem: str) -> Optional[List[float]]:
    """Wire y-positions (% height) from a midspan photo's location file, or None."""
    import re
    from src import photo_id_layout as _pil
    if _pil.ENABLED:
        return _pil.wire_ys(stem)
    lbl = Path("data/data_midspan/Labels") / f"{stem}_location.txt"
    if not lbl.exists():
        return None
    ys, inw = [], False
    for ln in lbl.read_text().splitlines():
        if ln.startswith("# Wire measurements"):
            inw = True
            continue
        if inw and ln.startswith("wire"):
            ys.append(float(re.findall(r"[-\d.]+", ln)[-1]))
        elif inw and ln.startswith("#") and "bounding" in ln:
            break
    return sorted(set(round(y, 1) for y in ys)) if ys else None


def resolve_gt_frame(span: Dict, mid_photos: List[str], tol_pct: float = 2.0) -> Tuple[Optional[str], float]:
    """Pick the midspan BURST FRAME the Stage-0 GT was annotated on.

    A section's burst photos are shot with the camera moving, so the same wires project to
    DIFFERENT image heights per frame. The Stage-0 GT markers are frame-specific, so detection
    must run on the matching frame or every association spuriously fails. We pick the disk frame
    whose location-file wire heights best match the GT markers. Returns (photo, dist); photo=None
    if no frame matches within tol (the GT can't be aligned to any available photo -> exclude).
    """
    gt_ys = sorted(set(round(m["y"], 1) for m in span["sides"]["M"] if m.get("y") is not None))
    if not gt_ys:
        return (mid_photos[0] if mid_photos else None), 0.0
    best, best_d = None, 1e9
    for ph in mid_photos:
        fy = _loc_wire_ys(Path(ph).stem)
        if not fy:
            continue
        d = sum(min(abs(g - f) for f in fy) for g in gt_ys) / len(gt_ys)
        if d < best_d:
            best_d, best = d, ph
    if best is None or best_d > tol_pct:
        return None, best_d
    return best, best_d


def resolve_span_photos(span: Dict, pole_dir: str = POLE_PHOTOS, mid_dir: str = MID_PHOTOS) -> Dict:
    a = resolve_pole_photo(span["job"], span["pole_a"]["scid"], pole_dir)
    b = resolve_pole_photo(span["job"], span["pole_b"]["scid"], pole_dir)
    m = resolve_midspan_photos(span["job"], span["pole_a"]["scid"], span["pole_b"]["scid"], mid_dir)
    return {"A": a, "B": b, "M": m, "resolvable": bool(a and b and m)}


def section_disk_photos(span: Dict, mid_dir: str = MID_PHOTOS):
    """Group a multi-section span's resolved midspan disk photos BY SECTION (ordered A→B).

    A multi-section connection's photos all share one ``(scidA)-to-(scidB)`` name, so
    resolve_midspan_photos returns them lumped. The Stage-0 builder already partitioned them per
    section (``sides.M_sections[s].photo_ids`` = each section's Katapult photo_id set, resolved by
    the ruler-keypoint re-keying). We map each resolved disk photo to its photo_id (the disk stem
    IS the pid under the photo_id layout, else via the disk→pid reverse index) and bucket it into
    the section that owns it. Returns ``(grouped, leftover)`` where ``grouped[s]`` aligns to
    ``sides.M_sections[s]`` and ``leftover`` holds photos that mapped to no section (or None when
    the photo_id index is unavailable — caller falls back to single-section)."""
    secs = (span.get("sides") or {}).get("M_sections") or []
    all_photos = (span.get("_photos") or resolve_span_photos(span, mid_dir=mid_dir)).get("M") or []
    from src import photo_id_layout as _pil

    def _pid_of(path):
        stem = Path(path).stem
        if _pil.ENABLED:
            return stem                     # data/Photos/<pid>.jpg
        try:
            return _pil.pid_for_disk_stem(stem)
        except Exception:
            return None

    try:
        sec_pid_sets = [set(s.get("photo_ids") or []) for s in secs]
        grouped: List[List[str]] = [[] for _ in secs]
        leftover: List[str] = []
        for ph in all_photos:
            pid = _pid_of(ph)
            placed = False
            if pid is not None:
                for si, pset in enumerate(sec_pid_sets):
                    if pid in pset:
                        grouped[si].append(ph)
                        placed = True
                        break
            if not placed:
                leftover.append(ph)
        return grouped, leftover
    except Exception:
        return None, all_photos             # photo_id index unavailable → caller degrades


# --------------------------------------------------------------------------- #
# Detector inference
# --------------------------------------------------------------------------- #

@dataclass
class Detectors:
    pole: object
    device: str = "cpu"
    pole_conf: float = 0.25
    # Pole point source: "unified" = unified_pole_detection joint-class pose model (the only
    # supported source). See detect_pole_points.
    pole_source: str = "unified"
    pole_crop_imgsz: int = 960   # detector imgsz on the pole crop (match hw model train res)
    # Collapse near-coincident pole detections within this image-height band (kind-aware,
    # keep max-conf) before matching. Conductors merge by height only (wire_class ignored);
    # down_guy is never deduped; conductor never merges with guy. Phantom duplicates
    # (deadend+pin on one insulator) split/steal matches without dedup. 0 = off.
    pole_dedup_y: float = 0.6
    # PHYSICAL (inch) dedup band via the projective ruler model: when set (>0), two
    # near-coincident pole detections merge only if within this many INCHES of true
    # height (ruler_fit_for_photo), instead of the position-dependent percent band
    # above. None = legacy percent dedup (pole_dedup_y). A fixed % band merges a
    # different real spacing depending on where in the image the points sit; inches
    # make it consistent and let us preserve a ~1ft-spaced stacked rack. Falls back
    # to pole_dedup_y for any photo with no fittable ruler. DEPLOYED DEFAULT 4.0in:
    # swept-optimal, +0.40pp e2e chain (0.5440->0.5480) over the 0.6% percent band AND
    # higher node recall (no tradeoff). See sweep in eval_wire_tracing_e2e --sweep-dedup-inch.
    pole_dedup_inch: Optional[float] = 4.0
    # OPT-IN down_guy dedup + anchor-count guidance (annotation path; None = legacy
    # never-dedup, e2e byte-identical). Pipeline (needs down_guy detected at a low floor,
    # e.g. per-class conf 0.05): height dedup within down_guy_dedup_inch INCHES, with a
    # GUARD — if the anchor-inventory K (down_guy_expected: photo-stem ->
    # PoleDownGuyExpectation from src.pole_anchor_down_guy.build_photo_expectations) says
    # more down_guys exist than survive the merge, merged-away ones are re-admitted (conf
    # order) up to K, preserving genuine same-height twins. Then gate at
    # down_guy_conf_gate, RELAXED back down to the floor until K is met (a sub-gate
    # candidate is admitted only when the inventory proves one is missing). Val-tuned
    # 4.0in/0.20 on armboost: test kp-F1@6" 0.660 -> 0.717 (P .597->.728, R .737->.705);
    # ft2 0.683 -> 0.714.
    down_guy_dedup_inch: Optional[float] = None
    down_guy_conf_gate: float = 0.20
    down_guy_expected: Optional[Dict] = None
    # Midspan source: "strip" = HRNet ruler-column heatmap (the only supported source;
    # needs ruler + strip below; matcher should use w_x=0 / y-axis).
    midspan_source: str = "strip"
    ruler: object = None
    strip: object = None
    ruler_conf: float = 0.01
    # strip peak-extraction thresholds (None -> config defaults). Lowering the height gate
    # recovers faint top wires (e.g. crossarm primaries against bright sky) the model responds
    # to but below the default 0.6 gate; prominence still suppresses noise.
    strip_peak_height: Optional[float] = None
    strip_peak_prom: Optional[float] = None
    # COUNT-GUIDED ADAPTIVE strip extraction: when strip_min_peaks is set, peaks are
    # re-extracted at each strip_relax_ladder height (same heatmap, no extra model pass)
    # until at least that many are found. trace_span sets strip_min_peaks per span from the
    # detected pole conductor counts (min(#A, #B), pred_mult-weighted, guying excluded) —
    # a missed midspan wire is unrecoverable downstream, while a false extra peak is
    # absorbed by the matcher dustbin. Validated +0.9pp e2e chain (crossarm +6.9pp),
    # detector-robust (mi_clean + armboost). None = fixed-threshold legacy extraction.
    strip_adaptive: bool = False        # let trace_span set strip_min_peaks from pole counts
    strip_min_peaks: Optional[int] = None
    strip_relax_ladder: Tuple[float, ...] = (0.30, 0.20, 0.10)
    # Widen the ruler-column crop about its centre (1.0 = legacy ruler-width). MUST match the
    # width_expand the strip weights were trained at (dataset prep). >1 lets a wire occluded at
    # the ruler column register where it is visible to either side. See extract_ruler_column_strip.
    strip_width_expand: float = 1.0
    # Strip GEOMETRY: "column" = legacy full-height ruler-bbox x-range (ruler YOLO at
    # inference); "ruler-line" = rectified 3ft strip along the CALIBRATION ruler-tick line,
    # ground line -> photo top (data_utils.extract_ruler_line_strip). Ticks come from the
    # label store / job JSON (calibration precedes tracing in the product flow — NO ruler
    # inference), so train and inference crops are byte-consistent. Photos with no tick
    # anchors fall back to the column path. Must match the strip checkpoint's training mode.
    strip_mode: str = "column"
    # (H, W) input/heatmap resolution of the strip checkpoint when it differs from the config
    # 3480x96 (e.g. (1740, 96)); peak min-distance auto-scales in infer_wires_on_strip.
    strip_resize_hw: Optional[Tuple[int, int]] = None
    # unified_pole_detection: ONE joint-class pose model (hardware x cable_type x crossarm-K).
    # pole_source="unified" decodes each detection to tier (hw_token) + wire_class + K. Run at the
    # unified_conf floor and keep a detection only if its conf clears its per-class threshold
    # (the tuned operating point, perclass_conf.json). None per-class -> single-threshold.
    unified: object = None
    unified_conf: float = 0.20
    unified_conf_per_class: Optional[Dict[str, float]] = None
    unified_imgsz: int = 960
    # Class-name list for decoding the unified model's class indices. None -> the deployed
    # 17-class UNIFIED_POLE_DETECTION_CLASS_NAMES. Set to the 14-class MERGED list to decode
    # the open_sec->neutral / catv,telco,fiber->comm variant (idea #1). MUST match the weights.
    unified_class_names: Optional[List[str]] = None


def load_detectors(device: str = "cuda", weights: Optional[Dict] = None,
                   midspan_source: str = "strip",
                   strip_resize_hw: Optional[Tuple[int, int]] = None) -> Detectors:
    """Load the production detector set: pole (crop), unified (nodes), ruler+strip (midspan)."""
    from ultralytics import YOLO
    w = weights or {}

    def _maybe(key, default):
        """Load a YOLO detector, or None if its weights are absent (e.g. pruned); a path
        passed explicitly in `weights` still raises if missing (caller asked for it)."""
        p = w.get(key, default)
        if key in w or Path(p).exists():
            return YOLO(p)
        return None

    det = Detectors(
        pole=_maybe("pole", "runs/pole_detection/weights/best.pt"),
        device=device,
        midspan_source=midspan_source,
    )
    if w.get("unified"):
        det.unified = YOLO(w["unified"])
    if midspan_source == "strip":
        import torch
        from src.inference_utils import load_wire_strip_model
        det.ruler = YOLO(w.get("ruler", "runs/ruler_detection/weights/best.pt"))
        det.strip = load_wire_strip_model(weights_path=w.get("strip"), device=torch.device(device),
                                          heatmap_size=strip_resize_hw)
        det.strip_resize_hw = strip_resize_hw
    return det


def _is_down_guy(d: Dict) -> bool:
    return d.get("hw_token") == "down_guy"


def _dedup_merge_group(d: Dict):
    """Merge-group key for height dedup (``down_guy`` handled separately — never merges)."""
    if d.get("kind") == "guying" or d.get("hw_token") in ("guy", "down_guy"):
        return d.get("hw_token")  # "guy", "down_guy", or None
    return "cond"  # all conductors merge by height (wire_class ignored — class is too noisy)


def dedup_pole_points_by_height(pts: List[Dict], band_pct: float,
                                fit=None, inch_tol: Optional[float] = None,
                                down_guy_inch: Optional[float] = None,
                                down_guy_expected: Optional[int] = None) -> List[Dict]:
    """Collapse near-coincident pole detections, keeping the highest-conf per cluster.

    Hybrid merge rules (within the height band):
      * **Conductors** merge by height only (position / inch band) — ignores ``wire_class``
        so phantom duplicates (deadend+pin, wire+hw double-fire) still collapse despite
        noisy class labels (+2.5pp e2e vs class-gated dedup).
      * **Aerial ``guy``** merges only with another ``guy`` (never with conductors).
      * **`down_guy`` is NEVER deduped by default** — every detection is kept even at
        identical height. OPT-IN ``down_guy_inch``: down_guy merges with down_guy within
        that height band, with an **anchor-inventory guard** — if ``down_guy_expected``
        (K from the job-JSON anchor ``sizes_of_attached_dn_guys``) says more down_guys
        exist than survive the merge, the merged-away ones are re-admitted (conf order)
        up to K, so genuine side-by-side guys at one height are preserved. Annotation-path
        lever (val-tuned, test kp-F1@6" 0.683→0.718); zero e2e effect (down_guy is
        auto-dustbinned in tracing).
      * Conductors never merge with guying nodes (anchor guys sit at conductor height).

    Two band modes:
      * INCH (physical): when ``fit`` (a projective HeightFit) and ``inch_tol`` are
        given, two points merge only if within ``inch_tol`` INCHES of true height.
        Falls back to ``band_pct`` for any point whose height the fit can't resolve.
      * PERCENT (legacy): ``|Δy| <= band_pct`` image-height. Validated +2.4pp e2e at
        0.6%; see Detectors.pole_dedup_y.
    """
    use_inch = fit is not None and inch_tol is not None and inch_tol > 0
    dg_dedup = down_guy_inch is not None and down_guy_inch > 0
    if (not use_inch and (not band_pct or band_pct <= 0) and not dg_dedup) or len(pts) < 2:
        return pts

    def same_group(i: int, k: int) -> bool:
        if _is_down_guy(pts[i]) or _is_down_guy(pts[k]):
            return dg_dedup and _is_down_guy(pts[i]) and _is_down_guy(pts[k])
        gi, gk = _dedup_merge_group(pts[i]), _dedup_merge_group(pts[k])
        if gi == "cond" or gk == "cond":
            return gi == gk
        return gi == gk or gi is None or gk is None  # guying: same token, or unread absorbs

    h_in: List[Optional[float]] = [None] * len(pts)
    if use_inch:
        from src.ruler_height_model import height_in_at
        h_in = [height_in_at(fit, d["y"]) for d in pts]

    def close(i: int, k: int) -> bool:
        # inches when both heights resolve, else the percent band (robust fallback)
        tol = down_guy_inch if (dg_dedup and _is_down_guy(pts[i])) else inch_tol
        if fit is not None and tol and h_in[i] is not None and h_in[k] is not None:
            return abs(h_in[i] - h_in[k]) <= tol
        if not band_pct or band_pct <= 0:
            return False
        return abs(pts[i]["y"] - pts[k]["y"]) <= band_pct

    if dg_dedup and fit is not None and not use_inch:
        from src.ruler_height_model import height_in_at
        h_in = [height_in_at(fit, d["y"]) for d in pts]

    order = sorted(range(len(pts)), key=lambda i: -(pts[i].get("conf") or 0.0))
    keep: List[int] = []
    dg_dropped: List[int] = []
    for i in order:
        if _is_down_guy(pts[i]) and not dg_dedup:
            keep.append(i)
            continue
        if any(same_group(i, k) and close(i, k) for k in keep):
            if _is_down_guy(pts[i]):
                dg_dropped.append(i)
            continue
        keep.append(i)
    if dg_dedup and down_guy_expected is not None and dg_dropped:
        # anchor-K guard: re-admit merged-away down_guys (conf order) up to the
        # inventory count — dedup can't tell a duplicate from a genuine same-height twin.
        n_dg = sum(1 for i in keep if _is_down_guy(pts[i]))
        if n_dg < down_guy_expected:
            keep.extend(dg_dropped[: down_guy_expected - n_dg])
    keep_set = set(keep)
    return [d for i, d in enumerate(pts) if i in keep_set]


def dedup_pole_points_for_photo(pts: List[Dict], photo: str, det: Detectors) -> List[Dict]:
    """Apply the configured dedup (inch via ruler model, else percent) to raw points.

    Sub-gate candidates (EXP-0007 ``_subgate``) bypass dedup entirely — they are not
    pass-1 points; trace_span decides their admission in the tier-corroborated pass."""
    subgate = [p for p in pts if p.get("_subgate")]
    if subgate:
        pts = [p for p in pts if not p.get("_subgate")]
    inch = getattr(det, "pole_dedup_inch", None)
    dg_inch = getattr(det, "down_guy_dedup_inch", None)
    fit = ruler_fit_for_photo(photo, "pole") if (inch or dg_inch) else None
    dg_k = None
    exps = getattr(det, "down_guy_expected", None)
    if dg_inch and exps is not None:
        exp = exps.get(Path(photo).stem)
        if exp is not None and getattr(exp, "mode", None) in ("anchor_count", "zero"):
            dg_k = exp.count
    out = dedup_pole_points_by_height(pts, getattr(det, "pole_dedup_y", 0.6),
                                      fit=fit, inch_tol=inch,
                                      down_guy_inch=dg_inch, down_guy_expected=dg_k)
    if dg_inch:
        # gate + anchor-RELAX: keep down_guys clearing the gate; admit sub-gate (floor-
        # detected) ones, conf order, only up to the inventory count K.
        gate = getattr(det, "down_guy_conf_gate", 0.20)
        dg = sorted((p for p in out if _is_down_guy(p)), key=lambda p: -(p.get("conf") or 0.0))
        keep = [p for p in dg if (p.get("conf") or 0.0) >= gate]
        if dg_k is not None and len(keep) < dg_k:
            keep += [p for p in dg if (p.get("conf") or 0.0) < gate][: dg_k - len(keep)]
        keep_ids = {id(p) for p in keep}
        out = [p for p in out if not _is_down_guy(p) or id(p) in keep_ids]
    return out + subgate


def detect_pole_points(photo: str, det: Detectors) -> List[Dict]:
    """Detected pole attachment points (kind-aware height-deduped per the Detectors config)."""
    return dedup_pole_points_for_photo(_detect_pole_points_raw(photo, det), photo, det)


_SPAN_METRIC_META: Optional[Dict] = None


def load_span_metric_meta(path: str = "datasets/wire_tracing_dataset/spans_metric.jsonl") -> Dict:
    """(job, connection_id) -> {e_a, e_b, e_mid, length_ft} from spans_metric.jsonl `_meta`.

    Supplies the USGS ground elevations (ground_A/B/M) and span length the chord-sag gate needs
    in the absolute frame. Cached. Missing file / entry -> the gate falls back to flat-earth (e=0).
    """
    global _SPAN_METRIC_META
    if _SPAN_METRIC_META is not None:
        return _SPAN_METRIC_META
    out: Dict = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            meta = s.get("_meta") or {}
            geom = s.get("geometry") or {}
            length_m = geom.get("length_m")
            out[(s.get("job"), s.get("connection_id"))] = {
                "e_a": meta.get("ground_A"), "e_b": meta.get("ground_B"), "e_mid": meta.get("ground_M"),
                "length_ft": (length_m * 3.280839895) if length_m else None,
            }
    _SPAN_METRIC_META = out
    return out


def apply_sag_gate(detA: List[Dict], detM: List[Dict], detB: List[Dict],
                   preds: Dict, fitA, fitB, fitM,
                   sag_min: float = -4.0, sag_max: float = 20.0, tol: float = 4.0,
                   e_a: float = 0.0, e_b: float = 0.0, e_mid: float = 0.0) -> int:
    """Two-sided chord-sag gate in ABSOLUTE ruler FEET (USGS-elevation-corrected), post-match.
    Mutates ``preds`` in place. Mirrors the sdk v3 ``classify_sag_endpoints``.

    A conductor sags DOWNWARD between its two attachments, so the midspan absolute height can
    neither sit above the A-B chord nor implausibly far below it. For each detected midspan matched
    to BOTH an A and a B pole point, lift each above-ground ruler height by its ground elevation
    (``H_x = h_x + e_x``) and test ``sag = (H_a + H_b)/2 - H_m``:

      * ``sag < sag_min`` (midspan bows ABOVE the chord = fly-over): drop each endpoint the wire
        overshoots by > ``tol`` (``H_m - H_x > tol``).
      * ``sag > sag_max`` (implausibly deep sag): drop the whole pairing (outlier reject). Pass a
        span-length-dependent ``sag_max`` for a physical bound (longer span sags more).

    Using ground elevations is what keeps the test correct on sloped terrain, where a midpoint can
    legitimately sit above the LOWER pole yet still below the chord — a flat-earth (e=0) sag injects
    the terrain delta as error. Returns the number of endpoints dropped; a no-op for any chain whose
    three heights don't all resolve.
    """
    from src.ruler_height_model import height_ft_at
    if fitM is None:
        return 0
    dropped = 0
    for r in range(len(detM)):
        a_idx = preds["A"][r] if r < len(preds["A"]) else None
        b_idx = preds["B"][r] if r < len(preds["B"]) else None
        if a_idx is None or b_idx is None:
            continue  # need both endpoints to form a chord
        ym = detM[r].get("y")
        h_m = height_ft_at(fitM, ym) if ym is not None else None
        h_a = height_ft_at(fitA, detA[a_idx]["y"]) if (fitA and a_idx < len(detA)) else None
        h_b = height_ft_at(fitB, detB[b_idx]["y"]) if (fitB and b_idx < len(detB)) else None
        if h_m is None or h_a is None or h_b is None:
            continue
        H_a, H_b, H_m = h_a + e_a, h_b + e_b, h_m + e_mid   # absolute (elevation-corrected) feet
        sag = (H_a + H_b) / 2.0 - H_m
        if sag < sag_min:
            if (H_m - H_a) > tol:
                preds["A"][r] = None; dropped += 1
            if (H_m - H_b) > tol:
                preds["B"][r] = None; dropped += 1
        elif sag > sag_max:
            preds["A"][r] = None; preds["B"][r] = None; dropped += 2
    return dropped


def _detect_pole_points_raw(photo: str, det: Detectors) -> List[Dict]:
    """Detected pole attachment points in image-percent coords (unified joint-class source).

    Returns [{x, y, kind, hw_token, conf, wire_class, pred_mult, tier3}]. kind='guying' for
    guy/down_guy (matcher excludes); everything else 'insulator'.
    """
    import cv2
    from src.data_utils import _compute_pole_upper70_2x5_crop

    img = cv2.imread(photo)
    if img is None:
        return []
    H, W = img.shape[:2]
    pres = det.pole(img, conf=det.pole_conf, max_det=1, verbose=False, imgsz=960, device=det.device)[0]
    if pres.boxes is None or len(pres.boxes) == 0:
        return []
    x1, y1, x2, y2 = map(int, pres.boxes.xyxy[0].cpu().numpy())
    crop_res = _compute_pole_upper70_2x5_crop(img, (x1, y1, x2, y2), W, H)
    if crop_res is None:
        return []
    crop, cx1, cy1, cx2, cy2, _cw, _ch = crop_res

    def to_pct(kx, ky):
        return (100.0 * (cx1 + kx) / W, 100.0 * (cy1 + ky) / H)

    src = getattr(det, "pole_source", "unified")
    if src != "unified":
        raise ValueError(f"pole_source={src!r} is no longer supported — only 'unified'")
    if det.unified is None:
        raise RuntimeError("pole_source='unified' but no unified model loaded (weights['unified'])")

    # unified_pole_detection: ONE joint-class pose model is the node source. Each detection's
    # class decodes to tier (hw_token) + wire_class + crossarm-K.
    ures = det.unified(crop, conf=det.unified_conf, max_det=60, verbose=False,
                       imgsz=det.unified_imgsz, device=det.device)[0]
    pts: List[Dict] = []
    cls_names = getattr(det, "unified_class_names", None) or UNIFIED_POLE_DETECTION_CLASS_NAMES
    if ures.boxes is not None and len(ures.boxes):
        ucls = ures.boxes.cls.cpu().numpy().astype(int)
        uconf = ures.boxes.conf.cpu().numpy()
        uxywh = ures.boxes.xywh.cpu().numpy()        # crop-pixel cx,cy,w,h (box-center fallback)
        kxy = None
        if getattr(ures, "keypoints", None) is not None and ures.keypoints.xy is not None:
            kxy = ures.keypoints.xy.cpu().numpy()    # (N, K, 2) crop pixels
        per_class = det.unified_conf_per_class
        for i in range(len(ucls)):
            if ucls[i] >= len(cls_names):
                continue
            name = cls_names[ucls[i]]
            c = float(uconf[i])
            below_gate = per_class is not None and c < per_class.get(name, det.unified_conf)
            # SUB-GATE retention (EXP-0007): keep conductor dets in [subgate_floor, gate) as
            # FLAGGED candidates — excluded from pass-1 matching/dedup; trace_span may admit
            # them in its tier-corroborated second pass. None (default) = byte-identical.
            sg_floor = getattr(det, "subgate_floor", None)
            is_subgate = below_gate and sg_floor is not None and c >= sg_floor
            if below_gate and not is_subgate:
                continue
            if kxy is not None and i < len(kxy) and len(kxy[i]) and (kxy[i][0] != 0).any():
                kx, ky = float(kxy[i][0][0]), float(kxy[i][0][1])
            else:
                kx, ky = float(uxywh[i][0]), float(uxywh[i][1])
            xp, yp = to_pct(kx, ky)
            pt = _unified_point(name, xp, yp, c)
            if pt is not None:
                if is_subgate:
                    if pt.get("kind") == "guying":
                        continue          # guys are never span endpoints — nothing to rescue
                    pt["_subgate"] = True
                pts.append(pt)
    return pts


def _ruler_anchor_pts_for_photo(photo: str) -> List[Tuple[float, float, float]]:
    """CALIBRATION ruler tick anchors ``(height_ft, percent_x, percent_y)`` for a photo.

    Sourced from the label store / job JSON (the ticks the calibration step already
    produced — no inference). Empty list when the photo has none."""
    stem = Path(photo).stem
    from src import photo_id_layout as _pil
    from src.height_calculations import RULER_ANCHOR_FEET
    pts: List[Tuple[float, float, float]] = []
    lines = _pil.location_lines(stem) if _pil.ENABLED else None
    for line in lines or []:
        parts = line.split(',')
        if len(parts) != 3:
            continue
        try:
            ft, px, py = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if ft in RULER_ANCHOR_FEET:
            pts.append((ft, px, py))
    return pts


_TICK_LINE_CACHE: Dict[str, Optional[Tuple[float, float]]] = {}


def _ruler_tick_line_pct(photo: str) -> Optional[Tuple[float, float]]:
    """Least-squares ruler-tick line in PERCENT coords: x_pct = m·y_pct + c, or None.

    The axis all midspan detections are projected onto (ticks + wires collinear).
    Cached per photo; None when the photo has <2 tick anchors."""
    if photo in _TICK_LINE_CACHE:
        return _TICK_LINE_CACHE[photo]
    pts = _ruler_anchor_pts_for_photo(photo)
    line: Optional[Tuple[float, float]] = None
    if len(pts) >= 2 and len({round(p[2], 6) for p in pts}) >= 2:
        import numpy as _np
        m, c = _np.polyfit([p[2] for p in pts], [p[1] for p in pts], 1)
        line = (float(m), float(c))
    _TICK_LINE_CACHE[photo] = line
    return line


def _midspan_points_ruler_line(photo: str, img, det: Detectors, dev) -> Optional[List[Dict]]:
    """Ruler-line strip inference for one photo (see Detectors.strip_mode).

    Crop via data_utils.extract_ruler_line_strip on the CALIBRATION tick anchors (same
    code path as dataset prep -> byte-consistent geometry), infer, then map each peak's
    in-strip y_norm (ground-line bottom) back to full-photo percent. None = no anchors/fit
    (caller falls back to the column path)."""
    from src.data_utils import extract_ruler_line_strip
    from src.inference_utils import infer_wires_on_strip
    import cv2
    pts_a = _ruler_anchor_pts_for_photo(photo)
    fit = ruler_fit_for_photo(photo, "midspan")
    if len(pts_a) < 2 or fit is None:
        return None
    out = extract_ruler_line_strip(img, pts_a, fit)
    if out is None:
        return None
    strip, lmeta = out
    H, W = img.shape[:2]
    skw = {}
    if det.strip_peak_height is not None:
        skw["height"] = det.strip_peak_height
    if det.strip_peak_prom is not None:
        skw["prominence"] = det.strip_peak_prom
    if getattr(det, "strip_min_peaks", None):
        skw["min_peaks"] = det.strip_min_peaks
        skw["relax_heights"] = det.strip_relax_ladder
    if getattr(det, "strip_resize_hw", None):
        skw["resize_hw"] = det.strip_resize_hw
    strip_rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
    wires, _ = infer_wires_on_strip(det.strip, strip_rgb, dev, **skw)
    gy = lmeta["ground_y_px"]
    pts = []
    for wseg in wires:
        y_px = wseg["y_norm"] * gy                       # strip bottom = ground line
        x_px = lmeta["line_m"] * y_px + lmeta["line_c"]  # ruler axis x at that height
        pts.append({"x": 100.0 * x_px / W, "y": 100.0 * y_px / H, "conf": wseg["conf"]})
    return pts


def detect_midspan_points_strip(photos: List[str], det: Detectors) -> List[Dict]:
    """Midspan wire points from the HRNet ruler-column STRIP detector.

    Per photo: detect the ruler -> crop the full-height column at the ruler x-range ->
    run the strip heatmap (extract_strip_wire_peaks) -> wire y-positions. Every wire shares
    the ruler-column x (so x is the ruler centre and matching/scoring must be height-only:
    cfg.w_x=0, mid_assoc_axis="y"). Dedup across the section's burst photos by most wires.

    Detectors.strip_mode="ruler-line" replaces the crop with the calibration-tick
    ruler-line strip (_midspan_points_ruler_line); photos without tick anchors fall back
    to the legacy column path below.
    """
    import cv2
    import torch
    from src.inference_utils import infer_wires_on_strip
    dev = torch.device(det.device)
    best_pts: List[Dict] = []
    for photo in photos:
        img = cv2.imread(photo)
        if img is None:
            continue
        if getattr(det, "strip_mode", "column") == "ruler-line":
            pts = _midspan_points_ruler_line(photo, img, det, dev)
            if pts is not None:
                if len(pts) > len(best_pts):
                    best_pts = pts
                continue
        H, W = img.shape[:2]
        rres = det.ruler(img, conf=det.ruler_conf, max_det=5, verbose=False,
                         imgsz=RULER_DETECTION_CONFIG["imgsz"], device=det.device)[0]
        if rres.boxes is None or len(rres.boxes) == 0:
            continue
        rx1, _ry1, rx2, _ry2 = rres.boxes.xyxy[0].cpu().numpy()
        if getattr(det, "strip_width_expand", 1.0) != 1.0:
            cx = 0.5 * (rx1 + rx2)
            half = 0.5 * (rx2 - rx1) * det.strip_width_expand
            rx1, rx2 = cx - half, cx + half
        rx1, rx2 = int(max(0, rx1)), int(min(W, rx2))
        if rx2 - rx1 < 4:
            continue
        strip_rgb = cv2.cvtColor(img[:, rx1:rx2], cv2.COLOR_BGR2RGB)   # full-height column
        skw = {}
        if det.strip_peak_height is not None:
            skw["height"] = det.strip_peak_height
        if det.strip_peak_prom is not None:
            skw["prominence"] = det.strip_peak_prom
        if getattr(det, "strip_min_peaks", None):
            skw["min_peaks"] = det.strip_min_peaks
            skw["relax_heights"] = det.strip_relax_ladder
        wires, _ = infer_wires_on_strip(det.strip, strip_rgb, dev, **skw)
        # x: project every detection onto the CALIBRATION ruler-tick line (all midspan
        # points + ticks collinear — the wire markers are placed on the ruler, so this is
        # the physically correct x). Falls back to the column centre when a photo has no
        # tick anchors. e2e-neutral: strip matching/scoring is height-only (w_x=0).
        xc = 100.0 * (rx1 + rx2) / 2 / W
        line = _ruler_tick_line_pct(photo)
        pts = []
        for wseg in wires:
            y_pct = 100.0 * wseg["y_norm"]
            x_pct = (line[0] * y_pct + line[1]) if line is not None else xc
            pts.append({"x": x_pct, "y": y_pct, "conf": wseg["conf"]})
        if len(pts) > len(best_pts):
            best_pts = pts
    return best_pts


# --------------------------------------------------------------------------- #
# Matcher-format construction + association + scoring
# --------------------------------------------------------------------------- #

def to_matcher_side(detected: List[Dict], is_pole: bool,
                    mult: Optional[Dict[int, int]] = None) -> List[Dict]:
    """Build matcher point dicts (one trace per detected point; multiplicity 1 unless `mult`
    supplies a per-detection count — e.g. an arm wire-count head, or a GT oracle)."""
    side = []
    for i, d in enumerate(detected):
        if is_pole:
            spec = TOKEN_TO_SPEC.get(d.get("hw_token"))
            traces = [{"insulator_spec": spec, "cable_type": None}]
            kind = d["kind"]
        else:
            traces = [{"insulator_spec": None, "cable_type": None}]
            kind = "wire"
        k = (mult or {}).get(i, d.get("pred_mult", 1))   # pred_mult = unified crossarm-K (else 1)
        side.append({"x": d["x"], "y": d["y"], "kind": kind, "multiplicity": max(1, k),
                     "traces": traces, "i": i, "wire_class": d.get("wire_class"),
                     "conf": d.get("conf"),          # detection confidence (learned-matcher feature)
                     "elev_ft": d.get("elev_ft"),    # absolute elevation (catenary-sag matcher term)
                     "tier3": d.get("tier3")})        # midspan-predicted 3-class tier (Experiment 2; None = no signal)
    return side


def _associate(detected: List[Dict], gt_points: List[Dict], tol_pct: float,
               axis: str = "xy", tol_per_gt: Optional[Dict[int, float]] = None,
               fit: object = None) -> Dict[int, int]:
    """det_idx -> gt_idx via OPTIMAL assignment against multiplicity-expanded GT slots.

    A crossarm GT point (multiplicity K) exposes K coincident slots, so up to K detected
    points may map to it (group-level), while clean points (mult 1) pair bijectively — this
    avoids the greedy collapse of near-duplicate points (close midspan wires, arm phases).

    axis="y" associates by height only (for strip-detected midspan points, which all share
    the ruler-column x and so carry no meaningful horizontal coordinate).

    tol_per_gt overrides tol_pct for specific GT indices — e.g. a looser tolerance on
    crossarm-bundle midspan points, whose vertical GT position is fuzzy because the wires
    are horizontally parallel.

    fit (PROJECTION MODEL): when given (with axis="y"), vertical distance is measured in
    INCHES via the projective percent_y->inches HeightFit (height_in_at) — the physically
    correct, perspective-aware metric. tol_pct / tol_per_gt are then INCH tolerances (PPI is
    not used). A point whose projective height is non-physical (None) cannot be inch-associated.
    """
    from scipy.optimize import linear_sum_assignment
    slots = []  # (x, y, gt_idx, tol)
    for g in gt_points:
        if g.get("x") is None:
            continue
        t = tol_per_gt.get(g["i"], tol_pct) if tol_per_gt else tol_pct
        for _ in range(max(1, g.get("multiplicity", 1))):
            slots.append((g["x"], g["y"], g["i"], t))
    R, C = len(detected), len(slots)
    if R == 0 or C == 0:
        return {}
    cost = np.full((R, C + R), 1e6, dtype=float)
    dust = max((s[3] for s in slots), default=tol_pct)
    use_proj = fit is not None and axis == "y"
    if use_proj:
        from src.ruler_height_model import height_in_at
        det_h = [height_in_at(fit, d["y"]) if d.get("y") is not None else None for d in detected]
        slot_h = [height_in_at(fit, s[1]) for s in slots]
    for r, d in enumerate(detected):
        cost[r, C + r] = dust                    # leave this detection unassociated
        if d.get("x") is None:
            continue
        for c in range(C):
            if use_proj:
                if det_h[r] is None or slot_h[c] is None:
                    continue                     # non-physical projective height -> no match
                dist = abs(det_h[r] - slot_h[c])
            else:
                dist = abs(d["y"] - slots[c][1]) if axis == "y" else \
                    math.hypot(d["x"] - slots[c][0], d["y"] - slots[c][1])
            if dist <= slots[c][3]:
                cost[r, c] = dist
    rows, cols = linear_sum_assignment(cost)
    out = {}
    for r, c in zip(rows, cols):
        if c < C:
            out[r] = slots[c][2]
    return out


def score_span_e2e(span: Dict, detA: List[Dict], detM: List[Dict], detB: List[Dict],
                   cfg: MatchConfig, assoc_tol_pct: float = 6.0,
                   mid_assoc_axis: str = "xy", bundle_crossarm: bool = True,
                   collect_records: bool = False,
                   tol_A: Optional[float] = None, tol_B: Optional[float] = None,
                   tol_M: Optional[float] = None, tol_M_crossarm: Optional[float] = None,
                   tol_A_crossarm: Optional[float] = None, tol_B_crossarm: Optional[float] = None,
                   exclude_above_pole_top: bool = False, oracle_crossarm_mult: bool = False,
                   sag_fits: Optional[Tuple] = None, sag_min: float = -4.0,
                   sag_max: float = 20.0, sag_tol: float = 4.0,
                   sag_elev: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                   fit_A: object = None, fit_B: object = None, fit_M: object = None) -> Dict:
    """Run the matcher on detected points and score recovery of each GT chain.

    mid_assoc_axis="y" scores midspan association by height only (use with strip-detected
    midspan points, which share the ruler-column x).

    fit_A/fit_B/fit_M (PROJECTION MODEL): per-photo projective percent_y->inches HeightFits.
    When given, the tol_* arguments are INCH tolerances (e.g. 3"/6"/36") and association is by
    projective inch distance (height_in_at) — no PPI. This is the deployed path; the legacy
    percent-band tolerance (PPI-derived via inch_tol_pct) is used only when fits are absent.

    bundle_crossarm=True scores crossarm groups at BUNDLE level: a group of coincident-
    midspan chains sharing the same (arm-A, arm-B) endpoints is one unit, credited when
    arm-A ↔ bundle ↔ arm-B is recovered (within-bundle permutation is don't-care, since
    the K wires are pixel-coincident at midspan and not visually separable). The legacy
    per-chain scoring counts K chains and can recover at most the 1 detectable midspan
    point, which structurally under-credits crossarms.
    """
    # oracle: stamp each detected pole point with the multiplicity of the GT point it associates to
    # (tests the ceiling of an arm wire-count head, which would supply K at inference).
    multA = multB = None
    if oracle_crossarm_mult:
        pA = _associate(detA, span["sides"]["A"], tol_A if tol_A is not None else assoc_tol_pct, axis="y", fit=fit_A)
        pB = _associate(detB, span["sides"]["B"], tol_B if tol_B is not None else assoc_tol_pct, axis="y", fit=fit_B)
        multA = {di: span["sides"]["A"][gi].get("multiplicity", 1) for di, gi in pA.items()}
        multB = {di: span["sides"]["B"][gi].get("multiplicity", 1) for di, gi in pB.items()}
    mA, mM, mB = to_matcher_side(detA, True, multA), to_matcher_side(detM, False), to_matcher_side(detB, True, multB)
    det_span = {"sides": {"A": mA, "M": mM, "B": mB}}
    preds = match_span(det_span, cfg, return_conf=collect_records)
    # Two-sided chord-sag gate (projective ruler feet): drop geometrically-impossible endpoints
    # AFTER matching (fly-over above the chord / implausibly deep sag). Opt-in via sag_fits.
    n_sag_dropped = 0
    if sag_fits is not None:
        ea, eb, em = sag_elev
        n_sag_dropped = apply_sag_gate(detA, detM, detB, preds, sag_fits[0], sag_fits[1], sag_fits[2],
                                       sag_min=sag_min, sag_max=sag_max, tol=sag_tol,
                                       e_a=ea or 0.0, e_b=eb or 0.0, e_mid=em or 0.0)
    records = []  # per chain/bundle: {bucket, proposed, correct, conf} (when collect_records)

    A_gt, M_gt, B_gt = span["sides"]["A"], span["sides"]["M"], span["sides"]["B"]

    def _above_top(c):
        # a wire passing OVER the pole top is above the upper-70% crop -> undetectable by design
        return exclude_above_pole_top and (
            (c["A"] is not None and A_gt[c["A"]].get("above_pole_top")) or
            (c["B"] is not None and B_gt[c["B"]].get("above_pole_top")))

    # crossarm-bundle points get a looser tolerance — the collapsed arm point and the
    # horizontally-parallel midspan wires have a fuzzy GT position; clean (vertically-
    # separated, single) attachments keep the tight tolerance.
    _chains = span["gt"]["chains"]
    def _xa(tol, key):
        return {c[key]: tol for c in _chains if c["group_ambiguous"] and c.get(key) is not None} if tol is not None else None
    tol_per_gt_M = _xa(tol_M_crossarm, "M")
    tol_per_gt_A = _xa(tol_A_crossarm, "A")
    tol_per_gt_B = _xa(tol_B_crossarm, "B")
    # Association tolerances (3"/6"/36") are VERTICAL distances — height above ground is the
    # measured quantity; horizontal offset is irrelevant (the detected insulator can sit a few %
    # off the GT arm-connection in x). So associate by |Δy| only, on every side.
    assocA = _associate(detA, A_gt, tol_A if tol_A is not None else assoc_tol_pct, axis="y", tol_per_gt=tol_per_gt_A, fit=fit_A)
    assocB = _associate(detB, B_gt, tol_B if tol_B is not None else assoc_tol_pct, axis="y", tol_per_gt=tol_per_gt_B, fit=fit_B)
    assocM = _associate(detM, M_gt, tol_M if tol_M is not None else assoc_tol_pct,
                        axis="y", tol_per_gt=tol_per_gt_M, fit=fit_M)
    # gt_idx -> set(det_idx)
    from collections import defaultdict
    gtA = defaultdict(set); gtB = defaultdict(set); gtM = defaultdict(set)
    for di, gi in assocA.items(): gtA[gi].add(di)
    for di, gi in assocB.items(): gtB[gi].add(di)
    for di, gi in assocM.items(): gtM[gi].add(di)

    res = {"clean": {"n": 0, "A": 0, "B": 0, "chain": 0},
           "ambig": {"n": 0, "A": 0, "B": 0, "chain": 0},
           "midspan_detected": 0, "midspan_total": 0, "sag_dropped": n_sag_dropped,
           # endpoint PRECISION (of A/B markers the matcher PROPOSES for a detected midspan
           # whose GT endpoint exists, how many associate to the correct GT pole point). This
           # is the axis a drop-gate (sag) can move — chain accuracy structurally cannot, since
           # dropping a match never CREATES a correct chain.
           "prec": {"prop_A": 0, "corr_A": 0, "prop_B": 0, "corr_B": 0}}

    def side_ok(gt_idx, pred_idx, gt_assoc):
        # gt_assoc: det_idx -> gt_idx. A one-sided GT chain (gt_idx None) is correct only
        # if the matcher dustbinned this side; otherwise the predicted detected pole point
        # must associate back to the correct GT pole point.
        if gt_idx is None:
            return pred_idx is None
        if pred_idx is None:
            return False
        return gt_assoc.get(pred_idx) == gt_idx

    def _margin(m_det):
        if "A_conf" not in preds:
            return None
        return min(preds["A_conf"][m_det], preds["B_conf"][m_det])

    def _detconf(m_det):
        # weakest detection confidence among the midspan point and the matched pole points
        mc = detM[m_det].get("conf", 1.0)
        pa, pb = preds["A"][m_det], preds["B"][m_det]
        ac = detA[pa].get("conf", 1.0) if pa is not None else 0.0
        bc = detB[pb].get("conf", 1.0) if pb is not None else 0.0
        return min(mc, ac, bc)

    def chain_ok(mi, A, B):
        """For GT-midspan index mi: (a_ok, b_ok, margin, det_conf) if detected, else None."""
        m_dets = gtM.get(mi, set())
        if not m_dets:
            return None
        m_gtpt = M_gt[mi]
        m_det = min(m_dets, key=lambda di: abs(detM[di]["y"] - m_gtpt["y"]))   # vertical only
        # endpoint-precision tally: among proposed (non-None) A/B markers for a real GT endpoint,
        # how many associate to the CORRECT GT pole point (the axis the sag drop-gate moves).
        pa, pb = preds["A"][m_det], preds["B"][m_det]
        if A is not None:
            res["prec"]["prop_A"] += int(pa is not None)
            res["prec"]["corr_A"] += int(pa is not None and assocA.get(pa) == A)
        if B is not None:
            res["prec"]["prop_B"] += int(pb is not None)
            res["prec"]["corr_B"] += int(pb is not None and assocB.get(pb) == B)
        return (side_ok(A, preds["A"][m_det], assocA), side_ok(B, preds["B"][m_det], assocB),
                _margin(m_det), _detconf(m_det))

    def _rec(bucket, r):
        if not collect_records:
            return
        if r is None:
            records.append({"bucket": bucket, "proposed": False, "correct": False,
                            "conf": None, "det_conf": None})
        else:
            records.append({"bucket": bucket, "proposed": True, "correct": bool(r[0] and r[1]),
                            "conf": r[2], "det_conf": r[3]})

    # clean (per-trace) chains: scored individually
    for c in span["gt"]["chains"]:
        if c["group_ambiguous"] or _above_top(c):
            continue
        res["clean"]["n"] += 1
        res["midspan_total"] += 1
        r = chain_ok(c["M"], c["A"], c["B"])
        _rec("clean", r)
        if r is None:
            continue
        res["midspan_detected"] += 1
        res["clean"]["A"] += int(r[0]); res["clean"]["B"] += int(r[1]); res["clean"]["chain"] += int(r[0] and r[1])

    # crossarm groups
    ambig = [c for c in span["gt"]["chains"] if c["group_ambiguous"] and not _above_top(c)]
    if bundle_crossarm:
        from collections import defaultdict as _dd
        bundles = _dd(list)
        for c in ambig:
            bundles[(c["A"], c["B"])].append(c)        # one bundle per (arm-A, arm-B)
        for (A, B), cs in bundles.items():
            res["ambig"]["n"] += 1
            res["midspan_total"] += 1
            det_mi = [c["M"] for c in cs if gtM.get(c["M"])]   # bundle-midspan points detected
            if not det_mi:
                _rec("crossarm", None)
                continue
            res["midspan_detected"] += 1
            a_any = b_any = ok = False
            best_conf = best_dc = None
            for mi in det_mi:
                r = chain_ok(mi, A, B)
                if r is None:
                    continue
                a_any |= r[0]; b_any |= r[1]; ok |= (r[0] and r[1])
                if r[2] is not None and (best_conf is None or r[2] > best_conf):
                    best_conf = r[2]
                if r[3] is not None and (best_dc is None or r[3] > best_dc):
                    best_dc = r[3]
            res["ambig"]["A"] += int(a_any); res["ambig"]["B"] += int(b_any); res["ambig"]["chain"] += int(ok)
            if collect_records:
                records.append({"bucket": "crossarm", "proposed": True, "correct": bool(ok),
                                "conf": best_conf, "det_conf": best_dc})
    else:
        for c in ambig:
            res["ambig"]["n"] += 1
            res["midspan_total"] += 1
            r = chain_ok(c["M"], c["A"], c["B"])
            _rec("crossarm", r)
            if r is None:
                continue
            res["midspan_detected"] += 1
            res["ambig"]["A"] += int(r[0]); res["ambig"]["B"] += int(r[1]); res["ambig"]["chain"] += int(r[0] and r[1])
    if collect_records:
        res["records"] = records
    return res


# --------------------------------------------------------------------------- #
# Multi-section scoring (pole-A -> M1 -> ... -> Mk -> pole-B)
# --------------------------------------------------------------------------- #

def _side_ok(gt_idx, pred_idx, gt_assoc) -> bool:
    """A predicted endpoint is correct iff (one-sided GT) the matcher dustbinned it, or the
    predicted detected pole point associates back to the correct GT pole point."""
    if gt_idx is None:
        return pred_idx is None
    if pred_idx is None:
        return False
    return gt_assoc.get(pred_idx) == gt_idx


def _score_multi(span: Dict, preds: Dict, assocA: Dict[int, int], assocB: Dict[int, int],
                 assocM: List[Dict[int, int]], det_sections: List[List[Dict]]) -> Dict:
    """Score gt.chains_multi given the matcher's multi prediction + per-side/per-section
    associations. The spine-anchored matcher can only trace a wire its SPINE section observed, so
    a chain is SCORED iff its spine-section midspan point exists (== it is a legacy gt.chain on the
    spine); chains seen only off-spine are reported as coverage loss, not scored. det_sections may
    be the GT section points themselves (identity association) for the realizable ceiling."""
    sp = preds["spine"]
    predA, predB = preds["A"], preds["B"]
    secs = span["sides"]["M_sections"]
    K = len(secs)
    spine_gt_to_det: Dict[int, set] = {}
    for di, gi in assocM[sp].items():
        spine_gt_to_det.setdefault(gi, set()).add(di)

    res = {"clean": {"n": 0, "A": 0, "B": 0, "chain": 0},
           "ambig": {"n": 0, "A": 0, "B": 0, "chain": 0},
           "chains_total": 0, "scored": 0, "detected_any": 0, "off_spine_only": 0}
    for c in span["gt"]["chains_multi"]:
        res["chains_total"] += 1
        det_any = any(c["M_path"][s] is not None and c["M_path"][s] in assocM[s].values()
                      for s in range(K))
        res["detected_any"] += int(det_any)
        g_sp = c["M_path"][sp]
        if g_sp is None:
            res["off_spine_only"] += int(det_any)   # observed off-spine: spine matcher can't trace
            continue
        bucket = res["ambig"] if c["group_ambiguous"] else res["clean"]
        bucket["n"] += 1
        res["scored"] += 1
        dets = spine_gt_to_det.get(g_sp, set())
        if not dets:
            continue                                 # spine midspan not detected -> chain miss
        r = min(dets, key=lambda di: abs(det_sections[sp][di]["y"] - secs[sp]["points"][g_sp]["y"]))
        a_ok = _side_ok(c["A"], predA[r], assocA)
        b_ok = _side_ok(c["B"], predB[r], assocB)
        bucket["A"] += int(a_ok); bucket["B"] += int(b_ok); bucket["chain"] += int(a_ok and b_ok)
    return res


def score_span_multi_gt(span: Dict, cfg: MatchConfig) -> Dict:
    """Realizable multi-section ceiling: run match_span_multi on the GT points themselves and score
    gt.chains_multi with identity association (perfect detection). The multi analogue of the
    single-section realizable ceiling — measures whether the spine-anchored matcher CAN thread the
    full pole-A → M1 → … → Mk → pole-B path, independent of detector noise."""
    preds = match_span_multi(span, cfg)
    secs = span["sides"]["M_sections"]
    det_sections = [s["points"] for s in secs]
    ident = [{p["i"]: p["i"] for p in s["points"]} for s in secs]
    assocA = {p["i"]: p["i"] for p in span["sides"]["A"]}
    assocB = {p["i"]: p["i"] for p in span["sides"]["B"]}
    return _score_multi(span, preds, assocA, assocB, ident, det_sections)


def score_span_multi(span: Dict, detA: List[Dict], det_sections: List[List[Dict]],
                     detB: List[Dict], cfg: MatchConfig, assoc_tol_pct: float = 6.0,
                     tol_A: Optional[float] = None, tol_B: Optional[float] = None,
                     tol_M: Optional[float] = None) -> Dict:
    """Detection-aware multi-section scoring: build the matcher input from the per-section detected
    points, run match_span_multi, associate detections to GT per side and per section, and score
    gt.chains_multi. Mirrors score_span_e2e's association rules (height-only, axis='y'). Pass the GT
    section points as ``det_sections`` (and GT pole points as detA/detB) to reproduce the ceiling."""
    secs = span["sides"]["M_sections"]
    K = len(secs)
    matcher_sections = [{"section_id": secs[i].get("section_id"),
                         "points": to_matcher_side(pts, False), "i": i}
                        for i, pts in enumerate(det_sections)]
    spine = max(range(K), key=lambda i: (len(matcher_sections[i]["points"]), -i)) if K else 0
    det_span = {"sides": {"A": to_matcher_side(detA, True), "B": to_matcher_side(detB, True),
                          "M": matcher_sections[spine]["points"] if K else [],
                          "M_sections": matcher_sections}}
    preds = match_span_multi(det_span, cfg)
    tolA = tol_A if tol_A is not None else assoc_tol_pct
    tolB = tol_B if tol_B is not None else assoc_tol_pct
    tolM = tol_M if tol_M is not None else assoc_tol_pct
    assocA = _associate(detA, span["sides"]["A"], tolA, axis="y")
    assocB = _associate(detB, span["sides"]["B"], tolB, axis="y")
    assocM = [_associate(det_sections[s], secs[s]["points"], tolM, axis="y") for s in range(K)]
    return _score_multi(span, preds, assocA, assocB, assocM, det_sections)


# --- span-level scoring helpers (moved from the deleted viz scripts 2026-07-29) ---

def clean_chains(span):
    """GT chains with both pole ends present (clean + crossarm), in a stable order."""
    return [c for c in span["gt"]["chains"] if c["A"] is not None and c["B"] is not None]


def score_span(span, preds, aA, aB, aM):
    """Per GT clean (non-ambiguous) chain: correct iff predicted A & B associate to the GT
    pole points. Returns (n_ok, n_total, per_chain_correct dict keyed by M idx)."""
    m_by_gt = {}
    for di, gi in aM.items():
        m_by_gt.setdefault(gi, di)
    ok = tot = 0
    correct = {}
    for c in clean_chains(span):
        if c["group_ambiguous"]:
            continue
        tot += 1
        dm = m_by_gt.get(c["M"])
        good = False
        if dm is not None:
            pA, pB = preds["A"][dm], preds["B"][dm]
            good = (aA.get(pA) == c["A"]) and (aB.get(pB) == c["B"]) \
                if (pA is not None and pB is not None) else False
        correct[c["M"]] = good
        ok += int(good)
    return ok, tot, correct

