#!/usr/bin/env python3
"""
Per-pole fidelity eval for the `unified_pole_detection` YOLO-pose model.

Each crop is ONE pole. The model emits 17 joint classes (hardware x cable_type x
crossarm-K, see ``config.UNIFIED_POLE_DETECTION_CLASS_NAMES``) + 1 attachment
keypoint per detection. The YOLO-pose label files in the dataset split ARE the
ground truth (keystone-encoded joint classes) — they are used directly here, the
GT is NOT re-derived from raw Katapult JSON.

It mirrors the matching / PR machinery of ``scripts/eval_wire_hw_f1.py`` /
``scripts/eval_perclass_conf.py`` (predict the split, greedily match preds->GT by
conf desc using keypoint proximity with a box-IoU fallback), then reports:

  (1) Per-class P/R/F1 for all 17 classes (+ support), so weak fine classes are
      visible (pin/post/davit; secondary/open_secondary/neutral; catv/telco/fiber).
  (2) CROSSARM-K accuracy: of GT arm{2,3,4plus}, how often the matched prediction
      carries the right K; plus an arm-vs-non-arm confusion block.
  (3) Coarse-TIER P/R: collapse the 17 classes to {power, secondary, comm, guy,
      unspecified} via ``decode_unified_class`` to separate detection quality from
      fine-class confusion.
  (4) Overall per-pole (= per-image) fidelity: fraction of images whose PREDICTED
      class multiset equals the GT class multiset.

  down_guy (eval-only): for jobs with anchor metadata, GT count = sum of
  ``sizes_of_attached_dn_guys`` on connected anchors (excludes ``new anchor`` and
  TelecomCo proposed guys). Poles with no anchors expect 0. High-conf preds are
  ranked by confidence vs that count. Training labels are unchanged.

Usage:
    python scripts/eval_unified_pole.py --weights models/production/unified_pole_detection/production/model.pt
    python scripts/eval_unified_pole.py --weights .../best.pt --split val --conf 0.25
    python scripts/eval_unified_pole.py --self-test     # no weights/dataset needed

If ``--weights`` is omitted the self-test runs instead (it also runs when the
dataset split is missing). Missing weights/split on the real path -> clear
message + nonzero exit (no traceback).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config  # noqa: E402


# --------------------------------------------------------------------------- #
# unified-class config  (single source of truth = src.config)
# --------------------------------------------------------------------------- #
# Prefer the project's src.config (the keystone source of truth). This isolated
# worktree's src/config.py may predate the unified symbols, so we fall back to a
# verbatim mirror of the keystone definitions (config.py commit f986428) — the same
# UNIFIED_POLE_DECODE / class names / dataset dir / imgsz — so the harness runs here
# AND stays correct once the keystone config is present.
_FALLBACK_CLASS_NAMES = [
    'pin', 'post', 'davit', 'deadend', 'arm2', 'arm3', 'arm4plus', 'primary',
    'secondary', 'open_secondary', 'neutral',
    'catv', 'telco', 'fiber',
    'guy', 'down_guy',
    'unspecified',
]
_FALLBACK_DECODE = {
    'pin':            ('pin', 'primary', 1, 'Pin Insulator'),
    'post':           ('post', 'primary', 1, 'Post Insulator'),
    'davit':          ('davit', 'primary', 1, 'Davit Arm'),
    'deadend':        ('deadend', 'primary', 1, 'Deadend'),
    'arm2':           ('arm', 'primary', 2, 'Crossarm x2'),
    'arm3':           ('arm', 'primary', 3, 'Crossarm x3'),
    'arm4plus':       ('arm', 'primary', 4, 'Crossarm x4+'),
    'primary':        (None, 'primary', 1, 'Primary (hardware unread)'),
    'secondary':      ('spool', 'secondary', 1, 'Spool (Secondary)'),
    'open_secondary': ('spool', 'open_secondary', 1, 'Spool (Open Secondary)'),
    'neutral':        ('spool', 'neutral', 1, 'Spool (Neutral)'),
    'catv':           ('three_bolt', 'catv', 1, 'Three-Bolt (CATV)'),
    'telco':          ('three_bolt', 'telco', 1, 'Three-Bolt (Telco)'),
    'fiber':          ('three_bolt', 'fiber', 1, 'Three-Bolt (Fiber)'),
    'guy':            (None, 'guy', 1, 'Guy'),
    'down_guy':       (None, 'down_guy', 1, 'Down Guy'),
    'unspecified':    (None, None, 1, 'Unspecified Wire'),
}
_FALLBACK_IMGSZ = 960
_FALLBACK_DATA_DIR = PROJECT_ROOT / "datasets" / "unified_pole_detection"


def _cfg_class_names():
    return list(getattr(config, 'UNIFIED_POLE_DETECTION_CLASS_NAMES', _FALLBACK_CLASS_NAMES))


def _cfg_decode(name):
    fn = getattr(config, 'decode_unified_class', None)
    if fn is not None:
        return fn(name)
    return _FALLBACK_DECODE.get(name)


def _cfg_default_imgsz():
    cfg = getattr(config, 'UNIFIED_POLE_DETECTION_CONFIG', None)
    if isinstance(cfg, dict) and 'imgsz' in cfg:
        return cfg['imgsz']
    return _FALLBACK_IMGSZ


def _cfg_default_data_dir():
    dirs = getattr(config, 'DATASET_DIRS', None)
    if isinstance(dirs, dict) and 'unified_pole_detection' in dirs:
        return dirs['unified_pole_detection']
    return _FALLBACK_DATA_DIR


NAMES = _cfg_class_names()
NAME_TO_IDX = {n: i for i, n in enumerate(NAMES)}
ARM_CLASSES = ('arm2', 'arm3', 'arm4plus')
GUYING_CLASSES = ('guy', 'down_guy')
TIER_ORDER = ('power', 'secondary', 'comm', 'guy', 'unspecified')


def _kind(cls_idx):
    """Coarse KIND for cross-kind match gating: guying vs conductor.

    Anchor guys attach at the SAME pole height as the conductor they brace, so
    ~68% of guy/down_guy GT keypoints are co-located (<0.04) with a conductor.
    Class-agnostic matching lets a confident conductor pred steal the guy GT.
    Gating matches by kind keeps conductor within-tier confusion intact while
    stopping cross-kind theft.
    """
    return 'guying' if NAMES[cls_idx] in GUYING_CLASSES else 'conductor'


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def coarse_tier(name):
    """Collapse a unified class name to its coarse tier via decode_unified_class.

    decode -> (hardware, cable_type, K, display); the cable_type carries the tier.
    """
    dec = _cfg_decode(name)
    if dec is None:
        return 'unspecified'
    _hw, ct, _k, _disp = dec
    if ct in ('guy', 'down_guy'):
        return 'guy'
    if ct == 'primary':
        return 'power'
    if ct in ('secondary', 'open_secondary', 'neutral'):
        return 'secondary'
    if ct in ('catv', 'telco', 'fiber'):
        return 'comm'
    return 'unspecified'          # ct is None


def arm_k(name):
    """K for an arm class (via decode), else None for non-arm classes."""
    if name not in ARM_CLASSES:
        return None
    dec = _cfg_decode(name)
    return dec[2] if dec else None


def iou_xywhn(a, b):
    """IoU of two normalized xywh boxes (cx, cy, w, h)."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


# --------------------------------------------------------------------------- #
# detection container + GT/prediction loaders
# --------------------------------------------------------------------------- #
class Det:
    """One detection (GT or pred). Box is normalized xywh; kp is normalized (x, y)."""
    __slots__ = ('cls', 'box', 'kp', 'conf')

    def __init__(self, cls, box, kp, conf=1.0):
        self.cls = int(cls)
        self.box = box            # (cx, cy, w, h)
        self.kp = kp              # (x, y) or None
        self.conf = float(conf)


def parse_pose_label_line(line):
    """Parse one YOLO-pose label line: `cls cx cy w h kx ky [v] ...` -> Det | None.

    Single keypoint per the unified dataset spec; tolerant of a missing visibility
    flag. Returns None for blank / malformed lines.
    """
    f = line.split()
    if len(f) < 5:
        return None
    cls = int(float(f[0]))
    box = (float(f[1]), float(f[2]), float(f[3]), float(f[4]))
    kp = None
    if len(f) >= 7:
        kp = (float(f[5]), float(f[6]))
    return Det(cls, box, kp, conf=1.0)


def load_gt(label_path):
    out = []
    if label_path.exists():
        for ln in label_path.read_text().splitlines():
            d = parse_pose_label_line(ln)
            if d is not None:
                out.append(d)
    return out


def load_preds_from_result(res):
    """Extract Det list from an ultralytics pose Results object."""
    preds = []
    boxes = getattr(res, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return preds
    xywhn = boxes.xywhn.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    kps = None
    kp_obj = getattr(res, 'keypoints', None)
    if kp_obj is not None and getattr(kp_obj, 'xyn', None) is not None:
        kxyn = kp_obj.xyn.cpu().numpy()      # (N, K, 2) normalized
        if kxyn.size:
            kps = kxyn
    for i in range(len(cls)):
        kp = None
        if kps is not None and i < len(kps) and len(kps[i]):
            kp = (float(kps[i][0][0]), float(kps[i][0][1]))
        preds.append(Det(cls[i], (float(xywhn[i][0]), float(xywhn[i][1]),
                                  float(xywhn[i][2]), float(xywhn[i][3])), kp,
                         float(conf[i])))
    return preds


# --------------------------------------------------------------------------- #
# matching: greedy by conf desc, keypoint proximity primary + box-IoU fallback
# --------------------------------------------------------------------------- #
def match_image(preds, gts, kp_thresh, iou_thresh, kind_gate=True):
    """Greedily match preds -> gts (class-agnostic, by conf desc).

    Cost: keypoint Euclidean distance (normalized) when both have keypoints; a
    box-IoU>=iou_thresh fallback otherwise. A pred matches a GT only if it is
    within kp_thresh (kp path) OR IoU>=iou_thresh (box path).

    Class-agnostic geometry lets us see fine-class CONFUSION (a localized pred
    that named the wrong class still matches the GT and is scored a per-class
    miss for the GT class + a per-class FP for the predicted class).

    Returns: list of (pred_idx, gt_idx), set of unmatched_pred_idx,
             set of unmatched_gt_idx.
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i].conf)
    used_gt = set()
    matches = []
    for pi in order:
        p = preds[pi]
        best_j, best_cost = -1, None
        for gj, g in enumerate(gts):
            if gj in used_gt:
                continue
            if kind_gate and _kind(p.cls) != _kind(g.cls):
                continue           # a guy GT can only be claimed by a guy-kind pred
            cost = None
            if p.kp is not None and g.kp is not None:
                d = ((p.kp[0] - g.kp[0]) ** 2 + (p.kp[1] - g.kp[1]) ** 2) ** 0.5
                if d <= kp_thresh:
                    cost = d
            if cost is None:
                v = iou_xywhn(p.box, g.box)
                if v >= iou_thresh:
                    cost = 1.0 - v + 1.0      # box matches rank below any kp match
            if cost is not None and (best_cost is None or cost < best_cost):
                best_cost, best_j = cost, gj
        if best_j >= 0:
            used_gt.add(best_j)
            matches.append((pi, best_j))
    matched_p = {pi for pi, _ in matches}
    unmatched_p = set(range(len(preds))) - matched_p
    unmatched_g = set(range(len(gts))) - used_gt
    return matches, unmatched_p, unmatched_g


# --------------------------------------------------------------------------- #
# scoring core: consume per-image (preds, gts) -> all metric tables
# --------------------------------------------------------------------------- #
def _score_down_guy_count(n_gt: int, preds_dg) -> Tuple[int, int, int]:
    """Count-based down_guy TP/FP/FN (high-conf preds = all preds, ranked by conf)."""
    n_pred = len(preds_dg)
    tp = min(n_gt, n_pred)
    fp = max(0, n_pred - n_gt)
    fn = max(0, n_gt - n_pred)
    return tp, fp, fn


def score(images, kp_thresh, iou_thresh, kind_gate=True,
          dg_expectations=None, photo_stems=None):
    """images: list of (preds, gts). Returns a dict of metric tables.

    When ``dg_expectations`` maps ``photo_stem`` -> ``PoleDownGuyExpectation`` with
    mode ``anchor_count`` or ``zero``, down_guy is scored by inventory count (top
  conf-ranked detections vs anchor sum); other classes use keypoint matching.
    """
    dg_expectations = dg_expectations or {}
    photo_stems = photo_stems or [None] * len(images)
    dg_idx = NAME_TO_IDX['down_guy']
    n_cls = len(NAMES)
    tp = [0] * n_cls
    fp = [0] * n_cls
    fn = [0] * n_cls
    gt_support = [0] * n_cls

    tier_tp = Counter(); tier_fp = Counter(); tier_fn = Counter(); tier_support = Counter()

    arm_k_total = 0
    arm_k_correct = 0
    arm_gt_total = 0
    arm_matched_as_arm = 0
    arm_matched_nonarm = 0
    arm_unmatched = 0
    nonarm_gt_matched_as_arm = 0

    pole_total = 0
    pole_exact = 0
    dg_anchor_poles = 0

    for (preds, gts), stem in zip(images, photo_stems):
        pole_total += 1
        exp = dg_expectations.get(stem) if stem else None
        use_anchor_dg = exp is not None and exp.mode in ("anchor_count", "zero")

        if use_anchor_dg:
            dg_anchor_poles += 1
            n_dg = exp.count
            preds_dg = sorted([p for p in preds if p.cls == dg_idx], key=lambda p: -p.conf)
            preds_rest = [p for p in preds if p.cls != dg_idx]
            gts_rest = [g for g in gts if g.cls != dg_idx]

            tpd, fpd, fnd = _score_down_guy_count(n_dg, preds_dg)
            gt_support[dg_idx] += n_dg
            tier_support['guy'] += n_dg
            tp[dg_idx] += tpd
            fp[dg_idx] += fpd
            fn[dg_idx] += fnd
            tier_tp['guy'] += tpd
            tier_fp['guy'] += fpd
            tier_fn['guy'] += fnd

            for g in gts_rest:
                gt_support[g.cls] += 1
                tier_support[coarse_tier(NAMES[g.cls])] += 1
                if NAMES[g.cls] in ARM_CLASSES:
                    arm_gt_total += 1

            matches, unmatched_p, unmatched_g = match_image(
                preds_rest, gts_rest, kp_thresh, iou_thresh, kind_gate)

            for pi, gj in matches:
                p, g = preds_rest[pi], gts_rest[gj]
                pn, gn = NAMES[p.cls], NAMES[g.cls]
                if p.cls == g.cls:
                    tp[g.cls] += 1
                else:
                    fp[p.cls] += 1
                    fn[g.cls] += 1
                pt, gtt = coarse_tier(pn), coarse_tier(gn)
                if pt == gtt:
                    tier_tp[gtt] += 1
                else:
                    tier_fp[pt] += 1
                    tier_fn[gtt] += 1
                g_is_arm, p_is_arm = gn in ARM_CLASSES, pn in ARM_CLASSES
                if g_is_arm:
                    arm_k_total += 1
                    if p_is_arm:
                        arm_matched_as_arm += 1
                        if arm_k(pn) == arm_k(gn):
                            arm_k_correct += 1
                    else:
                        arm_matched_nonarm += 1
                elif p_is_arm:
                    nonarm_gt_matched_as_arm += 1

            for pi in unmatched_p:
                p = preds_rest[pi]
                fp[p.cls] += 1
                tier_fp[coarse_tier(NAMES[p.cls])] += 1
            for gj in unmatched_g:
                g = gts_rest[gj]
                fn[g.cls] += 1
                tier_fn[coarse_tier(NAMES[g.cls])] += 1
                if NAMES[g.cls] in ARM_CLASSES:
                    arm_unmatched += 1

            gt_cnt = Counter(g.cls for g in gts_rest)
            gt_cnt[dg_idx] = n_dg
            if Counter(p.cls for p in preds) == gt_cnt:
                pole_exact += 1
            continue

        for g in gts:
            gt_support[g.cls] += 1
            tier_support[coarse_tier(NAMES[g.cls])] += 1
            if NAMES[g.cls] in ARM_CLASSES:
                arm_gt_total += 1

        matches, unmatched_p, unmatched_g = match_image(preds, gts, kp_thresh, iou_thresh, kind_gate)

        for pi, gj in matches:
            p, g = preds[pi], gts[gj]
            pn, gn = NAMES[p.cls], NAMES[g.cls]
            if p.cls == g.cls:
                tp[g.cls] += 1
            else:
                fp[p.cls] += 1
                fn[g.cls] += 1
            pt, gtt = coarse_tier(pn), coarse_tier(gn)
            if pt == gtt:
                tier_tp[gtt] += 1
            else:
                tier_fp[pt] += 1
                tier_fn[gtt] += 1
            g_is_arm, p_is_arm = gn in ARM_CLASSES, pn in ARM_CLASSES
            if g_is_arm:
                arm_k_total += 1
                if p_is_arm:
                    arm_matched_as_arm += 1
                    if arm_k(pn) == arm_k(gn):
                        arm_k_correct += 1
                else:
                    arm_matched_nonarm += 1
            elif p_is_arm:
                nonarm_gt_matched_as_arm += 1

        for pi in unmatched_p:
            p = preds[pi]
            fp[p.cls] += 1
            tier_fp[coarse_tier(NAMES[p.cls])] += 1
        for gj in unmatched_g:
            g = gts[gj]
            fn[g.cls] += 1
            tier_fn[coarse_tier(NAMES[g.cls])] += 1
            if NAMES[g.cls] in ARM_CLASSES:
                arm_unmatched += 1

        if Counter(p.cls for p in preds) == Counter(g.cls for g in gts):
            pole_exact += 1

    return dict(
        tp=tp, fp=fp, fn=fn, gt_support=gt_support,
        tier_tp=tier_tp, tier_fp=tier_fp, tier_fn=tier_fn, tier_support=tier_support,
        arm_k_total=arm_k_total, arm_k_correct=arm_k_correct, arm_gt_total=arm_gt_total,
        arm_matched_as_arm=arm_matched_as_arm, arm_matched_nonarm=arm_matched_nonarm,
        arm_unmatched=arm_unmatched, nonarm_gt_matched_as_arm=nonarm_gt_matched_as_arm,
        pole_total=pole_total, pole_exact=pole_exact,
        dg_anchor_poles=dg_anchor_poles,
    )


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_report(m):
    n_cls = len(NAMES)
    print("\n" + "=" * 72)
    print("(1) PER-CLASS  precision / recall / F1   (all 17 unified classes)")
    print("=" * 72)
    print(f"  {'class':<16}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'P':>8}{'R':>8}{'F1':>8}")
    macro = []
    tot_tp = tot_fp = tot_fn = 0
    for c in range(n_cls):
        tp, fp, fn, sup = m['tp'][c], m['fp'][c], m['fn'][c], m['gt_support'][c]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = _f1(p, r)
        tot_tp += tp; tot_fp += fp; tot_fn += fn
        if sup:
            macro.append(f)
        flag = "" if sup else "  (no GT)"
        print(f"  {NAMES[c]:<16}{sup:>6}{tp:>6}{fp:>6}{fn:>6}"
              f"{p:>8.3f}{r:>8.3f}{f:>8.3f}{flag}")
    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    print("  " + "-" * 62)
    print(f"  {'MACRO-F1 (classes w/ GT)':<40}{np.mean(macro) if macro else 0.0:>8.3f}")
    print(f"  {'MICRO P / R / F1':<40}{micro_p:>8.3f}{micro_r:>8.3f}{_f1(micro_p, micro_r):>8.3f}")

    print("\n" + "=" * 72)
    print("(2) CROSSARM-K accuracy  +  arm-vs-non-arm confusion")
    print("=" * 72)
    kacc = m['arm_k_correct'] / m['arm_k_total'] if m['arm_k_total'] else 0.0
    print(f"  GT arms total ............... {m['arm_gt_total']}")
    print(f"  GT arms matched (to any pred) {m['arm_k_total']}")
    print(f"    matched to an ARM class ... {m['arm_matched_as_arm']}")
    print(f"      of those, K correct ..... {m['arm_k_correct']}")
    print(f"    matched to a NON-arm class  {m['arm_matched_nonarm']}")
    print(f"  GT arms unmatched (missed) .. {m['arm_unmatched']}")
    print(f"  non-arm GT matched as ARM ... {m['nonarm_gt_matched_as_arm']}  (false arm)")
    print(f"  --> K-accuracy (correct K / GT arms matched to any pred) = {kacc:.3f}")

    print("\n" + "=" * 72)
    print("(3) COARSE-TIER  precision / recall / F1   {power, secondary, comm, guy, unspecified}")
    print("=" * 72)
    print(f"  {'tier':<14}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}{'F1':>8}")
    for t in TIER_ORDER:
        tp, fp, fn = m['tier_tp'][t], m['tier_fp'][t], m['tier_fn'][t]
        sup = m['tier_support'][t]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  {t:<14}{sup:>6}{tp:>6}{fp:>6}{fn:>6}{p:>8.3f}{r:>8.3f}{_f1(p, r):>8.3f}")

    print("\n" + "=" * 72)
    print("(4) OVERALL PER-POLE (= per-image) FIDELITY")
    print("=" * 72)
    fid = m['pole_exact'] / m['pole_total'] if m['pole_total'] else 0.0
    print(f"  images (poles) .............. {m['pole_total']}")
    if m.get('dg_anchor_poles'):
        print(f"  down_guy anchor-inventory .. {m['dg_anchor_poles']} poles (count-based GT)")
    print(f"  exact class-multiset match .. {m['pole_exact']}")
    print(f"  --> per-pole fidelity ....... {fid:.3f}")
    print()
    return fid, kacc


# --------------------------------------------------------------------------- #
# anchor-inventory down_guy expectations (eval-only; training unchanged)
# --------------------------------------------------------------------------- #
def _load_dg_expectations(enabled: bool):
    if not enabled:
        return {}
    from src.pole_anchor_down_guy import build_photo_expectations
    exps, jobs = build_photo_expectations()
    print(f"[down_guy] anchor-inventory GT: {len(exps)} pole photos across {len(jobs)} jobs")
    return exps


# --------------------------------------------------------------------------- #
# per-class confidence tuning (tune on one split, apply on another)
# --------------------------------------------------------------------------- #
def _infer_split(model, data_root, split, imgsz, conf, device):
    """Run the model over a split at a (low) floor conf -> (images, stems)."""
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split
    imgs = sorted([p for ext in ("*.jpg", "*.jpeg", "*.png") for p in img_dir.glob(ext)])
    images, stems = [], []
    for k, img in enumerate(imgs):
        gts = load_gt(lbl_dir / (img.stem + ".txt"))
        res = model.predict(str(img), imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
        images.append((load_preds_from_result(res), gts))
        stems.append(img.stem)
        if (k + 1) % 200 == 0:
            print(f"  ...{k + 1}/{len(imgs)}")
    return images, stems


def filter_images(images, thresholds):
    """Drop preds whose conf < thresholds[pred.cls]. thresholds: list[float] per class."""
    return [([p for p in preds if p.conf >= thresholds[p.cls]], gts) for preds, gts in images]


def tune_per_class(images, kp_thresh, iou_thresh, default=0.20):
    """Per-class F1-optimal conf threshold via CLASS-AWARE greedy matching.

    For each class c, match its preds to its GT (class-aware), record each pred's
    (conf, is_TP), then sweep the threshold over observed confs to maximize F1.
    Classes with no GT/preds keep `default`. Returns (thresholds list, info dict).
    """
    n_cls = len(NAMES)
    cand = [[] for _ in range(n_cls)]      # per class: list of (conf, is_tp)
    gt_count = [0] * n_cls
    for preds, gts in images:
        for c in range(n_cls):
            preds_c = [p for p in preds if p.cls == c]
            gts_c = [g for g in gts if g.cls == c]
            gt_count[c] += len(gts_c)
            if not preds_c:
                continue
            matches, _, _ = match_image(preds_c, gts_c, kp_thresh, iou_thresh)
            matched = {pi for pi, _ in matches}
            for i, p in enumerate(preds_c):
                cand[c].append((p.conf, i in matched))
    thresholds = [default] * n_cls
    info = {}
    for c in range(n_cls):
        if not cand[c] or gt_count[c] == 0:
            info[c] = (thresholds[c], 0.0, gt_count[c], False)
            continue
        confs = sorted({cc for cc, _ in cand[c]})
        best_f1, best_t = -1.0, confs[0]
        for t in confs:
            tp = sum(1 for cc, is_tp in cand[c] if cc >= t and is_tp)
            fp = sum(1 for cc, is_tp in cand[c] if cc >= t and not is_tp)
            fn = gt_count[c] - tp
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = _f1(p, r)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = round(float(best_t), 4)
        info[c] = (thresholds[c], best_f1, gt_count[c], True)
    return thresholds, info


def _micro(m):
    tp, fp, fn = sum(m['tp']), sum(m['fp']), sum(m['fn'])
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, _f1(p, r)


def run_tuned(args):
    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        return 2
    data_root = Path(args.data)
    for split in (args.tune_split, args.split):
        if not (data_root / "images" / split).is_dir():
            print(f"ERROR: split missing: {data_root/'images'/split}")
            return 2
    try:
        from ultralytics import YOLO
    except Exception as e:                       # pragma: no cover
        print(f"ERROR: ultralytics not importable: {e}")
        return 2

    model = YOLO(str(weights))
    floor = args.floor_conf
    dg_exp = _load_dg_expectations(getattr(args, "anchor_down_guy", True))
    score_kw = dict(kp_thresh=args.kp_thresh, iou_thresh=args.iou_thresh,
                    kind_gate=args.kind_gate, dg_expectations=dg_exp)

    print(f"[tune] infer '{args.tune_split}' @ floor conf {floor} (imgsz {args.imgsz}) ...")
    val_imgs, _val_stems = _infer_split(model, data_root, args.tune_split, args.imgsz, floor, args.device)
    thresholds, info = tune_per_class(val_imgs, args.kp_thresh, args.iou_thresh, default=args.conf)

    print(f"\nPer-class F1-optimal conf (tuned on '{args.tune_split}'):")
    print(f"  {'class':<16}{'conf>=':>8}{'valF1':>8}{'GT':>7}")
    for c in range(len(NAMES)):
        t, f1, sup, tuned = info[c]
        tag = "" if tuned else "  (default; no val GT/preds)"
        print(f"  {NAMES[c]:<16}{t:>8.3f}{f1:>8.3f}{sup:>7}{tag}")
    if args.save_conf:
        import json
        payload = {NAMES[c]: thresholds[c] for c in range(len(NAMES))}
        Path(args.save_conf).write_text(json.dumps(payload, indent=2))
        print(f"  saved per-class conf -> {args.save_conf}")

    print(f"\n[eval] infer '{args.split}' @ floor conf {floor} ...")
    test_imgs, test_stems = _infer_split(model, data_root, args.split, args.imgsz, floor, args.device)
    base = score(filter_images(test_imgs, [args.conf] * len(NAMES)),
                 photo_stems=test_stems, **score_kw)
    tuned = score(filter_images(test_imgs, thresholds),
                  photo_stems=test_stems, **score_kw)

    print("\n" + "#" * 72)
    print(f"# TUNED per-class operating point  —  '{args.split}' split")
    print("#" * 72)
    print_report(tuned)

    bp, br, bf = _micro(base)
    tp_, tr, tf = _micro(tuned)
    b_fid = base['pole_exact'] / base['pole_total'] if base['pole_total'] else 0.0
    t_fid = tuned['pole_exact'] / tuned['pole_total'] if tuned['pole_total'] else 0.0
    b_k = base['arm_k_correct'] / base['arm_k_total'] if base['arm_k_total'] else 0.0
    t_k = tuned['arm_k_correct'] / tuned['arm_k_total'] if tuned['arm_k_total'] else 0.0
    print("=" * 72)
    print(f"COMPARISON on '{args.split}'   flat conf {args.conf}  ->  per-class tuned")
    print("=" * 72)
    print(f"  micro P     {bp:.3f}  ->  {tp_:.3f}")
    print(f"  micro R     {br:.3f}  ->  {tr:.3f}")
    print(f"  micro F1    {bf:.3f}  ->  {tf:.3f}")
    print(f"  crossarm-K  {b_k:.3f}  ->  {t_k:.3f}")
    print(f"  per-pole    {b_fid:.3f}  ->  {t_fid:.3f}")
    print()

    if getattr(args, "save_report", None):
        import json
        n_cls = len(NAMES)
        per_class = {}
        for c in range(n_cls):
            tp, fp, fn, sup = tuned['tp'][c], tuned['fp'][c], tuned['fn'][c], tuned['gt_support'][c]
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            per_class[NAMES[c]] = {"gt": sup, "tp": tp, "fp": fp, "fn": fn,
                                   "precision": round(p, 4), "recall": round(r, 4),
                                   "f1": round(_f1(p, r), 4)}
        payload = {
            "split": args.split, "weights": str(args.weights), "data": str(args.data),
            "micro": {"p": round(tp_, 4), "r": round(tr_, 4) if False else round(tr, 4), "f1": round(tf, 4)},
            "macro_f1": round(float(np.mean([v["f1"] for v in per_class.values() if v["gt"]])) if any(v["gt"] for v in per_class.values()) else 0.0, 4),
            "crossarm_k": round(t_k, 4), "per_pole_fidelity": round(t_fid, 4),
            "per_class": per_class,
        }
        Path(args.save_report).write_text(json.dumps(payload, indent=2))
        print(f"  saved machine-readable report -> {args.save_report}")
    return 0


# --------------------------------------------------------------------------- #
# real run
# --------------------------------------------------------------------------- #
def run_real(args):
    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        return 2
    data_root = Path(args.data)
    img_dir = data_root / "images" / args.split
    lbl_dir = data_root / "labels" / args.split
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        print(f"ERROR: dataset split missing.\n  images: {img_dir} (exists={img_dir.is_dir()})"
              f"\n  labels: {lbl_dir} (exists={lbl_dir.is_dir()})")
        return 2
    imgs = sorted([p for ext in ("*.jpg", "*.jpeg", "*.png")
                   for p in img_dir.glob(ext)])
    if not imgs:
        print(f"ERROR: no images found in {img_dir}")
        return 2

    try:
        from ultralytics import YOLO
    except Exception as e:                       # pragma: no cover
        print(f"ERROR: ultralytics not importable: {e}")
        return 2

    print(f"[unified_pole_detection] weights={weights}  imgsz={args.imgsz}  "
          f"split={args.split}  conf={args.conf}  images={len(imgs)}")
    model = YOLO(str(weights))
    dg_exp = _load_dg_expectations(getattr(args, "anchor_down_guy", True))

    images = []
    stems = []
    for k, img in enumerate(imgs):
        gts = load_gt(lbl_dir / (img.stem + ".txt"))
        res = model.predict(str(img), imgsz=args.imgsz, conf=args.conf,
                            verbose=False, device=args.device)[0]
        preds = load_preds_from_result(res)
        images.append((preds, gts))
        stems.append(img.stem)
        if (k + 1) % 200 == 0:
            print(f"  ...{k + 1}/{len(imgs)}")

    m = score(images, kp_thresh=args.kp_thresh, iou_thresh=args.iou_thresh,
              kind_gate=args.kind_gate, dg_expectations=dg_exp, photo_stems=stems)
    print_report(m)
    return 0


# --------------------------------------------------------------------------- #
# self-test (no weights / dataset needed) — exercises the SCORING logic
# --------------------------------------------------------------------------- #
def _mk(name, x, y):
    """Build a Det from a class NAME at keypoint (x, y) with a small box around it."""
    return Det(NAME_TO_IDX[name], (x, y, 0.05, 0.05), (x, y), conf=0.9)


def _synthetic_gt():
    """Two poles with a representative mix incl. an arm3 crossarm."""
    pole_a = [
        _mk('arm3', 0.50, 0.20),
        _mk('pin', 0.50, 0.30),
        _mk('secondary', 0.50, 0.55),
        _mk('catv', 0.50, 0.75),
        _mk('down_guy', 0.30, 0.80),
    ]
    pole_b = [
        _mk('deadend', 0.50, 0.25),
        _mk('neutral', 0.50, 0.60),
        _mk('fiber', 0.50, 0.80),
    ]
    return [pole_a, pole_b]


def _clone_as_preds(gt_poles):
    out = []
    for pole in gt_poles:
        out.append([Det(d.cls, d.box, d.kp, conf=0.9) for d in pole])
    return out


def run_self_test(kp_thresh, iou_thresh):
    print("\n" + "#" * 72)
    print("# SELF-TEST  (synthetic data; scoring logic only, no model)")
    print("#" * 72)
    gt = _synthetic_gt()
    ok = True

    # (a) preds == GT  -> everything perfect
    preds = _clone_as_preds(gt)
    images = list(zip(preds, gt))
    m = score(images, kp_thresh, iou_thresh)
    print("\n[a] feed GT as predictions (expect all P/R=1.0, K-acc=1.0, fidelity=1.0)")
    fid, kacc = print_report(m)
    per_class_perfect = all(
        (m['tp'][c] / (m['tp'][c] + m['fp'][c]) if (m['tp'][c] + m['fp'][c]) else 1.0) == 1.0
        and (m['tp'][c] / (m['tp'][c] + m['fn'][c]) if (m['tp'][c] + m['fn'][c]) else 1.0) == 1.0
        for c in range(len(NAMES)))
    a_ok = per_class_perfect and abs(kacc - 1.0) < 1e-9 and abs(fid - 1.0) < 1e-9
    print(f"  [a] PASS={a_ok}  (per-class P/R all 1.0={per_class_perfect}, "
          f"K-acc={kacc:.3f}, fidelity={fid:.3f})")
    ok = ok and a_ok

    # (b) inject a class swap (catv->telco on pole A) and an arm-K mismatch (arm3->arm2)
    preds_b = _clone_as_preds(gt)
    preds_b[0][3].cls = NAME_TO_IDX['telco']    # was catv  -> fine-class swap (same tier)
    preds_b[0][0].cls = NAME_TO_IDX['arm2']     # was arm3  -> wrong K (still an arm)
    images_b = list(zip(preds_b, gt))
    mb = score(images_b, kp_thresh, iou_thresh)
    print("\n[b] inject class swap (catv->telco) + arm-K mismatch (arm3->arm2)")
    fid_b, kacc_b = print_report(mb)

    catv_i, telco_i = NAME_TO_IDX['catv'], NAME_TO_IDX['telco']
    checks = [
        # class swap: catv recall drops (its GT is now a miss), telco precision drops
        ("catv now has an FN (missed)", mb['fn'][catv_i] == 1),
        ("telco now has an FP (spurious)", mb['fp'][telco_i] == 1),
        # tier UNCHANGED for the catv->telco swap (both 'comm'): comm tier still perfect
        ("comm tier stays perfect (swap is within-tier)",
         mb['tier_fn']['comm'] == 0 and mb['tier_fp']['comm'] == 0),
        # arm-K: still matched as an arm, but K now wrong -> K-acc < 1.0
        ("arm matched as arm but K wrong -> K-acc<1.0", kacc_b < 1.0),
        ("arm K-mismatch counted (>=1 arm matched, fewer correct)",
         mb['arm_k_total'] >= 1 and mb['arm_k_correct'] < mb['arm_k_total']),
        # per-pole fidelity drops; pole A broke, pole B still exact -> 1/2 = 0.5
        ("per-pole fidelity dropped below 1.0", fid_b < 1.0),
        ("fidelity == 0.5 (pole B still exact)", abs(fid_b - 0.5) < 1e-9),
    ]

    print("\n  [b] directional checks:")
    b_ok = True
    for desc, passed in checks:
        b_ok = b_ok and passed
        print(f"    [{'PASS' if passed else 'FAIL'}]  {desc}")
    ok = ok and b_ok

    print("\n" + "#" * 72)
    print(f"# SELF-TEST OVERALL: {'PASS' if ok else 'FAIL'}")
    print("#" * 72 + "\n")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None, help="path to best.pt (omit -> self-test)")
    ap.add_argument("--imgsz", type=int, default=_cfg_default_imgsz())
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--data", default=str(_cfg_default_data_dir()))
    ap.add_argument("--device", default="0")
    ap.add_argument("--kp-thresh", type=float, default=0.04,
                    help="max normalized keypoint distance for a match")
    ap.add_argument("--iou-thresh", type=float, default=0.3,
                    help="box-IoU fallback threshold when keypoints are absent")
    ap.add_argument("--self-test", action="store_true",
                    help="run scoring-logic self-test on synthetic data and exit")
    ap.add_argument("--tune", action="store_true",
                    help="tune per-class conf on --tune-split, then eval --split with it")
    ap.add_argument("--tune-split", choices=["val", "test"], default="val",
                    help="split to tune per-class thresholds on (default: val)")
    ap.add_argument("--floor-conf", type=float, default=0.01,
                    help="low conf floor for inference during tuning (candidates)")
    ap.add_argument("--save-conf", default=None,
                    help="write the tuned per-class thresholds to this JSON path")
    ap.add_argument("--save-report", default=None,
                    help="write the machine-readable per-class F1 report to this JSON path")
    ap.add_argument("--no-kind-gate", dest="kind_gate", action="store_false",
                    help="disable kind-gated matching (let conductor preds match guy GTs)")
    ap.add_argument("--merged", action="store_true",
                    help="score the 14-class MERGED model (open_secondary->neutral, "
                         "catv/telco/fiber->comm; idea #1). Defaults --data to the merged dataset.")
    ap.add_argument("--no-anchor-down-guy", dest="anchor_down_guy", action="store_false",
                    help="disable anchor-inventory down_guy count GT (use label keypoints only)")
    ap.add_argument("--hwfirst", action="store_true",
                    help="score the 10-class HARDWARE-FIRST model (spool/three_bolt consolidated, "
                         "dead classes dropped). Defaults --data to the hwfirst dataset.")
    ap.set_defaults(kind_gate=True, anchor_down_guy=True)
    args = ap.parse_args()

    if args.merged or args.hwfirst:
        # Swap the module-level class-name globals BEFORE any scoring runs (every scoring fn reads
        # NAMES/NAME_TO_IDX at call time). decode_unified_class handles all variant names.
        global NAMES, NAME_TO_IDX, ARM_CLASSES
        if args.hwfirst:
            NAMES = list(config.UNIFIED_POLE_DETECTION_CLASS_NAMES_HWFIRST)
            ARM_CLASSES = ('arm2', 'arm3plus')   # hwfirst folds arm4plus -> arm3plus
            sub = "unified_pole_detection_hwfirst"
        else:
            NAMES = list(config.UNIFIED_POLE_DETECTION_CLASS_NAMES_MERGED)
            sub = "unified_pole_detection_merged"
        NAME_TO_IDX = {n: i for i, n in enumerate(NAMES)}
        if args.data == str(_cfg_default_data_dir()):
            args.data = str(_cfg_default_data_dir().parent / sub)
        print(f"[{'hwfirst' if args.hwfirst else 'merged'}] {len(NAMES)} classes; data={args.data}")

    if args.self_test or not args.weights:
        if not args.weights and not args.self_test:
            print("No --weights given; running self-test (scoring logic on synthetic data).")
        sys.exit(run_self_test(args.kp_thresh, args.iou_thresh))

    if args.tune:
        sys.exit(run_tuned(args))

    sys.exit(run_real(args))


if __name__ == "__main__":
    main()
