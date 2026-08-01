#!/usr/bin/env python3
"""
F1 evaluation + operating-point sweep for the HRNet midspan-wire STRIP detector.

Training only ever tracked *recall* (at 1/2/3-inch tolerance) and picked the best
checkpoint on val *loss* — precision/F1 were never measured, so neither the column
profile nor the peak extractor was ever F1-tuned. This script (all post-hoc, no
training):

  1. Predicts the strip heatmap once per image; caches three 1-D column profiles
     (max over all columns, max over the central band, mean over the central band).
  2. Extracts peaks two ways: the shipped greedy segment-scan (``_heatmap_y_peaks``)
     and ``scipy.signal.find_peaks`` (height + distance + PROMINENCE — prominence is
     what kills the spurious peaks on an elevated baseline).
  3. Matches pred peaks to the *label* wire-ys (honest denominator — NOT the GT
     heatmap, which merges close wires and inflates recall) within an inch-tolerance
     derived from each strip's PPI.
  4. Reports the shipped baseline, sweeps val for max-F1, and reports the test split
     at the val-optimal operating point.

Usage:
    python scripts/eval_midspan_strip_f1.py
    python scripts/eval_midspan_strip_f1.py --weights runs/.../best.pth --tol-inch 2.0
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_DIRS, WIRE_STRIP_HEATMAP_HEIGHT, WIRE_STRIP_HEATMAP_WIDTH,
    WIRE_STRIP_PEAK_MIN_DISTANCE, WIRE_STRIP_PEAK_THRESHOLD,
    INFERENCE_MIDSPAN_WIRE_STRIP_WEIGHTS,
)
from src.datasets import _parse_wire_strip_label
from src.inference_utils import load_wire_strip_model, WIRE_STRIP_PREPROCESS

HM_H = WIRE_STRIP_HEATMAP_HEIGHT
CEN = WIRE_STRIP_HEATMAP_WIDTH // 2
BAND = max(WIRE_STRIP_HEATMAP_WIDTH // 6, 4)  # central column half-width (~16)


def _match(gt, pred, tol):
    """Greedy 1-1 nearest match in the same coordinate space. Returns TP count."""
    used, tp = set(), 0
    for g in gt:
        best_j, best_d = None, tol + 1e-9
        for j, p in enumerate(pred):
            if j in used:
                continue
            d = abs(g - p)
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None:
            tp += 1
            used.add(best_j)
    return tp


@torch.no_grad()
def predict(model, ds, split, device):
    """Return per-image dicts with 3 column profiles + GT ys + ppi + strip height."""
    img_dir, lbl_dir = Path(ds) / "images" / split, Path(ds) / "labels" / split
    out, files = [], sorted(img_dir.glob("*.jpg"))
    for k, img_path in enumerate(files):
        gt_ys, ppi = _parse_wire_strip_label(lbl_dir / f"{img_path.stem}.txt")
        if not gt_ys or ppi <= 0:
            continue
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h = img.shape[0]
        tensor = WIRE_STRIP_PREPROCESS(img).unsqueeze(0).to(device)
        hm = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()           # (HM_H, HM_W)
        band = hm[:, CEN - BAND:CEN + BAND]
        out.append({
            "max_all": hm.max(axis=1),
            "max_cen": band.max(axis=1),
            "mean_cen": band.mean(axis=1),
            "gt": np.array(gt_ys), "ppi": ppi, "h": h,
        })
        if (k + 1) % 100 == 0:
            print(f"  ...{k+1}/{len(files)} predicted")
    return out


def _greedy_peaks(profile, min_distance, threshold):
    """The shipped extractor: walk in min_distance steps, take each window's argmax."""
    peaks, n, i = [], len(profile), 0
    while i < n:
        seg = profile[i:min(i + min_distance, n)]
        if seg.size == 0:
            break
        lp = int(seg.argmax()) + i
        if profile[lp] >= threshold:
            peaks.append(lp)
            i = lp + min_distance
        else:
            i += min_distance
    return peaks


def prf1(preds, extractor, profile_key, tol_inch):
    """Aggregate P/R/F1 over a prediction set at one operating point + tolerance."""
    TP = FP = FN = 0
    for p in preds:
        peaks = extractor(p[profile_key])
        pred_norm = [pk / max(HM_H - 1, 1) for pk in peaks]
        gt_norm = list(p["gt"])
        tol_norm = tol_inch * p["ppi"] / max(p["h"] - 1, 1)
        tp = _match(gt_norm, pred_norm, tol_norm)
        TP += tp
        FP += len(pred_norm) - tp
        FN += len(gt_norm) - tp
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F1, TP, FP, FN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(INFERENCE_MIDSPAN_WIRE_STRIP_WEIGHTS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol-inch", type=float, default=2.0, help="primary F1 tolerance")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = DATASET_DIRS["midspan_wire_strip_detection"]

    print(f"== predicting heatmaps (weights={args.weights}) ==")
    model = load_wire_strip_model(weights_path=args.weights, device=device)
    val = predict(model, ds, "val", device)
    test = predict(model, ds, "test", device)
    print(f"val={len(val)} test={len(test)} strips with GT")

    # ---- baseline: shipped greedy extractor on max_all profile ----
    base = lambda prof: _greedy_peaks(prof, WIRE_STRIP_PEAK_MIN_DISTANCE, WIRE_STRIP_PEAK_THRESHOLD)
    print(f"\n== BASELINE  greedy(thr={WIRE_STRIP_PEAK_THRESHOLD}, min_d={WIRE_STRIP_PEAK_MIN_DISTANCE}), profile=max_all ==")
    for name, preds in (("val", val), ("test", test)):
        line = f"  {name:5s}"
        for t in (1.0, 2.0, 3.0):
            P, R, F1, *_ = prf1(preds, base, "max_all", t)
            line += f" | {t:.0f}in P={P:.3f} R={R:.3f} F1={F1:.3f}"
        print(line)

    # ---- sweep find_peaks (height/distance/prominence) x profile, max F1 on val ----
    print(f"\n== SWEEP find_peaks on val (max F1 @ {args.tol_inch:.0f}-inch) ==")
    heights = [0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75]
    distances = [8, 10, 12, 15, 20, 25, 30, 40]
    prominences = [0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    profiles = ["max_all", "max_cen", "mean_cen"]
    best = (-1.0, None)
    for prof in profiles:
        for h in heights:
            for d in distances:
                for pr in prominences:
                    ext = lambda profile, _h=h, _d=d, _pr=pr: find_peaks(
                        profile, height=_h, distance=_d, prominence=_pr)[0].tolist()
                    P, R, F1, *_ = prf1(val, ext, prof, args.tol_inch)
                    if F1 > best[0]:
                        best = (F1, (prof, h, d, pr, P, R))
    bf1, (bprof, bh, bd, bpr, bP, bR) = best
    print(f"  best val: profile={bprof} height={bh} distance={bd} prominence={bpr}")
    print(f"            -> P={bP:.3f} R={bR:.3f} F1={bf1:.3f}")

    # ---- report test at val-optimal find_peaks point ----
    best_ext = lambda profile: find_peaks(profile, height=bh, distance=bd, prominence=bpr)[0].tolist()
    print(f"\n== TEST @ val-optimal find_peaks (profile={bprof}, h={bh}, d={bd}, prom={bpr}) ==")
    for t in (1.0, 2.0, 3.0):
        P, R, F1, TP, FP, FN = prf1(test, best_ext, bprof, t)
        print(f"  {t:.0f}in  P={P:.3f} R={R:.3f} F1={F1:.3f}   (TP={TP} FP={FP} FN={FN})")

    F1b = prf1(test, base, "max_all", args.tol_inch)[2]
    F1o = prf1(test, best_ext, bprof, args.tol_inch)[2]
    print(f"\n  TEST F1@{args.tol_inch:.0f}in:  shipped {F1b:.3f}  ->  tuned find_peaks {F1o:.3f}  (Δ {F1o - F1b:+.3f})")


if __name__ == "__main__":
    main()
