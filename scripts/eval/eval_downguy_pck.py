#!/usr/bin/env python3
"""
Box-size-INVARIANT down_guy F1 for the wire_attachment_hw_detection box-shape sweep.

Why not `eval_wire_hw_f1.py`?
    The down_guy box is SYNTHETIC — a fixed-feet rectangle around the attachment
    keypoint, not a real object extent. So ultralytics Box-IoU-F1 rewards bigger boxes
    mechanically (easier IoU; OKS sigma scales with box area), which would make a 4ft box
    "win" for the wrong reason. The product only cares about the attachment KEYPOINT
    (location + class), so we score that directly and ignore the predicted box entirely:

      * GT  = down_guy keypoints from the BASELINE 1ft×2ft labels (same crops for every
              variant). Each GT instance's baseline box gives the per-photo inch scale
              (box height h == 12 in vertical; w/2 == 12 in horizontal; pixels are square).
      * pred = predicted down_guy (class 7) keypoints + conf, from ANY checkpoint.
      * match = greedy nearest by EUCLIDEAN inches within a tolerance; TP/FP/FN -> P/R/F1.
      * conf is swept; we report the F1-optimal operating point per tolerance.

    Because GT keypoints + inch scale are fixed and only the predicted keypoint is read,
    the comparison across box shapes is fair: a bigger training box helps ONLY if it
    genuinely improves where/whether the model fires.

Usage (run AFTER the sweep trains; needs the GPU for inference):
    python scripts/eval/eval_downguy_pck.py \
        --weights runs/wire_hw_dg_base_1x2c/weights/best.pt runs/wire_hw_dg_v2x1c/weights/best.pt \
        --labels base_1x2c v2x1c --split val --imgsz 1024
    # or glob every sweep run:
    python scripts/eval/eval_downguy_pck.py --runs-glob 'runs/wire_hw_dg_*/weights/best.pt' --split val
    # logic-only sanity check, no model / no GPU:
    python scripts/eval/eval_downguy_pck.py --self-test
"""
import argparse
import glob
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DOWN_GUY_CLASS_ID = 7
TOLS_IN = (3.0, 6.0, 12.0)        # euclidean-inch tolerances to report


def _read_gt_downguys(label_path: Path):
    """[(kp_x, kp_y, unit_x, unit_y)] for down_guy GT in a BASELINE label file.

    unit_x/unit_y = crop-normalized length of 1 foot in X / Y (from the baseline 1ft×2ft
    box: h == 1ft in Y, w/2 == 1ft in X). 12 inches = unit.
    """
    out = []
    if not label_path.exists():
        return out
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if not p or int(p[0]) != DOWN_GUY_CLASS_ID:
            continue
        w, h = float(p[3]), float(p[4])
        kpx, kpy = float(p[5]), float(p[6])
        if h <= 0 or w <= 0:
            continue
        out.append((kpx, kpy, w / 2.0, h))
    return out


def _match(gts, preds, tol_in):
    """Greedy 1:1 match of preds->gts by euclidean inches. Returns (tp, fp, fn).

    gts:   [(kpx, kpy, unit_x, unit_y)]   preds: [(kpx, kpy, conf)]
    Distance for a (gt, pred) pair uses the GT's per-foot units (true inches, square px).
    """
    pairs = []
    for gi, (gx, gy, ux, uy) in enumerate(gts):
        for pi, (px, py, _c) in enumerate(preds):
            dx_in = abs(px - gx) / ux * 12.0
            dy_in = abs(py - gy) / uy * 12.0
            d = math.hypot(dx_in, dy_in)
            if d <= tol_in:
                pairs.append((d, gi, pi))
    pairs.sort()
    g_used, p_used = set(), set()
    tp = 0
    for d, gi, pi in pairs:
        if gi in g_used or pi in p_used:
            continue
        g_used.add(gi); p_used.add(pi); tp += 1
    fn = len(gts) - tp
    fp = len(preds) - tp
    return tp, fp, fn


def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _sweep_operating_point(per_image, tol_in, conf_grid):
    """Best-F1 conf over a grid. per_image: [(gts, [(kpx,kpy,conf)])]. Returns dict."""
    best = {"f1": -1.0}
    for c in conf_grid:
        TP = FP = FN = 0
        for gts, preds in per_image:
            pf = [pp for pp in preds if pp[2] >= c]
            tp, fp, fn = _match(gts, pf, tol_in)
            TP += tp; FP += fp; FN += fn
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = _f1(prec, rec)
        if f1 > best["f1"]:
            best = {"conf": c, "f1": f1, "P": prec, "R": rec, "TP": TP, "FP": FP, "FN": FN}
    return best


def evaluate(weights, labels, split, imgsz, device, base_dataset, conf_floor):
    from ultralytics import YOLO

    base = Path(base_dataset)
    img_dir = base / "images" / split
    lbl_dir = base / "labels" / split
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    # GT per image (down_guy keypoints + inch scale) from baseline labels
    gt_by_img = {p.name: _read_gt_downguys(lbl_dir / f"{p.stem}.txt") for p in images}
    n_gt = sum(len(v) for v in gt_by_img.values())
    n_gt_imgs = sum(1 for v in gt_by_img.values() if v)

    conf_grid = [round(x, 3) for x in [i / 100 for i in range(2, 71, 2)]]
    print(f"\nGT: {n_gt} down_guy keypoints across {n_gt_imgs}/{len(images)} {split} images")

    rows = []
    for w, lab in zip(weights, labels):
        m = YOLO(w)
        per_image = []
        # per-image predict: a list source is treated as ONE batch by this ultralytics
        # version (588 images -> 11GB OOM), so loop explicitly to keep memory flat
        for img_path in images:
            r = m.predict(source=str(img_path), imgsz=imgsz, conf=conf_floor,
                          device=device, verbose=False)[0]
            gts = gt_by_img[img_path.name]
            preds = []
            if r.keypoints is not None and r.boxes is not None and len(r.boxes):
                cls = r.boxes.cls.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                kxy = r.keypoints.xy.cpu().numpy()      # (N, K, 2) pixel coords
                H, W = r.orig_shape
                for i in range(len(cls)):
                    if int(cls[i]) != DOWN_GUY_CLASS_ID:
                        continue
                    kx, ky = kxy[i, 0]
                    preds.append((kx / W, ky / H, float(conf[i])))
            per_image.append((gts, preds))

        res = {"label": lab}
        for tol in TOLS_IN:
            res[tol] = _sweep_operating_point(per_image, tol, conf_grid)
        rows.append(res)

    # report
    print(f"\n{'='*78}\nDOWN_GUY keypoint F1 (box-invariant; euclidean-inch tol, F1-optimal conf)\n{'='*78}")
    header = f"{'variant':<14}" + "".join([f"  | tol {int(t)}\"  P/R/F1@conf" for t in TOLS_IN])
    print(header)
    for res in rows:
        line = f"{res['label']:<14}"
        for tol in TOLS_IN:
            b = res[tol]
            line += f"  | {b['P']:.2f}/{b['R']:.2f}/{b['f1']:.3f}@{b['conf']:.2f}"
        print(line)
    # rank by 6" F1
    rows_sorted = sorted(rows, key=lambda r: r[6.0]["f1"], reverse=True)
    print(f"\nRanked by 6\" F1:")
    for i, res in enumerate(rows_sorted):
        b = res[6.0]
        print(f"  {i+1}. {res['label']:<14} F1={b['f1']:.3f}  P={b['P']:.3f} R={b['R']:.3f}  "
              f"(TP={b['TP']} FP={b['FP']} FN={b['FN']} @conf {b['conf']:.2f})")
    return rows


def self_test():
    """Validate matching logic on synthetic data (no model, no GPU)."""
    # one image: 1 GT at (0.5,0.5) with unit_x=unit_y=0.05 (1ft=5% of crop). 12in=0.05.
    gts = [(0.5, 0.5, 0.05, 0.05)]
    # pred exactly on it -> TP
    tp, fp, fn = _match(gts, [(0.5, 0.5, 0.9)], 3.0)
    assert (tp, fp, fn) == (1, 0, 0), (tp, fp, fn)
    # pred 4.8in below (dy=0.02 -> 4.8in): outside 3" tol, inside 6"
    tp3, _, _ = _match(gts, [(0.5, 0.52, 0.9)], 3.0)
    tp6, _, _ = _match(gts, [(0.5, 0.52, 0.9)], 6.0)
    assert tp3 == 0 and tp6 == 1, (tp3, tp6)
    # extra spurious pred far away -> FP
    tp, fp, fn = _match(gts, [(0.5, 0.5, 0.9), (0.1, 0.1, 0.9)], 3.0)
    assert (tp, fp, fn) == (1, 1, 0), (tp, fp, fn)
    # two preds near one GT -> only one matches (1:1)
    tp, fp, fn = _match(gts, [(0.5, 0.5, 0.9), (0.5, 0.501, 0.9)], 3.0)
    assert (tp, fp, fn) == (1, 1, 0), (tp, fp, fn)
    # conf sweep picks the threshold that drops the FP without losing the TP
    per_image = [(gts, [(0.5, 0.5, 0.8), (0.1, 0.1, 0.3)])]
    b = _sweep_operating_point(per_image, 3.0, [0.1, 0.5, 0.9])
    assert b["f1"] == 1.0 and 0.3 < b["conf"] <= 0.8, b
    print("self-test ✓ (matching, tolerance gating, 1:1, conf sweep all correct)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="*", default=[], help="checkpoint(s) to score")
    ap.add_argument("--labels", nargs="*", default=None, help="display name per --weights")
    ap.add_argument("--runs-glob", default=None,
                    help="glob for checkpoints, e.g. 'runs/wire_hw_dg_*/weights/best.pt'")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--base-dataset", default="datasets/wire_attachment_hw_detection",
                    help="dataset providing GT down_guy keypoints + inch scale (the 1ft baseline)")
    ap.add_argument("--conf-floor", type=float, default=0.02, help="inference conf floor (swept above this)")
    ap.add_argument("--class-id", type=int, default=7,
                    help="down_guy class id in --base-dataset AND in the checkpoints' class space "
                         "(7=wire_hw 8-class, 15=unified 17-class)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    global DOWN_GUY_CLASS_ID
    DOWN_GUY_CLASS_ID = args.class_id

    if args.self_test:
        self_test()
        return

    weights = list(args.weights)
    labels = list(args.labels) if args.labels else []
    if args.runs_glob:
        for w in sorted(glob.glob(args.runs_glob)):
            weights.append(w)
            labels.append(Path(w).parent.parent.name
                          .replace("wire_hw_dg_", "").replace("unified_dg_", ""))
    if not weights:
        sys.exit("provide --weights or --runs-glob (or --self-test)")
    if len(labels) != len(weights):
        labels = [Path(w).parent.parent.name for w in weights]

    base = args.base_dataset if Path(args.base_dataset).is_absolute() else str(PROJECT_ROOT / args.base_dataset)
    evaluate(weights, labels, args.split, args.imgsz, args.device, base, args.conf_floor)


if __name__ == "__main__":
    main()
