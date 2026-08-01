#!/usr/bin/env python3
"""
Apples-to-apples E2E comparison: single YOLO-pose equipment model vs. the deployed
two-stage YOLO-box + per-class HRNet pipeline.

Both pipelines are evaluated on the SAME manifest test images (never seen in either
model's training), through the SAME pole-crop front-end and the SAME scoring function
(evaluation_attachment_equipment._compute_class_metrics), so detection P/R/F1, mAP and
keypoint PCK (vertical error in inches via PPI) are directly comparable.

Per-class confidence is tuned independently for each pipeline on the val split
(F1-maximising via the same E2E metric), then locked and reported on test — mirroring
the baseline's "sweep on val, report on test" protocol.

Usage:
    python scripts/eval/eval_equipment_pose_e2e.py \
        --pose-weights runs/equipment_pose_detection/weights/best.pt
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EQUIPMENT_CLASS_NAMES,
    EQUIPMENT_DETECTION_CONFIG,
    EQUIPMENT_KEYPOINT_CONFIGS,
    EQUIPMENT_POSE_CLASS_NUM_KP,
    POLE_LABELS_DIR,
    INFERENCE_POLE_CONF_THRESHOLD,
    INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET,
)
from src.data_utils import (
    get_e2e_test_images,
    get_e2e_val_images,
    parse_equipment_with_keypoints,
    load_ppi_from_label,
)
from src.height_calculations import fit_height_from_location_file
from src.evaluation_attachment_equipment import (
    _load_pole_detector,
    _load_equipment_models,
    _extract_equipment_crop,
    _detect_and_kp_on_crop,
    _compute_class_metrics,
    _equipment_gt_normalizer,
)

BASE_CONF = 0.02   # run detectors permissively; per-class thresholds applied at scoring time
CONF_GRID = np.round(np.arange(0.05, 0.85 + 1e-9, 0.02), 4)


def _gt_normalizer_equip(inst):
    num_kp = EQUIPMENT_KEYPOINT_CONFIGS.get(inst['class_name'], (None, 2, None))[1]
    return _equipment_gt_normalizer(inst, num_kp)


def _apply_sdl_max(preds):
    """Keep at most INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET highest-conf secondary_drip_loop preds."""
    sdl = [d for d in preds if d['cls_name'] == 'secondary_drip_loop']
    if len(sdl) > INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET:
        sdl.sort(key=lambda d: d['conf'], reverse=True)
        keep = {id(d) for d in sdl[:INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET]}
        preds = [d for d in preds if d['cls_name'] != 'secondary_drip_loop' or id(d) in keep]
    return preds


def _pose_detect_on_crop(crop_bgr, crop_x1, crop_y1, pose_model, base_conf, imgsz):
    """Run the single YOLO-pose model on a crop; return preds in full-image coords.

    Keypoints come straight from the pose head (no second crop), truncated to each
    class's real keypoint count so padded slots are never scored. Same dict shape and
    same sdl max-det as the two-stage predictor for an exact apples-to-apples score.
    """
    res = pose_model(crop_bgr, conf=base_conf, max_det=20, verbose=False, imgsz=imgsz)[0]
    preds = []
    if res.boxes is None or len(res.boxes) == 0:
        return preds
    kpts_xy = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
    kpts_cf = res.keypoints.conf.cpu().numpy() if (res.keypoints is not None and res.keypoints.conf is not None) else None
    for i in range(len(res.boxes)):
        bbox = res.boxes.xyxy[i].cpu().numpy()
        conf = float(res.boxes.conf[i].cpu().numpy())
        cls_id = int(res.boxes.cls[i].cpu().numpy())
        cls_name = EQUIPMENT_CLASS_NAMES[cls_id] if cls_id < len(EQUIPMENT_CLASS_NAMES) else 'unknown'
        ex1, ey1, ex2, ey2 = map(int, bbox)
        x1f, y1f, x2f, y2f = crop_x1 + ex1, crop_y1 + ey1, crop_x1 + ex2, crop_y1 + ey2
        det = {'cls_id': cls_id, 'cls_name': cls_name, 'bbox': (x1f, y1f, x2f, y2f), 'conf': conf, 'keypoints': []}
        n_real = EQUIPMENT_POSE_CLASS_NUM_KP.get(cls_name, 0)
        if kpts_xy is not None and i < len(kpts_xy):
            for k in range(min(n_real, kpts_xy.shape[1])):
                kx = float(kpts_xy[i, k, 0]) + crop_x1
                ky = float(kpts_xy[i, k, 1]) + crop_y1
                kc = float(kpts_cf[i, k]) if kpts_cf is not None else 1.0
                det['keypoints'].append({'name': f'kp{k}', 'x': kx, 'y': ky, 'conf': kc})
        preds.append(det)
    return _apply_sdl_max(preds)


def _filter_cached(cached, per_class_conf):
    """Return a shallow copy with preds filtered by per-class confidence."""
    out = []
    for r in cached:
        kept = [p for p in r['preds'] if p['conf'] >= per_class_conf.get(p['cls_name'], 0.0)]
        out.append({**r, 'preds': kept})
    return out


def tune_per_class_conf(cached_val):
    """Pick the F1-maximising confidence per class on val using the E2E metric."""
    best = {}
    for cls in EQUIPMENT_CLASS_NAMES:
        best_f1, best_c = -1.0, CONF_GRID[0]
        for c in CONF_GRID:
            filt = _filter_cached(cached_val, {cls: float(c)})
            m = _compute_class_metrics(filt, cls, _gt_normalizer_equip)
            f1 = m['detection']['f1']
            if f1 > best_f1:
                best_f1, best_c = f1, float(c)
        best[cls] = best_c
    return best


def score(cached_test, per_class_conf):
    filt = _filter_cached(cached_test, per_class_conf)
    return {cls: _compute_class_metrics(filt, cls, _gt_normalizer_equip) for cls in EQUIPMENT_CLASS_NAMES}


def _fmt_pct(x):
    return f"{x:5.1f}" if x is not None else "  -  "


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument('--pose-weights', default=str(PROJECT_ROOT / 'runs/equipment_pose_detection/weights/best.pt'))
    ap.add_argument('--out', default=str(PROJECT_ROOT / 'results/equipment/pose_vs_hrnet_e2e.json'))
    ap.add_argument('--imgsz', type=int, default=EQUIPMENT_DETECTION_CONFIG['imgsz'])
    ap.add_argument('--limit', type=int, default=0, help='Cap images per split (0 = all); for smoke tests')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from ultralytics import YOLO

    pole_detector = _load_pole_detector()
    box_detector, hrnet_models = _load_equipment_models(device)
    pose_model = YOLO(args.pose_weights)

    # _detect_and_kp_on_crop needs the full-image RGB for the HRNet second crop, so the
    # predictors close over the per-image RGB built inside run_both.
    val_images = get_e2e_val_images('equipment')
    test_images = get_e2e_test_images('equipment')
    if args.limit:
        val_images = val_images[:args.limit]
        test_images = test_images[:args.limit]
    print(f"Val images: {len(val_images)}  |  Test images: {len(test_images)}")

    import cv2

    def make_baseline_predictor(img_rgb):
        def _pred(crop, cx1, cy1):
            return _detect_and_kp_on_crop(
                crop, img_rgb, cx1, cy1, box_detector, BASE_CONF, args.imgsz,
                EQUIPMENT_CLASS_NAMES, hrnet_models, device,
            )
        return _pred

    def make_pose_predictor(_img_rgb):
        def _pred(crop, cx1, cy1):
            return _pose_detect_on_crop(crop, cx1, cy1, pose_model, BASE_CONF, args.imgsz)
        return _pred

    def run_both(images, desc):
        """One pole pass per image; run both predictors on the same crop."""
        cached_base, cached_pose = [], []
        for img_path in tqdm(images, desc=desc):
            img_bgr = cv2.imread(str(img_path))
            empty = {'preds': [], 'ppi': None, 'h': 0, 'w': 0, 'gt': []}
            if img_bgr is None:
                cached_base.append(empty); cached_pose.append(empty); continue
            h_img, w_img = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            from src import photo_id_layout as _pil
            lbl = _pil.loc_path(POLE_LABELS_DIR, img_path.stem)
            _have = lbl is not None and lbl.exists()
            ppi = load_ppi_from_label(lbl) if _have else None
            gt = parse_equipment_with_keypoints(lbl) if _have else []
            fit = fit_height_from_location_file(lbl) if _have else None
            base = {'ppi': ppi, 'h': h_img, 'w': w_img, 'gt': gt, 'fit': fit}
            pres = pole_detector(img_bgr, conf=INFERENCE_POLE_CONF_THRESHOLD, max_det=1, verbose=False, imgsz=960)[0]
            if pres.boxes is None or len(pres.boxes) == 0:
                cached_base.append({**base, 'preds': []}); cached_pose.append({**base, 'preds': []}); continue
            px1, py1, px2, py2 = map(int, pres.boxes.xyxy[0].cpu().numpy())
            crop, bounds = _extract_equipment_crop(img_bgr, (px1, py1, px2, py2))
            if crop is None:
                cached_base.append({**base, 'preds': []}); cached_pose.append({**base, 'preds': []}); continue
            cx1, cy1 = bounds[0], bounds[1]
            cached_base.append({**base, 'preds': make_baseline_predictor(img_rgb)(crop, cx1, cy1)})
            cached_pose.append({**base, 'preds': make_pose_predictor(img_rgb)(crop, cx1, cy1)})
        return cached_base, cached_pose

    print("\n[1/2] Inference on VAL (for per-class conf tuning)...")
    val_base, val_pose = run_both(val_images, "val")
    conf_base = tune_per_class_conf(val_base)
    conf_pose = tune_per_class_conf(val_pose)
    print(f"  baseline per-class conf: {conf_base}")
    print(f"  pose     per-class conf: {conf_pose}")

    print("\n[2/2] Inference on TEST...")
    test_base, test_pose = run_both(test_images, "test")
    res_base = score(test_base, conf_base)
    res_pose = score(test_pose, conf_pose)

    # ---- report ----
    def macro(results, sub, key):
        vals = [results[c][sub][key] for c in EQUIPMENT_CLASS_NAMES if results[c][sub][key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    print("\n" + "=" * 100)
    print("EQUIPMENT E2E — single YOLO-pose  vs.  YOLO-box + per-class HRNet  (test split, identical images)")
    print("=" * 100)
    hdr = f"{'class':20s} {'pipeline':9s} | {'gt':>4s} {'P':>5s} {'R':>5s} {'F1':>5s} {'mAP50':>6s} {'mIoU':>5s} | {'PCK3':>5s} {'PCK2':>5s} {'PCK1':>5s} {'PCK.5':>5s} {'medErr':>6s}"
    print(hdr); print("-" * len(hdr))
    for cls in EQUIPMENT_CLASS_NAMES:
        for tag, r in (('HRNet', res_base), ('YOLOpose', res_pose)):
            d, k = r[cls]['detection'], r[cls]['keypoint']
            print(f"{cls if tag=='HRNet' else '':20s} {tag:9s} | {d['gt_count']:>4d} "
                  f"{d['precision']*100:5.1f} {d['recall']*100:5.1f} {d['f1']*100:5.1f} "
                  f"{(d['map_0_5'] or 0)*100:6.1f} {d['mean_iou']*100:5.1f} | "
                  f"{_fmt_pct(k['pck_3_inch'])} {_fmt_pct(k['pck_2_inch'])} {_fmt_pct(k['pck_1_inch'])} {_fmt_pct(k['pck_0_5_inch'])} "
                  f"{(k['median_error_inches'] if k['median_error_inches'] is not None else 0):6.2f}")
        print("-" * len(hdr))

    print(f"\n{'MACRO (mean over classes)':30s}  detF1     PCK@2in   PCK@1in   medErr(in)")
    for tag, r in (('HRNet (2-stage)', res_base), ('YOLO-pose (1-model)', res_pose)):
        print(f"  {tag:28s}  {macro(r,'detection','f1')*100:5.1f}     {macro(r,'keypoint','pck_2_inch'):5.1f}     "
              f"{macro(r,'keypoint','pck_1_inch'):5.1f}     {macro(r,'keypoint','median_error_inches'):5.2f}")

    out = {
        'evaluation_date': datetime.now().isoformat(),
        'test_images': len(test_images),
        'pose_weights': args.pose_weights,
        'conf_baseline': conf_base,
        'conf_pose': conf_pose,
        'per_class': {cls: {'hrnet': res_base[cls], 'yolo_pose': res_pose[cls]} for cls in EQUIPMENT_CLASS_NAMES},
        'macro': {
            'hrnet': {'det_f1': macro(res_base, 'detection', 'f1'), 'pck_2in': macro(res_base, 'keypoint', 'pck_2_inch'),
                      'pck_1in': macro(res_base, 'keypoint', 'pck_1_inch'), 'median_err_in': macro(res_base, 'keypoint', 'median_error_inches')},
            'yolo_pose': {'det_f1': macro(res_pose, 'detection', 'f1'), 'pck_2in': macro(res_pose, 'keypoint', 'pck_2_inch'),
                          'pck_1in': macro(res_pose, 'keypoint', 'pck_1_inch'), 'median_err_in': macro(res_pose, 'keypoint', 'median_error_inches')},
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n✓ saved {args.out}")


if __name__ == '__main__':
    main()
