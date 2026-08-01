"""
Compare equipment ONNX pipeline vs PyTorch E2E on sample images.

Run from project root:

    USE_PRODUCTION_MODELS=true python sdk/equipment_annotation_sdk/tools/parity_check.py [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]   # tools/ -> v1/ -> equipment_annotation_sdk/ -> sdk/ -> repo
SDK_ROOT = Path(__file__).resolve().parents[1]   # the versioned SDK dir (…/v1)

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SDK_ROOT))

os.environ.setdefault("USE_PRODUCTION_MODELS", "true")


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_preds(onnx_preds: list, torch_preds: list, iou_thresh: float = 0.5) -> list:
    pairs = []
    used_t = set()
    for o in onnx_preds:
        best_iou, best_j = 0.0, -1
        for j, t in enumerate(torch_preds):
            if j in used_t or o["cls_name"] != t["cls_name"]:
                continue
            iou = _bbox_iou(o["bbox"], t["bbox"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            used_t.add(best_j)
            pairs.append((o, torch_preds[best_j], best_iou))
    return pairs


def _kp_distance(onnx_kps: list, torch_kps: list) -> list[float]:
    dists = []
    torch_by_name = {k["name"]: k for k in torch_kps}
    for ok in onnx_kps:
        tk = torch_by_name.get(ok["name"])
        if tk is None:
            continue
        dists.append(float(np.hypot(ok["x"] - tk["x"], ok["y"] - tk["y"])))
    return dists


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Max images to check")
    args = parser.parse_args()

    import torch
    from src.config import (
        EQUIPMENT_CLASS_NAMES,
        EQUIPMENT_DETECTION_CONFIG,
        EQUIPMENT_E2E_IMAGES_DIR,
        INFERENCE_EQUIPMENT_CONF_PER_CLASS,
    )
    from src.data_utils import get_e2e_test_images
    from src.evaluation_attachment_equipment import (
        _load_equipment_models,
        _load_pole_detector,
        _run_e2e_single_image,
    )
    from equipment_annotation.pipeline import EquipmentAnnotationPipeline

    device = torch.device("cpu")
    pole_detector = _load_pole_detector()
    equip_detector, kp_models = _load_equipment_models(device)
    base_conf = min(INFERENCE_EQUIPMENT_CONF_PER_CLASS.values())
    imgsz = EQUIPMENT_DETECTION_CONFIG["imgsz"]

    images = get_e2e_test_images("equipment")
    if not images:
        images = sorted(EQUIPMENT_E2E_IMAGES_DIR.glob("*.jpg"))
    images = images[: args.limit]

    if not images:
        print("No test images found.", file=sys.stderr)
        return 1

    pole_onnx = REPO_ROOT / "sdk/calibration_sdk/v2/calibration/weights/pole_detection.onnx"
    if not pole_onnx.exists():
        print(f"Missing pole ONNX: {pole_onnx}", file=sys.stderr)
        return 1

    pipe = EquipmentAnnotationPipeline(pole_weights_path=pole_onnx)
    pipe.warmup()

    print(f"Checking {len(images)} image(s)...\n")
    all_ious: list[float] = []
    all_kp_dists: list[float] = []

    for img_path in images:
        torch_preds, _, _, _ = _run_e2e_single_image(
            Path(img_path),
            pole_detector,
            equip_detector,
            kp_models,
            device,
            base_conf,
            imgsz,
            EQUIPMENT_CLASS_NAMES,
        )
        onnx_result = pipe.run(str(img_path))
        onnx_preds = onnx_result["equipment"]

        pairs = _match_preds(onnx_preds, torch_preds)
        print(f"{Path(img_path).name}: onnx={len(onnx_preds)} torch={len(torch_preds)} matched={len(pairs)}")
        for o, t, iou in pairs:
            all_ious.append(iou)
            dists = _kp_distance(o.get("keypoints") or [], t.get("keypoints") or [])
            if dists:
                all_kp_dists.extend(dists)
                print(f"  {o['cls_name']}: IoU={iou:.3f} kp_L2 mean={np.mean(dists):.1f}px max={max(dists):.1f}px")

    if all_ious:
        print(f"\nBBox IoU: mean={np.mean(all_ious):.3f} min={np.min(all_ious):.3f}")
    if all_kp_dists:
        print(f"Keypoint L2 (px): mean={np.mean(all_kp_dists):.1f} max={np.max(all_kp_dists):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
