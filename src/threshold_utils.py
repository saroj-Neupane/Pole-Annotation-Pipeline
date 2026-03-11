"""
Shared threshold sweep and config update logic for YOLO detection models.
Used by scripts/threshold_sweep.py and evaluation_attachment_equipment.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from .config import PROJECT_ROOT

# Model config: (weights_relative_path, data_yaml_relative_path)
YOLO_DETECTION_MODELS: Dict[str, tuple] = {
    "pole_detection": (
        "runs/pole_detection/weights/best.pt",
        "datasets/pole_detection/data.yaml",
    ),
    "ruler_detection": (
        "runs/ruler_detection/weights/best.pt",
        "datasets/ruler_detection/data.yaml",
    ),
    "equipment_detection": (
        "runs/equipment_detection/weights/best.pt",
        "datasets/equipment_detection/data.yaml",
    ),
    "attachment_detection": (
        "runs/attachment_detection/weights/best.pt",
        "datasets/attachment_detection/data.yaml",
    ),
}


def run_threshold_sweep(
    model_name: str,
    weights_path: Path,
    data_yaml: Path,
    split: str = "val",
    verbose: bool = False,
    return_curves: bool = False,
) -> dict:
    """
    Run YOLO val, extract optimal conf per class and overall (F1-maximizing).
    split: 'val' or 'test' - which split to run on (use 'val' for threshold selection).
    If return_curves=True, also returns px, f1_curve, p_curve, r_curve, prec_values, names
    for chart generation.
    """
    from ultralytics import YOLO

    if not weights_path.exists():
        return {"error": f"Weights not found: {weights_path}"}

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        plots=False,
        verbose=verbose,
    )

    box = metrics.box
    if not hasattr(box, "f1_curve") or box.f1_curve is None:
        return {"error": "No F1 curve in validation results"}

    px = np.array(box.px)
    f1_curve = np.array(box.f1_curve)
    names = metrics.names or {}

    # Build class index -> name (ap_class_index order)
    ap_class_index = getattr(box, "ap_class_index", list(range(f1_curve.shape[0])))
    if hasattr(ap_class_index, "tolist"):
        ap_class_index = ap_class_index.tolist()
    names_ordered = [names.get(i, f"class_{i}") for i in ap_class_index]

    result: Dict[str, Any] = {
        "model": model_name,
        "optimal_overall": None,
        "optimal_per_class": {},
    }

    # Overall: mean F1 across classes
    f1_mean = f1_curve.mean(0)
    opt_idx = int(np.argmax(f1_mean))
    result["optimal_overall"] = {
        "confidence": round(float(px[opt_idx]), 4),
        "f1": round(float(f1_mean[opt_idx]), 4),
    }

    # Per-class
    for i in range(f1_curve.shape[0]):
        class_name = names_ordered[i] if i < len(names_ordered) else names.get(ap_class_index[i], f"class_{i}")
        idx = int(np.argmax(f1_curve[i]))
        result["optimal_per_class"][class_name] = {
            "confidence": round(float(px[idx]), 4),
            "f1": round(float(f1_curve[i, idx]), 4),
        }

    if return_curves:
        p_curve = np.array(box.p_curve) if hasattr(box, "p_curve") and box.p_curve is not None else None
        r_curve = np.array(box.r_curve) if hasattr(box, "r_curve") and box.r_curve is not None else None
        prec_values = np.array(box.prec_values) if hasattr(box, "prec_values") and box.prec_values is not None else None
        result["px"] = px.tolist()
        result["f1_curve"] = f1_curve.tolist()
        result["p_curve"] = p_curve.tolist() if p_curve is not None else None
        result["r_curve"] = r_curve.tolist() if r_curve is not None else None
        result["prec_values"] = prec_values.tolist() if prec_values is not None else None
        result["names"] = {int(k): v for k, v in names.items()}
        result["names_ordered"] = names_ordered

    return result


def run_val_on_split(
    weights_path: Path,
    data_yaml: Path,
    split: str,
    conf: float,
    optimal_per_class: Dict[str, Any],
    names_ordered: list,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run YOLO val on a specific split (e.g. 'test') and return chart data:
    px, f1_curve, p_curve, r_curve, prec_values, confusion_matrix, and
    optimal_per_class with F1 values computed on this split at val-chosen confs.

    Uses conf=0.001 for the F1/PR curves (full sweep from low to high conf, same as
    val during training). Runs a second val with conf=optimal for confusion matrix.
    """
    from ultralytics import YOLO

    if not weights_path.exists():
        return {"error": f"Weights not found: {weights_path}"}

    model = YOLO(str(weights_path))
    # Use low conf for curves so we get full F1-confidence sweep (like val in runs/)
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        plots=False,
        verbose=verbose,
        conf=0.001,
    )

    box = metrics.box
    if not hasattr(box, "f1_curve") or box.f1_curve is None:
        return {"error": "No F1 curve in validation results"}

    px = np.array(box.px)
    f1_curve = np.array(box.f1_curve)
    p_curve = np.array(box.p_curve) if hasattr(box, "p_curve") and box.p_curve is not None else None
    r_curve = np.array(box.r_curve) if hasattr(box, "r_curve") and box.r_curve is not None else None
    prec_values = np.array(box.prec_values) if hasattr(box, "prec_values") and box.prec_values is not None else None
    names = metrics.names or {}

    # Per-class F1 at val-chosen conf (interpolate from curve)
    per_class_f1 = {}
    for i, cls_name in enumerate(names_ordered):
        if i >= f1_curve.shape[0]:
            break
        conf_val = optimal_per_class.get(cls_name, {}).get("confidence")
        if conf_val is None:
            continue
        # Find F1 at conf_val: interpolate or nearest
        idx = np.searchsorted(px, conf_val)
        idx = min(idx, len(px) - 1) if idx >= len(px) else max(0, idx)
        if idx > 0 and px[idx] != px[idx - 1]:
            # Linear interpolate
            t = (conf_val - px[idx - 1]) / (px[idx] - px[idx - 1])
            f1_at = (1 - t) * f1_curve[i, idx - 1] + t * f1_curve[i, idx]
        else:
            f1_at = float(f1_curve[i, idx])
        per_class_f1[cls_name] = {"confidence": conf_val, "f1": round(f1_at, 4)}

    # Confusion matrix at optimal conf (plots=True needed for Ultralytics to populate it)
    metrics_cm = model.val(
        data=str(data_yaml),
        split=split,
        plots=True,
        verbose=False,
        conf=float(conf),
    )
    confusion_matrix = None
    if hasattr(metrics_cm, "confusion_matrix") and metrics_cm.confusion_matrix is not None:
        cm = getattr(metrics_cm.confusion_matrix, "matrix", None)
        if hasattr(cm, "tolist"):
            confusion_matrix = cm.tolist()

    map50 = float(metrics.box.map50) if hasattr(metrics.box, "map50") else None
    return {
        "px": px.tolist(),
        "f1_curve": f1_curve.tolist(),
        "p_curve": p_curve.tolist() if p_curve is not None else None,
        "r_curve": r_curve.tolist() if r_curve is not None else None,
        "prec_values": prec_values.tolist() if prec_values is not None else None,
        "names": {int(k): v for k, v in names.items()},
        "names_ordered": names_ordered,
        "confusion_matrix": confusion_matrix,
        "optimal_per_class": per_class_f1,
        "optimal_overall": {
            "confidence": conf,
            "f1": round(float(f1_curve.mean(0)[np.argmin(np.abs(px - conf))]), 4) if len(px) > 0 else 0,
        },
        "map_0_5": map50,
    }


def update_equipment_config(optimal_overall: float, optimal_per_class: Dict[str, float]) -> None:
    """Update src/config.py with equipment detection optimal thresholds (F1-maximizing per class)."""
    config_path = PROJECT_ROOT / "src" / "config.py"
    with open(config_path) as f:
        content = f.read()

    content = re.sub(
        r"INFERENCE_EQUIPMENT_CONF_THRESHOLD\s*=\s*[\d.]+",
        f"INFERENCE_EQUIPMENT_CONF_THRESHOLD = {optimal_overall}",
        content,
    )

    inner = ",\n".join(f"    '{k}': {v}" for k, v in optimal_per_class.items())
    content = re.sub(
        r"INFERENCE_EQUIPMENT_CONF_PER_CLASS\s*=\s*\{[^}]+\}",
        f"INFERENCE_EQUIPMENT_CONF_PER_CLASS = {{\n{inner}\n}}",
        content,
        count=1,
        flags=re.DOTALL,
    )

    with open(config_path, "w") as f:
        f.write(content)
    print(f"  Updated config: INFERENCE_EQUIPMENT_CONF_THRESHOLD = {optimal_overall}, per-class = {optimal_per_class}")


def update_attachment_config(optimal_overall: float, optimal_per_class: Dict[str, float]) -> None:
    """Update src/config.py with attachment detection optimal thresholds."""
    config_path = PROJECT_ROOT / "src" / "config.py"
    with open(config_path) as f:
        content = f.read()

    content = re.sub(
        r"INFERENCE_ATTACHMENT_CONF_THRESHOLD\s*=\s*[\d.]+",
        f"INFERENCE_ATTACHMENT_CONF_THRESHOLD = {optimal_overall}",
        content,
    )

    def replace_per_class(m):
        inner = ",\n".join(f"    '{k}': {v}" for k, v in optimal_per_class.items())
        return f"INFERENCE_ATTACHMENT_CONF_PER_CLASS = {{\n{inner}\n}}"

    content = re.sub(
        r"INFERENCE_ATTACHMENT_CONF_PER_CLASS\s*=\s*\{[^}]+\}",
        replace_per_class,
        content,
        count=1,
        flags=re.DOTALL,
    )

    with open(config_path, "w") as f:
        f.write(content)
    print(f"  Updated config: INFERENCE_ATTACHMENT_CONF_THRESHOLD = {optimal_overall}, per-class = {optimal_per_class}")


def update_config_all(results: Dict[str, dict]) -> None:
    """Update config for all models in results (pole, ruler, equipment, attachment)."""
    config_path = PROJECT_ROOT / "src" / "config.py"
    with open(config_path) as f:
        content = f.read()

    if "pole_detection" in results and "error" not in results["pole_detection"]:
        conf = results["pole_detection"]["optimal_overall"]["confidence"]
        content = re.sub(
            r"INFERENCE_POLE_CONF_THRESHOLD\s*=\s*[\d.]+",
            f"INFERENCE_POLE_CONF_THRESHOLD = {conf}",
            content,
        )
        print(f"  Updated config: INFERENCE_POLE_CONF_THRESHOLD = {conf}")

    if "ruler_detection" in results and "error" not in results["ruler_detection"]:
        conf = results["ruler_detection"]["optimal_overall"]["confidence"]
        content = re.sub(
            r"INFERENCE_RULER_CONF_THRESHOLD\s*=\s*[\d.]+",
            f"INFERENCE_RULER_CONF_THRESHOLD = {conf}",
            content,
        )
        print(f"  Updated config: INFERENCE_RULER_CONF_THRESHOLD = {conf}")

    if "equipment_detection" in results and "error" not in results["equipment_detection"]:
        r = results["equipment_detection"]
        conf_overall = r["optimal_overall"]["confidence"]
        per_class = r["optimal_per_class"]
        content = re.sub(
            r"INFERENCE_EQUIPMENT_CONF_THRESHOLD\s*=\s*[\d.]+",
            f"INFERENCE_EQUIPMENT_CONF_THRESHOLD = {conf_overall}",
            content,
        )
        new_per_class = {
            "riser": per_class.get("riser", {}).get("confidence", 0.2182),
            "transformer": per_class.get("transformer", {}).get("confidence", 0.3714),
            "street_light": per_class.get("street_light", {}).get("confidence", 0.1011),
            "secondary_drip_loop": per_class.get("secondary_drip_loop", {}).get("confidence", 0.3093),
        }
        inner = ",\n".join(f"    '{k}': {v}" for k, v in new_per_class.items())
        content = re.sub(
            r"INFERENCE_EQUIPMENT_CONF_PER_CLASS\s*=\s*\{[^}]*\}",
            f"INFERENCE_EQUIPMENT_CONF_PER_CLASS = {{\n{inner}\n}}",
            content,
            count=1,
            flags=re.DOTALL,
        )
        print(f"  Updated config: INFERENCE_EQUIPMENT_CONF_THRESHOLD, INFERENCE_EQUIPMENT_CONF_PER_CLASS")

    if "attachment_detection" in results and "error" not in results["attachment_detection"]:
        r = results["attachment_detection"]
        content = re.sub(
            r"INFERENCE_ATTACHMENT_CONF_THRESHOLD\s*=\s*[\d.]+",
            f"INFERENCE_ATTACHMENT_CONF_THRESHOLD = {r['optimal_overall']['confidence']}",
            content,
        )
        per_class = {k: v["confidence"] for k, v in r["optimal_per_class"].items()}
        inner = ",\n".join(f"    '{k}': {v}" for k, v in per_class.items())
        content = re.sub(
            r"INFERENCE_ATTACHMENT_CONF_PER_CLASS\s*=\s*\{[^}]*\}",
            f"INFERENCE_ATTACHMENT_CONF_PER_CLASS = {{\n{inner}\n}}",
            content,
            count=1,
        )
        print(f"  Updated config: INFERENCE_ATTACHMENT_CONF_THRESHOLD, INFERENCE_ATTACHMENT_CONF_PER_CLASS")

    with open(config_path, "w") as f:
        f.write(content)
    print(f"  Config updated: {config_path}")
