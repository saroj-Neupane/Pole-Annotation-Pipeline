"""Format SDK pipeline results for the demo UI.

Small pure-python helpers replacing the old torch-based batched_inference
module. The display shapes match what the demo/landing page JS consumes:
keypoints carry x, y, percentX, percentY, conf (+ height for ruler ticks).
"""
from typing import Any, Dict, Optional


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and tuples to JSON-safe types."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _pct(value: float, total: int) -> float:
    return (float(value) / total) * 100 if total > 0 else 0.0


def calibration_display(result: Dict[str, Any], img_w: int, img_h: int) -> Dict[str, Any]:
    """CalibrationPipeline.run() result -> demo display dict."""
    markings = None
    if result.get("ruler_keypoints"):
        markings = []
        for kp in result["ruler_keypoints"]:
            markings.append({
                "name": kp.get("name"),
                # ruler tick names are decimal feet ("2.5" .. "16.5"); the UI
                # formats kp.height as a feet-inches label
                "height": kp.get("name"),
                "x": float(kp["x"]),
                "y": float(kp["y"]),
                "percentX": _pct(kp["x"], img_w),
                "percentY": _pct(kp["y"], img_h),
                "conf": round(float(kp.get("conf", 0.0)), 4),
            })
    pole_top: Optional[Dict[str, Any]] = None
    if result.get("pole_top"):
        pt = result["pole_top"]
        pole_top = {
            "x": float(pt["x"]),
            "y": float(pt["y"]),
            "percentX": _pct(pt["x"], img_w),
            "percentY": _pct(pt["y"], img_h),
            "conf": round(float(pt.get("conf", 0.0)), 4),
        }
    return {"Ruler Markings": markings, "Pole Top": pole_top}


def equipment_display(result: Dict[str, Any], img_w: int, img_h: int) -> Dict[str, Any]:
    """EquipmentAnnotationPipeline.run() result -> demo display dict."""
    equipment = []
    for det in result.get("equipment", []):
        entry: Dict[str, Any] = {
            "type": det.get("cls_name", "?"),
            "conf": round(float(det.get("conf", 0.0)), 3),
        }
        kps = det.get("keypoints") or []
        if kps:
            entry["keypoints"] = [
                {
                    "name": kp.get("name"),
                    "x": float(kp["x"]),
                    "y": float(kp["y"]),
                    "percentX": _pct(kp["x"], img_w),
                    "percentY": _pct(kp["y"], img_h),
                    "conf": round(float(kp.get("conf", 0.0)), 4),
                }
                for kp in kps
            ]
        equipment.append(entry)
    return {"Equipment": equipment or None}
