"""ONNX SDK model loader.

Loads the three production SDK pipelines (calibration, equipment annotation,
wire tracer) as lazy singletons. Weights are bundled under sdk/ in development;
when absent (e.g. a fresh Hugging Face Space container) they are downloaded
from the public HF model repo at startup.

No torch and no src/ imports — the deployed app runs entirely on the SDKs.
"""
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SDK_ROOT = PROJECT_ROOT / "sdk"

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "nsaroj789/pole-annotation-models")

# Production SDK versions — resolved from models/registry.json -> "sdks" pointers
# (the SSOT; a hardcoded copy here went stale at the v2.9 promotion).
def _registry_sdk_paths():
    import json
    fallback = {"calibration": "calibration_sdk/v2", "equipment": "equipment_annotation_sdk/v1",
                "wire_tracer": "wire_tracer_sdk/v2.9.1"}
    try:
        sdks = json.loads((PROJECT_ROOT / "models" / "registry.json").read_text())["sdks"]
        return {"calibration": sdks["calibration_sdk"]["path"],
                "equipment": sdks["equipment_annotation_sdk"]["path"],
                "wire_tracer": sdks["wire_tracer_sdk"]["path"]}
    except (OSError, KeyError, ValueError):
        return fallback


SDK_DIRS = {name: SDK_ROOT / rel for name, rel in _registry_sdk_paths().items()}

# One representative weight per SDK; if any is missing we fetch from HF.
_WEIGHT_SENTINELS = (
    SDK_DIRS["calibration"] / "calibration" / "weights" / "pole_detection.onnx",
    SDK_DIRS["equipment"] / "equipment_annotation" / "weights" / "equipment_detection.onnx",
    SDK_DIRS["wire_tracer"] / "wire_tracer" / "weights" / "unified_pole_detection.onnx",
)

_lock = threading.Lock()
_pipelines: Dict[str, Any] = {}
_weights_ready = False


def _ensure_import_paths() -> None:
    for sdk_dir in SDK_DIRS.values():
        path_str = str(sdk_dir)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_weights() -> None:
    """Download SDK bundles from the HF model repo if weights are missing."""
    global _weights_ready
    if _weights_ready:
        return
    missing = [p for p in _WEIGHT_SENTINELS if not p.exists()]
    if not missing:
        _weights_ready = True
        return

    logger.info("SDK weights missing (%d); downloading from %s ...", len(missing), HF_MODEL_REPO)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_MODEL_REPO,
        local_dir=str(SDK_ROOT),
        allow_patterns=[f"{d.relative_to(SDK_ROOT)}/**" for d in SDK_DIRS.values()],
    )
    still_missing = [p for p in _WEIGHT_SENTINELS if not p.exists()]
    if still_missing:
        raise FileNotFoundError(f"Weights still missing after HF download: {still_missing}")
    _weights_ready = True
    logger.info("SDK weights ready.")


def get_calibration_pipeline():
    with _lock:
        if "calibration" not in _pipelines:
            _ensure_import_paths()
            ensure_weights()
            from calibration import CalibrationPipeline

            _pipelines["calibration"] = CalibrationPipeline()
            logger.info("Calibration pipeline initialized.")
    return _pipelines["calibration"]


def get_equipment_pipeline():
    with _lock:
        if "equipment" not in _pipelines:
            _ensure_import_paths()
            ensure_weights()
            from equipment_annotation import EquipmentAnnotationPipeline

            _pipelines["equipment"] = EquipmentAnnotationPipeline()
            logger.info("Equipment annotation pipeline initialized.")
    return _pipelines["equipment"]


def get_wire_tracer_pipeline():
    with _lock:
        if "wire_tracer" not in _pipelines:
            _ensure_import_paths()
            ensure_weights()
            from wire_tracer import WireTracerPipeline

            _pipelines["wire_tracer"] = WireTracerPipeline()
            logger.info("Wire tracer pipeline initialized.")
    return _pipelines["wire_tracer"]


def load_all(warmup: bool = True) -> None:
    """Construct all pipelines (called at startup unless SKIP_STARTUP_MODEL_LOAD).

    warmup=True pre-loads every ONNX session so the first request is fast.
    """
    calibration = get_calibration_pipeline()
    equipment = get_equipment_pipeline()
    tracer = get_wire_tracer_pipeline()
    if warmup:
        calibration.warmup()
        equipment.warmup()
        tracer.warmup()
        logger.info("All ONNX sessions warmed up.")


def is_models_loaded() -> bool:
    return len(_pipelines) == 3


def clear_models() -> None:
    """Drop pipeline singletons (tests / memory management)."""
    global _weights_ready
    with _lock:
        _pipelines.clear()
        _weights_ready = False
