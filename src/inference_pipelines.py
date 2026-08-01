"""
Inference pipelines for the production stacks:

- calibration_pipeline: pole → ruler → ruler keypoints → pole top
  Use for pole/midspan photos with ruler (height calibration).

- annotation_pipeline: pole → 70% 2:5 crop → equipment + attachment (combined, optimized)
  Use for production; runs equipment and attachment on shared crop (load + pole once).

Usage:
    from src.inference_pipelines import calibration_pipeline, annotation_pipeline
    calibration_pipeline.run(image_path, models, ...)
    equip_preds, attach_preds, ppi = annotation_pipeline.run_single(img_path, pole_detector, ...)
"""

from pathlib import Path
from typing import Dict, Any

from .config import INFERENCE_POLE_CONF_THRESHOLD

# -----------------------------------------------------------------------------
# Calibration Pipeline
# -----------------------------------------------------------------------------


def run_calibration_pipeline(
    image_path: Path,
    models: Dict[str, Any],
    use_tta: bool = True,
) -> Dict[str, Any]:
    """
    Run calibration pipeline: pole → ruler → ruler keypoints → pole top.
    For pole/midspan photos (ruler visible).
    """
    from .inference import run_end_to_end_inference_simple
    return run_end_to_end_inference_simple(image_path, models, use_tta=use_tta)


def run_calibration_batch(
    images_dir: Path,
    output_dir: Path,
    models: Dict[str, Any],
    use_tta: bool = True,
    save_annotated: bool = True,
    save_labels: bool = True,
) -> list:
    """Run calibration pipeline on a batch of images."""
    from .inference import run_batch_inference
    return run_batch_inference(
        images_dir, output_dir, models,
        use_tta=use_tta,
        save_annotated=save_annotated,
        save_labels=save_labels
    )


# -----------------------------------------------------------------------------
# Annotation Pipeline (Equipment + Attachment, optimized)
# -----------------------------------------------------------------------------


def run_annotation_pipeline_single(
    image_path: Path,
    pole_detector,
    equip_detector,
    attach_detector,
    equip_kp_models: Dict,
    attach_kp_models: Dict,
    device,
    pole_conf: float = INFERENCE_POLE_CONF_THRESHOLD,
) -> tuple:
    """
    Run annotation pipeline on one image: equipment + attachment on shared crop.
    Returns (equip_preds, attach_preds, ppi).
    """
    from .evaluation_attachment_equipment import run_e2e_annotation_single_image
    return run_e2e_annotation_single_image(
        image_path, pole_detector, equip_detector, attach_detector,
        equip_kp_models, attach_kp_models, device, pole_conf=pole_conf
    )


# Convenience namespace objects for import
calibration_pipeline = type('CalibrationPipeline', (), {'run': staticmethod(run_calibration_pipeline), 'run_batch': staticmethod(run_calibration_batch)})()
annotation_pipeline = type('AnnotationPipeline', (), {'run_single': staticmethod(run_annotation_pipeline_single)})()
