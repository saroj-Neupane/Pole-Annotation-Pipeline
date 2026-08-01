"""Inference endpoints, backed by the ONNX SDK pipelines."""
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from deploy.demo_format import calibration_display, equipment_display, jsonable
from deploy.model_loader import (
    get_calibration_pipeline,
    get_equipment_pipeline,
    get_wire_tracer_pipeline,
)
from deploy.utils import image_to_base64_data_url

router = APIRouter(tags=["inference"])

MAX_FILE_SIZE = 10 * 1024 * 1024

# Same-input inference results are deterministic: cache demo responses by photo
# hash so re-opening a sample span is instant (bounded FIFO, in-memory).
_RESULT_CACHE_MAX = 256
_result_cache: "dict[str, dict]" = {}


def _cache_get(key: str):
    return _result_cache.get(key)


def _cache_put(key: str, value: dict) -> None:
    if len(_result_cache) >= _RESULT_CACHE_MAX:
        _result_cache.pop(next(iter(_result_cache)))
    _result_cache[key] = value


def _read_image_bgr(image: UploadFile) -> "tuple[np.ndarray, str]":
    """Decode an uploaded image; returns (bgr_array, sha256-of-bytes)."""
    import hashlib

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = image.file.read(MAX_FILE_SIZE)
    if len(contents) >= MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image file too large. Maximum size is 10MB.")
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image format or corrupted image")
    return img_bgr, hashlib.sha256(contents).hexdigest()


@router.post("/demo/predict")
def demo_predict(
    image: UploadFile = File(...),
    image_type: str = Query("pole", description="Image type: 'pole' or 'midspan'"),
    pipeline: str = Query("calibration", description="Pipeline: 'calibration' or 'annotation'"),
    include_images: bool = Query(True, description="Include original and annotated images in response"),
):
    if image_type not in {"pole", "midspan"}:
        raise HTTPException(status_code=400, detail="image_type must be 'pole' or 'midspan'")
    if pipeline not in {"calibration", "annotation", "attachments"}:
        raise HTTPException(status_code=400, detail="pipeline must be 'calibration', 'annotation' or 'attachments'")
    if pipeline == "attachments" and image_type != "pole":
        raise HTTPException(status_code=400, detail="attachments pipeline requires image_type=pole")
    img_bgr, img_sha = _read_image_bgr(image)
    cache_key = f"predict|{pipeline}|{image_type}|{img_sha}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return JSONResponse(content=cached)
    try:
        img_height, img_width = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_data_url = image_to_base64_data_url(img_bgr)

        if pipeline == "annotation" and image_type == "midspan":
            # Midspan annotation = strip wire crossings + calibration heights.
            from deploy.span_trace import midspan_wire_display

            wires = midspan_wire_display(
                get_wire_tracer_pipeline(), get_calibration_pipeline(), img_rgb
            )
            response = {
                "original_image": original_data_url,
                "annotated_image": original_data_url,
                "results": {"Wires": wires or None},
                "image_width": img_width,
                "image_height": img_height,
            }
            _cache_put(cache_key, response)
            return JSONResponse(content=response)

        if pipeline == "attachments":
            # Single-pole attachment inventory from the wire tracer's pole stage
            # (unified joint-class model on the upper-70% crop). No span needed.
            from wire_tracer.constants import INSULATOR_DISPLAY
            from wire_tracer.pipeline import _select_down_guys

            pts = get_wire_tracer_pipeline()._detect_pole_points(img_rgb)
            # run() applies the down-guy dedup+gate after detection; mirror it here
            # (detection runs at the 0.05 floor and would otherwise flood the view)
            pts = _select_down_guys(pts, None)
            atts = []
            for p in pts:
                token = p.get("hw_token")
                is_guy = p.get("kind") == "guying" or token in ("guy", "down_guy")
                name = ("Guy" if token == "guy" else "Down Guy") if is_guy \
                    else INSULATOR_DISPLAY.get(token, INSULATOR_DISPLAY[None])
                atts.append({
                    "insulator_name": name,
                    "hardware": token,
                    "cable_type_hint": p.get("wire_class"),
                    "cable_type_fine": p.get("cable_fine"),
                    "role": "guying" if is_guy else "single",
                    "crossarm_k": int(p.get("pred_mult", 1) or 1),
                    "x": round(float(p["x"]), 2),
                    "y": round(float(p["y"]), 2),
                    "conf": round(float(p.get("conf", 0.0)), 3),
                })
            response = {
                "original_image": original_data_url,
                "annotated_image": original_data_url,
                "results": {"Attachments": atts or None},
                "image_width": img_width,
                "image_height": img_height,
            }
            _cache_put(cache_key, response)
            return JSONResponse(content=response)

        if pipeline == "calibration":
            result = get_calibration_pipeline().run(
                img_rgb, detect_pole=(image_type == "pole")
            )
            response = {
                "original_image": original_data_url,
                # the demo page draws the calibration overlay client-side
                "annotated_image": original_data_url,
                "results": calibration_display(result, img_width, img_height),
                "image_width": img_width,
                "image_height": img_height,
            }
        else:
            result = get_equipment_pipeline().run(img_rgb, return_annotated=True)
            annotated_rgb = result.pop("annotated_image", None)
            response = {
                "original_image": original_data_url,
                "annotated_image": original_data_url,
                "results": equipment_display(result, img_width, img_height),
                "image_width": img_width,
                "image_height": img_height,
            }
            if annotated_rgb is not None:
                response["server_annotated_image"] = image_to_base64_data_url(
                    cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                )

        _cache_put(cache_key, response)
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Models not available: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc).strip() or "Internal server error.") from exc


@router.post("/predict")
def predict(
    image: UploadFile = File(...),
    image_type: str = Query("pole", description="Image type: 'pole' or 'midspan'"),
    include_images: bool = Query(False, description="Include original and annotated images in response"),
):
    """Raw calibration API: pole bbox + ruler + tick keypoints + pole top."""
    if image_type not in {"pole", "midspan"}:
        raise HTTPException(status_code=400, detail="image_type must be 'pole' or 'midspan'")

    img_bgr, _ = _read_image_bgr(image)
    try:
        img_height, img_width = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = get_calibration_pipeline().run(
            img_rgb,
            detect_pole=(image_type == "pole"),
            return_annotated=include_images,
        )
        annotated_rgb = result.pop("annotated_image", None)

        response = {
            "results": jsonable(result),
            "image_width": img_width,
            "image_height": img_height,
        }
        if include_images:
            response["original_image"] = image_to_base64_data_url(img_bgr)
            if annotated_rgb is not None:
                response["annotated_image"] = image_to_base64_data_url(
                    cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                )
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@router.post("/demo/trace")
def demo_trace(
    pole_a: UploadFile = File(..., description="Pole A photo"),
    pole_b: UploadFile = File(..., description="Pole B photo"),
    midspans: List[UploadFile] = File(..., description="Midspan photo(s), left-to-right"),
):
    """Span trace for the demo UI: pole-mid[-mid]-pole per-photo annotations.

    Every photo gets its own annotation set (attachments / wire crossings +
    height labels from the calibration ruler ticks); traces link the columns.
    """
    from deploy.span_trace import build_span_payload

    a_bgr, a_sha = _read_image_bgr(pole_a)
    b_bgr, b_sha = _read_image_bgr(pole_b)
    mids = [_read_image_bgr(m) for m in midspans]
    mid_bgrs = [m[0] for m in mids]
    if not mid_bgrs:
        raise HTTPException(status_code=400, detail="At least one midspan photo is required")
    if len(mid_bgrs) > 4:
        raise HTTPException(status_code=400, detail="At most 4 midspan photos per span")
    cache_key = "trace|" + "|".join([a_sha] + [m[1] for m in mids] + [b_sha])
    cached = _cache_get(cache_key)
    if cached is not None:
        return JSONResponse(content=cached)

    try:
        payload = build_span_payload(
            get_wire_tracer_pipeline(),
            get_calibration_pipeline(),
            cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB),
            [cv2.cvtColor(m, cv2.COLOR_BGR2RGB) for m in mid_bgrs],
            cv2.cvtColor(b_bgr, cv2.COLOR_BGR2RGB),
        )
        response = jsonable(payload)
        _cache_put(cache_key, response)
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Models not available: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc).strip() or "Internal server error.") from exc


@router.post("/api/trace")
def trace_span(
    pole_a: UploadFile = File(..., description="Pole A photo"),
    pole_b: UploadFile = File(..., description="Pole B photo"),
    midspans: List[UploadFile] = File(..., description="Midspan photo(s) for the span"),
    include_images: bool = Query(True, description="Include the annotated span grid in the response"),
):
    """Trace one span (pole A ↔ midspan ↔ pole B) with the wire tracer SDK.

    The response uses a sections envelope so multi-section spans
    (pole–mid–mid–pole) can be added without a breaking change; this endpoint
    currently accepts a single section.
    """
    a_bgr, _ = _read_image_bgr(pole_a)
    b_bgr, _ = _read_image_bgr(pole_b)
    mid_bgrs = [_read_image_bgr(m)[0] for m in midspans]
    if not mid_bgrs:
        raise HTTPException(status_code=400, detail="At least one midspan photo is required")

    try:
        tracer = get_wire_tracer_pipeline()
        result = tracer.run(
            cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB),
            [cv2.cvtColor(m, cv2.COLOR_BGR2RGB) for m in mid_bgrs],
            cv2.cvtColor(b_bgr, cv2.COLOR_BGR2RGB),
            return_annotated=include_images,
        )
        annotated_bgr = result.pop("annotated_image", None)

        section = {"trace": jsonable(result)}
        if include_images and annotated_bgr is not None:
            section["annotated_image"] = image_to_base64_data_url(annotated_bgr)

        return JSONResponse(content={"sections": [section]})
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Models not available: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc).strip() or "Internal server error.") from exc
