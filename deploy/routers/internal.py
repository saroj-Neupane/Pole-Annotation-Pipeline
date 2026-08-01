"""Internal browser-backend routes used by the product UI."""
import random
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from deploy.demo_format import calibration_display, equipment_display
from deploy.demo_images import get_demo_image_info, list_demo_images, list_demo_images_for_api
from deploy.model_loader import get_calibration_pipeline, get_equipment_pipeline
from deploy.shared import LOGOS_DIR, MIDSPAN_IMAGES_DIR, POLE_IMAGES_DIR, SPANS_DIR
from deploy.utils import image_to_base64_data_url, serve_image_file

router = APIRouter(tags=["internal"])


def _load_span_manifest() -> list:
    import json

    manifest_path = SPANS_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return []


@router.get("/api/spans/list")
async def list_spans():
    """Sample spans for the demo: each with pole_a / midspans / pole_b photo URLs."""
    spans = []
    for entry in _load_span_manifest():
        span_id = entry.get("id")
        files = entry.get("files") or {}
        if not span_id or not files.get("pole_a") or not files.get("pole_b"):
            continue
        base = f"/api/spans/{span_id}"
        spans.append({
            "id": span_id,
            "job": entry.get("job"),
            "pole_a_scid": entry.get("pole_a_scid"),
            "pole_b_scid": entry.get("pole_b_scid"),
            "pole_a_latlon": entry.get("pole_a_latlon"),
            "pole_b_latlon": entry.get("pole_b_latlon"),
            "photos": (
                [{"role": "pole_a", "url": f"{base}/{files['pole_a']}"}]
                + [{"role": "midspan", "url": f"{base}/{m}"} for m in files.get("midspans", [])]
                + [{"role": "pole_b", "url": f"{base}/{files['pole_b']}"}]
            ),
        })
    return {"spans": spans}


_showcase_cache: dict = {}


@router.get("/api/demo/span_showcase")
def span_showcase():
    """Landing-page hero: one sample span's full trace payload, cached.

    Runs the wire tracer + calibration once for a fixed sample span, persists the
    payload beside the span photos, and serves it thereafter — the landing page
    renders the pole-mid-pole strip client-side from this.
    """
    import json
    import os

    entries = _load_span_manifest()
    if not entries:
        raise HTTPException(status_code=404, detail="No sample spans available")
    want = os.getenv("SHOWCASE_SPAN_ID", "COAR-FR01_131-130")
    entry = next((e for e in entries if e.get("id") == want), entries[0])
    span_id = entry["id"]

    if span_id in _showcase_cache:
        return JSONResponse(content=_showcase_cache[span_id])

    cache_file = SPANS_DIR / span_id / ".showcase.json"
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text())
            _showcase_cache[span_id] = payload
            return JSONResponse(content=payload)
        except (OSError, ValueError):
            pass

    from deploy.demo_format import jsonable
    from deploy.model_loader import get_calibration_pipeline, get_wire_tracer_pipeline
    from deploy.span_trace import build_span_payload

    files = entry["files"]
    span_dir = SPANS_DIR / span_id

    def _read(name):
        img = cv2.imread(str(span_dir / name))
        if img is None:
            raise HTTPException(status_code=500, detail=f"Showcase photo unreadable: {name}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        trace = build_span_payload(
            get_wire_tracer_pipeline(),
            get_calibration_pipeline(),
            _read(files["pole_a"]),
            [_read(m) for m in files.get("midspans", [])],
            _read(files["pole_b"]),
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Models not available: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc).strip() or "Showcase failed.") from exc

    base = f"/api/spans/{span_id}"
    urls = ([f"{base}/{files['pole_a']}"]
            + [f"{base}/{m}" for m in files.get("midspans", [])]
            + [f"{base}/{files['pole_b']}"])
    payload = jsonable({
        "span_id": span_id,
        "job": entry.get("job"),
        "pole_a_scid": entry.get("pole_a_scid"),
        "pole_b_scid": entry.get("pole_b_scid"),
        "photo_urls": urls,
        "trace": trace,
    })
    try:
        cache_file.write_text(json.dumps(payload))
    except OSError:
        pass
    _showcase_cache[span_id] = payload
    return JSONResponse(content=payload)


@router.get("/api/spans/{span_id}/gt")
async def get_span_gt(span_id: str):
    """Ground-truth overlay for a sample span (local label store only).

    Returns {"available": false} on deployments without the data/ label store
    (e.g. the public Space) — the demo hides the GT layer in that case.
    """
    from deploy.gt_overlay import load_span_gt

    safe_id = Path(span_id).name
    entry = next((e for e in _load_span_manifest() if e.get("id") == safe_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Span not found")
    return JSONResponse(content=load_span_gt(entry, SPANS_DIR / safe_id))


@router.get("/api/spans/{span_id}/{filename}")
async def get_span_image(span_id: str, filename: str):
    known = {e.get("id") for e in _load_span_manifest()}
    safe_id = Path(span_id).name
    if safe_id not in known:
        raise HTTPException(status_code=404, detail="Span not found")
    return serve_image_file(SPANS_DIR / safe_id, filename)


@router.get("/api/images/list")
async def list_images():
    return {
        "midspan": list_demo_images_for_api("midspan"),
        "pole": list_demo_images_for_api("pole"),
    }


@router.get("/api/images/midspan/{filename}")
async def get_midspan_image(filename: str):
    return serve_image_file(MIDSPAN_IMAGES_DIR, filename)


@router.get("/api/images/pole/{filename}")
async def get_pole_image(filename: str):
    return serve_image_file(POLE_IMAGES_DIR, filename)


@router.get("/api/logos/{filename}")
async def get_logo(filename: str):
    return serve_image_file(LOGOS_DIR, filename)


@router.get("/api/demo/info")
async def get_demo_info():
    return JSONResponse(content=get_demo_image_info())


@router.get("/api/demo/random")
def get_random_demo(
    pipeline: str = Query(
        "calibration",
        description="Landing demo pipeline: 'calibration' or 'annotation'.",
    ),
):
    if pipeline not in {"calibration", "annotation"}:
        raise HTTPException(status_code=400, detail="pipeline must be 'calibration' or 'annotation'")

    pole_images = list_demo_images("pole")
    if not pole_images:
        info = get_demo_image_info()
        raise HTTPException(
            status_code=404,
            detail=f"No pole demo images available. {info['total']} demo image(s) of other types found.",
        )

    random_image_path = random.choice(pole_images)
    try:
        img_bgr = cv2.imread(str(random_image_path))
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to read image")

        img_height, img_width = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if pipeline == "calibration":
            result = get_calibration_pipeline().run(img_rgb, return_annotated=True)
            display = calibration_display(result, img_width, img_height)
        else:
            result = get_equipment_pipeline().run(img_rgb, return_annotated=True)
            display = equipment_display(result, img_width, img_height)

        annotated_rgb = result.get("annotated_image")
        annotated_bgr = (
            cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR) if annotated_rgb is not None else img_bgr
        )
        return JSONResponse(
            content={
                "original_image": image_to_base64_data_url(img_bgr),
                "annotated_image": image_to_base64_data_url(annotated_bgr),
                "results": display,
                "image_width": img_width,
                "image_height": img_height,
            }
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Models not available: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
