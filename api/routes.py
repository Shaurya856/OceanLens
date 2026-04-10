import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from api.scheduler import schedule
from api.pipeline import (
    run_video_enhance_pipeline,
    run_video_detect_pipeline,
    run_image_enhance_detect_pipeline,
)
from inference.scheduler import schedule_inference
from video.processor import extract_frames
from api.websocket_manager import WebSocketManager
from api.validators import validate_request
from core.utils import generate_image_id
from core.config import BATCH_SIZE_DEFAULT, RESULTS_DIR, INFER_RESULTS_DIR, FRAMES_DIR

router = APIRouter()
ws_manager = WebSocketManager()


# ── Enhancement pipeline ──────────────────────────────────────────────────────

@router.post("/enhance", status_code=202)
async def enhance(
    images: list[UploadFile] = File(...),
    job_id: str | None = Form(None),
    mode: str = Form("custom"),
    techniques: str = Form(...),
    params: str = Form("{}"),
    batch_size: int = Form(BATCH_SIZE_DEFAULT),
):
    if job_id is None:
        job_id = generate_image_id()

    techniques_list: list = json.loads(techniques)
    params_dict: dict     = json.loads(params)

    try:
        validate_request(mode, techniques_list, params_dict)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    image_objs: list[dict[str, bytes | str]] = [
        {"filename": img.filename or "image.png", "content": await img.read()}
        for img in images
    ]

    asyncio.create_task(schedule(
        image_objs, job_id, mode, techniques_list, params_dict, batch_size, ws_manager
    ))

    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "accepted"})


@router.get("/frames/{job_id}/{filename}")
def download_frame(job_id: str, filename: str):
    safe_job_id   = Path(job_id).name
    safe_filename = Path(filename).name
    if not safe_job_id or not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid job_id or filename")
    path = Path(FRAMES_DIR) / safe_job_id / safe_filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(str(path), filename=safe_filename)


@router.get("/download/{job_id}/{filename}")
def download_image(job_id: str, filename: str):
    safe_job_id   = Path(job_id).name
    safe_filename = Path(filename).name
    if not safe_job_id or not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid job_id or filename")
    path = Path(RESULTS_DIR) / safe_job_id / safe_filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), filename=safe_filename)


# ── Video ingestion ───────────────────────────────────────────────────────────

@router.post("/video/ingest", status_code=202)
async def ingest_video(
    video: UploadFile = File(...),
    job_id: str | None = Form(None),
    sample_fps: float = Form(2.0),
):
    if job_id is None:
        job_id = generate_image_id()
    if sample_fps <= 0:
        raise HTTPException(status_code=422, detail="sample_fps must be positive")

    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=422, detail="Uploaded video is empty")

    loop = asyncio.get_running_loop()
    try:
        frames: list[dict] = await loop.run_in_executor(
            None, lambda: extract_frames(video_bytes, job_id, sample_fps)
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not frames:
        raise HTTPException(status_code=422, detail="No frames could be extracted from video")

    return JSONResponse(
        status_code=202,
        content={
            "job_id":      job_id,
            "status":      "accepted",
            "frame_count": len(frames),
            "frames":      [{"filename": f["filename"], "path": f["path"]} for f in frames],
        },
    )


# ── Species detection / inference ─────────────────────────────────────────────

@router.post("/infer", status_code=202)
async def infer(
    images: list[UploadFile] = File(...),
    job_id: str | None = Form(None),
    batch_size: int = Form(4),
):
    if job_id is None:
        job_id = generate_image_id()

    image_objs: list[dict[str, bytes | str]] = [
        {"filename": img.filename or "image.png", "content": await img.read()}
        for img in images
    ]

    asyncio.create_task(schedule_inference(image_objs, job_id, ws_manager, batch_size))

    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "accepted"})


@router.get("/infer/{job_id}/results")
def get_inference_results(job_id: str):
    safe_job_id = Path(job_id).name
    if not safe_job_id:
        raise HTTPException(status_code=400, detail="Invalid job_id")

    results_path = Path(INFER_RESULTS_DIR) / safe_job_id / "results.json"
    if not results_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Results not found — job may still be running or job_id is invalid",
        )

    with open(results_path) as f:
        data = json.load(f)

    return JSONResponse(status_code=200, content={"job_id": job_id, "results": data})


# ── Combined pipelines (single request → final output only) ──────────────────
# These replace manual frontend chaining.  Each runs all stages in the backend
# and streams progress via WebSocket.  The original single-stage endpoints
# (/video/ingest, /enhance, /infer) are kept for when intermediate outputs
# are explicitly needed.

@router.post("/pipeline/video/enhance", status_code=202)
async def pipeline_video_enhance(
    video: UploadFile = File(...),
    job_id: str | None = Form(None),
    sample_fps: float = Form(2.0),
    techniques: str = Form("[]"),
    params: str = Form("{}"),
):
    """Video → enhanced frames.  Extraction and enhancement run entirely in
    the backend; the caller receives download URLs for the final enhanced PNGs."""
    if job_id is None:
        job_id = generate_image_id()
    if sample_fps <= 0:
        raise HTTPException(status_code=422, detail="sample_fps must be positive")

    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=422, detail="Uploaded video is empty")

    techniques_list: list = json.loads(techniques)
    params_dict: dict     = json.loads(params)

    asyncio.create_task(run_video_enhance_pipeline(
        video_bytes, job_id, sample_fps, techniques_list, params_dict, ws_manager
    ))
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "accepted"})


@router.post("/pipeline/video/detect", status_code=202)
async def pipeline_video_detect(
    video: UploadFile = File(...),
    job_id: str | None = Form(None),
    sample_fps: float = Form(2.0),
    techniques: str = Form("[]"),
    params: str = Form("{}"),
):
    """Video → species detections.  Extraction, optional enhancement, and
    inference all run in the backend.  Pass an empty techniques list to skip
    enhancement and infer on raw frames."""
    if job_id is None:
        job_id = generate_image_id()
    if sample_fps <= 0:
        raise HTTPException(status_code=422, detail="sample_fps must be positive")

    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=422, detail="Uploaded video is empty")

    techniques_list: list = json.loads(techniques)
    params_dict: dict     = json.loads(params)

    asyncio.create_task(run_video_detect_pipeline(
        video_bytes, job_id, sample_fps, techniques_list, params_dict, ws_manager
    ))
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "accepted"})


@router.post("/pipeline/image/enhance-detect", status_code=202)
async def pipeline_image_enhance_detect(
    images: list[UploadFile] = File(...),
    job_id: str | None = Form(None),
    techniques: str = Form("[]"),
    params: str = Form("{}"),
):
    """Images → enhance → species detections.  Enhancement and inference run
    in the backend; the caller receives only the final detection results.
    Pass an empty techniques list to skip enhancement."""
    if job_id is None:
        job_id = generate_image_id()

    techniques_list: list = json.loads(techniques)
    params_dict: dict     = json.loads(params)

    image_objs = [
        {"filename": img.filename or "image.png", "content": await img.read()}
        for img in images
    ]

    asyncio.create_task(run_image_enhance_detect_pipeline(
        image_objs, job_id, techniques_list, params_dict, ws_manager
    ))
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "accepted"})
