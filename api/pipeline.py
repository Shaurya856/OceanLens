"""
Composed backend pipelines.

Each pipeline runs multiple processing stages entirely within the backend.
The caller submits a single request and receives only the final output —
no intermediate frames or enhanced images are returned unless a separate
single-stage endpoint is explicitly called.

Pipelines
─────────
video_enhance        video → frames (in memory) → enhance → write enhanced PNGs
video_detect         video → frames (in memory) → enhance → infer → write results.json
image_enhance_detect images → enhance (in memory) → infer → write results.json

Enhancement is optional for the detect pipelines: pass an empty techniques
list to run inference on raw frames / images.
"""
import asyncio
import functools
import json
import logging
import os
from typing import Any

from core.config import MAX_CONCURRENCY, RESULTS_DIR, INFER_RESULTS_DIR
from core.utils import build_enhanced_filename
from enhancement.pipelines import run_custom
from inference.runner import run_inference
from video.processor import extract_frames_memory
from api.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

# Pipeline semaphore: limits combined pipeline (enhance + infer) concurrency.
# Separate from api/scheduler._semaphore (enhancement-only) and
# inference/scheduler._semaphore (inference-only) — each pool is independently
# bounded at MAX_CONCURRENCY matching its workload's resource pressure.
_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


async def _extract_frames(
    video_bytes: bytes,
    sample_fps: float,
    job_id: str,
    ws_manager: WebSocketManager,
) -> list[dict[str, Any]] | None:
    """Extract frames in memory; send WS error on failure and return None."""
    loop = asyncio.get_running_loop()
    await ws_manager.send(job_id, {"status": "extracting_frames", "job_id": job_id})
    try:
        frames: list[dict[str, Any]] = await loop.run_in_executor(
            None, lambda: extract_frames_memory(video_bytes, sample_fps)
        )
    except ValueError as exc:
        await ws_manager.send(job_id, {"status": "failed", "error": str(exc), "job_id": job_id})
        return None
    if not frames:
        await ws_manager.send(job_id, {
            "status": "failed", "error": "No frames could be extracted", "job_id": job_id,
        })
        return None
    return frames


# ── Video → enhanced frames ───────────────────────────────────────────────────

async def run_video_enhance_pipeline(
    video_bytes: bytes,
    job_id: str,
    sample_fps: float,
    techniques: list[str],
    params: dict[str, dict],
    ws_manager: WebSocketManager,
) -> None:
    """Extract frames in memory, enhance each frame, write enhanced PNGs to disk."""
    loop   = asyncio.get_running_loop()
    frames = await _extract_frames(video_bytes, sample_fps, job_id, ws_manager)
    if frames is None:
        return

    total   = len(frames)
    out_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    await ws_manager.send(job_id, {
        "status": "enhancing", "total_frames": total, "job_id": job_id
    })

    completed: list[dict] = []

    async def _process(frame: dict[str, Any], idx: int) -> None:
        async with _semaphore:
            filename: str   = frame["filename"]
            content:  bytes = frame["content"]
            try:
                if techniques:
                    content = await loop.run_in_executor(
                        None, functools.partial(run_custom, content, techniques, params)
                    )
                out_name = build_enhanced_filename(filename) if techniques else filename
                out_path = os.path.join(out_dir, out_name)
                with open(out_path, "wb") as f:
                    f.write(content)
                entry = {
                    "filename":     out_name,
                    "frame_index":  idx,
                    "download_url": f"/download/{job_id}/{out_name}",
                }
                completed.append(entry)
                await ws_manager.send(job_id, {
                    "status":       "frame_done",
                    "frame_index":  idx,
                    "total_frames": total,
                    **entry,
                })
            except Exception as exc:
                await ws_manager.send(job_id, {
                    "status":      "frame_failed",
                    "frame_index": idx,
                    "filename":    filename,
                    "error":       str(exc),
                })

    await asyncio.gather(*[_process(f, i) for i, f in enumerate(frames)])

    await ws_manager.send(job_id, {
        "status":    "job_done",
        "job_id":    job_id,
        "total":     total,
        "completed": len(completed),
        "files":     completed,
    })


# ── Video → detection results ─────────────────────────────────────────────────

async def run_video_detect_pipeline(
    video_bytes: bytes,
    job_id: str,
    sample_fps: float,
    techniques: list[str],
    params: dict[str, dict],
    ws_manager: WebSocketManager,
) -> None:
    """Extract frames in memory, optionally enhance, run inference.
    Writes results.json to INFER_RESULTS_DIR/{job_id}/.
    """
    loop   = asyncio.get_running_loop()
    frames = await _extract_frames(video_bytes, sample_fps, job_id, ws_manager)
    if frames is None:
        return

    total    = len(frames)
    out_dir  = os.path.join(INFER_RESULTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    await ws_manager.send(job_id, {
        "status": "processing", "total_frames": total, "job_id": job_id
    })

    all_results: list[dict] = []

    async def _process(frame: dict[str, Any], idx: int) -> None:
        async with _semaphore:
            filename: str   = frame["filename"]
            content:  bytes = frame["content"]
            try:
                if techniques:
                    content = await loop.run_in_executor(
                        None, functools.partial(run_custom, content, techniques, params)
                    )
                result: dict = await loop.run_in_executor(
                    None, functools.partial(run_inference, content)
                )
                result["frame_index"] = idx
                result["filename"]    = filename
                all_results.append(result)
                await ws_manager.send(job_id, {
                    "status":          "frame_done",
                    "frame_index":     idx,
                    "total_frames":    total,
                    "filename":        filename,
                    "detection_count": len(result["detections"]),
                    "novel_count":     sum(1 for d in result["detections"] if d["is_novel"]),
                })
            except Exception as exc:
                await ws_manager.send(job_id, {
                    "status":      "frame_failed",
                    "frame_index": idx,
                    "filename":    filename,
                    "error":       str(exc),
                })

    await asyncio.gather(*[_process(f, i) for i, f in enumerate(frames)])

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f)

    await ws_manager.send(job_id, {
        "status":      "job_done",
        "job_id":      job_id,
        "total":       total,
        "completed":   len(all_results),
        "results_url": f"/infer/{job_id}/results",
    })


# ── Images → enhance → detect ─────────────────────────────────────────────────

async def run_image_enhance_detect_pipeline(
    images: list[dict[str, Any]],
    job_id: str,
    techniques: list[str],
    params: dict[str, dict],
    ws_manager: WebSocketManager,
) -> None:
    """Enhance images in memory then run inference.
    Writes results.json to INFER_RESULTS_DIR/{job_id}/.
    """
    loop  = asyncio.get_running_loop()
    total = len(images)

    out_dir = os.path.join(INFER_RESULTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    await ws_manager.send(job_id, {
        "status": "processing", "total_images": total, "job_id": job_id
    })

    all_results: list[dict] = []

    async def _process(image_obj: dict[str, Any], idx: int) -> None:
        async with _semaphore:
            filename: str   = image_obj["filename"]
            content:  bytes = image_obj["content"]
            try:
                if techniques:
                    content = await loop.run_in_executor(
                        None, functools.partial(run_custom, content, techniques, params)
                    )
                result: dict = await loop.run_in_executor(
                    None, functools.partial(run_inference, content)
                )
                result["image_index"] = idx
                result["filename"]    = filename
                all_results.append(result)
                await ws_manager.send(job_id, {
                    "status":          "image_done",
                    "image_index":     idx,
                    "total_images":    total,
                    "filename":        filename,
                    "detection_count": len(result["detections"]),
                    "novel_count":     sum(1 for d in result["detections"] if d["is_novel"]),
                })
            except Exception as exc:
                await ws_manager.send(job_id, {
                    "status":      "image_failed",
                    "image_index": idx,
                    "filename":    filename,
                    "error":       str(exc),
                })

    await asyncio.gather(*[_process(img, i) for i, img in enumerate(images)])

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f)

    await ws_manager.send(job_id, {
        "status":      "job_done",
        "job_id":      job_id,
        "total":       total,
        "completed":   len(all_results),
        "results_url": f"/infer/{job_id}/results",
    })
