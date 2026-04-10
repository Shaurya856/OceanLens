"""
Inference batch processor — runs the detection pipeline on a batch of images,
writes results incrementally to disk, and sends WebSocket progress events.

Results are written to:
    INFER_RESULTS_DIR/{job_id}/results.json
"""
import asyncio
import functools
import json
import os
from typing import Any

from core.config import INFER_RESULTS_DIR
from inference.runner import run_inference
from core.utils import generate_image_id
from api.websocket_manager import WebSocketManager


async def process_inference_batch(
    batch: list[dict[str, Any]],
    job_id: str,
    ws_manager: WebSocketManager,
) -> None:
    out_dir      = os.path.join(INFER_RESULTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.json")
    loop         = asyncio.get_running_loop()

    for image_obj in batch:
        image_id = generate_image_id()
        filename: str   = image_obj["filename"]
        content:  bytes = image_obj["content"]

        await ws_manager.send(job_id, {
            "image_id": image_id,
            "status":   "processing",
            "filename": filename,
        })

        try:
            result: dict = await loop.run_in_executor(
                None, functools.partial(run_inference, content)
            )
            result["image_id"] = image_id
            result["filename"] = filename
            _append_result(results_path, result)

            await ws_manager.send(job_id, {
                "image_id":        image_id,
                "status":          "completed",
                "filename":        filename,
                "detection_count": len(result["detections"]),
                "novel_count":     sum(1 for d in result["detections"] if d["is_novel"]),
            })

        except Exception as exc:
            await ws_manager.send(job_id, {
                "image_id": image_id,
                "status":   "failed",
                "filename": filename,
                "error":    str(exc),
            })

    await ws_manager.send(job_id, {"status": "batch_done", "job_id": job_id})


def _append_result(path: str, result: dict) -> None:
    """Read existing results list, append the new result, write back."""
    data: list = []
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
    data.append(result)
    with open(path, "w") as f:
        json.dump(data, f)
