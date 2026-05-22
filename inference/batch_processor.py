"""
Inference batch processor — runs the detection pipeline on a batch of images,
sends WebSocket progress events, and returns results in-memory.

Results are returned to the caller (inference/scheduler.py) rather than written
to disk here, so that concurrent batches never race on the same results.json file.
"""
import asyncio
import functools
import logging
from typing import Any

from inference.runner import run_inference
from core.utils import generate_image_id
from api.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


async def process_inference_batch(
    batch: list[dict[str, Any]],
    job_id: str,
    ws_manager: WebSocketManager,
) -> list[dict]:
    """Process a batch of images concurrently and return their results.

    Images within the batch are dispatched to the thread-pool in parallel.
    Results are returned — the caller writes them to disk once all batches
    are done, avoiding any file-level race between concurrent batches.
    """
    loop = asyncio.get_running_loop()

    async def _process_one(image_obj: dict[str, Any]) -> dict | None:
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

            await ws_manager.send(job_id, {
                "image_id":        image_id,
                "status":          "completed",
                "filename":        filename,
                "detection_count": len(result["detections"]),
                "novel_count":     sum(1 for d in result["detections"] if d["is_novel"]),
            })
            return result

        except Exception as exc:
            logger.error("Inference failed for %s in job %s: %s", filename, job_id, exc)
            await ws_manager.send(job_id, {
                "image_id": image_id,
                "status":   "failed",
                "filename": filename,
                "error":    str(exc),
            })
            return None

    image_results = await asyncio.gather(*[_process_one(img) for img in batch])
    return [r for r in image_results if r is not None]
