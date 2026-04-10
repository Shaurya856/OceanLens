"""
Inference scheduler — splits the image list into batches, runs them
concurrently up to MAX_CONCURRENCY, then sends a final "job_done" WS event.
"""
import asyncio
from typing import Any

from core.config import MAX_CONCURRENCY, INFER_BATCH_SIZE
from inference.batch_processor import process_inference_batch
from api.websocket_manager import WebSocketManager

_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


async def schedule_inference(
    images: list[dict[str, Any]],
    job_id: str,
    ws_manager: WebSocketManager,
    batch_size: int = INFER_BATCH_SIZE,
) -> None:
    async def guarded(batch: list[dict[str, Any]]) -> None:
        async with _semaphore:
            await process_inference_batch(batch, job_id, ws_manager)

    tasks = [
        asyncio.create_task(guarded(images[i:i + batch_size]))
        for i in range(0, len(images), batch_size)
    ]
    await asyncio.gather(*tasks)

    await ws_manager.send(job_id, {
        "status":      "job_done",
        "job_id":      job_id,
        "results_url": f"/infer/{job_id}/results",
    })
