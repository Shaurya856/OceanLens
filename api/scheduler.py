import asyncio
import logging
from typing import Any

from api.batch_processor import process_batch
from api.websocket_manager import WebSocketManager
from core.config import MAX_CONCURRENCY

logger = logging.getLogger(__name__)

# Enhancement semaphore: limits concurrent CPU-bound enhancement workers.
# Kept independent from inference/scheduler._semaphore so enhancement (CPU/OpenCV)
# and inference (GPU) can each scale to MAX_CONCURRENCY without sharing a pool.
_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


async def schedule(
    images: list[dict[str, Any]],
    job_id: str,
    mode: str,
    techniques: list[str],
    params: dict[str, dict],
    batch_size: int,
    ws_manager: WebSocketManager,
) -> None:
    async def guarded(batch):
        async with _semaphore:
            await process_batch(batch, job_id, mode, techniques, params, ws_manager)

    logger.info("Job %s: scheduling %d images across %d batches", job_id, len(images),
                (len(images) + batch_size - 1) // batch_size)
    tasks = [
        asyncio.create_task(guarded(images[i:i + batch_size]))
        for i in range(0, len(images), batch_size)
    ]
    await asyncio.gather(*tasks)
    logger.info("Job %s: all enhancement batches complete", job_id)
