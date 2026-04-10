import asyncio
from typing import Any

from api.batch_processor import process_batch
from api.websocket_manager import WebSocketManager
from core.config import MAX_CONCURRENCY

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

    tasks = [
        asyncio.create_task(guarded(images[i:i + batch_size]))
        for i in range(0, len(images), batch_size)
    ]
    await asyncio.gather(*tasks)
