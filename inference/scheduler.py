"""
Inference scheduler — splits the image list into batches, runs them
concurrently up to MAX_CONCURRENCY, collects all results, then writes
results.json once and sends a final "job_done" WS event.

Writing once at the end (rather than per-image in the batch processor)
eliminates the read-modify-write race that would occur when multiple
concurrent batches append to the same file simultaneously.
"""
import asyncio
import json
import logging
import os
from typing import Any

from core.config import MAX_CONCURRENCY, INFER_BATCH_SIZE, INFER_RESULTS_DIR
from inference.batch_processor import process_inference_batch
from api.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

# Inference semaphore: limits concurrent GPU/CPU workers independently of the
# enhancement semaphore in api/scheduler.py and the pipeline semaphore in
# api/pipeline.py. Keeping them separate lets enhancement (CPU-bound) and
# inference (GPU-bound) scale their own concurrency without sharing a pool.
_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


async def schedule_inference(
    images: list[dict[str, Any]],
    job_id: str,
    ws_manager: WebSocketManager,
    batch_size: int = INFER_BATCH_SIZE,
) -> None:
    out_dir = os.path.join(INFER_RESULTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    async def guarded(batch: list[dict[str, Any]]) -> list[dict]:
        async with _semaphore:
            return await process_inference_batch(batch, job_id, ws_manager)

    batch_results = await asyncio.gather(*[
        guarded(images[i:i + batch_size])
        for i in range(0, len(images), batch_size)
    ])

    all_results = [result for batch in batch_results for result in batch]

    # Write once — no concurrent batches can race here since gather is done.
    results_path = os.path.join(out_dir, "results.json")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _write_results, results_path, all_results)

    logger.info("Job %s done: %d results written to %s", job_id, len(all_results), results_path)

    await ws_manager.send(job_id, {
        "status":      "job_done",
        "job_id":      job_id,
        "results_url": f"/infer/{job_id}/results",
    })


def _write_results(path: str, results: list) -> None:
    with open(path, "w") as f:
        json.dump(results, f)
