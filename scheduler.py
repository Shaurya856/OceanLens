import asyncio
from batch_processor import process_batch
from config import MAX_CONCURRENCY

semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

async def schedule(
    images,
    job_id,
    mode,
    techniques,
    params,
    batch_size,
    ws_manager
):
    async def guarded(batch):
        async with semaphore:
            await process_batch(
                batch,
                job_id,
                mode,
                techniques,
                params,
                ws_manager
            )

    tasks = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tasks.append(asyncio.create_task(guarded(batch)))

    await asyncio.gather(*tasks)
