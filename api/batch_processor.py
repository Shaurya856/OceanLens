import asyncio
import functools
import logging
import os
from typing import Any

from enhancement.pipelines import run_single, run_custom
from core.utils import generate_image_id, build_enhanced_filename
from api.websocket_manager import WebSocketManager
from core.config import RESULTS_DIR

logger = logging.getLogger(__name__)


async def process_batch(
    batch: list[dict[str, Any]],
    job_id: str,
    mode: str,
    techniques: list[str],
    params: dict[str, dict],
    ws_manager: WebSocketManager,
) -> None:
    output_dir = f"{RESULTS_DIR}/{job_id}"
    os.makedirs(output_dir, exist_ok=True)

    loop = asyncio.get_running_loop()

    for image_obj in batch:
        image_id = generate_image_id()
        filename = image_obj["filename"]
        image    = image_obj["content"]

        await ws_manager.send(job_id, {
            "image_id": image_id,
            "status":   "processing",
            "filename": filename,
        })

        try:
            if mode == "single":
                image = await loop.run_in_executor(
                    None, functools.partial(run_single, image, techniques[0], params)
                )
            else:
                image = await loop.run_in_executor(
                    None, functools.partial(run_custom, image, techniques, params)
                )

            out_name = build_enhanced_filename(filename)
            out_path = f"{output_dir}/{out_name}"
            with open(out_path, "wb") as f:
                f.write(image)

            await ws_manager.send(job_id, {
                "image_id":     image_id,
                "status":       "completed",
                "filename":     out_name,
                "download_url": f"/download/{job_id}/{out_name}",
            })

        except Exception as e:
            logger.error("Enhancement failed for %s in job %s: %s", filename, job_id, e)
            await ws_manager.send(job_id, {
                "image_id": image_id,
                "status":   "failed",
                "filename": filename,
                "error":    str(e),
            })
