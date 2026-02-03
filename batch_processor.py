import os
from pipelines import run_single, run_custom
from utils import generate_image_id, build_enhanced_filename
from config import RESULTS_DIR

async def process_batch(
    batch,
    job_id,
    mode,
    techniques,
    params,
    ws_manager
):
    output_dir = f"{RESULTS_DIR}/{job_id}"
    os.makedirs(output_dir, exist_ok=True)

    for image_obj in batch:
        image_id = generate_image_id()
        filename = image_obj["filename"]
        image = image_obj["content"]

        await ws_manager.send(job_id, {
            "image_id": image_id,
            "status": "processing",
            "filename": filename
        })

        try:
            if mode == "single":
                image = run_single(image, techniques[0], params)
            else:
                image = run_custom(image, techniques, params)

            out_name = build_enhanced_filename(filename)
            out_path = f"{output_dir}/{out_name}"

            with open(out_path, "wb") as f:
                f.write(image)

            await ws_manager.send(job_id, {
                "image_id": image_id,
                "status": "completed",
                "filename": out_name,
                "download_url": f"/download/{job_id}/{out_name}"
            })

        except Exception as e:
            await ws_manager.send(job_id, {
                "image_id": image_id,
                "status": "failed",
                "filename": filename,
                "error": str(e)
            })
