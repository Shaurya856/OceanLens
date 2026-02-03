import json
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from scheduler import schedule
from websocket_manager import WebSocketManager
from validators import validate_request
from config import BATCH_SIZE_DEFAULT, RESULTS_DIR

router = APIRouter()
ws_manager = WebSocketManager()

@router.post("/enhance")
async def enhance(
    images: list[UploadFile] = File(...),
    job_id: str = Form(...),
    mode: str = Form(...),
    techniques: str = Form(...),
    params: str = Form("{}"),
    batch_size: int = Form(BATCH_SIZE_DEFAULT),
):
    techniques = json.loads(techniques)
    params = json.loads(params)

    validate_request(mode, techniques, params)

    image_objs = []
    for img in images:
        image_objs.append({
            "filename": img.filename,
            "content": await img.read()
        })

    await schedule(
        image_objs,
        job_id,
        mode,
        techniques,
        params,
        batch_size,
        ws_manager
    )

    return {"job_id": job_id, "status": "started"}

@router.get("/download/{job_id}/{filename}")
def download_image(job_id: str, filename: str):
    path = f"{RESULTS_DIR}/{job_id}/{filename}"
    return FileResponse(path, filename=filename)
