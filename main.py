from fastapi import FastAPI, WebSocket
from api import router, ws_manager

app = FastAPI()
app.include_router(router)

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        ws_manager.disconnect(job_id)
