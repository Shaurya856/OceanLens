import os

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router, ws_manager

app = FastAPI()

# CORS — set CORS_ALLOW_ORIGINS to a comma-separated list of origins in production.
# Defaults to "*" for local development.
_raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
_allow_origins = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        ws_manager.disconnect(job_id)

# Serve the frontend — must be mounted last (catch-all)
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")
