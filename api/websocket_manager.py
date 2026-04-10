from typing import Any
from fastapi import WebSocket


class WebSocketManager:
    def __init__(self) -> None:
        self.connections: dict[str, WebSocket] = {}

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections[job_id] = websocket

    async def send(self, job_id: str, payload: dict[str, Any]) -> None:
        ws = self.connections.get(job_id)
        if ws:
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect(job_id)

    def disconnect(self, job_id: str) -> None:
        self.connections.pop(job_id, None)
