from fastapi import WebSocket

class WebSocketManager:
    def __init__(self):
        self.connections = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[job_id] = websocket

    async def send(self, job_id: str, payload: dict):
        ws = self.connections.get(job_id)
        if ws:
            await ws.send_json(payload)

    def disconnect(self, job_id: str):
        self.connections.pop(job_id, None)
