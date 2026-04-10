# api/

FastAPI routes and the async processing layer for image enhancement and inference jobs.

| File | Purpose |
|------|---------|
| `routes.py` | API endpoints: `POST /enhance`, `POST /video/ingest`, `POST /infer`, `GET /download/...`, `GET /infer/.../results` |
| `validators.py` | Request validation — checks mode, technique names, and param shapes |
| `websocket_manager.py` | Per-job WebSocket connection registry; sends real-time progress events |
| `batch_processor.py` | Runs enhancement pipelines on a batch of images; writes results to disk |
| `scheduler.py` | Splits image list into batches and runs them concurrently up to `MAX_CONCURRENCY` |

## Async job flow

```
POST /enhance
  └─ api/scheduler.schedule()          ← asyncio.create_task (non-blocking)
       └─ api/batch_processor.process_batch()   for each batch
            └─ enhancement/pipelines.run_single / run_custom
                 └─ loop.run_in_executor()       CPU work off event loop
                      └─ WebSocket events sent per image
```

## WebSocket events

Jobs communicate progress over `ws://<host>/ws/{job_id}`.

| `status` | Trigger | Extra fields |
|----------|---------|--------------|
| `processing` | image picked up | `image_id`, `filename` |
| `completed` | image done | `image_id`, `filename`, `download_url` |
| `failed` | error | `image_id`, `filename`, `error` |
| `job_done` | all batches done | `job_id` |

See [API.md](../API.md) in the project root for the full event schema and endpoint reference.

## Start the server

```bash
source .venv/bin/activate            # activate venv (see root README)
uvicorn main:app --reload --port 8000
```
