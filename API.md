# Image API — Reference

Real-time progress for every job is streamed over WebSocket.  
Connect **before** submitting a job, then listen for events.

---

## WebSocket

### `GET /ws/{job_id}`

Open a WebSocket connection to receive live status events for a job.

**Connect before** calling any job-creating endpoint with the same `job_id`.

#### Event payloads — single-stage endpoints

| `status` field | Sent by | Extra fields |
|---|---|---|
| `processing` | enhance & infer | `image_id`, `filename` |
| `completed` | enhance | `image_id`, `filename`, `download_url` |
| `completed` | infer | `image_id`, `filename`, `detection_count`, `novel_count` |
| `failed` | enhance & infer | `image_id`, `filename`, `error` |
| `batch_done` | infer | `job_id` |
| `job_done` | infer (final) | `job_id`, `results_url` |

#### Event payloads — combined pipeline endpoints

| `status` field | Sent by | Extra fields |
|---|---|---|
| `extracting_frames` | video pipelines | `job_id` |
| `enhancing` | `/pipeline/video/enhance` | `job_id`, `total_frames` |
| `processing` | detect pipelines | `job_id`, `total_frames` or `total_images` |
| `frame_done` | video pipelines | `frame_index`, `total_frames`, `filename`, `download_url` (enhance) or `detection_count`, `novel_count` (detect) |
| `frame_failed` | video pipelines | `frame_index`, `filename`, `error` |
| `image_done` | image enhance-detect | `image_index`, `total_images`, `filename`, `detection_count`, `novel_count` |
| `image_failed` | image enhance-detect | `image_index`, `filename`, `error` |
| `job_done` | all pipelines | `job_id`, `total`, `completed`, `files` (enhance) or `results_url` (detect) |
| `failed` | all pipelines | `job_id`, `error` |

```json
{ "image_id": "uuid", "status": "completed", "filename": "frame_000001_enhanced.png",
  "download_url": "/download/job-123/frame_000001_enhanced.png" }
```

```json
{ "status": "frame_done", "frame_index": 5, "total_frames": 120,
  "filename": "frame_000005.png", "detection_count": 3, "novel_count": 1 }
```

```json
{ "status": "job_done", "job_id": "pipeline-abc", "total": 120,
  "completed": 120, "results_url": "/infer/pipeline-abc/results" }
```

---

## Enhancement Pipeline

### `POST /enhance`

Submit images for preprocessing. Runs asynchronously; progress via WebSocket.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `images` | file[] | yes | One or more image files |
| `job_id` | string | no | Client-chosen identifier; auto-generated UUID if omitted |
| `mode` | string | no | `"single"` or `"custom"` (default `"custom"`) |
| `techniques` | JSON string | yes | Array of technique names (see below) |
| `params` | JSON string | no | Per-technique param overrides (default `{}`) |
| `batch_size` | integer | no | Images per concurrent batch (default `8`) |

**`mode` rules**
- `single` — exactly **one** technique is applied.
- `custom` — techniques applied **in order**, each stage's output feeds the next.

**Available techniques**

| Name | Key params |
|---|---|
| `denoise` | `h` (filter strength, default 10) |
| `clahe` | `clip_limit` (default 2.0), `tile_grid_size` (default [8,8]) |
| `gamma_correction` | `gamma` (default 1.2) |
| `white_balance` | *(none required)* |
| `dehaze` | *(none required)* |
| `retinex` | `sigmas` (default [15,80,250]) |
| `superres` | `scale` (default 2) |

**Request example**
```
POST /enhance
Content-Type: multipart/form-data

images=@photo.jpg
job_id=job-abc
mode=custom
techniques=["denoise","clahe","gamma_correction"]
params={"denoise":{"h":15},"gamma_correction":{"gamma":1.4}}
```

**Response — 202 Accepted**
```json
{ "job_id": "job-abc", "status": "accepted" }
```

**Error responses**

| Code | Reason |
|---|---|
| `422` | Invalid mode, unknown technique, `single` mode with ≠ 1 technique, malformed params |

---

### `GET /frames/{job_id}/{filename}`

Download a raw extracted frame produced by `POST /video/ingest`.  
Filenames are returned in the `/video/ingest` response as `frames[].filename`.

| Path param | Description |
|---|---|
| `job_id` | Job identifier used during `/video/ingest` |
| `filename` | Filename from the `frames[].filename` field (e.g. `frame_000000.png`) |

**Response — 200 OK**  
Raw image bytes (`image/png`).

**Error responses**

| Code | Reason |
|---|---|
| `400` | Path traversal detected |
| `404` | Frame not found (wrong `job_id` or `filename`) |

---

### `GET /download/{job_id}/{filename}`

Download an enhanced image produced by `/enhance`.  
Filenames are returned in WebSocket `completed` events as `download_url`.

| Path param | Description |
|---|---|
| `job_id` | Job identifier used during `/enhance` |
| `filename` | Filename from the `download_url` field (e.g. `photo_enhanced.png`) |

**Response — 200 OK**  
Raw image bytes (`image/png`).

**Error responses**

| Code | Reason |
|---|---|
| `400` | Path traversal detected in `job_id` or `filename` |
| `404` | File not found (job not done yet, or wrong `job_id`/filename) |

---

## Video Ingestion

### `POST /video/ingest`

Extract frames from a video at a configurable sample rate.  
Frames are saved to disk and paths are returned immediately — no WebSocket for this step.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `video` | file | yes | Video file (mp4, avi, mov, mkv, …) |
| `job_id` | string | no | Auto-generated UUID if omitted |
| `sample_fps` | float | no | Frames to extract per second of video (default `2.0`) |

**Request example**
```
POST /video/ingest
Content-Type: multipart/form-data

video=@dive_footage.mp4
job_id=dive-001
sample_fps=1.0
```

**Response — 202 Accepted**
```json
{
  "job_id": "dive-001",
  "status": "accepted",
  "frame_count": 94,
  "frames": [
    { "filename": "frame_000000.png", "path": "/tmp/frames/dive-001/frame_000000.png" },
    { "filename": "frame_000001.png", "path": "/tmp/frames/dive-001/frame_000001.png" }
  ]
}
```

**Error responses**

| Code | Reason |
|---|---|
| `422` | `sample_fps` ≤ 0, empty file, or video could not be decoded |

**When to use:** only when you need the raw extracted frames as an intermediate output (e.g. to review frames before enhancement). For end-to-end workflows use the combined pipeline endpoints instead.

---

## Species Detection

### `POST /infer`

Run the seabed species detector on enhanced images. Runs asynchronously; progress via WebSocket.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `images` | file[] | yes | Enhanced image files (PNG recommended) |
| `job_id` | string | no | Identifier for WS events and result retrieval; auto-generated if omitted |
| `batch_size` | integer | no | Images per concurrent batch (default `4`) |

**Request example**
```
POST /infer
Content-Type: multipart/form-data

images=@frame_000000_enhanced.png
images=@frame_000001_enhanced.png
job_id=dive-001-infer
```

**Response — 202 Accepted**
```json
{ "job_id": "dive-001-infer", "status": "accepted" }
```

**Error responses**

| Code | Reason |
|---|---|
| `422` | No images provided |

---

### `GET /infer/{job_id}/results`

Retrieve the full detection results for a completed inference job.

**Response — 200 OK**
```json
{
  "job_id": "dive-001-infer",
  "results": [
    {
      "frame_id": "uuid",
      "image_id": "uuid",
      "filename": "frame_000000_enhanced.png",
      "detections": [
        {
          "detection_id": "uuid",
          "bbox": { "x1": 142.3, "y1": 88.7, "x2": 310.1, "y2": 265.4 },
          "confidence": 0.912,
          "taxonomy": {
            "phylum": "Chordata",
            "class_": "Actinopterygii",
            "order": "Syngnathiformes",
            "family": "Syngnathidae",
            "species": "Hippocampus kuda"
          },
          "is_novel": false,
          "novelty_score": 0.088
        },
        {
          "detection_id": "uuid",
          "bbox": { "x1": 520.0, "y1": 400.2, "x2": 640.0, "y2": 512.0 },
          "confidence": 0.341,
          "taxonomy": {
            "phylum": "Echinodermata",
            "class_": "Asteroidea",
            "order": "Forcipulatida",
            "family": "Asteriidae",
            "species": "novel_species"
          },
          "is_novel": true,
          "novelty_score": 0.659,
          "closest_known_species": "Asterias rubens"
        }
      ]
    }
  ]
}
```

**Novel species fields**  
When `is_novel` is `true`:
- `taxonomy.species` is set to `"novel_species"`
- `closest_known_species` holds the model's best guess for reference
- `novelty_score` → 0 = certain known species, 1 = completely unknown

**Error responses**

| Code | Reason |
|---|---|
| `400` | Invalid `job_id` |
| `404` | Results file not found — job still running or `job_id` wrong |

---

## Combined Pipelines

These endpoints run all stages entirely in the backend.  
The frontend makes **one request** and receives only the final output.

---

### `POST /pipeline/video/enhance`

Video → enhanced frames. Extracts frames in memory then applies the enhancement chain; no raw frames are written to disk.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `video` | file | yes | Video file (mp4, avi, mov, mkv, …) |
| `job_id` | string | no | Auto-generated UUID if omitted |
| `sample_fps` | float | no | Sample rate in frames per second (default `2.0`) |
| `techniques` | JSON string | no | Array of technique names in application order (default `[]`) |
| `params` | JSON string | no | Per-technique param overrides (default `{}`) |

Pass `techniques=[]` to extract frames without any enhancement.

**Request example**
```
POST /pipeline/video/enhance
Content-Type: multipart/form-data

video=@dive_footage.mp4
job_id=pipe-enh-001
sample_fps=1.0
techniques=["denoise","clahe","white_balance"]
params={"denoise":{"h":15}}
```

**Response — 202 Accepted**
```json
{ "job_id": "pipe-enh-001", "status": "accepted" }
```

**WebSocket `job_done` payload**
```json
{
  "status": "job_done",
  "job_id": "pipe-enh-001",
  "total": 94,
  "completed": 94,
  "files": [
    { "filename": "frame_000000_enhanced.png", "frame_index": 0,
      "download_url": "/download/pipe-enh-001/frame_000000_enhanced.png" }
  ]
}
```

**Error responses**

| Code | Reason |
|---|---|
| `422` | `sample_fps` ≤ 0, empty file, or malformed `techniques`/`params` JSON |

---

### `POST /pipeline/video/detect`

Video → species detections. Extracts frames in memory, optionally enhances them, then runs inference. No frames or enhanced images are written to disk — only `results.json`.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `video` | file | yes | Video file |
| `job_id` | string | no | Auto-generated UUID if omitted |
| `sample_fps` | float | no | Sample rate (default `2.0`) |
| `techniques` | JSON string | no | Enhancement chain applied before inference (default `[]` — skip enhancement) |
| `params` | JSON string | no | Per-technique param overrides (default `{}`) |

**Request example**
```
POST /pipeline/video/detect
Content-Type: multipart/form-data

video=@dive_footage.mp4
job_id=pipe-det-001
sample_fps=2.0
techniques=["denoise","clahe","white_balance"]
```

**Response — 202 Accepted**
```json
{ "job_id": "pipe-det-001", "status": "accepted" }
```

**WebSocket `job_done` payload**
```json
{
  "status": "job_done",
  "job_id": "pipe-det-001",
  "total": 188,
  "completed": 188,
  "results_url": "/infer/pipe-det-001/results"
}
```

Retrieve results with `GET /infer/{job_id}/results` once `job_done` is received.

**Error responses**

| Code | Reason |
|---|---|
| `422` | `sample_fps` ≤ 0, empty file, or malformed JSON fields |

---

### `POST /pipeline/image/enhance-detect`

Images → enhance → species detections. Enhancement and inference run in the backend in a single pass; only the detection results are returned.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `images` | file[] | yes | One or more image files |
| `job_id` | string | no | Auto-generated UUID if omitted |
| `techniques` | JSON string | no | Enhancement chain applied before inference (default `[]` — skip enhancement) |
| `params` | JSON string | no | Per-technique param overrides (default `{}`) |

**Request example**
```
POST /pipeline/image/enhance-detect
Content-Type: multipart/form-data

images=@frame_000000.png
images=@frame_000001.png
job_id=pipe-imgdet-001
techniques=["denoise","clahe"]
```

**Response — 202 Accepted**
```json
{ "job_id": "pipe-imgdet-001", "status": "accepted" }
```

**WebSocket `job_done` payload**
```json
{
  "status": "job_done",
  "job_id": "pipe-imgdet-001",
  "total": 2,
  "completed": 2,
  "results_url": "/infer/pipe-imgdet-001/results"
}
```

**Error responses**

| Code | Reason |
|---|---|
| `422` | No images provided or malformed JSON fields |

---

## Single-stage endpoints

Use these when you need intermediate outputs explicitly (e.g. inspect raw frames before enhancement, or download enhanced images before running inference).

| Endpoint | Purpose |
|---|---|
| `POST /video/ingest` | Extract frames to disk, return paths immediately |
| `POST /enhance` | Enhance uploaded images, stream progress via WS |
| `POST /infer` | Run inference on uploaded images, stream progress via WS |
| `GET /download/{job_id}/{filename}` | Download an enhanced image |
| `GET /infer/{job_id}/results` | Retrieve detection results JSON |

---

## Recommended workflows

**Video → enhanced frames (one request)**
```
1.  WS  /ws/{job_id}
2. POST /pipeline/video/enhance   → WS frame_done events → job_done with file list
3. GET  /download/{job_id}/{filename}   (for each file in job_done.files)
```

**Video → detections (one request)**
```
1.  WS  /ws/{job_id}
2. POST /pipeline/video/detect    → WS frame_done events → job_done with results_url
3. GET  /infer/{job_id}/results
```

**Images → enhance + detect (one request)**
```
1.  WS  /ws/{job_id}
2. POST /pipeline/image/enhance-detect  → WS image_done events → job_done with results_url
3. GET  /infer/{job_id}/results
```

**Step-by-step (intermediate outputs needed)**
```
1.  WS  /ws/{video-job}
2. POST /video/ingest              → frame paths returned immediately

3.  WS  /ws/{enhance-job}
4. POST /enhance                   → WS completed events per frame
5. GET  /download/{enhance-job}/{frame_N_enhanced.png}

6.  WS  /ws/{infer-job}
7. POST /infer                     → WS completed events per frame
8. GET  /infer/{infer-job}/results
```

---

## Model status

Two detector variants are available. Both are fully implemented and share the same inference API — the server loads one or the other based on the `USE_LITE_MODEL` environment variable.

### Selecting the active model

When `USE_LITE_MODEL` is not set, the server auto-selects based on which weights file is present: if `weights/detector.pt` is missing but `weights/detector_lite.pt` exists, SeabedLite is loaded automatically.

```bash
# Auto-detect (recommended) — picks whichever weights file is present
uvicorn main:app --port 8000

# Force SeabedLite
USE_LITE_MODEL=1 uvicorn main:app --port 8000

# Force full SeabedDetector
USE_LITE_MODEL=0 uvicorn main:app --port 8000
```

The API surface (`/infer`, `/pipeline/*`, WebSocket events) is identical for both.

### SeabedDetector (full)

ConvNeXt-S + Swin-T backbone, BiFPN neck, 6-layer DETR decoder, 300 queries. ~83M params.

| Item | Path | Notes |
|---|---|---|
| Fine-tuned weights | `weights/detector.pt` | Train with `python -m train.trainer --epochs 80` |
| Taxonomy label map | `weights/taxonomy_labels.json` | `{"species": ["name1", ...], ...}` per level |
| Prototype embeddings | embedded in `detector.pt` | Auto-populated after Phase 2 training |

### SeabedLite

MobileNetV3-Small backbone, simple top-down FPN, 2-layer DETR decoder, 50 queries. ~2.1M params. Inference ~10–20 ms on M3 MPS.

| Item | Path | Notes |
|---|---|---|
| Fine-tuned weights | `weights/detector_lite.pt` | Train with `python -m train.trainer --lite --epochs 50` |
| Taxonomy label map | `weights/taxonomy_labels.json` | Same file as full model |

**Both models require training before detections are meaningful.**  
Without a weights file, the server starts with ImageNet-pretrained backbone only — detection heads are randomly initialised and will output noise.
