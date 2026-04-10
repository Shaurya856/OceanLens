# Image API — Seabed Enhancement & Species Detection

A FastAPI server that provides:
- **Image enhancement** — 7 configurable techniques (CLAHE, denoising, dehazing, gamma correction, Retinex, super-resolution, white balance), chainable as pipelines.
- **Species detection** — SeabedDetector neural network (ConvNeXt-S + Swin-T backbone, BiFPN neck, DETR decoder, hierarchical taxonomy classifier, novelty detection).
- **Video ingestion** — frame extraction from uploaded video files.
- **Real-time progress** — per-job WebSocket events for all async operations.

---

## Project structure

```
Image_API/
├── main.py                # FastAPI server entry point
├── streamlit_app.py       # Local Streamlit UI (calls enhancement library directly)
├── requirements.txt
├── API.md                 # Full REST + WebSocket API reference
│
├── core/                  # Shared constants, utilities, retry logic
├── enhancement/           # Image enhancement techniques and pipeline runners
├── api/                   # FastAPI routes, async scheduler, WebSocket manager
├── inference/             # Species detection pipeline (model load + batch inference)
├── video/                 # Video frame extraction
├── model/                 # Neural network architecture
│   ├── detection/         # Backbone (ConvNeXt+Swin), BiFPN neck, DETR decoder
│   ├── classification/    # Hierarchical taxonomy classifier, novelty detector
│   └── lite/              # SeabedLite — lightweight variant for laptop/MPS
├── train/                 # Training pipeline (supports both SeabedDetector and SeabedLite)
├── data/                  # Annotations JSON + training images
│   └── images/            # Species image folders (e.g. data/images/Seahorse/)
└── weights/               # Trained model weights and taxonomy label map
```

---

## Server-client architecture

The system is designed with a clear server/client split so that:
- Enhancement and inference compute stays on a dedicated machine (GPU server).
- Any number of clients — a browser, a Python script, another service — can call the API over HTTP/WebSocket without touching the server's Python environment.

```
┌──────────────────────────────────────────┐
│               CLIENT SIDE                │
│                                          │
│  Browser (frontend/)   streamlit_app.py  │
│  curl / Python requests                  │
│  Any REST/WebSocket client               │
└────────────┬─────────────────────────────┘
             │  HTTP REST + WebSocket
             ▼
┌──────────────────────────────────────────┐
│               SERVER SIDE                │
│  main.py  (uvicorn / FastAPI)            │
│  ├─ api/        routes, scheduler        │
│  ├─ enhancement/  pipeline logic         │
│  ├─ inference/    model + batch runner   │
│  └─ video/        frame extraction       │
└──────────────────────────────────────────┘
```

**`streamlit_app.py`** is a special case: it imports the enhancement library directly
(no HTTP hop) for a fast, zero-latency local UI. It is **not** a client of the FastAPI
server — it runs the same enhancement code in-process.  Run it separately from the server.

---

## Quick start

### 1 — Set up virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows CMD
# .venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
```

### 2 — Start the API server

```bash
uvicorn main:app --reload --port 8000
```

The API is now available at `http://localhost:8000`.  
WebSocket endpoint: `ws://localhost:8000/ws/{job_id}`.  
Interactive docs: `http://localhost:8000/docs`.

**Model auto-selection:** if `USE_LITE_MODEL` is not set, the server picks the model based on which weights file is present:

| `weights/detector.pt` | `weights/detector_lite.pt` | Model loaded |
|---|---|---|
| exists | — | SeabedDetector (full) |
| missing | exists | SeabedLite (auto-fallback) |
| exists | exists | SeabedDetector (full) |
| missing | missing | SeabedDetector with random heads — no useful detections |

Override at any time:

```bash
USE_LITE_MODEL=1 uvicorn main:app --reload --port 8000   # force lite
USE_LITE_MODEL=0 uvicorn main:app --reload --port 8000   # force full
```

If neither weights file exists, the server starts with ImageNet-pretrained backbone only — detection heads are randomly initialised and detections will be noise until training is complete.

### 3 — Run the Streamlit UI (optional, local mode)

Open a separate terminal with the same venv active:

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## Model variants

Two model variants are available. They share the same training pipeline, dataset format, and inference API — only the neural network architecture differs.

| | SeabedLite | SeabedDetector (full) |
|---|---|---|
| **Purpose** | Laptop demo, development, MPS | Production, GPU server |
| **Backbone** | MobileNetV3-Small | ConvNeXt-Small + Swin-Tiny (dual-path) |
| **Input size** | 320 × 320 | 448 × 448 |
| **Neck** | Simple top-down FPN (128ch) | BiFPN × 3 iterations (256ch) |
| **Decoder** | 2 layers, 4 heads, 50 queries | 6 layers, 8 heads, 300 queries |
| **Params** | ~2.1M | ~83M |
| **Inference — M3 MPS** | ~10–20 ms | ~200–400 ms |
| **Inference — CPU** | ~80–120 ms | > 2 s |
| **Novelty detection** | Confidence gate only | Confidence gate + prototype distance |
| **Weights file** | `weights/detector_lite.pt` | `weights/detector.pt` |

Use `SeabedLite` for development, smoke-testing, and demos on your laptop.  
Use the full `SeabedDetector` when GPU hardware is available and accuracy matters.

---

## Training the detector

### Prerequisites

```bash
# 1. Organise your images (one subfolder per species):
#    data/images/Seahorse/img001.jpg
#    data/images/Sharks/img001.jpg  ...
#    (Current dataset: 13,711 images across 23 species)

# 2. Annotation file is already at data/annotations.json
#    To regenerate from the image folders:
python train/generate_annotations.py
```

### Option A — SeabedLite (laptop / Apple Silicon MPS)

Recommended for development and demos. Trains on an M3 MacBook Air in under an hour.

```bash
# Smoke-test first (1 epoch, 64 samples — confirms the pipeline works)
python -m train.trainer --lite \
    --epochs 1 --max-samples 64

# Recommended run (15 epochs, ~45 min on M3 MPS with batch 32 + autocast)
# Good enough for demos — validation loss typically plateaus by epoch 12–15.
python -m train.trainer --lite \
    --annotation-path data/annotations.json \
    --image-dir       data/images \
    --epochs 15 \
    --warmup-epochs 5 \
    --batch-size 32

# Higher-quality run (30 epochs, ~1.5 h on M3 MPS with batch 32 + autocast)
python -m train.trainer --lite \
    --annotation-path data/annotations.json \
    --image-dir       data/images \
    --epochs 30 \
    --warmup-epochs 5 \
    --batch-size 32
```

Weights are written to `weights/detector_lite.pt` automatically.

**Estimated training times on M3 MacBook Air (8 GB unified memory):**

| Epochs | Batch size | Steps/epoch | Est. time (with autocast) |
|---|---|---|---|
| 1 (smoke-test) | 32 | 429 | ~2–3 min |
| 15 | 32 | 429 | ~45 min |
| 30 | 32 | 429 | ~1.5 h |
| 50 | 32 | 429 | ~2–2.5 h |

> MPS acceleration and mixed-precision (`torch.autocast("mps", dtype=bfloat16)`) are applied automatically. `bfloat16` is used instead of `float16` — it is more stable on Apple Silicon when backpropagating through the Transformer decoder and unfrozen backbone simultaneously. Without autocast, each epoch takes ~11 min; with it, expect ~5–6 min/epoch.  
> Try `--batch-size 32` if memory allows — it halves steps/epoch with no accuracy cost.

### Option B — SeabedDetector full model (GPU server)

```bash
# Smoke-test
python -m train.trainer \
    --epochs 1 --max-samples 64

# Full training run (80 epochs, requires CUDA GPU)
python -m train.trainer \
    --annotation-path data/annotations.json \
    --image-dir       data/images \
    --epochs 80 \
    --warmup-epochs 10 \
    --batch-size 8
```

Weights are written to `weights/detector.pt` automatically.

### All CLI flags

```
python -m train.trainer [options]

  --annotation-path   Path to annotations JSON  (default: data/annotations.json)
  --image-dir         Root image directory       (default: data/images)
  --checkpoint-dir    Where to save checkpoints  (default: checkpoints/)
  --epochs            Total training epochs      (default: 80)
  --warmup-epochs     Backbone-frozen epochs     (default: 10)
  --batch-size        Images per batch           (default: 8)
  --head-lr           LR for neck/decoder/heads  (default: 1e-4)
  --backbone-lr       LR for backbone (Phase 2)  (default: 1e-5)
  --weight-decay      AdamW weight decay         (default: 1e-4)
  --val-split         Fraction held out for val  (default: 0.1)
  --num-workers       DataLoader workers         (default: 2; safe on macOS with __main__ guard)
  --max-samples       Cap dataset (smoke-tests)
  --lite              Train SeabedLite instead of the full model
```

---

## Making API requests

See [API.md](API.md) for the full reference. A minimal example with `curl`:

```bash
# Submit an enhancement job
JOB=job-001
curl -s -X POST http://localhost:8000/enhance \
  -F "images=@photo.jpg" \
  -F "job_id=$JOB" \
  -F "mode=custom" \
  -F 'techniques=["denoise","clahe","gamma_correction"]' \
  -F 'params={"gamma_correction":{"gamma":1.4}}'

# Download the result (filename from WS completed event)
curl -O http://localhost:8000/download/$JOB/photo_enhanced.png
```
