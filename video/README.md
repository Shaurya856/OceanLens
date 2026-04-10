# video/

Video ingestion — extracts frames from uploaded video files.

| File | Purpose |
|------|---------|
| `processor.py` | `extract_frames(video_bytes, job_id, sample_fps)` — writes frames as PNGs to `FRAMES_DIR/{job_id}/` |

## Frame extraction

`extract_frames` accepts raw video bytes (mp4, avi, mov, mkv, etc.) and samples frames at the requested rate:

```python
from video.processor import extract_frames

frames = extract_frames(video_bytes, job_id="dive-001", sample_fps=2.0)
# returns [{"filename": "frame_000000.png", "path": "/tmp/frames/dive-001/frame_000000.png"}, ...]
```

## Typical workflow

```
POST /video/ingest  →  frames written to FRAMES_DIR
POST /enhance       →  enhance the returned frame paths
POST /infer         →  detect species in enhanced frames
GET  /infer/{job}/results
```

See [API.md](../API.md) for the full HTTP reference.

## Setup

```bash
source .venv/bin/activate    # activate venv (see root README)
# opencv-python handles video decoding (included in requirements.txt)
```
