# enhancement/

Image enhancement pipeline — techniques, registry, and pipeline runners.

| File | Purpose |
|------|---------|
| `techniques.py` | All 7 enhancement functions: `apply_clahe`, `apply_denoise`, `apply_dehaze`, `apply_gamma_correction`, `apply_retinex`, `apply_superres`, `apply_white_balance` |
| `registry.py` | `TECHNIQUES` dict mapping technique names to their functions |
| `pipelines.py` | `run_single` and `run_custom` — apply one technique or chain many sequentially |

## Function signatures

Every technique follows the same contract:

```python
def apply_<name>(image: bytes, config: dict = {}) -> bytes:
    ...
```

`image` is raw image bytes (PNG/JPEG); `config` is an optional dict of technique-specific parameters. The function returns the processed image as PNG bytes.

## Available techniques

| Name | Key params (with defaults) |
|------|---------------------------|
| `clahe` | `clip_limit=2.0`, `tile_size=8` |
| `denoise` | `h=10`, `h_for_color=10`, `template_window=7`, `search_window=21` |
| `dehaze` | `omega=0.95`, `t0=0.1`, `radius=15` |
| `gamma_correction` | `gamma=1.2` |
| `retinex` | `sigmas=[15,80,250]`, `alpha=125` |
| `superres` | `scale=2.0` |
| `white_balance` | `percent=1.0` |

## Running a pipeline programmatically

```python
from enhancement.pipelines import run_single, run_custom

# Single technique
out = run_single(image_bytes, "clahe", {"clahe": {"clip_limit": 3.0}})

# Chained pipeline
out = run_custom(
    image_bytes,
    techniques=["denoise", "clahe", "gamma_correction"],
    params={"gamma_correction": {"gamma": 1.4}},
)
```

## Adding a new technique

1. Implement `apply_<name>(image: bytes, config: dict = {}) -> bytes` in `techniques.py`.
2. Add an entry to `TECHNIQUES` in `registry.py`.

That's it — validators, the Streamlit UI, and the REST API all pick it up automatically.

## Setup

```bash
source .venv/bin/activate        # activate venv (see root README)
# no additional steps — opencv-python and numpy are in requirements.txt
```
