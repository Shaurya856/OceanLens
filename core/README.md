# core/

Shared infrastructure used across all packages.  Every other package imports from here — nothing in `core/` depends on any sibling package.

| File | Purpose |
|------|---------|
| `config.py` | Global constants — batch sizes, directory paths, model hyperparameters, taxonomy settings |
| `utils.py` | Image encode/decode helpers (`decode_image`, `encode_image`) and ID/filename generators |
| `retry.py` | Exponential-backoff retry decorator — wraps CPU-bound enhancement calls |

## Key constants (`config.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE_DEFAULT` | 8 | Images per enhancement batch |
| `MAX_CONCURRENCY` | 4 | Max concurrent batch tasks |
| `MAX_RETRIES` | 2 | Retry attempts on transient failures |
| `RESULTS_DIR` | `/tmp/results` | Enhanced image output directory |
| `FRAMES_DIR` | `/tmp/frames` | Extracted video frame directory |
| `INFER_RESULTS_DIR` | `/tmp/infer_results` | Inference results directory |
| `MODEL_INPUT_SIZE` | 448 | Backbone input resolution (must be divisible by 28) |
| `MODEL_WEIGHTS_PATH` | `weights/detector.pt` | Trained detector weights |
| `TAXONOMY_LABELS_PATH` | `weights/taxonomy_labels.json` | Taxonomy class index map |

## Usage

```python
from core.config import MAX_CONCURRENCY, RESULTS_DIR
from core.utils import decode_image, encode_image, generate_image_id
from core.retry import retry
```

No setup beyond `pip install -r requirements.txt` (see root [README](../README.md)).
