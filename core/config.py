import os

BATCH_SIZE_DEFAULT = 8
MAX_CONCURRENCY = 4
MAX_RETRIES = 2

# Storage directories — override via env vars for persistent / cloud-mounted volumes.
# Defaults to /tmp which is ephemeral; set these to a persistent path in production.
RESULTS_DIR       = os.getenv("RESULTS_DIR",       "/tmp/results")
FRAMES_DIR        = os.getenv("FRAMES_DIR",        "/tmp/frames")
INFER_RESULTS_DIR = os.getenv("INFER_RESULTS_DIR", "/tmp/infer_results")

# Model
MODEL_INPUT_SIZE = 448  # must be divisible by Swin patch_size×window_size (4×7=28)
MODEL_NUM_QUERIES = 300
MODEL_D_MODEL = 256
MODEL_WEIGHTS_PATH = "weights/detector.pt"

# Taxonomy (sizes are dataset-dependent placeholders)
TAXONOMY_LEVELS = ["phylum", "class_", "order", "family", "species"]
TAXONOMY_SIZES: dict[str, int] = {
    "phylum":  8,
    "class_":  32,
    "order":   128,
    "family":  512,
    "species": 2048,
}
TAXONOMY_LABELS_PATH = "weights/taxonomy_labels.json"

# Novelty / OOD detection
NOVELTY_CONF_THRESHOLD = 0.5
NOVELTY_DIST_THRESHOLD = 0.7

# Inference
INFER_CONF_THRESHOLD = 0.3
INFER_BATCH_SIZE = 4

# Video ingestion
VIDEO_SAMPLE_FPS = 2.0

# ── SeabedLite — lightweight demo model ───────────────────────────────────────
# Runs on Apple Silicon MPS (M1/M2/M3) or CPU without a discrete GPU.
# Taxonomy sizes match the actual data/annotations.json class counts.
# Switch between models at runtime: USE_LITE_MODEL=1 uvicorn api.app:app
LITE_INPUT_SIZE      = 320          # no Swin patch/window constraint
LITE_NUM_QUERIES     = 50
LITE_D_MODEL         = 128
LITE_DECODER_LAYERS  = 2
LITE_DECODER_HEADS   = 4
LITE_WEIGHTS_PATH    = "weights/detector_lite.pt"
LITE_TAXONOMY_SIZES: dict[str, int] = {
    "phylum":  5,    # actual counts from data/annotations.json
    "class_":  13,
    "order":   19,
    "family":  23,
    "species": 23,
}

# ── TaxDETR — novel architecture (full model) ──────────────────────────────────
TAXDETR_INPUT_SIZE      = 448          # same constraint as SeabedDetector
TAXDETR_NUM_QUERIES     = 300
TAXDETR_D_MODEL         = 256
TAXDETR_BIFPN_ITERS     = 3
TAXDETR_CONF_THRESHOLD  = 0.5
TAXDETR_DISAGREE_THRESH = 0.3          # coarse–fine gap to trigger novelty
TAXDETR_WEIGHTS_PATH    = "weights/taxdetr.pt"

# ── TaxDETR-Lite — novel architecture (lightweight) ────────────────────────────
TAXDETR_LITE_INPUT_SIZE      = 320
TAXDETR_LITE_NUM_QUERIES     = 50
TAXDETR_LITE_D_MODEL         = 128
TAXDETR_LITE_CONF_THRESHOLD  = 0.5
TAXDETR_LITE_DISAGREE_THRESH = 0.3
TAXDETR_LITE_WEIGHTS_PATH    = "weights/taxdetr_lite.pt"

# Taxonomy tree: maps species_idx → phylum_idx.
# Populated from annotations at runtime; placeholder shown for 23-species lite dataset.
# Format: {species_idx: phylum_idx, ...}
# Replace with actual mappings derived from data/annotations.json.
TAXDETR_SPECIES_TO_PHYLUM: dict[int, int] = {
    # species 0-4  → phylum 0
    **{i: 0 for i in range(5)},
    # species 5-9  → phylum 1
    **{i: 1 for i in range(5, 10)},
    # species 10-14 → phylum 2
    **{i: 2 for i in range(10, 15)},
    # species 15-19 → phylum 3
    **{i: 3 for i in range(15, 20)},
    # species 20-22 → phylum 4
    **{i: 4 for i in range(20, 23)},
}
