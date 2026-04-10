# SeabedDetector — Model Architecture

A hybrid CNN–Transformer detection model for marine species identification in seabed imagery.  
Designed for high accuracy on fine-grained underwater specimens including detection of novel/unseen species.

---

## Directory layout

```
model/
├── detector.py                 SeabedDetector — full model assembly (~83M params)
├── detection/
│   ├── backbone.py             DualPathBackbone — ConvNeXt-S + Swin-T + CBAM fusion
│   ├── bifpn.py                BiFPN — bidirectional multi-scale feature pyramid
│   └── decoder.py              DETRDecoder — shared by both models
├── classification/
│   ├── classifier.py           HierarchicalClassifier — shared by both models
│   └── novelty.py              NoveltyDetector — shared by both models
└── lite/
    ├── backbone_lite.py        LiteBackbone — MobileNetV3-Small single-path
    ├── neck_lite.py            LiteFPN — single top-down FPN (128ch)
    └── detector_lite.py        SeabedLite — lightweight assembly (~2.1M params)
```

Both `SeabedDetector` and `SeabedLite` return an identical output dict and are compatible with the same training pipeline, loss function, and inference runner.

---

## End-to-end data flow

```
Input image  [B, 3, 448, 448]
│
├─ ConvNeXt-Small ──────────────────────────────────────────────────────┐
│   pretrained ImageNet-1k, features_only, out_indices=(1,2,3)          │  fused with
├─ Swin-Tiny (img_size=448) ────────────────────────────────────────────┤  learned softmax
│   pretrained ImageNet-1k, features_only, out_indices=(1,2,3)          │  weights + CBAM
└──────────────────────────────────────────────────────────────────────-┘
        │
        │   P3  [B, 192, 56, 56]   stride 8
        │   P4  [B, 384, 28, 28]   stride 16
        │   P5  [B, 768, 14, 14]   stride 32
        │
        ▼
    BiFPN  (3 iterations, d_model=256)
        │   Project 192/384/768 → 256 channels
        │   Create P6 from P5 via strided conv
        │   3× bidirectional top-down + bottom-up fusion
        │
        │   P3  [B, 256, 56, 56]
        │   P4  [B, 256, 28, 28]
        │   P5  [B, 256, 14, 14]
        │   P6  [B, 256,  7,  7]
        │
        ▼
    DETRDecoder  (P3+P4+P5 only, 6 layers, 8 heads)
        │   Memory = flatten & concat P3/P4/P5 with 2-D sinusoidal PE
        │   300 learnable object queries cross-attend over memory
        │
        │   query embeddings  [B, 300, 256]
        │
        ├──▶  bbox_head   MLP(256→256→4)  →  sigmoid  →  boxes  [B, 300, 4]
        │                                                  (cx, cy, w, h) normalised 0–1
        │
        ├──▶  conf_head   Linear(256→1)               →  confidence  [B, 300]
        │                                                  (raw logit; sigmoid at inference)
        │
        ├──▶  HierarchicalClassifier
        │         phylum   MLP(256→256→8)    →  [B, 300, 8]
        │         class_   MLP(256→256→32)   →  [B, 300, 32]
        │         order    MLP(256→256→128)  →  [B, 300, 128]
        │         family   MLP(256→256→512)  →  [B, 300, 512]
        │         species  MLP(256→256→2048) →  [B, 300, 2048]
        │
        └──▶  NoveltyDetector
                  confidence gate:   max_softmax(species logits) < 0.5  → novel
                  prototype gate:    cosine_sim(embedding, nearest_proto) < 0.7  → novel
                  novelty_score [B, 300]  =  1 − max_softmax_confidence
                  is_novel      [B, 300]  (bool, fires on either gate)
```

---

## Component reference

### 1. DualPathBackbone  (`detection/backbone.py`)

**Why two paths?**  
ConvNeXt-Small excels at local texture (scales, shell patterns, coral polyp detail). Swin-Tiny captures long-range spatial context (body shape, scene layout) via shifted-window self-attention. Neither architecture alone is sufficient for fine-grained underwater identification.

**Fusion**  
At each of the three output scales (P3/P4/P5) the feature maps are combined with a learned, per-scale, softmax-normalised weighted sum:

```
fused = softmax([w0, w1])[0] × convnext_feat + softmax([w0, w1])[1] × swin_feat
```

This lets the model learn how much to trust each path per scale — early layers often prefer ConvNeXt texture while deeper layers lean on Swin context.

**CBAM attention**  
After fusion, a Convolutional Block Attention Module (CBAM) re-weights channels then spatial positions independently:

```
channel attention: Sigmoid( FC(AvgPool(x)) + FC(MaxPool(x)) )  × x
spatial attention: Sigmoid( Conv([AvgPool_c(x), MaxPool_c(x)]) )  × x
```

**Input size constraint**  
Swin-Tiny uses patch size 4 and window size 7, so the model input must be divisible by 4 × 7 = 28. Default: `MODEL_INPUT_SIZE = 448`.

| Property | ConvNeXt-Small | Swin-Tiny |
|---|---|---|
| Architecture family | Pure CNN | Shifted-window transformer |
| Pretrained on | ImageNet-1k | ImageNet-1k |
| Output channels (P3/P4/P5) | 192 / 384 / 768 | 192 / 384 / 768 |
| Receptive field strength | Local (texture) | Global (context) |

---

### 2. BiFPN  (`detection/bifpn.py`)

Bidirectional Feature Pyramid Network (EfficientDet-style) for multi-scale feature aggregation.

**Why BiFPN over standard FPN?**  
A standard top-down FPN propagates information from coarse to fine scales only. BiFPN adds a bottom-up path so both directions are learned, allowing fine-scale features (P3) to inform coarse ones (P6) and vice versa. The learned per-connection fusion weights also allow the network to suppress uninformative paths.

**Structure per iteration**

```
Top-down pass:
  P5_td  = DWConv( w(P5,     upsample(P6))   )
  P4_td  = DWConv( w(P4,     upsample(P5_td)) )
  P3_out = DWConv( w(P3,     upsample(P4_td)) )

Bottom-up pass:
  P4_out = DWConv( w(P4, P4_td, downsample(P3_out)) )
  P5_out = DWConv( w(P5, P5_td, downsample(P4_out)) )
  P6_out = DWConv( w(P6,        downsample(P5_out)) )
```

Weights are ReLU-normalised (fast normalisation, avoids vanishing gradients during early training):

```python
w = relu(raw_w) + eps
w = w / w.sum()
```

Convolutions are depthwise-separable (DW 3×3 → PW 1×1 + BN + SiLU) to keep FLOP count low.

| Parameter | Value |
|---|---|
| Input channels (P3/P4/P5) | 192 / 384 / 768 |
| Output channels (all levels) | 256 (`d_model`) |
| Scales | P3, P4, P5, P6 |
| Iterations | 3 |
| Conv type | Depthwise-separable |

---

### 3. DETRDecoder  (`detection/decoder.py`)

DETR-style transformer decoder. The backbone+BiFPN output is treated as a sequence of visual tokens that 300 learnable object queries attend to.

**Memory construction**  
P3, P4, and P5 (not P6) are used as memory. Each scale is flattened to a token sequence and enriched with 2-D sinusoidal positional encodings before concatenation:

```
P3 → [B, 56×56, 256] = [B, 3136, 256]
P4 → [B, 28×28, 256] = [B,  784, 256]
P5 → [B, 14×14, 256] = [B,  196, 256]
─────────────────────────────────────
memory               = [B, 4116, 256]
```

**2-D sinusoidal positional encoding**  
Separate sin/cos encodings for the Y and X axes (each d_model/2 dims) are concatenated to form a d_model-dimensional position signal per token. This preserves the spatial structure of the feature map in the attention mechanism.

**Decoder layers**  
Each of 6 decoder layers is a standard pre-norm (`norm_first=True`) transformer decoder layer:

```
Self-attention on queries (300 × 300)
Cross-attention: queries attend to memory (300 × 4116)
Feed-forward: d_model → d_model×4 → d_model  (GELU)
```

Pre-norm improves gradient flow in deep transformers and allows training with larger learning rates in the early epochs.

| Hyperparameter | Value |
|---|---|
| `d_model` | 256 |
| Attention heads | 8 |
| Decoder layers | 6 |
| Object queries | 300 |
| FFN expansion | 4× |
| Activation | GELU |
| Dropout | 0.1 |

---

### 4. HierarchicalClassifier  (`classification/classifier.py`)

Five independent two-layer MLP heads, one per taxonomy level, operating on the same 300 query embeddings.

```
embeddings [B, 300, 256]
    ├── phylum  head:  Linear(256→256) → GELU → LayerNorm → Linear(256→   8)
    ├── class_  head:  Linear(256→256) → GELU → LayerNorm → Linear(256→  32)
    ├── order   head:  Linear(256→256) → GELU → LayerNorm → Linear(256→ 128)
    ├── family  head:  Linear(256→256) → GELU → LayerNorm → Linear(256→ 512)
    └── species head:  Linear(256→256) → GELU → LayerNorm → Linear(256→2048)
```

**Why independent heads?**  
Shared parameters across taxonomy levels risk the coarser levels being dominated by fine-level gradients. Independent heads let each level converge at its own rate. The loss weights in training (`train/loss.py`) also scale per level, with species carrying the highest weight.

**Taxonomy levels and class counts (defaults)**

| Level | Classes |
|---|---|
| phylum | 8 |
| class_ | 32 |
| order | 128 |
| family | 512 |
| species | 2048 |

These are configured in `core/config.py → TAXONOMY_SIZES` and must match the training dataset's label map.

---

### 5. NoveltyDetector  (`classification/novelty.py`)

Flags detections as novel (unseen species) using two complementary signals.

**Signal 1 — Confidence gate**

```
novelty_score = 1 − max( softmax(species_logits) )
is_novel      = max_softmax_confidence < NOVELTY_CONF_THRESHOLD  (default 0.5)
```

A detection with low confidence across all known species is a candidate for novelty. This fires immediately at inference without needing prototype data.

**Signal 2 — Prototype distance**

```python
sim = cosine_similarity(normalize(embedding), normalize(prototypes))  # [B, N, S]
max_sim = sim.max(dim=-1)
is_novel |= (max_sim < NOVELTY_DIST_THRESHOLD)  # default 0.7
```

After training, mean embedding vectors (prototypes) for each known species are computed via `update_prototypes()`. At inference, if no prototype is cosine-close enough to the query embedding, the detection is flagged as novel regardless of its classification confidence.

A detection is `is_novel=True` when **either** signal fires — conservative flagging is preferred for scientific use.

**Populating prototypes (post-training)**

```python
# After training, iterate training set with the frozen model:
for batch in train_loader:
    embeddings = model(batch["images"])["embeddings"]  # [B, N, d]
    labels     = batch["species_labels"]               # [B, N]
    model.novelty.update_prototypes(
        embeddings.flatten(0, 1),
        labels.flatten(0, 1)
    )
```

Prototypes are stored as `register_buffer` tensors and are saved/loaded with `model.state_dict()`.

| Parameter | Default | Effect |
|---|---|---|
| `NOVELTY_CONF_THRESHOLD` | 0.5 | Lower → more conservative (more novel flags) |
| `NOVELTY_DIST_THRESHOLD` | 0.7 | Lower → more conservative (more novel flags) |

---

### 6. Detection heads  (`detector.py`)

Two lightweight heads operate on the 300 query embeddings in parallel with the classifier:

**Bounding box head**

```
MLP: Linear(256→256) → GELU → Linear(256→4) → Sigmoid
Output: (cx, cy, w, h)  normalised to [0, 1]
```

Sigmoid ensures coordinates stay in valid range without any clipping.

**Confidence head**

```
Linear(256→1)
Output: raw logit  (apply sigmoid at inference, threshold at INFER_CONF_THRESHOLD=0.3)
```

Raw logits are kept during training for use with binary cross-entropy loss.

---

## Model output format

`SeabedDetector.forward(x)` returns a dict:

| Key | Shape | Description |
|---|---|---|
| `boxes` | `[B, 300, 4]` | Normalised (cx, cy, w, h), sigmoid applied |
| `confidence` | `[B, 300]` | Raw objectness logits |
| `class_logits` | `dict[str, [B, 300, C]]` | Raw logits per taxonomy level |
| `novelty_scores` | `[B, 300]` | 0 = certain known, 1 = fully novel |
| `is_novel` | `[B, 300]` | Boolean novel flag |
| `embeddings` | `[B, 300, 256]` | Raw query embeddings (for prototype update) |

---

## Inference post-processing (`inference/runner.py`)

```
1. Sigmoid on confidence logits
2. Filter detections where confidence > INFER_CONF_THRESHOLD (0.3)
3. Convert (cx,cy,w,h) → (x1,y1,x2,y2) in pixel space
4. Torchvision NMS (IoU threshold 0.5) to remove duplicates
5. Argmax on each taxonomy level → string label lookup
6. If is_novel: set species = "novel_species", record closest_known_species
```

---

## Training summary

Two-phase training is implemented in `train/trainer.py`:

**Phase 1 — Backbone frozen** (epochs 0–29)  
Only BiFPN, decoder, and heads are trained. The ImageNet backbone weights are preserved while the task-specific layers learn to use them.

**Phase 2 — Full fine-tune** (epochs 30–79)  
All layers unfrozen with layer-wise learning rate decay (backbone LR = head LR × 0.1).

**Loss functions** (`train/loss.py`)

| Component | Loss | Weight |
|---|---|---|
| Bounding box | L1 + GIoU | 5 / 2 |
| Confidence | Weighted BCE | 1 |
| species | Focal cross-entropy | 4 |
| family, order, class_, phylum | Cross-entropy | 1 (each) |

Hungarian bipartite matching (`train/matcher.py`) aligns predicted queries to ground-truth annotations before loss computation.

**Data augmentation** (`train/augmentations.py`)

Underwater-specific augmentations applied during training:
- Horizontal / vertical flip
- Random scale-crop
- Red-channel suppression (simulates depth-dependent colour shift)
- Gaussian blur and noise
- Brightness jitter and colour jitter

---

## Configuration

All tunable constants live in `core/config.py`:

```python
MODEL_INPUT_SIZE     = 448    # must be divisible by 28 (Swin: patch 4 × window 7)
MODEL_NUM_QUERIES    = 300
MODEL_D_MODEL        = 256
MODEL_WEIGHTS_PATH   = "weights/detector.pt"

TAXONOMY_LEVELS      = ["phylum", "class_", "order", "family", "species"]
TAXONOMY_SIZES       = {"phylum": 8, "class_": 32, "order": 128, "family": 512, "species": 2048}
TAXONOMY_LABELS_PATH = "weights/taxonomy_labels.json"

NOVELTY_CONF_THRESHOLD = 0.5
NOVELTY_DIST_THRESHOLD = 0.7
INFER_CONF_THRESHOLD   = 0.3
```

---

## Quick usage

```python
from model.detector import SeabedDetector
from core.config import TAXONOMY_SIZES, MODEL_D_MODEL, MODEL_NUM_QUERIES
import torch

model = SeabedDetector(
    taxonomy_sizes=TAXONOMY_SIZES,
    d_model=MODEL_D_MODEL,
    num_queries=MODEL_NUM_QUERIES,
)
model.eval()

x = torch.randn(1, 3, 448, 448)
with torch.no_grad():
    out = model(x)

# out["boxes"]         → [1, 300, 4]
# out["confidence"]    → [1, 300]
# out["class_logits"]  → {"phylum": ..., "species": ...}
# out["is_novel"]      → [1, 300]
```

For the production load + preprocess + postprocess flow see `inference/runner.py`.  
For training see `train/trainer.py` and the `train.py` CLI (`--smoke-test` flag for quick validation).

---

## SeabedLite — lightweight variant

SeabedLite is a purpose-built lightweight variant for development, demos, and training on Apple Silicon MPS (M1/M2/M3) without a discrete GPU. It shares the same output format, training pipeline, and inference runner as SeabedDetector — only the network components are smaller.

### Architecture comparison

| Component | SeabedDetector | SeabedLite |
|---|---|---|
| Backbone | ConvNeXt-Small + Swin-Tiny (dual-path + CBAM) | MobileNetV3-Small (single-path) |
| Input size | 448 × 448 (Swin constraint: divisible by 28) | 320 × 320 (no constraint) |
| Neck | BiFPN × 3 iterations, 256ch, 4 levels (P3–P6) | Top-down FPN × 1 pass, 128ch, 3 levels (P3–P5) |
| Decoder | 6 layers, 8 heads, 300 queries, d_model=256 | 2 layers, 4 heads, 50 queries, d_model=128 |
| Memory tokens | 4116 (P3: 3136 + P4: 784 + P5: 196) | 2100 (P3: 1600 + P4: 400 + P5: 100) |
| Classifier | Independent MLP heads, d_model=256 | Same structure, d_model=128 |
| Novelty detection | Confidence gate + prototype distance | Confidence gate only |
| Total params | ~83M | ~2.1M |
| Weights file | `weights/detector.pt` | `weights/detector_lite.pt` |

### Data flow

```
Input image  [B, 3, 320, 320]
│
└─ MobileNetV3-Small ──────────────────────────────────────────────────────────
    pretrained ImageNet-1k, features_only, out_indices=(2,3,4)
        │
        │   P3  [B, 24, 40, 40]   stride 8
        │   P4  [B, 48, 20, 20]   stride 16
        │   P5  [B, 96, 10, 10]   stride 32
        │
        ▼
    LiteFPN  (1 top-down pass, d_model=128)
        │   lat5: Conv1×1(96→128)
        │   lat4: Conv1×1(48→128)
        │   lat3: Conv1×1(24→128)
        │   td4 = lat4 + upsample(lat5)
        │   td3 = lat3 + upsample(td4)
        │   3×3 output convs (BN + SiLU) on each level
        │
        │   P3  [B, 128, 40, 40]
        │   P4  [B, 128, 20, 20]
        │   P5  [B, 128, 10, 10]
        │
        ▼
    DETRDecoder  (P3+P4+P5, 2 layers, 4 heads)  ← same module, smaller params
        │   Memory = [B, 2100, 128]  (1600 + 400 + 100 tokens)
        │   50 learnable object queries cross-attend over memory
        │
        │   query embeddings  [B, 50, 128]
        │
        ├──▶  bbox_head   MLP(128→64→4)   →  sigmoid  →  boxes  [B, 50, 4]
        ├──▶  conf_head   Linear(128→1)                →  confidence  [B, 50]
        ├──▶  HierarchicalClassifier (d_model=128)
        │         phylum   [B, 50,  5]
        │         class_   [B, 50, 13]
        │         order    [B, 50, 19]
        │         family   [B, 50, 23]
        │         species  [B, 50, 23]
        └──▶  NoveltyDetector (confidence gate only)
                  is_novel = max_softmax(species) < 0.5
```

### Training SeabedLite on Apple Silicon M3

```bash
# Smoke-test — 1 epoch, 64 samples (~2–3 minutes)
python -m train.trainer --lite --epochs 1 --max-samples 64

# Recommended run — 15 epochs (~45 min on M3 MPS with batch 32 + autocast)
# Val loss typically plateaus by epoch 12–15 on this 23-species dataset.
python -m train.trainer --lite \
    --annotation-path data/annotations.json \
    --image-dir       data/images \
    --epochs 15 \
    --warmup-epochs 5 \
    --batch-size 32

# Higher-quality run — 30 epochs (~1.5 h on M3 MPS with batch 32 + autocast)
python -m train.trainer --lite \
    --annotation-path data/annotations.json \
    --image-dir       data/images \
    --epochs 30 \
    --warmup-epochs 5 \
    --batch-size 32
```

**Estimated wall-clock times on M3 MacBook Air (8 GB unified memory):**

| Epochs | Batch size | Steps/epoch | Est. time (with autocast) |
|---|---|---|---|
| 1 (smoke) | 32 | 429 | ~2–3 min |
| 15 | 32 | 429 | ~45 min |
| 30 | 32 | 429 | ~1.5 h |
| 50 | 32 | 429 | ~2–2.5 h |

MPS acceleration and `torch.autocast("mps", dtype=bfloat16)` are applied automatically. `bfloat16` is used rather than `float16` — MPS can deadlock during Phase 2 (full fine-tune) when backpropagating through the Transformer decoder and unfrozen backbone together with `float16`. `bfloat16` has better numerical range and is fully stable on Apple Silicon.  
Measured baseline without autocast: ~11 min/epoch at batch 16.  
Try `--batch-size 32` to halve steps per epoch — memory usage stays well within 8 GB for SeabedLite.

### Expected accuracy (23-species dataset, 13,711 images)

| Metric | Expected range | Notes |
|---|---|---|
| Species top-1 accuracy | 75–85% | ~600 images/class; underwater variation limits ceiling |
| Phylum / Class accuracy | 90–95% | Only 5–13 classes |
| Detection mAP@0.5 | 40–55% | Sparse: one annotation per image |
| Novel species | qualitative only | No prototype distance check in lite model |

The full SeabedDetector is expected to reach ~55–70% mAP@0.5 on the same dataset with a GPU.

### Quick usage

```python
from model.lite.detector_lite import SeabedLite
from core.config import LITE_TAXONOMY_SIZES, LITE_D_MODEL, LITE_NUM_QUERIES
from core.config import LITE_DECODER_LAYERS, LITE_DECODER_HEADS
import torch

model = SeabedLite(
    taxonomy_sizes=LITE_TAXONOMY_SIZES,
    d_model=LITE_D_MODEL,
    num_queries=LITE_NUM_QUERIES,
    decoder_layers=LITE_DECODER_LAYERS,
    decoder_heads=LITE_DECODER_HEADS,
)
model.eval()

x = torch.randn(1, 3, 320, 320)
with torch.no_grad():
    out = model(x)

# out["boxes"]        → [1, 50, 4]
# out["confidence"]   → [1, 50]
# out["class_logits"] → {"phylum": ..., "species": ...}
# out["is_novel"]     → [1, 50]
```

To use SeabedLite via the inference runner (auto-selected by env var):

```bash
USE_LITE_MODEL=1 uvicorn main:app --port 8000
```

---

## Weights and labels

The model requires trained weights before detections are meaningful.  
ImageNet-pretrained backbones are loaded automatically from `timm`; all other heads are randomly initialised without the weights files.

| File | Model | Purpose |
|---|---|---|
| `weights/detector.pt` | SeabedDetector | Full model state dict (+ prototype embeddings after training) |
| `weights/detector_lite.pt` | SeabedLite | Lite model state dict |
| `weights/taxonomy_labels.json` | Both | `{"species": ["name0", ...], "family": [...], ...}` per level |

After training the full model, populate prototype embeddings:

```python
import torch
model.load_state_dict(torch.load("weights/detector.pt"))
model.eval()

for batch in train_loader:
    with torch.no_grad():
        out = model(batch["images"])
    model.novelty.update_prototypes(
        out["embeddings"].flatten(0, 1),
        batch["species_labels"].flatten(0, 1),
    )

torch.save(model.state_dict(), "weights/detector.pt")
```

SeabedLite uses confidence-gate-only novelty detection — no prototype population step is needed.
