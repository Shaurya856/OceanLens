"""
Inference runner — loads the model once (singleton) and exposes
`run_inference(image_bytes)` for use by the batch processor.

Post-processing:
  1. Sigmoid on confidence logits → filter by INFER_CONF_THRESHOLD
  2. Convert (cx,cy,w,h) normalised → (x1,y1,x2,y2) pixel coords
  3. Torchvision NMS to remove duplicate detections
  4. Attach taxonomy labels and novelty flags
  5. If is_novel → set species label to "novel_species" and record
     the model's closest guess as `closest_known_species`
"""
import json
import os
import threading
import uuid
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

from core.config import (
    MODEL_D_MODEL,
    MODEL_INPUT_SIZE,
    MODEL_NUM_QUERIES,
    MODEL_WEIGHTS_PATH,
    NOVELTY_CONF_THRESHOLD,
    NOVELTY_DIST_THRESHOLD,
    TAXONOMY_LABELS_PATH,
    TAXONOMY_LEVELS,
    TAXONOMY_SIZES,
    INFER_CONF_THRESHOLD,
    LITE_D_MODEL,
    LITE_INPUT_SIZE,
    LITE_NUM_QUERIES,
    LITE_DECODER_LAYERS,
    LITE_DECODER_HEADS,
    LITE_WEIGHTS_PATH,
    LITE_TAXONOMY_SIZES,
)
from model.detector import SeabedDetector
from core.utils import decode_image

# Set USE_LITE_MODEL=1 to load SeabedLite instead of the full SeabedDetector.
# If the env var is not set, auto-detect: use lite when full weights are absent
# but lite weights exist (typical during development on Apple Silicon).
def _resolve_use_lite() -> bool:
    env = os.getenv("USE_LITE_MODEL")
    if env is not None:
        return env == "1"
    full_exists = os.path.isfile(MODEL_WEIGHTS_PATH)
    lite_exists = os.path.isfile(LITE_WEIGHTS_PATH)
    return (not full_exists) and lite_exists

USE_LITE_MODEL: bool = _resolve_use_lite()

logger = logging.getLogger(__name__)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── Singleton models ──────────────────────────────────────────────────────────

_model: SeabedDetector | None = None
_lite_model = None
_device: torch.device | None = None
_taxonomy_labels: dict[str, list[str]] | None = None

# Locks prevent double-initialisation when run_in_executor dispatches concurrent callers.
_model_lock      = threading.Lock()
_lite_model_lock = threading.Lock()


def _load_taxonomy_labels() -> dict[str, list[str]]:
    if os.path.isfile(TAXONOMY_LABELS_PATH):
        with open(TAXONOMY_LABELS_PATH) as f:
            return json.load(f)
    logger.warning(
        "Taxonomy labels file not found at %s — using placeholder indices.",
        TAXONOMY_LABELS_PATH,
    )
    return {level: [str(i) for i in range(TAXONOMY_SIZES[level])] for level in TAXONOMY_LEVELS}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model() -> tuple[SeabedDetector, torch.device, dict[str, list[str]]]:
    """Return the (model, device, taxonomy_labels) singleton, loading on first call."""
    global _model, _device, _taxonomy_labels

    if _model is not None:
        return _model, _device, _taxonomy_labels  # type: ignore[return-value]

    with _model_lock:
        if _model is not None:  # re-check after acquiring lock
            return _model, _device, _taxonomy_labels  # type: ignore[return-value]

        _device = _get_device()
        _taxonomy_labels = _load_taxonomy_labels()

        _model = SeabedDetector(
            taxonomy_sizes=TAXONOMY_SIZES,
            d_model=MODEL_D_MODEL,
            num_queries=MODEL_NUM_QUERIES,
            conf_threshold=NOVELTY_CONF_THRESHOLD,
            dist_threshold=NOVELTY_DIST_THRESHOLD,
            pretrained_backbone=True,
        )

        if os.path.isfile(MODEL_WEIGHTS_PATH):
            state = torch.load(MODEL_WEIGHTS_PATH, map_location=_device, weights_only=True)
            _model.load_state_dict(state)
            logger.info("Loaded detector weights from %s", MODEL_WEIGHTS_PATH)
        else:
            logger.warning(
                "No weights found at %s — backbone uses ImageNet pretraining only.",
                MODEL_WEIGHTS_PATH,
            )

        _model.to(_device).eval()

    return _model, _device, _taxonomy_labels  # type: ignore[return-value]


def get_lite_model():
    """Return the (SeabedLite, device, taxonomy_labels) singleton."""
    global _lite_model, _device, _taxonomy_labels

    if _lite_model is not None:
        return _lite_model, _device, _taxonomy_labels

    with _lite_model_lock:
        if _lite_model is not None:  # re-check after acquiring lock
            return _lite_model, _device, _taxonomy_labels

        from model.lite.detector_lite import SeabedLite

        _device = _get_device()
        _taxonomy_labels = _load_taxonomy_labels()

        # Use real taxonomy sizes from labels file if available; fall back to config.
        lite_taxonomy = {
            level: len(_taxonomy_labels[level])
            for level in TAXONOMY_LEVELS
            if level in _taxonomy_labels
        } or LITE_TAXONOMY_SIZES

        _lite_model = SeabedLite(
            taxonomy_sizes=lite_taxonomy,
            d_model=LITE_D_MODEL,
            num_queries=LITE_NUM_QUERIES,
            decoder_layers=LITE_DECODER_LAYERS,
            decoder_heads=LITE_DECODER_HEADS,
            conf_threshold=NOVELTY_CONF_THRESHOLD,
            pretrained_backbone=True,
        )

        if os.path.isfile(LITE_WEIGHTS_PATH):
            state = torch.load(LITE_WEIGHTS_PATH, map_location=_device, weights_only=True)
            _lite_model.load_state_dict(state)
            logger.info("Loaded lite detector weights from %s", LITE_WEIGHTS_PATH)
        else:
            logger.warning(
                "No lite weights found at %s — backbone uses ImageNet pretraining only.",
                LITE_WEIGHTS_PATH,
            )

        _lite_model.to(_device).eval()

    return _lite_model, _device, _taxonomy_labels


# ── Pre / post-processing ─────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size))
    tensor      = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    tensor      = (tensor - _MEAN) / _STD
    return tensor.unsqueeze(0).to(device)


def _postprocess(
    outputs: dict,
    orig_h: int,
    orig_w: int,
    taxonomy_labels: dict[str, list[str]],
    conf_thresh: float,
) -> list[dict]:
    boxes_norm   = outputs["boxes"][0]
    conf_logits  = outputs["confidence"][0]
    is_novel     = outputs["is_novel"][0]
    novelty_sc   = outputs["novelty_scores"][0]
    class_logits = {k: v[0] for k, v in outputs["class_logits"].items()}

    conf = conf_logits.sigmoid()
    keep = conf > conf_thresh
    if not keep.any():
        return []

    boxes_norm   = boxes_norm[keep]
    conf         = conf[keep]
    is_novel     = is_novel[keep]
    novelty_sc   = novelty_sc[keep]
    class_logits = {k: v[keep] for k, v in class_logits.items()}

    cx, cy, w, h = boxes_norm.unbind(-1)
    boxes_xyxy   = torch.stack([
        (cx - w / 2) * orig_w,
        (cy - h / 2) * orig_h,
        (cx + w / 2) * orig_w,
        (cy + h / 2) * orig_h,
    ], dim=-1)

    nms_idx    = nms(boxes_xyxy, conf, iou_threshold=0.5)
    detections = []
    for i in nms_idx.tolist():
        bx       = boxes_xyxy[i].tolist()
        taxonomy = {
            level: taxonomy_labels[level][int(logits[i].argmax().item())]
            for level, logits in class_logits.items()
        }
        novel_flag = bool(is_novel[i].item())
        detection  = {
            "detection_id":  str(uuid.uuid4()),
            "bbox":          {"x1": bx[0], "y1": bx[1], "x2": bx[2], "y2": bx[3]},
            "confidence":    round(float(conf[i].item()), 4),
            "taxonomy":      taxonomy,
            "is_novel":      novel_flag,
            "novelty_score": round(float(novelty_sc[i].item()), 4),
        }
        if novel_flag:
            detection["closest_known_species"] = taxonomy.get("species", "unknown")
            detection["taxonomy"]["species"]   = "novel_species"
        detections.append(detection)

    return detections


# ── Public API ────────────────────────────────────────────────────────────────

def run_inference(image_bytes: bytes) -> dict:
    """Run the detector on a single image (bytes).

    Selects SeabedLite or SeabedDetector based on the USE_LITE_MODEL env var.

    Returns:
        {"frame_id": str, "detections": [{detection_id, bbox, confidence,
                                           taxonomy, is_novel, novelty_score, ...}]}
    """
    if USE_LITE_MODEL:
        model, device, taxonomy_labels = get_lite_model()
        input_size = LITE_INPUT_SIZE
    else:
        model, device, taxonomy_labels = get_model()
        input_size = MODEL_INPUT_SIZE

    img_bgr        = decode_image(image_bytes)
    orig_h, orig_w = img_bgr.shape[:2]
    tensor         = _preprocess(img_bgr, input_size, device)

    with torch.no_grad():
        outputs = model(tensor)

    detections = _postprocess(outputs, orig_h, orig_w, taxonomy_labels, INFER_CONF_THRESHOLD)
    return {"frame_id": str(uuid.uuid4()), "detections": detections}
