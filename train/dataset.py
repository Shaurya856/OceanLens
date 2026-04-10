"""
SeabedDataset — loads images and bounding-box annotations from a single
JSON file in the project's annotation format.

Annotation JSON schema
──────────────────────
{
  "taxonomy_labels": {          ← vocabulary for every taxonomy level
    "phylum":  ["Chordata", ...],
    "class_":  ["Actinopterygii", ...],
    "order":   [...],
    "family":  [...],
    "species": ["Hippocampus kuda", ...]
  },
  "images": [
    {"id": 1, "file_name": "frame_000001.png", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "bbox": [x, y, w, h],          ← pixel-space, top-left origin
      "taxonomy": {
        "phylum":  "Chordata",
        "class_":  "Actinopterygii",
        "order":   "Syngnathiformes",
        "family":  "Syngnathidae",
        "species": "Hippocampus kuda"
      }
    }
  ]
}

__getitem__ returns
───────────────────
  image  : torch.Tensor [3, H, W]  normalised RGB
  target : {
    "boxes"  : Tensor [N, 4]  (cx, cy, w, h) normalised to [0, 1]
    "labels" : { level: Tensor [N] int64 }
    "image_id": int
  }
"""
import json
import os
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.config import MODEL_INPUT_SIZE, TAXONOMY_LEVELS, TAXONOMY_LABELS_PATH

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SeabedDataset(Dataset):
    def __init__(
        self,
        annotation_path: str,
        image_dir: str,
        input_size: int = MODEL_INPUT_SIZE,
        transforms: Callable | None = None,
    ) -> None:
        with open(annotation_path) as f:
            data = json.load(f)

        self.image_dir  = image_dir
        self.input_size = input_size
        self.transforms = transforms

        # Build label → index mappings from the annotation file's vocabulary
        self.taxonomy_labels: dict[str, list[str]] = data["taxonomy_labels"]
        self.label2idx: dict[str, dict[str, int]] = {
            level: {name: i for i, name in enumerate(names)}
            for level, names in self.taxonomy_labels.items()
        }

        # Persist vocabulary so inference_runner can load it
        os.makedirs(os.path.dirname(TAXONOMY_LABELS_PATH) or ".", exist_ok=True)
        with open(TAXONOMY_LABELS_PATH, "w") as f:
            json.dump(self.taxonomy_labels, f, indent=2)

        # Index images
        self.images: dict[int, dict] = {img["id"]: img for img in data["images"]}
        self.image_ids: list[int] = [img["id"] for img in data["images"]]

        # Group annotations by image
        self.annotations: dict[int, list[dict]] = {img_id: [] for img_id in self.image_ids}
        for ann in data["annotations"]:
            self.annotations[ann["image_id"]].append(ann)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_id = self.image_ids[idx]
        meta   = self.images[img_id]

        # Load image
        path   = os.path.join(self.image_dir, meta["file_name"])
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {path}")

        orig_h, orig_w = img_bgr.shape[:2]
        anns = self.annotations[img_id]

        # Build raw boxes [N, 4] (x,y,w,h) pixel space
        boxes_px = np.array(
            [a["bbox"] for a in anns], dtype=np.float32
        ).reshape(-1, 4) if anns else np.zeros((0, 4), dtype=np.float32)

        # Apply augmentations (they receive BGR + pixel boxes)
        if self.transforms is not None:
            img_bgr, boxes_px = self.transforms(img_bgr, boxes_px)

        # Resize to model input
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res   = cv2.resize(img_rgb, (self.input_size, self.input_size))
        sx = self.input_size / orig_w
        sy = self.input_size / orig_h
        if boxes_px.shape[0]:
            boxes_px[:, 0] *= sx
            boxes_px[:, 2] *= sx
            boxes_px[:, 1] *= sy
            boxes_px[:, 3] *= sy

        # Convert (x,y,w,h) → (cx,cy,w,h) normalised
        boxes_norm = _xywhpx_to_cxcywh_norm(boxes_px, self.input_size, self.input_size)

        # Encode taxonomy labels
        labels: dict[str, torch.Tensor] = {}
        for level in TAXONOMY_LEVELS:
            indices = [
                self.label2idx[level][a["taxonomy"][level]]
                for a in anns
            ] if anns else []
            labels[level] = torch.tensor(indices, dtype=torch.long)

        # Normalise image tensor
        tensor = torch.from_numpy(img_res).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - _MEAN) / _STD

        target = {
            "boxes":    torch.from_numpy(boxes_norm),  # [N, 4]
            "labels":   labels,
            "image_id": img_id,
        }
        return tensor, target


# ── Box utilities ─────────────────────────────────────────────────────────────

def _xywhpx_to_cxcywh_norm(
    boxes: np.ndarray, img_w: int, img_h: int
) -> np.ndarray:
    """
    (x, y, w, h) pixel-space  →  (cx, cy, w, h) normalised to [0, 1].
    Returns zero-row array if input is empty.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes.copy()
    out[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / img_w   # cx
    out[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / img_h   # cy
    out[:, 2] = boxes[:, 2] / img_w                        # w
    out[:, 3] = boxes[:, 3] / img_h                        # h
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(
    batch: list[tuple[torch.Tensor, dict]]
) -> tuple[torch.Tensor, list[dict]]:
    """
    Stack images into [B, 3, H, W]; keep targets as a list (variable N per image).
    """
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
