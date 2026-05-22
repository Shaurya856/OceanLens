import os
import uuid
import cv2
import numpy as np
import torch


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes to BGR array."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def encode_image(img: np.ndarray, ext: str = ".png") -> bytes:
    """Encode BGR array to image bytes."""
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise ValueError("Could not encode image")
    return buf.tobytes()


def generate_image_id() -> str:
    return str(uuid.uuid4())


def build_enhanced_filename(filename: str) -> str:
    name, _ = os.path.splitext(filename)
    return f"{name}_enhanced.png"


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) → (x1, y1, x2, y2) — works for both normalised and pixel coords."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
