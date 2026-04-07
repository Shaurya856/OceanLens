import os
import uuid
import cv2
import numpy as np


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
    name, ext = os.path.splitext(filename)
    return f"{name}_enhanced{ext}"
