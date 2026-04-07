"""Dehazing via Dark Channel Prior (DCP)."""

import cv2
import numpy as np
from utils import decode_image, encode_image


def _dark_channel(img: np.ndarray, radius: int = 15) -> np.ndarray:
    """Compute dark channel: min over RGB, then min filter."""
    patch = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
    dark = np.min(img, axis=2)
    return cv2.erode(dark, patch)


def run(image: bytes, **kwargs) -> bytes:
    """Remove haze using Dark Channel Prior.

    Kwargs:
        omega: Keep a small amount of haze (default: 0.95).
        t0: Minimum transmission (default: 0.1).
        radius: Patch size for dark channel (default: 15).
    """
    omega = float(kwargs.get("omega", 0.95))
    t0 = float(kwargs.get("t0", 0.1))
    radius = int(kwargs.get("radius", 15))
    radius = max(1, min(radius, 31) | 1)

    img = decode_image(image).astype(np.float64) / 255.0

    dark = _dark_channel(img, radius)
    # Estimate atmospheric light
    top_percent = int(img.shape[0] * img.shape[1] * 0.001)
    flat_dark = dark.ravel()
    flat_img = img.reshape(-1, 3)
    idx = np.argpartition(flat_dark, -top_percent)[-top_percent:]
    A = np.mean(flat_img[idx], axis=0)
    A = np.clip(A, 0.05, 1.0)

    # Transmission
    normalized = img / (A + 1e-6)
    t = 1.0 - omega * _dark_channel(normalized, radius)
    t = np.maximum(t, t0)

    # Recover scene
    t_3 = np.stack([t, t, t], axis=2)
    J = (img - A) / (t_3 + 1e-6) + A
    J = np.clip(J, 0, 1)
    result = (J * 255).astype(np.uint8)

    return encode_image(result)
