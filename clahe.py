"""CLAHE (Contrast Limited Adaptive Histogram Equalization)."""

import cv2
from utils import decode_image, encode_image


def run(image: bytes, **kwargs) -> bytes:
    """Apply CLAHE in LAB color space for natural-looking enhancement.

    Kwargs:
        clip_limit: Contrast limiting (default: 2.0).
        tile_size: Grid size for adaptive regions (default: 8).
    """
    clip_limit = float(kwargs.get("clip_limit", 2.0))
    tile_size = int(kwargs.get("tile_size", 8))
    tile_size = max(2, min(tile_size, 64))

    img = decode_image(image)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)

    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    return encode_image(result)
