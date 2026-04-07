"""White balance via Gray World algorithm."""

import cv2
import numpy as np
from utils import decode_image, encode_image


def run(image: bytes, **kwargs) -> bytes:
    """Apply Gray World white balance.

    Scales each channel so the average is neutral gray (128).
    Kwargs:
        percent: Clip extreme values before computing mean (default: 1).
                 Higher values clip more and can reduce color cast.
    """
    percent = float(kwargs.get("percent", 1))
    percent = max(0, min(percent, 50))

    img = decode_image(image).astype(np.float64)
    if percent > 0:
        low = np.percentile(img, percent)
        high = np.percentile(img, 100 - percent)
        img = np.clip(img, low, high)

    avg = np.mean(img, axis=(0, 1))
    gray = 128.0
    scale = gray / (avg + 1e-6)
    result = img * scale
    result = np.clip(result, 0, 255).astype(np.uint8)

    return encode_image(result)
