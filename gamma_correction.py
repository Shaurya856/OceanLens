"""Gamma correction for exposure/contrast adjustment."""

import cv2
import numpy as np
from utils import decode_image, encode_image


def run(image: bytes, **kwargs) -> bytes:
    """Apply gamma correction: I_out = I_in ^ gamma.

    Kwargs:
        gamma: Gamma value. <1 brightens, >1 darkens (default: 1.2).
    """
    gamma = float(kwargs.get("gamma", 1.2))
    gamma = max(0.1, min(gamma, 5.0))

    img = decode_image(image).astype(np.float64) / 255.0
    result = np.power(img, 1.0 / gamma)
    result = (np.clip(result, 0, 1) * 255).astype(np.uint8)

    return encode_image(result)
