"""Multi-Scale Retinex (MSR) for image enhancement."""

import cv2
import numpy as np
from utils import decode_image, encode_image


def _single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    """Single Scale Retinex: R = log(I) - log(G(I))."""
    img = img.astype(np.float64) + 1e-6
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.log10(img) - np.log10(blur)


def run(image: bytes, **kwargs) -> bytes:
    """Apply Multi-Scale Retinex with color restoration.

    Kwargs:
        sigmas: List of sigma values for Gaussian scales (default: [15, 80, 250]).
        alpha: Color restoration factor (default: 125).
    """
    sigmas = kwargs.get("sigmas", [15, 80, 250])
    if isinstance(sigmas, (int, float)):
        sigmas = [float(sigmas)]
    else:
        sigmas = [float(s) for s in sigmas][:5]
    alpha = float(kwargs.get("alpha", 125))

    img = decode_image(image)
    img_float = img.astype(np.float64) / 255.0 + 1e-6

    msr = np.zeros_like(img_float)
    for sigma in sigmas:
        msr += _single_scale_retinex(img_float, sigma)
    msr /= len(sigmas)

    # Color restoration
    intensity = np.sum(img_float, axis=2, keepdims=True) + 1e-6
    cr = np.log10(alpha * img_float) - np.log10(intensity)

    # MSRCR formula
    gain = 1
    offset = 0
    result = gain * (msr * cr + offset)

    # Normalize to 0-255
    for i in range(3):
        ch = result[:, :, i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
        result[:, :, i] = ch

    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return encode_image(result)
