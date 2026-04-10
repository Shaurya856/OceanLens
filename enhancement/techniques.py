"""
Image enhancement techniques.

Each function accepts raw image bytes and a config dict, and returns
the processed image as bytes.  The config dict keys match the kwarg
names documented below; missing keys fall back to the listed defaults.
"""

import cv2
import numpy as np
from core.utils import decode_image, encode_image


# ── CLAHE ─────────────────────────────────────────────────────────────────────

def apply_clahe(image: bytes, config: dict = {}) -> bytes:
    """Contrast Limited Adaptive Histogram Equalization in LAB color space.

    Config keys:
        clip_limit: Contrast limiting threshold (default: 2.0).
        tile_size:  Grid size for adaptive regions (default: 8).
    """
    clip_limit = float(config.get("clip_limit", 2.0))
    tile_size  = int(config.get("tile_size", 8))
    tile_size  = max(2, min(tile_size, 64))

    img = decode_image(image)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe_obj.apply(l)

    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return encode_image(result)


# ── Denoise ───────────────────────────────────────────────────────────────────

def apply_denoise(image: bytes, config: dict = {}) -> bytes:
    """Non-Local Means denoising (colored).

    Config keys:
        h:               Luminance filter strength (default: 10).
        h_for_color:     Color filter strength (default: 10).
        template_window: Patch size for comparison (default: 7, must be odd).
        search_window:   Search window size (default: 21, must be odd).
    """
    h               = float(config.get("h", 10))
    h_for_color     = float(config.get("h_for_color", 10))
    template_window = int(config.get("template_window", 7))
    search_window   = int(config.get("search_window", 21))
    template_window = template_window if template_window % 2 == 1 else template_window + 1
    search_window   = search_window   if search_window   % 2 == 1 else search_window   + 1

    img    = decode_image(image)
    result = cv2.fastNlMeansDenoisingColored(img, None, h, h_for_color, template_window, search_window)
    return encode_image(result)


# ── Dehaze ────────────────────────────────────────────────────────────────────

def _dark_channel(img: np.ndarray, radius: int) -> np.ndarray:
    """Min over RGB channels, then minimum filter over a patch."""
    patch = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
    dark  = np.min(img, axis=2)
    return cv2.erode(dark, patch)


def apply_dehaze(image: bytes, config: dict = {}) -> bytes:
    """Dark Channel Prior dehazing.

    Config keys:
        omega:  Haze removal strength — keep a small residual haze (default: 0.95).
        t0:     Minimum transmission floor (default: 0.1).
        radius: Patch size for dark channel computation (default: 15).
    """
    omega  = float(config.get("omega", 0.95))
    t0     = float(config.get("t0", 0.1))
    radius = int(config.get("radius", 15))
    radius = max(1, min(radius, 31) | 1)

    img  = decode_image(image).astype(np.float64) / 255.0
    dark = _dark_channel(img, radius)

    # Estimate atmospheric light from the brightest dark-channel pixels
    top_n   = int(img.shape[0] * img.shape[1] * 0.001)
    flat_dark = dark.ravel()
    flat_img  = img.reshape(-1, 3)
    idx = np.argpartition(flat_dark, -top_n)[-top_n:]
    A   = np.clip(np.mean(flat_img[idx], axis=0), 0.05, 1.0)

    # Transmission map
    t   = np.maximum(1.0 - omega * _dark_channel(img / (A + 1e-6), radius), t0)
    t_3 = np.stack([t, t, t], axis=2)

    J = np.clip((img - A) / (t_3 + 1e-6) + A, 0, 1)
    return encode_image((J * 255).astype(np.uint8))


# ── Gamma correction ──────────────────────────────────────────────────────────

def apply_gamma_correction(image: bytes, config: dict = {}) -> bytes:
    """Power-law gamma correction: I_out = I_in ^ (1 / gamma).

    Config keys:
        gamma: Gamma value — <1 brightens, >1 darkens (default: 1.2).
    """
    gamma = float(config.get("gamma", 1.2))
    gamma = max(0.1, min(gamma, 5.0))

    img    = decode_image(image).astype(np.float64) / 255.0
    result = (np.clip(np.power(img, 1.0 / gamma), 0, 1) * 255).astype(np.uint8)
    return encode_image(result)


# ── Retinex ───────────────────────────────────────────────────────────────────

def _single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    """Single Scale Retinex: R = log(I) - log(GaussianBlur(I))."""
    img  = img.astype(np.float64) + 1e-6
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.log10(img) - np.log10(blur)


def apply_retinex(image: bytes, config: dict = {}) -> bytes:
    """Multi-Scale Retinex with Color Restoration (MSRCR).

    Config keys:
        sigmas: Gaussian sigma values for each scale (default: [15, 80, 250]).
        alpha:  Color restoration factor (default: 125).
    """
    sigmas = config.get("sigmas", [15, 80, 250])
    if isinstance(sigmas, (int, float)):
        sigmas = [float(sigmas)]
    else:
        sigmas = [float(s) for s in sigmas][:5]
    alpha = float(config.get("alpha", 125))

    img       = decode_image(image)
    img_float = img.astype(np.float64) / 255.0 + 1e-6

    msr = sum(_single_scale_retinex(img_float, s) for s in sigmas) / len(sigmas)

    intensity = np.sum(img_float, axis=2, keepdims=True) + 1e-6
    cr        = np.log10(alpha * img_float) - np.log10(intensity)
    result    = msr * cr

    for i in range(3):
        ch = result[:, :, i]
        result[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)

    return encode_image(np.clip(result * 255, 0, 255).astype(np.uint8))


# ── Super-resolution ──────────────────────────────────────────────────────────

def apply_superres(image: bytes, config: dict = {}) -> bytes:
    """Bicubic upscaling (super-resolution via interpolation).

    Config keys:
        scale: Upscaling factor between 1.0 and 4.0 (default: 2).
    """
    scale = max(1.0, min(float(config.get("scale", 2)), 4.0))

    img  = decode_image(image)
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return encode_image(upscaled)


# ── White balance ─────────────────────────────────────────────────────────────

def apply_white_balance(image: bytes, config: dict = {}) -> bytes:
    """Gray World white balance — scales channels so the mean is neutral gray.

    Config keys:
        percent: Percentile to clip before computing the channel mean (default: 1).
                 Higher values reduce influence of extreme highlights/shadows.
    """
    percent = max(0.0, min(float(config.get("percent", 1)), 50.0))

    img = decode_image(image).astype(np.float64)
    if percent > 0:
        low  = np.percentile(img, percent)
        high = np.percentile(img, 100 - percent)
        img  = np.clip(img, low, high)

    avg    = np.mean(img, axis=(0, 1))
    result = np.clip(img * (128.0 / (avg + 1e-6)), 0, 255).astype(np.uint8)
    return encode_image(result)
