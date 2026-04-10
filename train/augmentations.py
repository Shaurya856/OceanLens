"""
Underwater-specific augmentations for seabed imagery.

All transforms operate on:
    img  : np.ndarray  BGR uint8  [H, W, 3]
    boxes: np.ndarray  float32    [N, 4]  (x, y, w, h) pixel space

and return the same types so they can be chained arbitrarily.

Underwater-specific transforms address:
  - Depth-dependent colour absorption (red fades fastest)
  - Water turbidity (blur)
  - Low-light sensor noise
  - ROV/AUV lighting variance (brightness drops)
"""
import random

import cv2
import numpy as np


# ── Individual transforms ─────────────────────────────────────────────────────

def random_hflip(
    img: np.ndarray, boxes: np.ndarray, p: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    if random.random() > p:
        return img, boxes
    img = cv2.flip(img, 1)
    if boxes.shape[0]:
        w = img.shape[1]
        boxes = boxes.copy()
        boxes[:, 0] = w - boxes[:, 0] - boxes[:, 2]  # x = W - x - bw
    return img, boxes


def random_vflip(
    img: np.ndarray, boxes: np.ndarray, p: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Vertical flip — valid for benthic surveys where orientation may vary."""
    if random.random() > p:
        return img, boxes
    img = cv2.flip(img, 0)
    if boxes.shape[0]:
        h = img.shape[0]
        boxes = boxes.copy()
        boxes[:, 1] = h - boxes[:, 1] - boxes[:, 3]  # y = H - y - bh
    return img, boxes


def red_channel_suppression(
    img: np.ndarray,
    max_suppress: float = 0.4,
    p: float = 0.6,
) -> np.ndarray:
    """
    Simulate depth-dependent colour absorption.
    Red is absorbed first (~5 m), then orange/yellow, then green.
    Randomly scale the red and green channels downward.
    """
    if random.random() > p:
        return img
    img = img.astype(np.float32)
    # OpenCV stores BGR — channel 2 is Red, channel 1 is Green
    red_factor   = 1.0 - random.uniform(0.0, max_suppress)
    green_factor = 1.0 - random.uniform(0.0, max_suppress * 0.5)
    img[:, :, 2] *= red_factor
    img[:, :, 1] *= green_factor
    return np.clip(img, 0, 255).astype(np.uint8)


def random_blur(
    img: np.ndarray, max_sigma: float = 2.0, p: float = 0.5
) -> np.ndarray:
    """Gaussian blur to simulate water turbidity / particle scatter."""
    if random.random() > p:
        return img
    sigma = random.uniform(0.3, max_sigma)
    ksize = int(sigma * 3) * 2 + 1  # odd kernel
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def random_gaussian_noise(
    img: np.ndarray, max_std: float = 15.0, p: float = 0.4
) -> np.ndarray:
    """Additive Gaussian noise for low-light sensor simulation."""
    if random.random() > p:
        return img
    std  = random.uniform(2.0, max_std)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_brightness_drop(
    img: np.ndarray, min_factor: float = 0.5, p: float = 0.4
) -> np.ndarray:
    """Random brightness reduction to simulate uneven ROV lighting."""
    if random.random() > p:
        return img
    factor = random.uniform(min_factor, 1.0)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def color_jitter(
    img: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    p: float = 0.5,
) -> np.ndarray:
    if random.random() > p:
        return img
    img = img.astype(np.float32)

    # Brightness
    b = random.uniform(1 - brightness, 1 + brightness)
    img = img * b

    # Contrast (scale around mean)
    c = random.uniform(1 - contrast, 1 + contrast)
    mean = img.mean()
    img = (img - mean) * c + mean

    # Saturation (shift toward grey)
    s = random.uniform(1 - saturation, 1 + saturation)
    grey = img.mean(axis=2, keepdims=True)
    img  = grey + s * (img - grey)

    return np.clip(img, 0, 255).astype(np.uint8)


def random_scale_crop(
    img: np.ndarray,
    boxes: np.ndarray,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
    p: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random scale + crop back to original size.
    Boxes outside the crop region are discarded; partially overlapping
    boxes are clipped.
    """
    if random.random() > p or boxes.shape[0] == 0:
        return img, boxes

    h, w = img.shape[:2]
    scale = random.uniform(min_scale, max_scale)
    new_w, new_h = int(w * scale), int(h * scale)
    img_scaled = cv2.resize(img, (new_w, new_h))

    # Random crop origin within the scaled image
    x0 = random.randint(0, max(0, new_w - w))
    y0 = random.randint(0, max(0, new_h - h))
    x1, y1 = min(x0 + w, new_w), min(y0 + h, new_h)
    crop = img_scaled[y0:y1, x0:x1]
    # Pad back to (h, w) if smaller
    pad_h, pad_w = h - crop.shape[0], w - crop.shape[1]
    if pad_h > 0 or pad_w > 0:
        crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    # Adjust boxes
    if boxes.shape[0]:
        boxes = boxes.copy().astype(np.float32)
        boxes[:, 0] = boxes[:, 0] * scale - x0
        boxes[:, 1] = boxes[:, 1] * scale - y0
        boxes[:, 2] *= scale
        boxes[:, 3] *= scale

        # Clip to crop region
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - boxes[:, 0])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - boxes[:, 1])

        # Remove boxes that shrank to near-zero
        keep = (boxes[:, 2] > 2) & (boxes[:, 3] > 2)
        boxes = boxes[keep]

    return crop, boxes


# ── Composed pipeline ─────────────────────────────────────────────────────────

class UnderwaterAugmentation:
    """
    Full augmentation pipeline for seabed detection training.
    Operates on BGR uint8 image + pixel-space (x,y,w,h) boxes.
    """

    def __call__(
        self, img: np.ndarray, boxes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        img, boxes = random_hflip(img, boxes)
        img, boxes = random_vflip(img, boxes)
        img, boxes = random_scale_crop(img, boxes)
        img        = red_channel_suppression(img)
        img        = random_blur(img)
        img        = random_gaussian_noise(img)
        img        = random_brightness_drop(img)
        img        = color_jitter(img)
        return img, boxes
