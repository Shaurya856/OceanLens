"""Super-resolution via bicubic interpolation upscaling."""

import cv2
from utils import decode_image, encode_image


def run(image: bytes, **kwargs) -> bytes:
    """Upscale image using bicubic interpolation.

    Kwargs:
        scale: Upscaling factor (default: 2).
    """
    scale = float(kwargs.get("scale", 2))
    scale = max(1.0, min(scale, 4.0))

    img = decode_image(image)
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return encode_image(upscaled)
