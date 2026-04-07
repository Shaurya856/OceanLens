"""Image denoising via Non-Local Means (colored)."""

import cv2
from utils import decode_image, encode_image


def run(image: bytes, **kwargs) -> bytes:
    """Denoise image using cv2.fastNlMeansDenoisingColored.

    Kwargs:
        h: Filter strength for luminance (default: 10).
        h_for_color: Filter strength for color (default: 10).
        template_window: Size of patch used for comparison (default: 7).
        search_window: Size of search window (default: 21).
    """
    h = float(kwargs.get("h", 10))
    h_for_color = float(kwargs.get("h_for_color", 10))
    template_window = int(kwargs.get("template_window", 7))
    search_window = int(kwargs.get("search_window", 21))
    template_window = template_window if template_window % 2 == 1 else template_window + 1
    search_window = search_window if search_window % 2 == 1 else search_window + 1

    img = decode_image(image)
    result = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h,
        h_for_color,
        template_window,
        search_window,
    )
    return encode_image(result)
