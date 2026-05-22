import logging

from core.retry import retry
from enhancement.registry import TECHNIQUES

logger = logging.getLogger(__name__)


def run_single(image: bytes, technique: str, params: dict[str, dict]) -> bytes:
    """Apply a single named technique to image bytes."""
    def step() -> bytes:
        return TECHNIQUES[technique](image, params.get(technique, {}))
    result, attempts = retry(step)
    if attempts > 1:
        logger.warning("Technique %s succeeded after %d attempts", technique, attempts)
    return result


def run_custom(image: bytes, techniques: list[str], params: dict[str, dict]) -> bytes:
    """Chain multiple techniques sequentially, passing output of each as input to the next."""
    for tech in techniques:
        result, attempts = retry(lambda t=tech, img=image: TECHNIQUES[t](img, params.get(t, {})))
        if attempts > 1:
            logger.warning("Technique %s succeeded after %d attempts", tech, attempts)
        image = result
    return image
