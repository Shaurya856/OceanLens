from core.retry import retry
from enhancement.registry import TECHNIQUES


def run_single(image: bytes, technique: str, params: dict[str, dict]) -> bytes:
    """Apply a single named technique to image bytes."""
    def step() -> bytes:
        return TECHNIQUES[technique](image, params.get(technique, {}))
    return retry(step)[0]


def run_custom(image: bytes, techniques: list[str], params: dict[str, dict]) -> bytes:
    """Chain multiple techniques sequentially, passing output of each as input to the next."""
    for tech in techniques:
        def step(t: str = tech) -> bytes:
            return TECHNIQUES[t](image, params.get(t, {}))
        image, _ = retry(step)
    return image
