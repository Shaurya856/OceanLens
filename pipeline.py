from retry import retry
from registry import TECHNIQUES

def run_single(image: bytes, technique: str, params: dict) -> bytes:
    def step():
        return TECHNIQUES[technique](image, params.get(technique, {}))
    return retry(step)[0]

def run_custom(image: bytes, techniques: list[str], params: dict) -> bytes:
    for tech in techniques:
        def step():
            return TECHNIQUES[tech](image, params.get(tech, {}))
        image, _ = retry(step)
    return image
