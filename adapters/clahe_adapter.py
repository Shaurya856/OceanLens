import clahe

def run(image: bytes, config: dict) -> bytes:
    return clahe.run(image, **config)
