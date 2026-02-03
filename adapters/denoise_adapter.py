import denoise

def run(image: bytes, config: dict) -> bytes:
    return denoise.run(image, **config)
