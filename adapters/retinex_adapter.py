import retinex

def run(image: bytes, config: dict) -> bytes:
    return retinex.run(image, **config)
