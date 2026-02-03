import dehaze

def run(image: bytes, config: dict) -> bytes:
    return dehaze.run(image, **config)
