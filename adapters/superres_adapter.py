import superres

def run(image: bytes, config: dict) -> bytes:
    return superres.run(image, **config)
