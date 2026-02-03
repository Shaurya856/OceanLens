import white_balance

def run(image: bytes, config: dict) -> bytes:
    return white_balance.run(image, **config)
