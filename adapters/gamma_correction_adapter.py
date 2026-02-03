import gamma_correction

def run(image: bytes, config: dict) -> bytes:
    gamma = config.get("gamma", 1.0)
    return gamma_correction.run(image, gamma=gamma)
