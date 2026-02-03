from registry import TECHNIQUES

def validate_request(mode, techniques, params):
    if mode not in {"single", "custom"}:
        raise ValueError("mode must be 'single' or 'custom'")

    if not isinstance(techniques, list) or not techniques:
        raise ValueError("techniques must be a non-empty list")

    for t in techniques:
        if t not in TECHNIQUES:
            raise ValueError(f"Unknown technique: {t}")

    if not isinstance(params, dict):
        raise ValueError("params must be an object")

    for k, v in params.items():
        if not isinstance(v, dict):
            raise ValueError(f"Params for {k} must be an object")
