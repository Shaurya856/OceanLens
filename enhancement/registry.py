from enhancement.techniques import (
    apply_clahe,
    apply_denoise,
    apply_dehaze,
    apply_gamma_correction,
    apply_retinex,
    apply_superres,
    apply_white_balance,
)

TECHNIQUES: dict[str, callable] = {
    "clahe":             apply_clahe,
    "denoise":           apply_denoise,
    "dehaze":            apply_dehaze,
    "gamma_correction":  apply_gamma_correction,
    "retinex":           apply_retinex,
    "superres":          apply_superres,
    "white_balance":     apply_white_balance,
}
