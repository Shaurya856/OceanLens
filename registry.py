from adapters.denoise_adapter import run as denoise_run
from adapters.white_balance_adapter import run as wb_run
from adapters.gamma_correction_adapter import run as gamma_run
from adapters.clahe_adapter import run as clahe_run
from adapters.retinex_adapter import run as retinex_run
from adapters.dehaze_adapter import run as dehaze_run
from adapters.superres_adapter import run as superres_run

TECHNIQUES = {
    "denoise": denoise_run,
    "white_balance": wb_run,
    "gamma_correction": gamma_run,
    "clahe": clahe_run,
    "retinex": retinex_run,
    "dehaze": dehaze_run,
    "superres": superres_run,
}
