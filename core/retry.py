import time
from typing import Callable, TypeVar
from core.config import MAX_RETRIES

T = TypeVar("T")

def retry(fn: Callable[[], T]) -> tuple[T, int]:
    attempt = 0
    while True:
        try:
            return fn(), attempt + 1
        except Exception:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            time.sleep(0.5 * attempt)
