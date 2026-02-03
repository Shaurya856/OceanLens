import time
from config import MAX_RETRIES

def retry(fn):
    attempt = 0
    while True:
        try:
            return fn(), attempt + 1
        except Exception:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            time.sleep(0.5 * attempt)
