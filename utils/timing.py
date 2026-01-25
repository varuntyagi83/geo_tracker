import time
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    ms = int((end - start) * 1000)
    setattr(timer, "last_ms", ms)
