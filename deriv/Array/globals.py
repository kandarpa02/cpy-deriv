GRAD_MODE = True

from contextlib import contextmanager

@contextmanager
def no_grad():
    global GRAD_MODE
    prev = GRAD_MODE
    GRAD_MODE = False
    try:
        yield
    finally:
        GRAD_MODE = prev