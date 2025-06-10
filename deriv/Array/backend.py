_backend = None

def set_backend(name: str):
    global _backend
    if name == "cpu":
        import numpy as np
        _backend = np
    elif name == "cuda":
        try:
            import cupy as cp
            _backend = cp
        except ImportError:
            raise RuntimeError("CuPy is not installed. Install it with `pip install cupy`.")
    else:
        raise ValueError(f"Unknown backend '{name}'. Use 'cpu' or 'cuda'.")

def get_backend():
    if _backend is None:
        raise RuntimeError("Backend not set. Please call `set_backend('cpu')` or `set_backend('cuda')` first.")
    return _backend

def is_gpu():
    return get_backend().__name__ == "cupy"
