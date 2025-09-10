import numpy as np
from .ksg import mi_ksg

def windowed_ksg_mi(
    x, y, window_size: int, hop_size: int,
    k: int = 5, metric: str = "chebyshev",
    mask=None, jitter: float = 0.0, seed: int | None = None
):
    """Sliding-window KSG MI (stateless). Returns (t_idx, mi_vals)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive")
    if hop_size > window_size:
        raise ValueError("hop_size cannot exceed window_size")

    n = x.size
    starts = np.arange(0, max(n - window_size + 1, 0), hop_size, dtype=int)
    t_idx = starts + window_size - 1
    mi_vals = np.empty_like(t_idx, dtype=np.float64)

    for i, s in enumerate(starts):
        e = s + window_size
        mi_vals[i] = mi_ksg(
            x[s:e], y[s:e], k=k, metric=metric,
            mask=None if mask is None else np.asarray(mask[s:e], dtype=bool),
            jitter=jitter, seed=seed
        )
    return t_idx, mi_vals
