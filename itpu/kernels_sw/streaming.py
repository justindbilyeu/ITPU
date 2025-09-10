import numpy as np
from .hist import mi_hist

def windowed_mi(x, y, window_size: int, hop_size: int, bins: int = 128, mask=None):
    """Sliding-window histogram MI. Returns (t_idx, mi_vals)."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    if window_size <= 0 or hop_size <= 0 or hop_size > window_size:
        raise ValueError("window_size>0, hop_size>0, hop_size<=window_size")
    n = x.size
    starts = np.arange(0, max(n - window_size + 1, 0), hop_size, dtype=int)
    t_idx = starts + window_size - 1
    mi_vals = np.empty_like(t_idx, dtype=np.float64)
    for i, s in enumerate(starts):
        e = s + window_size
        valid = slice(s, e) if mask is None else (mask[s:e] & ~(np.isnan(x[s:e]) | np.isnan(y[s:e])))
        mi_vals[i] = mi_hist(x[s:e][valid], y[s:e][valid], bins=bins)
    return t_idx, mi_vals
