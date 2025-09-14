# itpu/utils/windowed.py
import numpy as np
from itpu.sdk import ITPU

def windowed_mi(x, y, window_size=2000, hop_size=400, bins=64, method="hist", **kwargs):
    """
    Compute sliding-window MI across x,y.
    Returns (starts, mi_vals) where 'starts' are window start indices.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, "x and y must be same length"
    n = len(x)
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive integers")
    if window_size > n:
        raise ValueError("window_size cannot exceed signal length")

    itpu = ITPU(device="software")
    starts = list(range(0, n - window_size + 1, hop_size))
    mi_vals = np.empty(len(starts), dtype=float)

    for i, s in enumerate(starts):
        e = s + window_size
        mi_vals[i] = itpu.mutual_info(x[s:e], y[s:e], method=method, bins=bins, **kwargs)

    return np.array(starts, dtype=int), mi_vals
