# itpu/utils/windowed.py
import numpy as np
from ..sdk import ITPU

def windowed_mi(
    x, y,
    window_size: int = 1000,
    hop_size: int = 200,
    method: str = "hist",
    bins: int = 128,
    k: int = 5
):
    """
    Sliding-window MI over x,y.
    Returns (starts, mi_values), where starts are sample indices for each window start.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have same length")
    if window_size <= 0 or hop_size <= 0 or hop_size > window_size:
        raise ValueError("window_size > 0, hop_size > 0, and hop_size <= window_size required")

    itpu = ITPU(device="software")  # swap to "fpga" later, same API
    starts = []
    mis = []

    n = len(x)
    for start in range(0, n - window_size + 1, hop_size):
        end = start + window_size
        xi = x[start:end]
        yi = y[start:end]
        if method == "hist":
            mi = itpu.mutual_info(xi, yi, method="hist", bins=bins)
        elif method == "ksg":
            mi = itpu.mutual_info(xi, yi, method="ksg", k=k)
        else:
            raise ValueError("method must be 'hist' or 'ksg'")
        starts.append(start)
        mis.append(mi)

    return np.array(starts), np.array(mis)
