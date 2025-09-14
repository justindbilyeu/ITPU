# itpu/utils/windowed.py
import numpy as np
from itpu.sdk import ITPU


def windowed_mi(
    x,
    y,
    window_size: int = 2000,
    step: int = 400,
    method: str = "hist",
    **kwargs,
):
    """Sliding-window mutual information.

    Returns ``(mi_vals, centers)`` where ``centers`` are window centre indices.
    Additional keyword arguments are passed to :meth:`ITPU.mutual_info`.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, "x and y must be same length"
    n = len(x)
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive")
    if window_size > n:
        raise ValueError("window_size cannot exceed signal length")

    itpu = ITPU(device="software")
    starts = np.arange(0, n - window_size + 1, step, dtype=int)
    centers = starts + window_size // 2
    mi_vals = np.empty(len(starts), dtype=float)

    for i, s in enumerate(starts):
        e = s + window_size
        mi_vals[i] = itpu.mutual_info(x[s:e], y[s:e], method=method, **kwargs)

    return mi_vals, centers
