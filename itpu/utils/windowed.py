from __future__ import annotations
import numpy as np

def _hist_mi(x, y, bins=64):
    # Simple histogram MI in nats
    H, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    pxy = H / H.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where((pxy > 0) & (px > 0) & (py > 0), np.log(pxy / (px * py)), 0.0)
    return float(np.nansum(pxy * log_term))

def windowed_mi(
    x, y, *, window_size: int, hop_size: int,
    method: str = "hist", bins: int = 64, k: int = 5, metric: str = "euclidean"
):
    """
    Sliding-window mutual information over time.
    Returns (start_indices, mi_values) with MI in nats.
    """
    x = np.asarray(x); y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive")
    if hop_size > window_size:
        raise ValueError("hop_size cannot exceed window_size")

    n = len(x)
    starts = list(range(0, max(n - window_size + 1, 0), hop_size))
    mi_vals = np.empty(len(starts), dtype=float)

    if method == "hist":
        for i, s in enumerate(starts):
            sl = slice(s, s + window_size)
            mi_vals[i] = _hist_mi(x[sl], y[sl], bins=bins)
        return np.array(starts), mi_vals

    elif method == "ksg":
        # Lazy import to keep hist-only users light
        from itpu.kernels_sw.ksg import ksg_mi_estimate
        for i, s in enumerate(starts):
            sl = slice(s, s + window_size)
            mi_vals[i], _ = ksg_mi_estimate(x[sl], y[sl], k=k, metric=metric)
        return np.array(starts), mi_vals

    else:
        raise ValueError(f"Unknown method: {method}")
