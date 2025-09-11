from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

_EPS = 1e-12
__all__ = ["ksg_mi_estimate", "windowed_ksg_mi"]

def _as_1d(a):
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("Expected 1D array")
    return a

def ksg_mi_estimate(
    x, y, k: int = 5, metric: str = "chebyshev"
) -> tuple[float, dict]:
    """
    Kraskov–Stögbauer–Grassberger MI estimator (variant I).
    Returns (mi_nats, stats).
    """
    x = _as_1d(x); y = _as_1d(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    N = len(x)
    if N <= k:
        return 0.0, dict(N=N, k=k, method="ksg", note="too few samples")

    z = np.column_stack((x, y))
    p = np.inf if metric == "chebyshev" else 2
    tree_z = cKDTree(z)
    dists, _ = tree_z.query(z, k=k+1, p=p, workers=-1)  # includes self
    radii = dists[:, k]

    tiny = 1e-12
    tree_x = cKDTree(x[:, None])
    tree_y = cKDTree(y[:, None])
    nx = np.array(tree_x.query_ball_point(x[:, None], radii - tiny, return_length=True)) - 1
    ny = np.array(tree_y.query_ball_point(y[:, None], radii - tiny, return_length=True)) - 1
    nx = np.maximum(nx, 0); ny = np.maximum(ny, 0)

    mi = digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    mi = float(max(mi, 0.0))
    stats = dict(N=N, k=k, metric=metric, method="ksg")
    if np.all(radii < _EPS):
        stats["note"] = "zero-variance or duplicate samples"
    return mi, stats

def windowed_ksg_mi(
    x, y,
    window_size: int = 1000,
    hop_size: int = 200,
    k: int = 5,
    metric: str = "chebyshev",
):
    """
    Sliding-window KSG MI by recomputing per window.
    Returns: starts (idx), mi_vals (nats), extras (params).
    """
    x = _as_1d(x); y = _as_1d(y)
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have same length")

    starts = np.arange(0, max(0, n - window_size + 1), hop_size, dtype=np.int64)
    mi_vals = np.zeros(len(starts), dtype=float)
    for i, s in enumerate(starts):
        seg_x = x[s:s+window_size]
        seg_y = y[s:s+window_size]
        mi, _ = ksg_mi_estimate(seg_x, seg_y, k=k, metric=metric)
        mi_vals[i] = mi
    extras = dict(window_size=window_size, hop_size=hop_size, k=k, metric=metric)
    return starts, mi_vals, extras
