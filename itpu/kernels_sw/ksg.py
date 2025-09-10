from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

_EPS = 1e-12

def _as_1d(a):
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("Expected 1D array")
    return a

def ksg_mi_estimate(
    x, y, k: int = 5, metric: str = "euclidean"
) -> tuple[float, dict]:
    """
    Kraskov–Stögbauer–Grassberger (KSG) mutual information estimator (variant I).

    Args:
        x, y: 1D arrays of equal length
        k: neighbor parameter (default 5)
        metric: 'euclidean' or 'chebyshev' (for KDTree distance norm)

    Returns:
        (mi, stats) where mi is in nats.
    """
    x = _as_1d(x); y = _as_1d(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n <= k:
        return 0.0, dict(N=n, k=k, method="ksg", note="too few samples")

    xy = np.column_stack((x, y))
    # Joint space tree
    p = 2 if metric == "euclidean" else np.inf
    tree_xy = cKDTree(xy)
    # Distance to k-th neighbor in joint space (exclude self → k+1)
    dists, _ = tree_xy.query(xy, k=k + 1, p=p)
    eps = dists[:, -1]  # radius to k-th neighbor

    # Marginal counts within that radius (exclude self via -1)
    tree_x = cKDTree(x[:, None])
    tree_y = cKDTree(y[:, None])
    nx = np.array([len(tree_x.query_ball_point([x[i]], eps[i] - _EPS)) - 1 for i in range(n)])
    ny = np.array([len(tree_y.query_ball_point([y[i]], eps[i] - _EPS)) - 1 for i in range(n)])

    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(mi, 0.0)), dict(N=n, k=k, metric=metric, method="ksg")
