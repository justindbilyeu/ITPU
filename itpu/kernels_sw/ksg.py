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
    x, y, k: int = 5, metric: str = "chebyshev"
) -> tuple[float, dict]:
    """
    Kraskov–Stögbauer–Grassberger (KSG) MI estimator (variant I, canonical).

    Args:
        x, y: 1D arrays of equal length
        k: neighbor parameter (default 5)
        metric: 'chebyshev' (default, p=∞) or 'euclidean' (p=2)

    Returns:
        (mi, stats) where:
          mi: estimated mutual information (nats)
          stats: dict with N, k, metric, method, flags
    """
    x = _as_1d(x); y = _as_1d(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    N = len(x)
    if N <= k:
        return 0.0, dict(N=N, k=k, method="ksg", note="too few samples")

    z = np.column_stack((x, y))
    tree_z = cKDTree(z)

    p = np.inf if metric == "chebyshev" else 2
    dists, _ = tree_z.query(z, k=k+1, p=p, workers=-1)
    radii = dists[:, k]

    # Marginal neighbor counts (strict inequality)
    tiny = 1e-12
    tree_x = cKDTree(x[:, None])
    tree_y = cKDTree(y[:, None])
    nx = np.array(tree_x.query_ball_point(x[:, None], radii - tiny, return_length=True)) - 1
    ny = np.array(tree_y.query_ball_point(y[:, None], radii - tiny, return_length=True)) - 1

    # Canonical KSG-1 formula
    mi = digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    mi = float(max(mi, 0.0))

    stats = dict(N=N, k=k, metric=metric, method="ksg")
    if np.all(radii < _EPS):
        stats["note"] = "zero-variance or duplicate samples"
    return mi, stats
