from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

_EPS = 1e-12

def _as_1d(a, name="array"):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return a

def _apply_mask_and_jitter(x, y, mask=None, jitter=0.0, seed=None):
    x = _as_1d(x, "x"); y = _as_1d(y, "y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length")
    valid = ~(np.isnan(x) | np.isnan(y))
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != x.shape[0]:
            raise ValueError("mask length must match x/y")
        valid &= mask
    x = x[valid]; y = y[valid]
    if jitter and x.size:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(scale=jitter, size=x.size)
        y = y + rng.normal(scale=jitter, size=y.size)
    return x, y

def _ksg_core(x, y, k=5, metric="chebyshev") -> tuple[float, dict]:
    """KSG-I MI estimator (nats). Default metric is L∞ (chebyshev)."""
    x = _as_1d(x, "x"); y = _as_1d(y, "y")
    n = x.size
    if n == 0 or n <= k:
        return 0.0, dict(N=n, k=k, method="ksg", note="too few samples")

    xy = np.column_stack((x, y))
    p = np.inf if metric == "chebyshev" else 2
    tree_xy = cKDTree(xy)
    tree_x  = cKDTree(x[:, None])
    tree_y  = cKDTree(y[:, None])

    # distance to k-th neighbor in joint space (exclude self → k+1)
    dists, _ = tree_xy.query(xy, k=k+1, p=p)
    eps = dists[:, -1]
    radii = np.maximum(eps - _EPS, 0.0)

    nx = np.fromiter((len(tree_x.query_ball_point([x[i]], r=radii[i])) - 1 for i in range(n)), count=n, dtype=np.int64)
    ny = np.fromiter((len(tree_y.query_ball_point([y[i]], r=radii[i])) - 1 for i in range(n)), count=n, dtype=np.int64)

    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(mi, 0.0)), dict(N=n, k=k, method="ksg", metric=metric)

def mi_ksg(
    x, y, k: int = 5, metric: str = "chebyshev",
    mask=None, jitter: float = 0.0, seed: int | None = None,
    return_stats: bool = False
):
    """Public KSG MI (nats)."""
    x, y = _apply_mask_and_jitter(x, y, mask=mask, jitter=jitter, seed=seed)
    mi, stats = _ksg_core(x, y, k=k, metric=metric)
    return (mi, stats) if return_stats else mi

def mi_matrix_ksg(X, pairs="all", k=5, metric="chebyshev", jitter: float = 0.0, seed: int | None = None):
    """MI matrix over columns of X using KSG."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")
    n, d = X.shape
    if pairs == "all":
        M = np.zeros((d, d), dtype=np.float64)
        for i in range(d):
            for j in range(i+1, d):
                M[i, j] = M[j, i] = mi_ksg(X[:, i], X[:, j], k=k, metric=metric, jitter=jitter, seed=seed)
        return M
    out = np.zeros(len(pairs), dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        out[idx] = mi_ksg(X[:, i], X[:, j], k=k, metric=metric, jitter=jitter, seed=seed)
    return out
