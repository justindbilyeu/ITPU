# itpu/sdk.py
import numpy as np
from math import log
from scipy.spatial import cKDTree

__all__ = ["ITPU"]

class ITPU:
    """
    Device-agnostic API. device="software" supported now; future targets will
    preserve this interface.
    """

    def __init__(self, device="software"):
        if device != "software":
            raise NotImplementedError("Only device='software' is supported today.")
        self.device = device

    # ---------- Public API ----------
    def mutual_info(self, x, y, method="hist", **kwargs):
        """
        Mutual information between 1D arrays x,y (nats).
        method: "hist" (discrete/histogram) or "ksg" (continuous kNN).
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have same length.")

        if method == "hist":
            bins = int(kwargs.get("bins", 64))
            return _mi_hist(x, y, bins=bins)
        elif method == "ksg":
            k = int(kwargs.get("k", 5))
            return _mi_ksg(x, y, k=k)
        else:
            raise ValueError(f"Unknown method: {method}")

# ---------- Histogram-based MI (nats) ----------
def _entropy_from_hist(counts):
    # counts: nonnegative, sum > 0
    p = counts.astype(float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p /= total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def _mi_hist(x, y, bins=64):
    # 1D histograms
    hx, _ = np.histogram(x, bins=bins)
    hy, _ = np.histogram(y, bins=bins)
    # 2D joint histogram
    hxy, _, _ = np.histogram2d(x, y, bins=bins)

    Hx = _entropy_from_hist(hx)
    Hy = _entropy_from_hist(hy)
    Hxy = _entropy_from_hist(hxy)  # flatten ok in helper

    return Hx + Hy - Hxy

# ---------- KSG (Kraskov-Stögbauer-Grassberger) MI (nats), variant 1 ----------
# Minimal, dependency-light implementation for 1D x,y.
# Uses Chebyshev (infinity) norm in joint space, digamma via scipy.special if available
try:
    from scipy.special import digamma
except Exception:
    # simple fallback: harmonic approximation
    def digamma(n):
        # Euler–Mascheroni gamma ~0.57721
        if n <= 0:
            raise ValueError("digamma requires n>0 in this fallback.")
        return -0.5772156649015329 + sum(1.0/i for i in range(1, int(n)))

def _mi_ksg(x, y, k=5):
    """
    KSG-1 estimator for continuous MI in nats.
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    n = x.shape[0]
    if n != y.shape[0]:
        raise ValueError("x,y length mismatch")
    if k <= 0 or k >= n:
        raise ValueError("k must be in [1, n-1)")

    # Joint space with Chebyshev (max) norm via KDTree by duplicating dims and using max radius
    xy = np.hstack([x, y])
    tree_joint = cKDTree(xy)
    # Distance to k-th neighbor in joint space (Chebyshev approximated by max of |dx|,|dy|)
    # We emulate Chebyshev by querying in Euclidean but we will count marginals strictly less than eps.
    # Query k+1 because point itself counts as neighbor.
    dists, _ = tree_joint.query(xy, k=k+1, workers=-1)
    eps = dists[:, -1]  # radius to k-th neighbor

    # Count neighbors in marginals within strictly less than eps (avoid boundary double counts)
    nx = _count_within_radius_1d(x, eps, strict=True)
    ny = _count_within_radius_1d(y, eps, strict=True)

    # KSG-1 formula (nats)
    # I = psi(k) - <psi(nx+1)+psi(ny+1)> + psi(n)  (with digamma)
    term = np.mean(digamma(k) - digamma(nx + 1) - digamma(ny + 1) + digamma(n))
    return float(term)

def _count_within_radius_1d(x, eps, strict=True):
    """
    For each point i, count number of other samples whose |x_j - x_i| < eps_i (strict) or <= eps_i.
    Returns counts excluding the point itself.
    """
    x = np.asarray(x).ravel()
    n = x.shape[0]
    # Sort to enable sliding window counts
    order = np.argsort(x)
    xs = x[order]
    counts = np.empty(n, dtype=int)

    j_left = 0
    for idx, i in enumerate(order):
        xi = xs[idx]
        # expand left boundary
        while j_left < n and (xi - xs[j_left] > eps[i] if strict else xi - xs[j_left] >= eps[i]):
            j_left += 1
        # expand right boundary
        j_right = idx + 1
        while j_right < n and (xs[j_right] - xi < eps[i] if strict else xs[j_right] - xi <= eps[i]):
            j_right += 1
        # total in (left,right) minus 1 (exclude self)
        counts[i] = (j_right - j_left) - 1
    # clamp minimum 0
    counts[counts < 0] = 0
    return counts
