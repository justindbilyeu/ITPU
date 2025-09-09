# SPDX-License-Identifier: Apache-2.0
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

def ksg_mi_1d(x, y, k=5):
    """
    Kraskov-Stögbauer-Grassberger MI estimator (KSG-1) for 1D x,y.
    Uses max-norm distance in joint space and counts in marginals.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    N = x.size
    if N < k + 2:
        raise ValueError("Not enough samples for KSG")

    # joint space
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    # distance to k-th neighbor (exclude the point itself)
    d, idx = tree.query(pts, k=k+1, p=np.inf)
    eps = d[:, -1]  # k-th distance in max-norm

    # counts within eps in marginals (exclude the point itself)
    nx = _count_within(x[:, None], eps, p=np.inf)
    ny = _count_within(y[:, None], eps, p=np.inf)

    # KSG-1 estimator
    mi = digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(mi)

def _count_within(colvec, eps, p=np.inf):
    """
    Counts how many points fall within eps (inclusive) along one axis,
    using broadcasting (O(N^2) for simplicity in baseline).
    """
    v = colvec.ravel()
    # |v_i - v_j| <= eps_i  -> count per i
    diff = np.abs(v[:, None] - v[None, :])
    within = diff <= eps[:, None]
    # exclude self
    np.fill_diagonal(within, False)
    return within.sum(axis=1)
