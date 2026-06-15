"""
KSG-I mutual-information estimator — spec-compliant reference implementation.

All six mandatory conventions from ITPU validation spec §2 are enforced here.
This is the estimator under test for the validation suite. It is intentionally
standalone (does not import from itpu.*) so the validation does not inherit
implementation assumptions from the SDK path.

Conventions
-----------
C1  Joint metric = Chebyshev (L∞ / p=inf). Not Euclidean.
C2  Use ψ (digamma), never ln.
C3  Marginal counts strict < ρ_k(i). Realized via searchsorted half-open.
C4  Tie-break jitter: 1e-10 * N(0,1) applied before search, seeded and logged.
C5  Standardize each marginal to unit variance (z-score) before search.
C6  cKDTree(p=inf) for joint search; np.searchsorted on sorted marginals for
    exact O(N log N) strict-< counting.

Formula (KSG-1, Kraskov et al. 2004, Eq. 8):
    I_hat = ψ(k) − (1/N) Σ_i [ ψ(n_x(i)+1) + ψ(n_y(i)+1) ] + ψ(N)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def ksg_mi(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 4,
    jitter_seed: int | None = 0,
) -> tuple[float, dict]:
    """
    KSG-I estimator with all C1-C6 conventions applied.

    Parameters
    ----------
    x, y : array-like, 1D
    k     : number of nearest neighbors
    jitter_seed : seed for C4 tie-break jitter. None disables jitter (not
                  recommended for production use — C4 matters for quantized data).

    Returns
    -------
    (mi_nats, stats)
        mi_nats : float — MI estimate in nats (unclipped; may be negative under H₀)
        stats   : dict  — provenance record for the run JSON
    """
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    N = len(x)
    if len(y) != N:
        raise ValueError("x and y must have the same length")
    if N <= k:
        return 0.0, {"N": N, "k": k, "note": "too few samples"}

    # C4 — tie-break jitter (seeded, logged)
    jitter_x = np.zeros(N)
    jitter_y = np.zeros(N)
    if jitter_seed is not None:
        rng_j = np.random.default_rng(jitter_seed)
        jitter_x = 1e-10 * rng_j.standard_normal(N)
        jitter_y = 1e-10 * rng_j.standard_normal(N)
        x += jitter_x
        y += jitter_y

    # C5 — standardize each marginal to unit variance
    std_x = x.std(ddof=0)
    std_y = y.std(ddof=0)
    mean_x = x.mean()
    mean_y = y.mean()
    x = (x - mean_x) / (std_x if std_x > 1e-15 else 1.0)
    y = (y - mean_y) / (std_y if std_y > 1e-15 else 1.0)

    # C1 / C6 — joint Chebyshev kNN via cKDTree(p=inf)
    z = np.column_stack([x, y])
    tree = cKDTree(z)
    # k+1 to include self (self-distance = 0, always the smallest)
    dists, _ = tree.query(z, k=k + 1, p=np.inf, workers=-1)
    radii = dists[:, k]  # k-th nearest neighbor distance (excluding self)

    # C3 / C6 — exact strict-< marginal counts via searchsorted
    #   We want #{j≠i : |x_i − x_j| < ρ_k(i)}.
    #   In sorted array sx:
    #     lo = searchsorted(sx, x_i − r, 'right')  → first idx with sx[lo] > x_i−r
    #     hi = searchsorted(sx, x_i + r, 'left')   → first idx with sx[hi] ≥ x_i+r
    #   Points in (x_i−r, x_i+r) span indices [lo, hi). Count = hi−lo.
    #   Subtract 1 to exclude self (x_i itself is inside the interval).
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    lo_x = np.searchsorted(x_sorted, x - radii, side="right")
    hi_x = np.searchsorted(x_sorted, x + radii, side="left")
    nx = np.maximum(hi_x - lo_x - 1, 0)

    lo_y = np.searchsorted(y_sorted, y - radii, side="right")
    hi_y = np.searchsorted(y_sorted, y + radii, side="left")
    ny = np.maximum(hi_y - lo_y - 1, 0)

    # C2 — digamma formula (KSG-1 Eq. 8)
    mi = float(
        digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    )

    stats = {
        "N": N,
        "k": k,
        "jitter_seed": jitter_seed,
        "conventions": "C1-C6",
        "mean_nx": float(np.mean(nx)),
        "mean_ny": float(np.mean(ny)),
        "mean_radius": float(np.mean(radii)),
    }
    return mi, stats
