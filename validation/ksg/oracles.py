"""
Validation oracles for the KSG suite.

    assert_digamma()  — T7 digamma identity gate (runs first).
    mi_bruteforce()   — O(N²) Chebyshev kNN for T6 agreement gate.

The brute-force implementation is intentionally un-clever so that it is
obviously correct. It applies the same C4/C5 preprocessing as ksg.py
(same jitter seed, same standardization) so that the T6 comparison to
≤ 1e-9 is achievable.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma

EULER_MASCHERONI = 0.5772156649015329


def assert_digamma() -> None:
    """
    T7 — Digamma identity gate.

    ψ(1) = −γ     →  |ψ(1) + γ| < 1e-12
    ψ(2) = 1 − γ  →  |ψ(2) − (1 − γ)| < 1e-12

    These identities distinguish correct digamma from ln substitution:
    ln(1) = 0 ≠ ψ(1) = −γ ≈ −0.5772.
    """
    gamma = EULER_MASCHERONI
    err1 = abs(digamma(1) + gamma)
    assert err1 < 1e-12, f"T7 FAIL: |ψ(1) + γ| = {err1:.3e} (expected < 1e-12)"

    err2 = abs(digamma(2) - (1.0 - gamma))
    assert err2 < 1e-12, f"T7 FAIL: |ψ(2) − (1−γ)| = {err2:.3e} (expected < 1e-12)"


def mi_bruteforce(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 4,
    jitter_seed: int | None = 0,
) -> float:
    """
    O(N²) KSG-I reference. No KDTree — direct pairwise computation.

    Applies the same C4/C5 preprocessing as ksg.py (identical jitter seed
    and standardization) so that |mi_bruteforce − ksg_mi| ≤ 1e-9 in T6.

    Parameters
    ----------
    x, y        : 1D array-like
    k           : neighbors (same as ksg_mi)
    jitter_seed : same as ksg_mi — must match for T6 agreement

    Returns
    -------
    float — MI estimate in nats
    """
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    N = len(x)

    # C4 — same jitter, same seed
    if jitter_seed is not None:
        rng_j = np.random.default_rng(jitter_seed)
        x += 1e-10 * rng_j.standard_normal(N)
        y += 1e-10 * rng_j.standard_normal(N)

    # C5 — same standardization
    std_x = x.std(ddof=0)
    std_y = y.std(ddof=0)
    x = (x - x.mean()) / (std_x if std_x > 1e-15 else 1.0)
    y = (y - y.mean()) / (std_y if std_y > 1e-15 else 1.0)

    # Full pairwise Chebyshev (L∞) distance matrix
    diff_x = np.abs(x[:, None] - x[None, :])   # (N, N)
    diff_y = np.abs(y[:, None] - y[None, :])   # (N, N)
    dist = np.maximum(diff_x, diff_y)           # Chebyshev

    # Exclude self-distance
    np.fill_diagonal(dist, np.inf)

    # k-th nearest neighbor distance for each point
    radii = np.partition(dist, k - 1, axis=1)[:, k - 1]

    # Strict-< marginal counts via searchsorted — identical arithmetic to
    # ksg.py so that floating-point boundary cases agree bit-for-bit.
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    lo_x = np.searchsorted(x_sorted, x - radii, side="right")
    hi_x = np.searchsorted(x_sorted, x + radii, side="left")
    nx = hi_x - lo_x - 1   # exclude self (always inside radius)
    lo_y = np.searchsorted(y_sorted, y - radii, side="right")
    hi_y = np.searchsorted(y_sorted, y + radii, side="left")
    ny = hi_y - lo_y - 1

    nx = np.maximum(nx, 0)
    ny = np.maximum(ny, 0)

    # C2 — digamma formula
    return float(
        digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    )
