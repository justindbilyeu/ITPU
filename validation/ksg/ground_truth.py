"""
Analytic MI and data generators for the KSG validation suite.

All values are in nats. The table is frozen at spec authoring time.
"""
from __future__ import annotations

import numpy as np

# ── Frozen ground-truth table (bivariate Gaussian, spec §3) ─────────────────
GAUSSIAN_TABLE: dict[float, float] = {
    0.3: 0.047155,
    0.5: 0.143841,
    0.7: 0.336672,
    0.9: 0.830366,
    0.99: 1.958518,
}

# Verified 2D cross-covariance matrix and its exact MI (spec §3)
C_2D = np.array([[0.6, 0.1], [0.1, 0.5]])
MI_2D_TRUE: float = 0.394719  # nats


def gaussian_mi(rho: float) -> float:
    """Exact MI for bivariate Gaussian with correlation ρ. Returns nats."""
    return -0.5 * float(np.log(1.0 - rho ** 2))


def generate_bivariate_gaussian(
    N: int,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate N samples from bivariate Gaussian via Cholesky factorization.

    X = Z₁,  Y = ρ·Z₁ + √(1−ρ²)·Z₂,  Z₁,Z₂ ~ N(0,1) iid.
    """
    z1 = rng.standard_normal(N)
    z2 = rng.standard_normal(N)
    return z1.copy(), rho * z1 + np.sqrt(1.0 - rho ** 2) * z2


def generate_bivariate_uniform(
    N: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Uniform(0,1)² — non-Gaussian null for T4."""
    return rng.uniform(0, 1, N), rng.uniform(0, 1, N)


def multidim_gaussian_mi(C: np.ndarray) -> float:
    """
    Exact MI for (X,Y) ∈ ℝ^d × ℝ^d with cross-covariance matrix C.

    I(X;Y) = −½ log det(I − CᵀC)  [nats]
    """
    d = C.shape[0]
    I = np.eye(d)
    det = np.linalg.det(I - C.T @ C)
    return float(-0.5 * np.log(det))


def generate_multidim_gaussian(
    N: int,
    C: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate N samples of (X,Y) ∈ ℝ^d × ℝ^d with cross-covariance C.

    Σ = [[I, C], [C^T, I]]  — marginally standard normal.
    """
    d = C.shape[0]
    Sigma = np.block([[np.eye(d), C], [C.T, np.eye(d)]])
    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal((N, 2 * d)) @ L.T
    return z[:, :d], z[:, d:]
