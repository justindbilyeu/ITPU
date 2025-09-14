"""Surrogate data generation utilities.

This module provides functions for generating Iterative Amplitude Adjusted
Fourier Transform (IAAFT) surrogates and coarse autocorrelation-preserving
block shuffles.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def iaaft_surrogate(
    x: np.ndarray,
    n_iter: int = 1000,
    tol: float = 1e-5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate an IAAFT surrogate of ``x``.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional input signal.
    n_iter : int, optional
        Maximum number of iterations to perform. Defaults to 1000.
    tol : float, optional
        Relative RMSE change threshold for early stopping. If the relative
        change in amplitude spectrum RMSE is below ``tol`` for three
        consecutive iterations the algorithm stops early. Defaults to
        ``1e-5``.
    rng : np.random.Generator, optional
        Optional random number generator for deterministic behaviour.

    Returns
    -------
    np.ndarray
        Surrogate signal with the same amplitude spectrum and rank order
        distribution as ``x``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array")

    rng = np.random.default_rng(rng)
    n = x.size
    target_amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)

    s = rng.permutation(x)  # initial guess preserves distribution
    prev_rmse = None
    stable_count = 0

    for _ in range(n_iter):
        # Enforce target amplitude spectrum
        s_fft = np.fft.rfft(s)
        phases = np.exp(1j * np.angle(s_fft))
        s = np.fft.irfft(target_amp * phases, n)

        # Enforce distribution via rank matching: replace the sorted values of
        # the surrogate with the sorted original values so that the rank order
        # matches the input.
        order = np.argsort(s)
        s = s.copy()
        s[order] = sorted_x

        # Compute relative RMSE of amplitude spectrum
        amp = np.abs(np.fft.rfft(s))
        rmse = np.sqrt(np.mean((amp - target_amp) ** 2))
        rmse /= np.sqrt(np.mean(target_amp ** 2))

        if prev_rmse is not None:
            rel_change = abs(prev_rmse - rmse) / (prev_rmse if prev_rmse != 0 else 1)
            if rel_change < tol:
                stable_count += 1
                if stable_count >= 3:
                    break
            else:
                stable_count = 0
        prev_rmse = rmse

    return s.copy()


def block_shuffle(
    x: np.ndarray,
    block_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Shuffle ``x`` in non-overlapping blocks.

    The array is partitioned into consecutive blocks of length ``block_size``.
    The order of the blocks is randomly permuted while preserving the order
    of elements within each block. The last block may be shorter than
    ``block_size``. This preserves the autocorrelation structure up to roughly
    ``block_size / 2`` samples.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional input array.
    block_size : int
        Size of each block. Must be positive.
    rng : np.random.Generator, optional
        Optional random number generator for deterministic behaviour.

    Returns
    -------
    np.ndarray
        Block-shuffled copy of ``x``.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n = x.size
    rng = np.random.default_rng(rng)
    blocks = [x[i : i + block_size] for i in range(0, n, block_size)]
    order = rng.permutation(len(blocks))
    shuffled = np.concatenate([blocks[i] for i in order])
    return shuffled.copy()
