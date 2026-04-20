from __future__ import annotations

import math

import numpy as np
from numpy.random import Generator


def shuffle_surrogate(
    x: np.ndarray,
    n_surrogates: int,
    rng: Generator | None = None,
) -> np.ndarray:
    """Generate surrogates by independently shuffling x.

    Each surrogate is a random permutation of the input array, destroying
    any temporal or spatial structure while preserving the marginal
    distribution.

    Parameters
    ----------
    x:
        1D input array to shuffle.
    n_surrogates:
        Number of surrogate samples to generate.
    rng:
        NumPy random Generator for reproducibility. If None, uses
        numpy.random.default_rng().

    Returns
    -------
    np.ndarray
        Shape (n_surrogates, len(x)). Each row is one shuffled surrogate.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(x)
    n = len(x)
    out = np.empty((n_surrogates, n), dtype=x.dtype)
    for i in range(n_surrogates):
        out[i] = rng.permutation(x)
    return out


def block_bootstrap_surrogate(
    x: np.ndarray,
    block_size: int,
    n_surrogates: int,
    rng: Generator | None = None,
) -> np.ndarray:
    """Generate surrogates via circular block bootstrap.

    Resamples contiguous blocks of x with replacement, preserving short-range
    autocorrelation structure while breaking long-range dependence with y.
    Uses circular (wrap-around) indexing so every position has equal
    probability of being a block start.

    Parameters
    ----------
    x:
        1D input array to resample.
    block_size:
        Length of each contiguous block.
    n_surrogates:
        Number of surrogate samples to generate.
    rng:
        NumPy random Generator for reproducibility. If None, uses
        numpy.random.default_rng().

    Returns
    -------
    np.ndarray
        Shape (n_surrogates, len(x)). Each row is one block-resampled surrogate.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(x)
    n = len(x)
    n_blocks = math.ceil(n / block_size)
    out = np.empty((n_surrogates, n), dtype=x.dtype)
    for i in range(n_surrogates):
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate([
            np.arange(s, s + block_size) % n for s in starts
        ])
        out[i] = x[indices[:n]]
    return out
