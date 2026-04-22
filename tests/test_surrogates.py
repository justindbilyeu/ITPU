from __future__ import annotations

import numpy as np
import pytest

from itpu.stats.surrogates import block_bootstrap_surrogate, shuffle_surrogate, iaaft_surrogate


RNG_SEED = 42
X = np.arange(20, dtype=float)


# ---------------------------------------------------------------------------
# shuffle_surrogate
# ---------------------------------------------------------------------------

def test_shuffle_output_shape():
    out = shuffle_surrogate(X, n_surrogates=50, rng=RNG_SEED)
    assert out.shape == (50, len(X))


def test_shuffle_each_row_is_permutation():
    out = shuffle_surrogate(X, n_surrogates=30, rng=RNG_SEED)
    for row in out:
        assert np.array_equal(np.sort(row), np.sort(X))


def test_shuffle_deterministic():
    a = shuffle_surrogate(X, n_surrogates=10, rng=RNG_SEED)
    b = shuffle_surrogate(X, n_surrogates=10, rng=RNG_SEED)
    assert np.array_equal(a, b)


def test_shuffle_different_seeds_differ():
    a = shuffle_surrogate(X, n_surrogates=10, rng=0)
    b = shuffle_surrogate(X, n_surrogates=10, rng=1)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# block_bootstrap_surrogate
# ---------------------------------------------------------------------------

def test_block_output_shape():
    out = block_bootstrap_surrogate(X, block_size=4, n_surrogates=50, rng=RNG_SEED)
    assert out.shape == (50, len(X))


def test_block_output_length_matches_input():
    for block_size in (1, 3, 7, 20, 25):
        out = block_bootstrap_surrogate(X, block_size=block_size, n_surrogates=5, rng=RNG_SEED)
        assert out.shape[1] == len(X), f"failed for block_size={block_size}"


def test_block_values_drawn_from_input():
    out = block_bootstrap_surrogate(X, block_size=4, n_surrogates=20, rng=RNG_SEED)
    x_set = set(X.tolist())
    for row in out:
        assert all(v in x_set for v in row)


def test_block_deterministic():
    a = block_bootstrap_surrogate(X, block_size=4, n_surrogates=10, rng=RNG_SEED)
    b = block_bootstrap_surrogate(X, block_size=4, n_surrogates=10, rng=RNG_SEED)
    assert np.array_equal(a, b)


def test_block_different_seeds_differ():
    a = block_bootstrap_surrogate(X, block_size=4, n_surrogates=10, rng=0)
    b = block_bootstrap_surrogate(X, block_size=4, n_surrogates=10, rng=1)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# IAAFT tests — will fail until iaaft_surrogate() is implemented in surrogates.py
# ---------------------------------------------------------------------------

def test_iaaft_output_shape():
    x = np.arange(64, dtype=float)
    assert iaaft_surrogate(x, n_surrogates=20, rng=RNG_SEED).shape == (20, 64)


def test_iaaft_values_in_domain():
    x = np.random.default_rng(RNG_SEED).standard_normal(128)
    out = iaaft_surrogate(x, n_surrogates=10, rng=RNG_SEED)
    for row in out:
        assert np.array_equal(np.sort(row), np.sort(x))


def test_iaaft_deterministic():
    x = np.arange(64, dtype=float)
    a = iaaft_surrogate(x, n_surrogates=10, rng=RNG_SEED)
    b = iaaft_surrogate(x, n_surrogates=10, rng=RNG_SEED)
    assert np.array_equal(a, b)


def test_iaaft_spectral_preservation():
    x = np.sin(np.linspace(0, 8 * np.pi, 256))
    out = iaaft_surrogate(x, n_surrogates=5, rng=RNG_SEED)
    P_orig = np.abs(np.fft.rfft(x)) ** 2
    for row in out:
        P_surr = np.abs(np.fft.rfft(row)) ** 2
        mean_relative_diff = np.mean(np.abs(P_surr - P_orig) / (P_orig + 1e-12))
        # LOCKED THRESHOLD — do not adjust post-hoc. IAAFT preserves FFT magnitude by construction.
        assert mean_relative_diff < 1e-10


def test_iaaft_breaks_phase_correlation():
    x = np.cumsum(np.random.default_rng(RNG_SEED).standard_normal(500))
    surrogate = iaaft_surrogate(x, n_surrogates=1, rng=RNG_SEED)[0]
    autocorr_original = np.corrcoef(x[:-1], x[1:])[0, 1]
    autocorr_surrogate = np.corrcoef(surrogate[:-1], surrogate[1:])[0, 1]
    # Confirms phase randomization occurred — implementation returning original signal
    # unchanged would pass tests 1-4 but fail here.
    assert abs(autocorr_surrogate - autocorr_original) > 0.05
    assert not np.array_equal(surrogate, x)
