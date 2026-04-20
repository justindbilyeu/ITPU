from __future__ import annotations

import numpy as np
import pytest

from itpu.stats.surrogates import block_bootstrap_surrogate, shuffle_surrogate


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
