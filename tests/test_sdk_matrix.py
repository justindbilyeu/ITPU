# tests/test_sdk_matrix.py
import numpy as np
from itpu.sdk import ITPU

def make_correlated_features(n=40_000, d=6, rho=0.55, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, d))
    X = rho * z + np.sqrt(1 - rho**2) * noise
    return X.astype(np.float64)

def test_full_matrix_shape_and_symmetry():
    itpu = ITPU(device="software")
    X = make_correlated_features()
    M = itpu.mutual_info_matrix(X, method="hist", bins=128, pairs="all")
    assert M.shape == (X.shape[1], X.shape[1])
    assert np.allclose(M, M.T, atol=1e-9)
    assert np.all(np.diag(M) == 0.0)

def test_pairs_subset_matches_full():
    itpu = ITPU(device="software")
    X = make_correlated_features(d=5)
    # full matrix
    M = itpu.mutual_info_matrix(X, method="hist", bins=64, pairs="all")
    # a few pairs
    sel = [(0,1), (1,3), (2,4)]
    out = itpu.mutual_info_matrix(X, method="hist", bins=64, pairs=sel)
    for (i,j) in sel:
        assert abs(out[(i,j)] - M[i,j]) <= max(1e-3, 0.03 * M[i,j])

def test_masking_reduces_samples_but_still_works():
    itpu = ITPU(device="software")
    X = make_correlated_features(n=30_000, d=4, rho=0.6)
    mask = np.ones(X.shape[0], dtype=bool)
    mask[:5_000] = False  # drop some rows
    M = itpu.mutual_info_matrix(X, method="hist", bins=64, pairs="all", mask=mask)
    assert np.isfinite(M).all()
    assert (M >= 0.0).all()
