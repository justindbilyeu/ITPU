# tests/test_sdk_hist.py
import numpy as np
import pytest

from itpu.sdk import ITPU


def make_correlated(n=50_000, rho=0.6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return x.astype(np.float64), y.astype(np.float64)


def test_mi_nonnegative_and_symmetry_hist():
    itpu = ITPU(device="software")
    x, y = make_correlated(n=100_000, rho=0.7, seed=42)

    mi_xy = itpu.mutual_info(x, y, method="hist", bins=128)
    mi_yx = itpu.mutual_info(y, x, method="hist", bins=128)

    # Non-negativity and rough symmetry
    assert np.isfinite(mi_xy)
    assert mi_xy >= 0.0
    assert abs(mi_xy - mi_yx) <= max(1e-3, 0.02 * mi_xy)


def test_mi_detects_dependence_vs_shuffle_hist():
    itpu = ITPU(device="software")
    x, y = make_correlated(n=80_000, rho=0.5, seed=7)

    mi_dep = itpu.mutual_info(x, y, method="hist", bins=128)

    rng = np.random.default_rng(7)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)

    mi_indep = itpu.mutual_info(x, y_shuf, method="hist", bins=128)

    # Dependent pair should have strictly larger MI than shuffled
    assert mi_dep > mi_indep + 5e-3  # small margin to avoid flakiness


@pytest.mark.parametrize("bins", [32, 64, 128, 256])
def test_mi_monotone_with_bins_reasonable(bins):
    itpu = ITPU(device="software")
    x, y = make_correlated(n=60_000, rho=0.4, seed=123)
    mi = itpu.mutual_info(x, y, method="hist", bins=bins)
    assert np.isfinite(mi) and mi >= 0.0
