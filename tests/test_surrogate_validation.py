# LOCKED THRESHOLDS — see decision log 2026-04-19
"""Validity tests for the surrogate_test() pipeline.

These are validity tests, not smoke tests. They verify statistical
properties of the surrogate testing procedure under known conditions:

- test_h0_pvalue_calibration: Under the null (x and y independent
  Gaussians), p-values returned by surrogate_test() must be uniformly
  distributed on [0, 1]. Calibration is verified via a Kolmogorov-Smirnov
  test against the uniform distribution.

- test_h1_power_detects_correlation: Under a strong linear alternative
  (rho=0.6), surrogate_test() must reject at p < 0.05.

The KS threshold (pvalue > 0.05) and trial count (400) are locked and
must not be adjusted post-hoc to make a failing calibration test pass.
If the calibration test fails, the surrogate procedure is miscalibrated
and surrogate_test() must be fixed, not the threshold.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from itpu.stats.surrogate_test import surrogate_test


@pytest.mark.slow
def test_h0_pvalue_calibration():
    """P-values under H₀ must be uniformly distributed.

    400 independent trials, each with n=1000 i.i.d. Gaussian x and y.
    KS test against Uniform[0,1]: assert pvalue > 0.05.
    LOCKED — do not adjust the threshold or trial count.
    """
    n_trials = 400
    p_values = np.empty(n_trials)

    for i in range(n_trials):
        rng = np.random.default_rng(i)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        result = surrogate_test(
            x, y, method="ksg", n_surrogates=999, surrogate_type="shuffle"
        )
        p_values[i] = result["p_value"]

    ks_result = stats.kstest(p_values, "uniform")
    assert ks_result.pvalue > 0.05, (
        f"P-values under H₀ are not uniform (KS pvalue={ks_result.pvalue:.4f}). "
        "The surrogate procedure is miscalibrated."
    )


def test_h1_power_detects_correlation():
    """Surrogate test must reject at p < 0.05 under strong linear dependence.

    x ~ N(0,1), y = 0.6*x + 0.4*N(0,1), n=500, n_surrogates=499.
    LOCKED — do not adjust the threshold or the correlation coefficient.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500)
    y = 0.6 * x + 0.4 * rng.standard_normal(500)

    result = surrogate_test(
        x, y, method="ksg", n_surrogates=499, surrogate_type="shuffle"
    )

    assert result["p_value"] < 0.05, (
        f"Failed to detect strong correlation (p={result['p_value']:.4f}). "
        "Expected p < 0.05 for rho=0.6 with n=500."
    )


def test_iaaft_surrogate_type_wiring():
    """surrogate_type='iaaft' is wired through surrogate_test() correctly.

    Wiring test only — not a calibration test. Verifies the return dict has
    all expected keys and p_value is a float in [0, 1].
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = 0.6 * x + 0.4 * rng.standard_normal(200)

    result = surrogate_test(x, y, method="ksg", n_surrogates=49, surrogate_type="iaaft")

    expected_keys = {"mi_observed", "null_distribution", "p_value", "power_estimate", "warnings"}
    assert set(result.keys()) == expected_keys
    assert isinstance(result["p_value"], float)
    assert 0.0 <= result["p_value"] <= 1.0
