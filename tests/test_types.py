"""Tests for EstimatorValue and SurrogateResult in itpu.types."""
import numpy as np
import pytest

from itpu.types import EstimatorValue, SurrogateResult


# ---------- EstimatorValue ----------

def test_same_estimator_equal():
    assert EstimatorValue(1.0, "hist") == EstimatorValue(1.0, "hist")


def test_cross_estimator_eq_raises_typeerror():
    with pytest.raises(TypeError, match="to_common_basis"):
        _ = EstimatorValue(1.0, "hist") == EstimatorValue(1.0, "ksg")


def test_float_conversion_backward_compat():
    assert float(EstimatorValue(1.0, "hist")) == 1.0


def test_is_float_subclass():
    v = EstimatorValue(0.5, "ksg")
    assert isinstance(v, float)


def test_comparisons_with_plain_float():
    v = EstimatorValue(0.5, "ksg")
    assert v > 0.3
    assert v < 0.8
    assert v >= 0.5
    assert v <= 0.5


def test_add_same_estimator_returns_estimator_value():
    a = EstimatorValue(0.3, "hist")
    b = EstimatorValue(0.2, "hist")
    result = a + b
    assert isinstance(result, EstimatorValue)
    assert result.estimator == "hist"
    assert float(result) == pytest.approx(0.5)


def test_add_plain_float_preserves_tag():
    v = EstimatorValue(0.3, "ksg")
    result = v + 0.1
    assert isinstance(result, EstimatorValue)
    assert result.estimator == "ksg"
    assert float(result) == pytest.approx(0.4)


def test_add_cross_estimator_raises_typeerror():
    with pytest.raises(TypeError, match="Cross-estimator arithmetic"):
        _ = EstimatorValue(0.3, "hist") + EstimatorValue(0.2, "ksg")


def test_repr_contains_estimator():
    v = EstimatorValue(0.223456, "ksg")
    r = repr(v)
    assert "ksg" in r
    assert "EstimatorValue" in r


def test_str_contains_estimator():
    v = EstimatorValue(0.223456, "hist")
    assert "hist" in str(v)


def test_hash_usable_in_set():
    a = EstimatorValue(1.0, "ksg")
    b = EstimatorValue(2.0, "ksg")
    s = {a, b}
    assert len(s) == 2


def test_numpy_array_from_estimator_values():
    vals = [EstimatorValue(0.1, "ksg"), EstimatorValue(0.2, "ksg")]
    arr = np.array(vals)
    assert arr.dtype == np.float64
    np.testing.assert_allclose(arr, [0.1, 0.2])


# ---------- to_common_basis ----------

def test_to_common_basis_validates_input_type():
    from itpu import to_common_basis
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    with pytest.raises(TypeError, match="EstimatorValue"):
        to_common_basis(1.0, "ksg", x, y)


def test_to_common_basis_returns_estimator_value():
    from itpu import ITPU, to_common_basis
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    y = 0.5 * x + rng.standard_normal(500)
    itpu = ITPU()
    mi_hist = itpu.mutual_info(x, y, method="hist", bins=32)
    result = to_common_basis(mi_hist, "ksg", x, y, k=5)
    assert isinstance(result, EstimatorValue)
    assert result.estimator == "ksg"


# ---------- SurrogateResult ----------

def test_surrogate_result_rejects_mismatched_estimator():
    mi = EstimatorValue(0.3, "hist")
    with pytest.raises(ValueError, match="estimator mismatch"):
        SurrogateResult(mi=mi, p_value=0.1, n_surrogates=99, estimator="ksg")


def test_surrogate_result_accepts_matched_estimator():
    mi = EstimatorValue(0.3, "ksg")
    result = SurrogateResult(mi=mi, p_value=0.1, n_surrogates=99, estimator="ksg")
    assert result.mi.estimator == "ksg"
    assert result.p_value == 0.1
    assert result.n_surrogates == 99


def test_surrogate_result_p_value_is_float():
    mi = EstimatorValue(0.5, "hist")
    result = SurrogateResult(mi=mi, p_value=0.04, n_surrogates=499, estimator="hist")
    assert isinstance(result.p_value, float)


def test_surrogate_result_ks_stat_defaults_to_nan():
    mi = EstimatorValue(0.2, "ksg")
    result = SurrogateResult(mi=mi, p_value=0.2, n_surrogates=49, estimator="ksg")
    assert np.isnan(result.ks_stat)
