from __future__ import annotations

import numpy as np
import pytest

from itpu.stats.multiple_testing import benjamini_hochberg


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_output_keys():
    result = benjamini_hochberg(np.array([0.1, 0.5, 0.9]))
    assert set(result.keys()) == {"corrected_p_values", "rejected", "n_rejected"}


def test_output_length_matches_input():
    for n in (1, 5, 100):
        p = np.random.default_rng(0).uniform(0, 1, n)
        result = benjamini_hochberg(p)
        assert len(result["corrected_p_values"]) == n
        assert len(result["rejected"]) == n


def test_n_rejected_consistent_with_rejected_array():
    p = np.array([0.001, 0.01, 0.3, 0.8])
    result = benjamini_hochberg(p, alpha=0.05)
    assert result["n_rejected"] == int(result["rejected"].sum())


# ---------------------------------------------------------------------------
# Rejection behaviour
# ---------------------------------------------------------------------------

def test_no_rejections_for_large_p_values():
    p = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    result = benjamini_hochberg(p, alpha=0.05)
    assert result["n_rejected"] == 0
    assert not result["rejected"].any()


def test_all_rejections_for_near_zero_p_values():
    p = np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6])
    result = benjamini_hochberg(p, alpha=0.05)
    assert result["n_rejected"] == len(p)
    assert result["rejected"].all()


def test_partial_rejection():
    # First two should be rejected, last two should not.
    p = np.array([0.001, 0.005, 0.6, 0.9])
    result = benjamini_hochberg(p, alpha=0.05)
    assert result["rejected"][0] and result["rejected"][1]
    assert not result["rejected"][2] and not result["rejected"][3]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_p_values_one():
    p = np.ones(10)
    result = benjamini_hochberg(p, alpha=0.05)
    assert result["n_rejected"] == 0
    assert np.allclose(result["corrected_p_values"], 1.0)


def test_all_p_values_zero():
    p = np.zeros(10)
    result = benjamini_hochberg(p, alpha=0.05)
    assert result["n_rejected"] == 10
    assert result["rejected"].all()


def test_single_p_value_rejected():
    result = benjamini_hochberg(np.array([0.01]), alpha=0.05)
    assert result["n_rejected"] == 1


def test_single_p_value_not_rejected():
    result = benjamini_hochberg(np.array([0.9]), alpha=0.05)
    assert result["n_rejected"] == 0


# ---------------------------------------------------------------------------
# Order preservation — the critical correctness invariant
# ---------------------------------------------------------------------------

def test_output_order_matches_input_order():
    # Deliberately unsorted input: smallest p-value is at index 2.
    p = np.array([0.5, 0.4, 0.001, 0.8, 0.02])
    result = benjamini_hochberg(p, alpha=0.05)
    # Index 2 (p=0.001) must be rejected; index 0 (p=0.5) must not.
    assert result["rejected"][2], "smallest p-value at index 2 should be rejected"
    assert not result["rejected"][0], "large p-value at index 0 should not be rejected"
    # corrected_p must be indexed the same way as input.
    assert result["corrected_p_values"][2] < result["corrected_p_values"][0]


def test_corrected_p_values_for_unsorted_input():
    # Verify adjusted values are mapped to original positions, not sorted positions.
    p_sorted = np.array([0.01, 0.03, 0.05, 0.2, 0.5])
    p_shuffled = p_sorted[[2, 0, 4, 1, 3]]  # known permutation

    r_sorted = benjamini_hochberg(p_sorted, alpha=0.05)
    r_shuffled = benjamini_hochberg(p_shuffled, alpha=0.05)

    # After reordering r_shuffled back to sorted order, corrected values must match.
    inv = np.argsort([2, 0, 4, 1, 3])
    np.testing.assert_allclose(
        r_shuffled["corrected_p_values"][inv],
        r_sorted["corrected_p_values"],
        err_msg="corrected_p_values not mapped back to original index order",
    )
