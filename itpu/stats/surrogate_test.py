from __future__ import annotations

from typing import Any


def surrogate_test(
    x,
    y,
    method: str = "ksg",
    n_surrogates: int = 1000,
    surrogate_type: str = "shuffle",
    fdr_alpha: float = 0.05,
) -> dict[str, Any]:
    """Test for statistical dependence between x and y using surrogate resampling.

    Estimates mutual information between x and y with the chosen MI estimator,
    then builds a null distribution by computing MI between x and n_surrogates
    surrogate copies of y (generated via surrogate_type). The empirical p-value
    is the fraction of null MI values >= the observed MI.

    Optionally applies Benjamini-Hochberg FDR correction when called in a
    multi-test context (reserved for future batch interface; fdr_alpha is
    stored in the returned dict for downstream use).

    Parameters
    ----------
    x:
        1D array, first variable.
    y:
        1D array, second variable. Must be the same length as x.
    method:
        MI estimator to use. Currently supported: "ksg".
    n_surrogates:
        Number of surrogate samples used to build the null distribution.
    surrogate_type:
        Resampling strategy for generating surrogates. One of:
        "shuffle" (independent permutation) or "block" (block bootstrap).
    fdr_alpha:
        FDR level passed to benjamini_hochberg() for downstream correction.
        Not applied internally to the single-test p-value.

    Returns
    -------
    dict with keys:
        mi_observed : float
            MI estimate (nats) between x and y.
        null_distribution : np.ndarray, shape (n_surrogates,)
            MI values computed under the null (x independent of shuffled y).
        p_value : float
            Empirical p-value: fraction of null MI >= mi_observed.
        power_estimate : float
            Estimated statistical power based on the separation between
            mi_observed and the null distribution (heuristic, [0, 1]).
        warnings : list[str]
            Any diagnostic messages raised during estimation (e.g. from
            ksg_mi_estimate) collected and surfaced here.
    """
    raise NotImplementedError
