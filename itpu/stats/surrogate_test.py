from __future__ import annotations

from typing import Any

import numpy as np

from itpu.sdk import ITPU
from itpu.stats.surrogates import block_bootstrap_surrogate, iaaft_surrogate, shuffle_surrogate


def surrogate_test(
    x,
    y,
    method: str = "ksg",
    n_surrogates: int = 1000,
    surrogate_type: str = "shuffle",
    fdr_alpha: float = 0.05,
    rng=None,
) -> dict[str, Any]:
    """Test for statistical dependence between x and y using surrogate resampling.

    Estimates mutual information between x and y with the chosen MI estimator,
    then builds a null distribution by computing MI between x and n_surrogates
    surrogate copies of y (generated via surrogate_type). The empirical p-value
    is the fraction of null MI values >= the observed MI.

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
        FDR level for downstream Benjamini-Hochberg correction. This parameter
        is accepted for API compatibility with future batch usage but is NOT
        applied internally — p_value is always the raw permutation p-value.
        Batch FDR correction is a follow-on feature, not implemented here.
    rng:
        Seed or numpy Generator for reproducibility. Passed to the surrogate
        generator. Default None produces non-deterministic results.

    Returns
    -------
    dict with keys:
        mi_observed : float
            MI estimate (nats) between x and y.
        null_distribution : np.ndarray, shape (n_surrogates,)
            MI values computed under the null (x vs shuffled y).
        p_value : float
            Empirical permutation p-value:
            (sum(null >= mi_observed) + 1) / (n_surrogates + 1).
            This formula is locked — do not substitute an alternative.
        power_estimate : float
            Proxy for power: proportion of null distribution below mi_observed,
            i.e. mean(null < mi_observed). This is not a formal power analysis.
        warnings : list[str]
            Diagnostic messages. Contains one entry if mi_observed is below
            the null mean (possible estimator bias or insufficient sample size).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    sdk = ITPU(device="software")
    mi_observed = sdk.mutual_info(x, y, method=method)

    if surrogate_type == "shuffle":
        surrogates = shuffle_surrogate(y, n_surrogates=n_surrogates, rng=rng)
    elif surrogate_type == "block":
        block_size = max(1, len(x) // 20)
        surrogates = block_bootstrap_surrogate(
            y, block_size=block_size, n_surrogates=n_surrogates, rng=rng
        )
    elif surrogate_type == "iaaft":
        surrogates = iaaft_surrogate(y, n_surrogates=n_surrogates, rng=rng)
    else:
        raise ValueError(f"Unknown surrogate_type: {surrogate_type!r}. Use 'shuffle', 'block', or 'iaaft'.")

    null_distribution = np.array([
        sdk.mutual_info(x, surrogates[i], method=method)
        for i in range(n_surrogates)
    ])

    p_value = (np.sum(null_distribution >= mi_observed) + 1) / (n_surrogates + 1)
    power_estimate = float(np.mean(null_distribution < mi_observed))

    warning_messages = []
    if mi_observed < np.mean(null_distribution):
        warning_messages.append(
            "Observed MI below null mean — possible estimator bias or insufficient sample size."
        )

    return {
        "mi_observed": float(mi_observed),
        "null_distribution": null_distribution,
        "p_value": float(p_value),
        "power_estimate": power_estimate,
        "warnings": warning_messages,
    }
