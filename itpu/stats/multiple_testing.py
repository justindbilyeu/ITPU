from __future__ import annotations

import numpy as np


def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Benjamini-Hochberg procedure to control the false discovery rate.

    Ranks p-values in ascending order and compares each against the BH
    critical value (rank / m) * alpha, where m is the total number of tests.
    All hypotheses up to and including the largest rank that satisfies the
    criterion are rejected.

    Parameters
    ----------
    p_values:
        1D array of raw p-values, one per hypothesis.
    alpha:
        Target false discovery rate (FDR) level. Must be in (0, 1).

    Returns
    -------
    corrected_p : np.ndarray
        BH-adjusted p-values (same length as p_values), computed as
        p * m / rank so that reject = corrected_p <= alpha.
    reject : np.ndarray of bool
        True for each hypothesis that is rejected at the given FDR level.
    """
    raise NotImplementedError
