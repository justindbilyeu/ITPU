from __future__ import annotations

import numpy as np


def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Apply the Benjamini-Hochberg procedure to control the false discovery rate.

    Ranks p-values in ascending order and compares each against the BH
    critical value (rank / m) * alpha, where m is the total number of tests.
    All hypotheses up to and including the largest rank that satisfies the
    criterion are rejected.

    Critical: output arrays are in the same order as the input p_values array,
    not sorted order. Sorting is done internally for the BH step; results are
    mapped back to original indices before returning.

    Parameters
    ----------
    p_values:
        1D array of raw p-values, one per hypothesis.
    alpha:
        Target false discovery rate (FDR) level. Must be in (0, 1).

    Returns
    -------
    dict with keys:
        corrected_p_values : np.ndarray
            BH-adjusted p-values, same length and order as input. Computed as
            min_{j >= rank_i} (m / j * p_(j)), clipped to [0, 1]. Reject when
            corrected_p_values[i] <= alpha.
        rejected : np.ndarray of bool
            True for each hypothesis rejected at the given FDR level.
        n_rejected : int
            Number of rejected hypotheses.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    # Sort ascending; track original positions for mapping back.
    order = np.argsort(p_values, kind="stable")
    sorted_p = p_values[order]

    # BH-adjusted p-value for rank i (1-based): p_(i) * m / i,
    # then take running minimum from the right so monotonicity is preserved.
    adjusted = sorted_p * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    np.clip(adjusted, 0.0, 1.0, out=adjusted)

    # Map back to original index order.
    corrected_p = np.empty(m, dtype=float)
    corrected_p[order] = adjusted

    rejected = corrected_p <= alpha

    return {
        "corrected_p_values": corrected_p,
        "rejected": rejected,
        "n_rejected": int(np.sum(rejected)),
    }
