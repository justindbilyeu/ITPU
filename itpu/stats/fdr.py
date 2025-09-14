"""False discovery rate (FDR) control utilities."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control.

    Parameters
    ----------
    pvals : np.ndarray
        Array of p-values.
    alpha : float, optional
        Desired false discovery rate. Defaults to 0.05.

    Returns
    -------
    rejected : np.ndarray of bool
        Boolean array indicating which hypotheses are rejected.
    qvals : np.ndarray of float
        Adjusted p-values (q-values).
    """
    pvals = np.asarray(pvals, dtype=float)
    if np.any(pvals < 0) or np.any(pvals > 1):
        raise ValueError("p-values must be within [0, 1]")

    n = pvals.size
    order = np.argsort(pvals)
    sorted_p = pvals[order]

    qvals = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = sorted_p[i] * n / rank
        if q > prev:
            q = prev
        prev = q
        qvals[i] = q

    qvals_orig = np.empty_like(qvals)
    qvals_orig[order] = qvals
    rejected = qvals_orig <= alpha
    return rejected, qvals_orig
