"""Permutation test utilities."""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Optional, Any


def perm_test(
    stat_fn: Callable[[Any, Any], float],
    X: Any,
    Y: Any,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray | float]:
    """Perform a permutation test on statistic ``stat_fn``.

    Parameters
    ----------
    stat_fn : callable
        Function computing a statistic ``stat_fn(X, Y)`` returning a float.
    X, Y : Any
        Data arrays. ``Y`` will be permuted along its first axis.
    n_perm : int, optional
        Number of permutations to sample for the null distribution. Defaults
        to 1000.
    rng : np.random.Generator, optional
        Optional random number generator for deterministic behaviour.

    Returns
    -------
    dict
        Dictionary with keys ``"obs"`` (observed statistic), ``"null"``
        (array of null statistics) and ``"p"`` (two-sided p-value).
    """
    rng = np.random.default_rng(rng)

    obs = float(stat_fn(X, Y))
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        Y_perm = rng.permutation(Y)
        null[i] = stat_fn(X, Y_perm)

    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"obs": obs, "null": null, "p": p}
