"""Statistical utilities for surrogate data, permutation tests, and FDR control."""

from .surrogates import iaaft_surrogate, block_shuffle
from .permutation import perm_test
from .fdr import fdr_bh

__all__ = [
    "iaaft_surrogate",
    "block_shuffle",
    "perm_test",
    "fdr_bh",
]
