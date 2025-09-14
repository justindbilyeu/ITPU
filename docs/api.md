# API

## iaaft_surrogate(x, n_iter=1000, tol=1e-5, rng=None)
Generate an Iterative Amplitude Adjusted Fourier Transform surrogate preserving
both the amplitude spectrum and rank distribution of the input.

## block_shuffle(x, block_size, rng=None)
Shuffle a 1D array in non-overlapping blocks to retain coarse autocorrelation
structure up to roughly half the block size.

## perm_test(stat_fn, X, Y, n_perm=1000, rng=None)
Permutation test harness returning the observed statistic, a null distribution
from permuting ``Y`` and a two-sided p-value.

## fdr_bh(pvals, alpha=0.05)
Benjamini–Hochberg procedure for controlling the false discovery rate, returning
both the rejection mask and associated q-values.
