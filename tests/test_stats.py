import numpy as np

from itpu.stats.permutation import perm_test
from itpu.stats.fdr import fdr_bh


def test_permutation_detects_effect():
    rng = np.random.default_rng(0)
    n = 20_000

    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]
    data = rng.multivariate_normal(mean, cov, size=n)
    X, Y = data.T

    def stat_fn(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    res = perm_test(stat_fn, X, Y, n_perm=200, rng=rng)
    assert res["p"] < 0.05

    X_ind = rng.standard_normal(n)
    Y_ind = rng.standard_normal(n)
    res_ind = perm_test(stat_fn, X_ind, Y_ind, n_perm=200, rng=rng)
    assert res_ind["p"] > 0.1


def test_fdr_bh_properties():
    rng = np.random.default_rng(1)
    pvals = rng.uniform(size=100)

    rej05, q05 = fdr_bh(pvals, alpha=0.05)
    rej01, _ = fdr_bh(pvals, alpha=0.01)
    assert rej01.sum() <= rej05.sum()

    # Sorting p-values should not change results
    order = np.argsort(pvals)
    inv = np.argsort(order)
    rej_sorted, q_sorted = fdr_bh(pvals[order], alpha=0.05)
    assert np.array_equal(rej_sorted[inv], rej05)
    assert np.allclose(q_sorted[inv], q05)

    assert np.all((q05 >= 0) & (q05 <= 1))
