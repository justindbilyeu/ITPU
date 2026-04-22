import numpy as np
from itpu.kernels_sw.ksg import ksg_mi_estimate

def test_independent_gaussians():
    np.random.seed(0)
    x = np.random.randn(50000)
    y = np.random.randn(50000)
    mi, _ = ksg_mi_estimate(x, y, k=5)
    assert abs(mi) < 0.03

def test_correlated_gaussian():
    np.random.seed(1)
    n = 50000
    rho = 0.6
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal([0, 0], cov, size=n)
    x, y = data[:, 0], data[:, 1]
    mi, _ = ksg_mi_estimate(x, y, k=5)
    assert 0.20 <= mi <= 0.25  # analytic MI ≈ 0.223

def test_zero_variance():
    import warnings
    x = np.ones(1000)
    y = np.ones(1000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mi, stats = ksg_mi_estimate(x, y, k=5)
    assert mi == 0.0
    assert any("near-zero radius" in str(warning.message) for warning in w)

def test_chebyshev_accuracy_vs_true_mi():
    # KSG-1 is derived for the L∞ (Chebyshev) metric; Euclidean is a different
    # variant that carries a systematic downward bias (~0.26 nats, constant
    # across rho values) because the L2 ball is larger than L∞ for the same k,
    # inflating marginal neighbor counts and reducing the MI estimate.
    # The two metrics should NOT agree — the original parity test was only
    # passing because both paths were using the same (wrong) Euclidean
    # implementation. This test verifies that Chebyshev tracks the true MI.
    # True MI for bivariate Gaussian: -0.5 * log(1 - rho^2).
    rng = np.random.default_rng(2)
    n, rho = 5000, 0.6
    data = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    x, y = data[:, 0], data[:, 1]
    true_mi = -0.5 * np.log(1 - rho**2)  # ≈ 0.223 nats
    mi, _ = ksg_mi_estimate(x, y, metric="chebyshev")
    assert abs(mi - true_mi) < 0.05
