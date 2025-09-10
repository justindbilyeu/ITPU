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
    x = np.ones(1000)
    y = np.random.randn(1000)
    mi, stats = ksg_mi_estimate(x, y, k=5)
    assert mi == 0.0
    assert "note" in stats

def test_metric_toggle_parity():
    np.random.seed(2)
    x = np.random.randn(2000)
    y = x + 0.1 * np.random.randn(2000)
    mi1, _ = ksg_mi_estimate(x, y, metric="chebyshev")
    mi2, _ = ksg_mi_estimate(x, y, metric="euclidean")
    assert abs(mi1 - mi2) < 0.05
