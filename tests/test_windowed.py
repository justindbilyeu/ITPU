# tests/test_windowed.py
import numpy as np
from itpu.windowed import windowed_mi

def test_windowed_shapes_and_monotone_change():
    # create a simple change in correlation strength
    n, switch = 20_000, 10_000
    rng = np.random.default_rng(1)
    z = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, 2))
    rho1, rho2 = 0.1, 0.7
    X = np.zeros((n, 2))
    X[:switch, :] = rho1 * z[:switch] + np.sqrt(1 - rho1**2) * noise[:switch]
    X[switch:, :] = rho2 * z[switch:] + np.sqrt(1 - rho2**2) * noise[switch:]
    x, y = X[:,0], X[:,1]

    mi, centers = windowed_mi(x, y, window_size=2000, step=200, method="hist", bins=64)
    assert mi.ndim == 1 and centers.ndim == 1
    assert mi.shape == centers.shape
    # later windows should on average have higher MI than early ones
    early = mi[centers < switch]
    late  = mi[centers >= switch]
    assert late.mean() > early.mean()
# tests/test_windowed.py
import numpy as np
from itpu.windowed import windowed_mi

def test_windowed_shapes_and_monotone_change():
    # create a simple change in correlation strength
    n, switch = 20_000, 10_000
    rng = np.random.default_rng(1)
    z = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, 2))
    rho1, rho2 = 0.1, 0.7
    X = np.zeros((n, 2))
    X[:switch, :] = rho1 * z[:switch] + np.sqrt(1 - rho1**2) * noise[:switch]
    X[switch:, :] = rho2 * z[switch:] + np.sqrt(1 - rho2**2) * noise[switch:]
    x, y = X[:,0], X[:,1]

    mi, centers = windowed_mi(x, y, window_size=2000, step=200, method="hist", bins=64)
    assert mi.ndim == 1 and centers.ndim == 1
    assert mi.shape == centers.shape
    # later windows should on average have higher MI than early ones
    early = mi[centers < switch]
    late  = mi[centers >= switch]
    assert late.mean() > early.mean()
