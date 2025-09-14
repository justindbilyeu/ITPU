# tests/test_mi_gaussian.py
import numpy as np
from itpu.sdk import ITPU

def analytic_mi_gaussian(rho):
    return -0.5*np.log(1 - rho**2)

def test_hist_matches_gaussian_target():
    rng = np.random.default_rng(0)
    n = 50_000
    x = rng.standard_normal(n)
    y = 0.6*x + 0.4*rng.standard_normal(n)  # rho ~ 0.832
    target = analytic_mi_gaussian(np.corrcoef(x,y)[0,1])
    itpu = ITPU()
    mi = itpu.mutual_info(x, y, method="hist", bins=64)
    assert abs(mi - target) < 0.1  # loose but meaningful

def test_ksg_null_small():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(20_000)
    y = rng.standard_normal(20_000)
    itpu = ITPU()
    mi = itpu.mutual_info(x, y, method="ksg", k=5)
    assert mi < 0.05
