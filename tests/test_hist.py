import numpy as np
from itpu.sdk import ITPU

def test_mi_independent_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000); y = rng.normal(size=5000)
    mi = ITPU(device="software").mutual_info(x, y, method="hist", bins=64)
    assert mi >= 0 and mi < 0.1

def test_mi_linear_positive():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000); y = 0.7*x + 0.3*rng.normal(size=5000)
    mi = ITPU(device="software").mutual_info(x, y, method="hist", bins=64)
    assert mi > 0.1
