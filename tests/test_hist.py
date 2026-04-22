import numpy as np
from itpu.sdk import ITPU

def test_mi_independent_near_zero():
    # Histogram MI has positive bias ~(bins-1)^2 / (2*N) (Miller-Madow).
    # At bins=64, N=5000: bias ~0.40 nats — far above the desired threshold.
    # Use bins=16, N=10000 where bias ~(15^2)/(20000) = 0.011 nats, keeping
    # the original intent: independent data should yield near-zero MI.
    rng = np.random.default_rng(0)
    x = rng.normal(size=10000); y = rng.normal(size=10000)
    mi = ITPU(device="software").mutual_info(x, y, method="hist", bins=16)
    assert mi >= 0 and mi < 0.05

def test_mi_linear_positive():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000); y = 0.7*x + 0.3*rng.normal(size=5000)
    mi = ITPU(device="software").mutual_info(x, y, method="hist", bins=64)
    assert mi > 0.1
