# SPDX-License-Identifier: Apache-2.0
import numpy as np
from itpu import ITPU

def test_hist_mi_positive_and_reasonable():
    rng = np.random.default_rng(1)
    N = 50_000
    x = rng.normal(size=N)
    y = 0.7 * x + 0.3 * rng.normal(size=N)

    itpu = ITPU()
    mi = itpu.mutual_info(x, y, method="hist", bins=128)

    rho = np.corrcoef(x, y)[0,1]
    mi_ref = -0.5 * np.log(1 - rho**2 + 1e-12)

    assert mi > 0
    # histogram MI will be biased low; allow 30% tolerance
    assert abs(mi - mi_ref) / mi_ref < 0.3
