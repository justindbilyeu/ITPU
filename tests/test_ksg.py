# SPDX-License-Identifier: Apache-2.0
import numpy as np
from itpu import ITPU

def test_ksg_mi_close_to_gaussian_reference():
    rng = np.random.default_rng(2)
    N = 20_000
    x = rng.normal(size=N)
    y = 0.6 * x + 0.4 * rng.normal(size=N)

    itpu = ITPU()
    mi = itpu.mutual_info(x, y, method="ksg", k=5)

    rho = np.corrcoef(x, y)[0,1]
    mi_ref = -0.5 * np.log(1 - rho**2 + 1e-12)

    # KSG should be closer; allow 20% tolerance
    assert abs(mi - mi_ref) / mi_ref < 0.2
