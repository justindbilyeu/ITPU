# SPDX-License-Identifier: Apache-2.0
import numpy as np
from itpu import ITPU

def main():
    rng = np.random.default_rng(0)
    N = 200_000
    x = rng.normal(size=N)
    y = 0.8 * x + 0.2 * rng.normal(size=N)

    itpu = ITPU(device="software")

    mi_hist = itpu.mutual_info(x, y, method="hist", bins=128)
    mi_ksg  = itpu.mutual_info(x, y, method="ksg",  k=5)

    # analytic MI for jointly Gaussian with corr rho:  -0.5*ln(1-rho^2)
    rho = np.corrcoef(x, y)[0,1]
    mi_gauss = -0.5 * np.log(1 - rho**2 + 1e-12)

    print(f"Histogram MI: {mi_hist:.3f}")
    print(f"KSG MI      : {mi_ksg:.3f}")
    print(f"Gaussian MI : {mi_gauss:.3f} (reference)")

if __name__ == "__main__":
    main()
