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
# examples/mi_demo.py
"""
Minimal MI demo: generates synthetic correlated features, computes full MI matrix,
and saves a heatmap to results/examples/mi_matrix_heatmap.png

Run:
  python examples/mi_demo.py --n 60000 --d 8 --rho 0.6 --method hist --bins 128
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from itpu.sdk import ITPU


def make_correlated_features(n=60_000, d=8, rho=0.6, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, d))
    X = rho * z + np.sqrt(1 - rho**2) * noise
    return X.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--method", choices=["hist","ksg"], default="hist")
    ap.add_argument("--bins", type=int, default=128)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="results/examples")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X = make_correlated_features(n=args.n, d=args.d, rho=args.rho, seed=42)

    itpu = ITPU(device="software")
    M = itpu.mutual_info_matrix(
        X,
        method=args.method,
        bins=args.bins,
        k=args.k,
        pairs="all",
    )

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(M, origin="lower", interpolation="nearest", aspect="auto")
    plt.colorbar(label="Mutual Information (nats)")
    plt.title(f"Pairwise MI ({args.method})")
    plt.xlabel("feature")
    plt.ylabel("feature")

    out_png = os.path.join(args.out_dir, "mi_matrix_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"Saved: {out_png}")

    # Optional: print a quick summary
    iu = np.triu_indices_from(M, k=1)
    print(f"mean off-diagonal MI = {M[iu].mean():.4f} nats")


if __name__ == "__main__":
    main()
