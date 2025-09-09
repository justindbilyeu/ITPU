# examples/windowed_demo.py
"""
Windowed MI demo with a change-point:
- Two signals are weakly correlated first, then strongly correlated.
- We compute MI over sliding windows and plot MI vs time index.

Run:
  python examples/windowed_demo.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from itpu.windowed import windowed_mi


def make_series(n=50_000, switch=25_000, rho1=0.2, rho2=0.8, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, 2))
    X = np.zeros((n, 2))
    X[:switch, :] = rho1 * z[:switch] + np.sqrt(1 - rho1**2) * noise[:switch]
    X[switch:, :] = rho2 * z[switch:] + np.sqrt(1 - rho2**2) * noise[switch:]
    return X[:, 0], X[:, 1]


def main():
    os.makedirs("results/examples", exist_ok=True)
    x, y = make_series()

    mi_vals, centers = windowed_mi(
        x, y, window_size=2000, step=200, method="hist", bins=64
    )

    plt.figure(figsize=(7.0, 3.2))
    plt.plot(centers, mi_vals, lw=1.5)
    plt.axvline(25_000, color="k", linestyle="--", linewidth=1)
    plt.xlabel("sample index (center of window)")
    plt.ylabel("MI (nats)")
    plt.title("Windowed MI with change-point at 25k")
    out_png = "results/examples/windowed_mi.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"Saved: {out_png}")
    print(f"mean MI first half: {mi_vals[centers < 25_000].mean():.4f}")
    print(f"mean MI second half: {mi_vals[centers >= 25_000].mean():.4f}")


if __name__ == "__main__":
    main()
