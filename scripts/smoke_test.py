# scripts/smoke_test.py
import numpy as np
from itpu.sdk import ITPU
from itpu.utils.windowed import windowed_mi

def main():
    rng = np.random.default_rng(42)
    n = 50_000
    x = rng.standard_normal(n)
    y = 0.6 * x + 0.4 * rng.standard_normal(n)

    itpu = ITPU(device="software")

    # Histogram MI (nats)
    mi_hist = itpu.mutual_info(x, y, method="hist", bins=64)
    print(f"[hist] MI (nats): {mi_hist:.4f}")

    # Sliding-window MI (hist)
    starts, mi_vals = windowed_mi(x, y, window_size=2000, hop_size=400, bins=64)
    print(f"[hist/windowed] windows: {len(mi_vals)}, first 3: {mi_vals[:3]}")

    # Optional: KSG demo (kept small for speed)
    n_small = 4000
    xs = x[:n_small]
    ys = y[:n_small]
    mi_ksg = itpu.mutual_info(xs, ys, method="ksg", k=5)
    print(f"[ksg] MI (nats): {mi_ksg:.4f}")

if __name__ == "__main__":
    main()
