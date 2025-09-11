import numpy as np
from itpu.sdk import ITPU
from itpu.utils.windowed import windowed_mi

def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50_000)
    y = 0.6 * x + 0.4 * rng.normal(size=50_000)

    itpu = ITPU(device="software")

    mi_hist = itpu.mutual_info(x, y, method="hist", bins=64)
    print(f"[hist] MI (nats): {mi_hist:.3f}")

    starts, vals = windowed_mi(x, y, window_size=5000, hop_size=1000, bins=64)
    print(f"[hist] windowed count: {len(vals)}, mean MI: {vals.mean():.3f}")

    # Optional: try KSG if available
    try:
        mi_ksg = itpu.mutual_info(x, y, method="ksg", k=5)
        print(f"[ksg]  MI (nats): {mi_ksg:.3f}")
    except Exception as e:
        print(f"[ksg]  skipped ({e})")

if __name__ == "__main__":
    main()
