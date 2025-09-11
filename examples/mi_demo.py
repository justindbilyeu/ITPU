import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from itpu.sdk import ITPU
from itpu.utils.windowed import windowed_mi

def main(no_plot: bool):
    os.makedirs("results/examples", exist_ok=True)
    rng = np.random.default_rng(42)
    n = 60_000
    x = rng.normal(size=n)
    y = 0.7 * x + 0.3 * rng.normal(size=n)

    itpu = ITPU(device="software")
    mi_hist = itpu.mutual_info(x, y, method="hist", bins=64)
    try:
        mi_ksg = itpu.mutual_info(x, y, method="ksg", k=5)
    except Exception:
        mi_ksg = None

    starts, mi_series = windowed_mi(x, y, window_size=4000, hop_size=800, bins=64)

    print(f"[hist] point MI (nats): {mi_hist:.3f}")
    if mi_ksg is not None:
        print(f"[ksg ] point MI (nats): {mi_ksg:.3f}")

    if no_plot:
        return

    plt.figure(figsize=(9,4))
    plt.plot(starts, mi_series, lw=2)
    plt.title("Sliding-window MI (hist)")
    plt.xlabel("sample"); plt.ylabel("MI (nats)")
    out = "results/examples/mi_demo.png"
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"Wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true", help="skip figure generation")
    args = ap.parse_args()
    main(args.no_plot)
