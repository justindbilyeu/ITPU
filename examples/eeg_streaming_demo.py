import numpy as np, matplotlib.pyplot as plt, os
from itpu.sdk import ITPU

def synthetic_eeg(n=50000, fs=128, alpha=10, beta=20, state_split=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)/fs
    x = np.sin(2*np.pi*alpha*t) + 0.4*rng.normal(size=n)
    y = 0.6*x + 0.4*np.sin(2*np.pi*beta*t) + 0.4*rng.normal(size=n)
    labels = np.zeros(n, dtype=int)
    labels[int(n*state_split):] = 1  # 0=eyes closed, 1=open (toy)
    return x, y, labels

def run_demo():
    out_dir = "results/demos"; os.makedirs(out_dir, exist_ok=True)
    x, y, labels = synthetic_eeg()

    itpu = ITPU(device="software")
    # histogram streaming first (fast)
    from itpu.kernels_sw.streaming import windowed_mi
    t_idx, mi_vals = windowed_mi(x, y, window_size=1000, hop_size=200, bins=128)

    plt.figure(figsize=(10,4))
    plt.plot(t_idx, mi_vals, label="MI (hist)")
    plt.title("Streaming MI (synthetic EEG)")
    plt.xlabel("Sample"); plt.ylabel("MI (nats)")
    plt.legend(); plt.tight_layout()
    fig_path = os.path.join(out_dir, "eeg_mi_timeseries.png")
    plt.savefig(fig_path, dpi=160)
    print("Saved:", fig_path)

if __name__ == "__main__":
    run_demo()
