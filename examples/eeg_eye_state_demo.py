# examples/eeg_eye_state_demo.py
"""
EEG Eye State demo:
- Tries to load data/eeg_eye_state.csv with 14 EEG channels + binary label.
- Computes pairwise MI across channels and MI(channel, label).
- Saves heatmap + bar chart under results/examples/.

If the CSV is missing, falls back to synthetic data so the script still works.

Run:
  python examples/eeg_eye_state_demo.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from itpu.sdk import ITPU

DATA_PATH = "data/eeg_eye_state.csv"


def try_load_csv(path):
    if not os.path.exists(path):
        return None
    # Try to load with or without header; numeric only
    try:
        data = np.genfromtxt(path, delimiter=",", dtype=float, names=None)
        if data.ndim == 1:
            # Single row? not useful
            return None
        return data
    except Exception:
        return None


def make_fallback(n=15000, d=14, seed=7):
    # synthetic: 14 channels with mild correlations + binary label tied to ch0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, d))
    X = 0.5 * z + np.sqrt(1 - 0.5**2) * noise
    # label from ch0 threshold
    y = (X[:, 0] + 0.2 * rng.standard_normal(n) > 0).astype(float)
    return X, y


def main():
    os.makedirs("results/examples", exist_ok=True)

    raw = try_load_csv(DATA_PATH)
    if raw is None:
        print("EEG CSV not found; using synthetic fallback.")
        X, y = make_fallback()
    else:
        # If file has header, genfromtxt returns 2D float array already.
        # Expect at least 15 columns: 14 channels + label
        if raw.shape[1] < 15:
            print("EEG CSV has <15 columns; using synthetic fallback.")
            X, y = make_fallback()
        else:
            X = raw[:, :14].astype(float)
            y = raw[:, 14].astype(float)
            print(f"Loaded EEG data: X shape={X.shape}, y shape={y.shape}")

    itpu = ITPU(device="software")

    # Pairwise MI across channels
    M = itpu.mutual_info_matrix(X, method="hist", bins=128, pairs="all")

    # MI(channel, label)
    mi_ch_label = []
    for j in range(X.shape[1]):
        mij = itpu.mutual_info(X[:, j], y, method="hist", bins=128)
        mi_ch_label.append(mij)
    mi_ch_label = np.asarray(mi_ch_label)

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(M, origin="lower", interpolation="nearest", aspect="auto")
    plt.colorbar(label="MI (nats)")
    plt.title("EEG channels: pairwise MI")
    plt.xlabel("channel")
    plt.ylabel("channel")
    out_heat = "results/examples/eeg_mi_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_heat, dpi=160)
    print(f"Saved: {out_heat}")

    # Plot MI(channel, label)
    plt.figure(figsize=(6.5, 3.2))
    plt.bar(np.arange(len(mi_ch_label)), mi_ch_label)
    plt.xlabel("channel")
    plt.ylabel("MI(channel, label)")
    plt.title("Channel-to-label MI")
    out_bar = "results/examples/eeg_mi_to_label.png"
    plt.tight_layout()
    plt.savefig(out_bar, dpi=160)
    print(f"Saved: {out_bar}")

    # Small console summary
    iu = np.triu_indices_from(M, k=1)
    print(f"mean off-diagonal MI = {M[iu].mean():.4f} nats")
    print(f"top-3 channels by MI to label: {np.argsort(mi_ch_label)[-3:][::-1]}")


if __name__ == "__main__":
    main()
