# examples/mi_quickstart_eeg_eye_state.py
import os, numpy as np, matplotlib.pyplot as plt
import pandas as pd

try:
    # optional convenience: pip install ucimlrepo
    from ucimlrepo import fetch_ucirepo
    _HAS_UCI = True
except Exception:
    _HAS_UCI = False

from itpu.sdk import ITPU

def load_eeg_eye_state():
    """
    Returns (X, colnames). X: (n_samples, n_channels) float array of 14 EEG channels.
    If ucimlrepo is unavailable, tries local CSV at data/eeg_eye_state.csv.
    """
    if _HAS_UCI:
        ds = fetch_ucirepo(id=264)  # EEG Eye State
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    else:
        path = "data/eeg_eye_state.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(
                "EEG Eye State not found. Either `pip install ucimlrepo` or put CSV at data/eeg_eye_state.csv"
            )
        df = pd.read_csv(path)

    # Expect 14 EEG channels + label; drop label if present
    label_cols = [c for c in df.columns if str(c).lower() in {"eye", "eye_state", "label", "target"}]
    df_feat = df.drop(columns=label_cols, errors="ignore")
    # keep first 14 numerical columns
    df_feat = df_feat.select_dtypes(include=[np.number]).iloc[:, :14]
    X = df_feat.to_numpy(dtype=float)
    colnames = [str(c) for c in df_feat.columns]
    return X, colnames

def mi_matrix_hist(X, bins=64):
    """
    Compute pairwise MI matrix using ITPU().mutual_info(method='hist')
    """
    itpu = ITPU(device="software")  # uses software path today
    n = X.shape[1]
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            mi = itpu.mutual_info(X[:, i], X[:, j], method="hist", bins=bins)
            M[i, j] = M[j, i] = float(mi)
    return M

def main():
    os.makedirs("results/demos", exist_ok=True)
    X, names = load_eeg_eye_state()
    M = mi_matrix_hist(X, bins=128)

    # Save CSV
    out_csv = "results/demos/eeg_eye_state_mi.csv"
    pd.DataFrame(M, index=names, columns=names).to_csv(out_csv, float_format="%.6f")

    # Plot heatmap
    plt.figure(figsize=(7, 6))
    im = plt.imshow(M, interpolation="nearest", aspect="auto")
    plt.xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Mutual information (nats)")
    plt.title("EEG Eye State — Channel MI (histogram, bins=128)")
    plt.tight_layout()
    out_png = "results/demos/eeg_eye_state_mi.png"
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_csv}\nSaved: {out_png}")

if __name__ == "__main__":
    main()
