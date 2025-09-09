# examples/mi_grid.py
"""
Sweep MI settings quickly and write a tiny CSV summary.
- Uses EEG Eye State if available; otherwise synthetic fallback.
- Computes pairwise MI matrix with histogram method for several bin counts.
- Writes results/examples/mi_grid.csv with a simple scalar summary per config.

Run:
  python examples/mi_grid.py
"""
import os
import csv
import numpy as np
from itpu.sdk import ITPU
from itpu.data import load_eeg_eye_state

OUT_CSV = "results/examples/mi_grid.csv"

def offdiag_mean(M: np.ndarray) -> float:
    iu = np.triu_indices_from(M, k=1)
    return float(M[iu].mean())

def main():
    os.makedirs("results/examples", exist_ok=True)
    X, y, fb = load_eeg_eye_state()
    itpu = ITPU(device="software")

    bins_list = [32, 64, 128, 256]
    rows = []
    for bins in bins_list:
        M = itpu.mutual_info_matrix(X, method="hist", bins=bins, pairs="all")
        rows.append({
            "method": "hist",
            "bins": bins,
            "metric": "offdiag_mean_mi",
            "value": f"{offdiag_mean(M):.6f}",
            "fallback_data": str(fb)
        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT_CSV}")
    for r in rows:
        print(r)

if __name__ == "__main__":
    main()
