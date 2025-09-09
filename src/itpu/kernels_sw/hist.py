# SPDX-License-Identifier: Apache-2.0
import numpy as np

def joint_hist2d(x, y, bins=128, range=None):
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range)
    return H, xedges, yedges

def entropy_from_hist(H):
    H = np.asarray(H, dtype=float)
    total = H.sum()
    if total <= 0:
        return 0.0
    p = H / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def mi_from_hist(H):
    H = np.asarray(H, dtype=float)
    total = H.sum()
    if total <= 0:
        return 0.0
    px = H.sum(axis=1)
    py = H.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pxy = H / total
        px = px / total
        py = py / total
        # avoid zeros
        mask = pxy > 0
        ratio = np.zeros_like(pxy)
        ratio[mask] = pxy[mask] / (px[:, None] * py[None, :])[mask]
        log_ratio = np.zeros_like(pxy)
        valid = mask & (ratio > 0)
        log_ratio[valid] = np.log(ratio[valid])
        return float(np.sum(pxy * log_ratio))
