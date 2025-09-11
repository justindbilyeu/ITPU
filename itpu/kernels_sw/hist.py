# SPDX-License-Identifier: Apache-2.0
"""
Histogram-based entropy and mutual information estimators.
"""

from __future__ import annotations
import numpy as np

def mutual_info_hist(
    x, y, 
    bins: int = 64,
    base: float = np.e
) -> tuple[float, dict]:
    """
    Estimate mutual information using histogram method.
    
    Parameters:
        x, y: 1D arrays of equal length
        bins: number of histogram bins
        base: logarithm base (np.e for nats, 2 for bits)
        
    Returns:
        mi: mutual information value
        stats: dictionary with additional statistics
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    
    # Convert to probabilities
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    
    # Calculate entropies
    def entropy(p):
        p_pos = p[p > 0]
        return -np.sum(p_pos * np.log(p_pos) / np.log(base))
    
    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(pxy)
    
    mi = hx + hy - hxy
    
    stats = {
        "hx": hx,
        "hy": hy, 
        "hxy": hxy,
        "bins": bins,
        "base": base
    }
    
    return max(mi, 0.0), stats

def windowed_mi(
    x, y,
    window_size: int = 1000,
    hop_size: int = 200,
    bins: int = 32,
    base: float = np.e
):
    """
    Sliding window mutual information.
    
    Returns:
        starts: array of window start indices
        mi_vals: array of MI values
        extras: metadata dict
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = min(len(x), len(y))
    
    starts = np.arange(0, max(0, n - window_size + 1), hop_size)
    mi_vals = np.zeros(len(starts))
    
    for i, start in enumerate(starts):
        end = start + window_size
        mi_vals[i], _ = mutual_info_hist(
            x[start:end], y[start:end], 
            bins=bins, base=base
        )
    
    extras = {"window_size": window_size, "hop_size": hop_size, "bins": bins}
    return starts, mi_vals, extras
