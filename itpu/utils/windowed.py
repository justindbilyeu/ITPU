# SPDX-License-Identifier: Apache-2.0
"""
Windowed and streaming utilities.
"""

from ..kernels_sw.hist import windowed_mi as _windowed_mi_hist

def windowed_mi(x, y, method="hist", **kwargs):
    """
    Sliding window mutual information.
    
    Parameters:
        x, y: 1D arrays
        method: "hist" (only option in v0.1)
        **kwargs: passed to implementation
        
    Returns:
        starts: window start indices
        mi_vals: MI values per window
    """
    if method == "hist":
        starts, mi_vals, _ = _windowed_mi_hist(x, y, **kwargs)
        return starts, mi_vals
    else:
        raise NotImplementedError("Only method='hist' available")
