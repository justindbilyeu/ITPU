"""
Windowed/streaming utilities for ITPU.
Re-exports and convenience functions for sliding-window MI analysis.
"""

from ..kernels_sw.hist import windowed_mi as _windowed_mi_hist

def windowed_mi(x, y, method="hist", **kwargs):
    """
    Compute mutual information over sliding windows.
    
    Parameters:
        x, y: 1D arrays of equal length
        method: str, "hist" (default) or "ksg" (when available)
        **kwargs: passed to underlying implementation
        
    Returns:
        starts: array of window start indices
        mi_vals: array of MI values per window
        
    For method="hist", additional kwargs:
        window_size: int, samples per window (default 1000)
        hop_size: int, samples between windows (default 200)  
        bins: int, histogram bins (default 64)
        base: float, log base for MI units (default np.e for nats)
    """
    if method == "hist":
        starts, mi_vals, _ = _windowed_mi_hist(x, y, **kwargs)
        return starts, mi_vals
    elif method == "ksg":
        # TODO: implement windowed KSG when ksg.py is ready
        raise NotImplementedError("Windowed KSG coming soon")
    else:
        raise ValueError(f"Unknown method: {method}")

# Convenience alias
streaming_mi = windowed_mi
