# SPDX-License-Identifier: Apache-2.0
"""
Device-agnostic SDK for ITPU operations.
"""

from __future__ import annotations
import numpy as np
from .kernels_sw.hist import mutual_info_hist

class ITPU:
    """Information-Theoretic Processing Unit interface."""
    
    def __init__(self, device: str = "software"):
        if device not in ("software",):
            raise ValueError("Only 'software' backend available in v0.1")
        self.device = device

    def mutual_info(self, x, y, method: str = "hist", **kwargs):
        """
        Compute mutual information between x and y.
        
        Parameters:
            x, y: 1D arrays
            method: "hist" (only option in v0.1)
            **kwargs: passed to underlying implementation
            
        Returns:
            float: MI in nats
        """
        if method == "hist":
            mi, _ = mutual_info_hist(x, y, **kwargs)
            return mi
        else:
            raise NotImplementedError("Only method='hist' available in v0.1")

# Convenience function
def mutual_info(x, y, **kwargs):
    """Compute mutual information (convenience function)."""
    return mutual_info_hist(x, y, **kwargs)[0]
