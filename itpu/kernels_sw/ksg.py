# inside itpu/sdk.py
import numpy as np
from .kernels_sw.hist import mi_from_hist_2d  # whatever your hist MI helper is named
from .kernels_sw.ksg import ksg_mi_estimate

class ITPU:
    def __init__(self, device="software"):
        self.device = device  # "software" for now

    def mutual_info(self, x, y, method="hist", bins=128, k=5, metric="euclidean"):
        """
        Compute MI between 1D arrays x and y.
        method: "hist" or "ksg"
        returns MI (nats) as float
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have same length")

        if method == "hist":
            # your existing histogram path (example call shown)
            return float(mi_from_hist_2d(x, y, bins=bins))
        elif method == "ksg":
            mi, _stats = ksg_mi_estimate(x, y, k=int(k), metric=metric)
            return float(mi)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'hist' or 'ksg'.")
