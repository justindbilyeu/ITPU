# SPDX-License-Identifier: Apache-2.0
import numpy as np
from .kernels_sw.hist import joint_hist2d, entropy_from_hist, mi_from_hist
from .kernels_sw.ksg import ksg_mi_1d

class ITPU:
    """
    Software SDK baseline for the ITPU coprocessor.
    Use device="software" today; later swap to "card0" without changing your code.
    """
    def __init__(self, device: str = "software"):
        if device not in {"software"}:
            # hardware backends will be added later
            raise ValueError("Only device='software' is available in this baseline.")
        self.device = device

    # ---- Histograms / Entropy -------------------------------------------------

    def build_hist(self, x, y=None, bins=128, range=None):
        """
        Build joint (and marginals) histograms.
        Returns (hist2d, p_x, p_y, total_count)
        """
        x = np.asarray(x)
        if y is None:
            # 1D histogram
            H, edges = np.histogram(x, bins=bins, range=range)
            return H, edges
        y = np.asarray(y)
        H, xedges, yedges = joint_hist2d(x, y, bins=bins, range=range)
        px = H.sum(axis=1)
        py = H.sum(axis=0)
        return H, px, py, H.sum()

    def entropy(self, x, bins=128, range=None):
        H, edges = np.histogram(np.asarray(x), bins=bins, range=range)
        return entropy_from_hist(H)

    def mutual_info(self, x, y, method="hist", bins=128, k=5):
        """
        Mutual information I(X;Y)
        method: "hist" (plug-in), "ksg" (Kraskov 1D)
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        assert x.shape == y.shape, "x and y must have same length"

        if method == "hist":
            H, _, _, total = self.build_hist(x, y, bins=bins)
            return mi_from_hist(H)
        elif method == "ksg":
            return ksg_mi_1d(x, y, k=k)
        else:
            raise ValueError("method must be 'hist' or 'ksg'")

    # ---- Streaming derivatives (placeholder API) ------------------------------

    def witness_flux_deriv(self, series, dt):
        """
        Simple centered finite-difference derivative.
        Hardware path will provide streaming implementation.
        """
        s = np.asarray(series, dtype=float)
        d = np.gradient(s, float(dt))
        return d
