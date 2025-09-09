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
# --- Additions for batched MI -----------------------------------------------

from typing import Iterable, Optional, Tuple, Union

import numpy as np


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be 1D or 2D: (n_samples,) or (n_samples, n_features)")
    return X


class ITPU:  # (ensure this matches your existing class name)
    # ... keep your existing __init__ and mutual_info() here ...

    def mutual_info_matrix(
        self,
        X: np.ndarray,
        *,
        method: str = "hist",
        bins: int = 128,
        k: int = 5,
        pairs: Union[str, Iterable[Tuple[int, int]]] = "all",
        mask: Optional[np.ndarray] = None,
        metric: str = "euclidean",
        device: Optional[str] = None,
    ):
        """
        Batched MI for many pairs or a full symmetric matrix.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples,)
            Input features. If 1D, treated as single feature column.
        method : {"hist","ksg"}
            Backend estimator to use.
        bins : int
            Histogram bins (if method="hist").
        k : int
            k for KSG (if method="ksg").
        pairs : "all" or iterable of (i,j)
            If "all", returns full (d x d) symmetric matrix with diag=0.
            If iterable, returns dict {(i,j): mi_ij}.
        mask : array shape (n_samples,), optional
            Boolean mask of valid rows. (Simple 1D sample mask.)
        metric : str
            Distance metric for KSG (reserved; software path loops to mutual_info).
        device : str or None
            Optional override of device (e.g., "software", "fpga" later).

        Returns
        -------
        np.ndarray or dict
            - If pairs="all": (d x d) numpy array of MI values (diag=0).
            - Else: dict mapping (i,j) -> MI.
        """
        X = _as_2d(X)
        n, d = X.shape

        if device is not None:
            # optional override; for now we only support "software"
            pass

        # simple sample mask support
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (n,):
                raise ValueError("mask must be shape (n_samples,)")
            X = X[mask]
            n = X.shape[0]
            if n < 2:
                raise ValueError("Not enough valid samples after mask.")

        # If specific pairs are provided, compute and return a dict
        if pairs != "all":
            out = {}
            for (i, j) in pairs:
                xi = X[:, i]
                yj = X[:, j]
                mi_ij = self.mutual_info(xi, yj, method=method, bins=bins, k=k)
                out[(i, j)] = float(mi_ij)
            return out

        # Otherwise compute full symmetric matrix
        M = np.zeros((d, d), dtype=float)
        for i in range(d):
            xi = X[:, i]
            for j in range(i + 1, d):
                yj = X[:, j]
                mij = self.mutual_info(xi, yj, method=method, bins=bins, k=k)
                M[i, j] = M[j, i] = float(mij)

        # By convention keep diag at 0 (MI(X,X) is H(X), not what we want here)
        return M
