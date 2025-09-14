"""Core SDK providing mutual information utilities."""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, Dict, Union

from .kernels_sw.ksg import ksg_mi_estimate

__all__ = ["ITPU"]


def _entropy_from_hist(counts: np.ndarray) -> float:
    p = counts.astype(float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p /= total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _mi_hist(x: np.ndarray, y: np.ndarray, bins: int = 64) -> Tuple[float, float, float, float]:
    hx, _ = np.histogram(x, bins=bins)
    hy, _ = np.histogram(y, bins=bins)
    hxy, _, _ = np.histogram2d(x, y, bins=bins)

    Hx = _entropy_from_hist(hx)
    Hy = _entropy_from_hist(hy)
    Hxy = _entropy_from_hist(hxy)

    mi_raw = Hx + Hy - Hxy
    kx = np.count_nonzero(hx)
    ky = np.count_nonzero(hy)
    kxy = np.count_nonzero(hxy)
    bias = ((kx - 1) * (ky - 1) - (kxy - 1)) / (2 * len(x))
    mi = float(max(mi_raw - bias, 0.0))
    Hxy_corr = Hxy + bias
    return mi, Hx, Hy, Hxy_corr


class ITPU:
    """Device-agnostic API for mutual information computations."""

    def __init__(self, device: str = "software") -> None:
        if device != "software":
            raise NotImplementedError("Only device='software' is supported today.")
        self.device = device

    def mutual_info(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "hist",
        output: str = "mi",
        **kwargs,
    ) -> Union[float, Tuple[float, float, float], Dict[str, float]]:
        """Mutual information between 1D arrays ``x`` and ``y``.

        Parameters
        ----------
        x, y : array-like
            Input vectors of equal length.
        method : {"hist", "ksg"}
            Estimation method.
        output : {"mi", "entropies", "all"}
            Output format for histogram method. KSG only supports "mi".
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have same length.")

        if method == "hist":
            bins = int(kwargs.get("bins", 64))
            mi, Hx, Hy, Hxy = _mi_hist(x, y, bins=bins)
            if output == "mi":
                return mi
            elif output == "entropies":
                return {"Hx": Hx, "Hy": Hy, "Hxy": Hxy}
            elif output == "all":
                return {"mi": mi, "Hx": Hx, "Hy": Hy, "Hxy": Hxy}
            else:
                raise ValueError("Unknown output option")
        elif method == "ksg":
            k = int(kwargs.get("k", 5))
            metric = kwargs.get("metric", "chebyshev")
            mi, _ = ksg_mi_estimate(x, y, k=k, metric=metric)
            if output == "mi":
                return mi
            elif output in ("entropies", "all"):
                raise NotImplementedError("Entropy outputs not supported for KSG")
            else:
                raise ValueError("Unknown output option")
        else:
            raise ValueError(f"Unknown method: {method}")

    def mutual_info_matrix(
        self,
        X: np.ndarray,
        method: str = "hist",
        bins: int = 64,
        pairs: Union[str, Iterable[Tuple[int, int]]] = "all",
        mask: np.ndarray | None = None,
    ) -> Union[np.ndarray, Dict[Tuple[int, int], float]]:
        """Pairwise mutual information matrix for columns of ``X``."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if mask is not None:
            X = X[np.asarray(mask, dtype=bool)]
        n_features = X.shape[1]

        if pairs == "all":
            M = np.zeros((n_features, n_features), dtype=float)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    mi = self.mutual_info(X[:, i], X[:, j], method=method, bins=bins)
                    M[i, j] = M[j, i] = mi
            return M
        else:
            out: Dict[Tuple[int, int], float] = {}
            for i, j in pairs:
                mi = self.mutual_info(X[:, i], X[:, j], method=method, bins=bins)
                out[(i, j)] = mi
            return out
