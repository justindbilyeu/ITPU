import numpy as np

from itpu.kernels_sw.ksg import ksg_mi_estimate

__all__ = ["ITPU"]


class ITPU:
    """
    Device-agnostic API. device="software" supported now; future targets will
    preserve this interface.
    """

    def __init__(self, device="software"):
        if device != "software":
            raise NotImplementedError("Only device='software' is supported today.")
        self.device = device

    # ---------- Public API ----------
    def mutual_info(self, x, y, method="hist", **kwargs):
        """
        Mutual information between 1D arrays x,y (nats).

        method: "hist" (discrete/histogram) or "ksg" (continuous kNN).

        Warning — histogram bias: method="hist" has an uncorrected plug-in
        positive bias of approximately (bins-1)^2 / (2*N) nats. At bins=64
        and N=5000 this is ~0.40 nats, larger than many real effects. Use
        method="ksg" for quantitative accuracy, or keep bins low and N large
        (rule of thumb: (bins-1)^2 / (2*N) < 0.01).
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have same length.")

        if method == "hist":
            bins = int(kwargs.get("bins", 64))
            return _mi_hist(x, y, bins=bins)
        elif method == "ksg":
            k = int(kwargs.get("k", 5))
            mi, _ = ksg_mi_estimate(x, y, k=k, clip_zero=False)
            return mi
        else:
            raise ValueError(f"Unknown method: {method}")


# ---------- Histogram-based MI (nats) ----------
def _entropy_from_hist(counts):
    p = counts.astype(float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p /= total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _mi_hist(x, y, bins=64):
    """Plug-in histogram MI estimator (nats). Has uncorrected positive bias
    of ~(bins-1)^2 / (2*N). Use method='ksg' for quantitative accuracy."""
    hx, _ = np.histogram(x, bins=bins)
    hy, _ = np.histogram(y, bins=bins)
    hxy, _, _ = np.histogram2d(x, y, bins=bins)

    Hx = _entropy_from_hist(hx)
    Hy = _entropy_from_hist(hy)
    Hxy = _entropy_from_hist(hxy)

    return Hx + Hy - Hxy
