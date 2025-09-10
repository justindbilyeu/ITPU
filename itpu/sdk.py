if method == "ksg":
    from .kernels_sw.ksg import mi_ksg
    return float(mi_ksg(
        x, y,
        k=kwargs.get("k", 5),
        metric=kwargs.get("metric", "chebyshev"),
        mask=kwargs.get("mask", None),
        jitter=kwargs.get("jitter", 0.0),
        seed=kwargs.get("seed", None)
    ))
# inside itpu/sdk.py
def mutual_info(self, x, y, method="hist", bins=64, k=5, metric="euclidean"):
    import numpy as np
    x = np.asarray(x); y = np.asarray(y)

    if method == "hist":
        # your existing histogram MI impl or call into kernels_sw.hist
        from itpu.utils.windowed import _hist_mi
        return _hist_mi(x, y, bins=bins)

    if method == "ksg":
        from itpu.kernels_sw.ksg import ksg_mi_estimate
        mi, _ = ksg_mi_estimate(x, y, k=k, metric=metric)
        return mi

    raise ValueError(f"Unknown method: {method}")
