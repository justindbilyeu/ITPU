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
