import numpy as np
import pytest

def test_public_imports():
    from itpu.sdk import ITPU
    from itpu.utils.windowed import windowed_mi
    assert ITPU is not None and callable(windowed_mi)

def test_ksg_null_is_small():
    from itpu.sdk import ITPU
    itpu = ITPU()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20_000)
    y = rng.standard_normal(20_000)
    mi = itpu.mutual_info(x, y, method="ksg", k=5)
    assert mi < 0.05
