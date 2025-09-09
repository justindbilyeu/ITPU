# tests/test_output_all.py
import numpy as np
from itpu.sdk import ITPU

def test_hist_returns_entropies_and_all_agree():
    rng = np.random.default_rng(0)
    n = 100_000
    # Correlated Gaussians -> discretize via histogram inside mutual_info
    z = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, 2))
    rho = 0.6
    X = rho * z + np.sqrt(1 - rho**2) * noise
    x = X[:,0]; y = X[:,1]

    itpu = ITPU(device="software")
    out_all = itpu.mutual_info(x, y, method="hist", bins=128, output="all")
    out_ent = itpu.mutual_info(x, y, method="hist", bins=128, output="entropies")
    mi_only = itpu.mutual_info(x, y, method="hist", bins=128, output="mi")

    assert set(out_all.keys()) == {"mi","Hx","Hy","Hxy"}
    assert set(out_ent.keys()) == {"Hx","Hy","Hxy"}

    # Hx + Hy - Hxy == MI (up to small numeric tolerance)
    lhs = out_all["Hx"] + out_all["Hy"] - out_all["Hxy"]
    assert abs(lhs - out_all["mi"]) < 1e-6
    assert abs(mi_only - out_all["mi"]) < 1e-6

def test_ksg_only_mi_supported():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000)
    y = 0.8 * x + np.sqrt(1 - 0.8**2) * rng.standard_normal(5000)
    itpu = ITPU(device="software")

    # Should not raise for MI
    _ = itpu.mutual_info(x, y, method="ksg", k=5, output="mi")

    # But entropies are not supported
    try:
        itpu.mutual_info(x, y, method="ksg", k=5, output="all")
        raised = False
    except NotImplementedError:
        raised = True
    assert raised
