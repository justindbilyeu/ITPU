import numpy as np
from scipy.stats import spearmanr

from itpu.stats.surrogates import iaaft_surrogate, block_shuffle


def _acf(x: np.ndarray, lag_max: int) -> np.ndarray:
    x = x - np.mean(x)
    acf = [1.0]
    denom = np.var(x)
    for lag in range(1, lag_max + 1):
        acf.append(np.correlate(x[:-lag], x[lag:])[0] / ((len(x) - lag) * denom))
    return np.array(acf)


def test_iaaft_preserves_spectrum_and_ranks():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024)
    s = iaaft_surrogate(x, rng=rng)

    orig = np.abs(np.fft.rfft(x))
    sur = np.abs(np.fft.rfft(s))
    orig /= np.linalg.norm(orig)
    sur /= np.linalg.norm(sur)
    rmse = np.sqrt(np.mean((orig - sur) ** 2))
    assert rmse <= 1e-3

    rho, _ = spearmanr(np.sort(x), np.sort(s))
    assert rho >= 0.99


def test_block_shuffle_preserves_coarse_acf():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    block_size = 64

    bs = block_shuffle(x, block_size, rng=rng)
    full = np.random.default_rng(1).permutation(x)

    lag_max = block_size // 2
    acf_x = _acf(x, lag_max)
    acf_bs = _acf(bs, lag_max)
    acf_full = _acf(full, lag_max)

    diff_bs = np.mean(np.abs(acf_x[1:] - acf_bs[1:]))
    diff_full = np.mean(np.abs(acf_x[1:] - acf_full[1:]))
    assert diff_bs < diff_full
