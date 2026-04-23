# ITPU — Information-Theoretic Processing Unit

**Fast entropy, mutual information, and k-NN statistics. Software today, silicon tomorrow.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/justindbilyeu/ITPU/actions/workflows/ci.yml/badge.svg)](https://github.com/justindbilyeu/ITPU/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-R1%20complete-green)]()

---

## What this is

Most compute hardware is optimized for matrix multiplication. Mutual information, entropy, and k-NN density estimation are fundamentally different workloads — irregular access patterns, branch-heavy, memory-bound — and they pay a steep penalty running on hardware built for transformers.

ITPU is building dedicated hardware for information-theoretic computation. The software SDK ships first: correct, calibrated, and designed so that swapping in an FPGA or ASIC backend later is a one-line change.

**What the software does right now:**
- Histogram MI and KSG (Kraskov–Stögbauer–Grassberger) MI, both returning nats
- Sliding-window MI for streaming and time-series data
- Surrogate testing with shuffle, block-bootstrap, and IAAFT null distributions
- IAAFT surrogates for autocorrelated/oscillatory data — preserves power spectrum and amplitude distribution
- Benjamini-Hochberg FDR correction
- Statistical calibration verified: KS=0.0565, p=0.1497 under H₀

**What's coming:**
- FPGA pathfinder (R2) — same SDK API, hardware backend
- Partner pilots: BCI/EEG, medical imaging, causal ML

---

## Quickstart

```bash
git clone https://github.com/justindbilyeu/ITPU
cd ITPU
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q -m "not slow"
```

---

## API

```python
import numpy as np
from itpu.sdk import ITPU
from itpu.utils.windowed import windowed_mi
from itpu.stats.surrogate_test import surrogate_test

rng = np.random.default_rng(0)
x = rng.standard_normal(50_000)
y = 0.6 * x + 0.4 * rng.standard_normal(50_000)

itpu = ITPU(device="software")

# Histogram MI — fast, works well at large n with appropriate bin count
mi_hist = itpu.mutual_info(x, y, method="hist", bins=32)

# KSG MI — non-parametric, Chebyshev metric, calibrated
# Note: keep n ≥ 10,000 for reliable estimates; KSG is slow at large n
mi_ksg = itpu.mutual_info(x[:10_000], y[:10_000], method="ksg", k=5)

# Sliding-window MI for time series
starts, mi_vals = windowed_mi(x, y, window_size=2000, hop_size=400)

# Surrogate test — calibrated permutation p-value
result = surrogate_test(x[:5_000], y[:5_000], method="ksg", n_surrogates=499)
print(f"MI = {result['mi_observed']:.3f} nats, p = {result['p_value']:.3f}")

# All values in nats. Divide by np.log(2) for bits.
```

---

## Estimator notes

**Histogram MI** has a positive finite-sample bias of approximately `(bins−1)² / (2N)` (Miller-Madow). At `bins=64, N=5000` this is ~0.40 nats — larger than many real effects. Rule of thumb: keep `(bins−1)² / (2N) < 0.01`, which means `bins=32` needs `N > 48,000`. Use fewer bins or more data when testing for near-zero MI.

**KSG MI** uses the L∞ (Chebyshev) metric, which is the theoretically correct choice for KSG variant 1. The Euclidean variant carries a constant ~0.26 nat downward bias regardless of true MI — don't use it for quantitative work. The default (`metric="chebyshev"`) is calibrated.

**Surrogate testing** uses a permutation p-value: `p = (#{null ≥ observed} + 1) / (n_surrogates + 1)`. The +1 smoothing is conservative and intentional. Calibration was verified with 400 independent H₀ trials at n=1000, n_surrogates=999 (KS test against Uniform[0,1]: statistic=0.0565, p=0.1497). For autocorrelated data, use `surrogate_type="iaaft"` — IAAFT preserves the power spectrum and amplitude distribution while randomizing phases. `shuffle` destroys temporal structure and will give miscalibrated results on autocorrelated data. See `docs/estimator_guide.md` for the surrogate selection decision tree.

---

## Repository layout

```
itpu/
  sdk.py                    # ITPU class — the public API
  kernels_sw/
    ksg.py                  # KSG MI (Chebyshev, calibrated, clip_zero param)
    hist.py                 # Histogram MI kernel
    streaming.py            # Streaming windowed MI
  utils/
    windowed.py             # Sliding-window MI via SDK
  stats/
    surrogate_test.py       # surrogate_test() — end-to-end test with p-value
    surrogates.py           # shuffle_surrogate, block_bootstrap_surrogate, iaaft_surrogate
    multiple_testing.py     # benjamini_hochberg (BH FDR correction)

tests/
  test_ksg.py               # KSG correctness + warning escalations
  test_hist.py              # Histogram MI correctness
  test_surrogates.py        # Surrogate shape, permutation, determinism, IAAFT (14 tests)
  test_multiple_testing.py  # BH correctness incl. order-preservation (12 tests)
  test_surrogate_validation.py  # Locked H₀ calibration + H₁ power tests
  ...

docs/
  estimator_guide.md      # surrogate selection decision tree, histogram bias, KSG notes
  roadmap.md
  kernels.md
```

---

## Roadmap

**R1 — Software SDK (complete)**
Correctness established. Surrogate testing framework shipped and calibrated, including IAAFT. CI green at 46 tests.

**R2 — FPGA Pathfinder (next)**
Profile the histogram and KSG kernels on target workloads. Spec a PCIe dev card. Same SDK API — `device="fpga"` — no user code changes.

**R3 — Partner Pilots**
BCI/EEG real-time MI, medical imaging registration, causal ML. The hardware needs a real workload to validate against.

---

## Contributing

```bash
pip install -e .
pytest -q -m "not slow"   # fast suite — runs in ~10s
pytest -q -m slow         # includes H₀ calibration (400 trials, ~1hr)
```

Public API stability contract: `ITPU.mutual_info()`, `windowed_mi()`, `surrogate_test()`. Add tests for new kernels. The calibration thresholds in `tests/test_surrogate_validation.py` are locked — if a code change fails calibration, fix the code, not the threshold.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
