# Information-Theoretic Processing Unit (ITPU)

**Accelerate entropy, mutual information (MI), and k-NN estimators — software today, hardware-ready tomorrow.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)

---

## Why This Matters

Most computers are optimized for **matrix multiplication**. But many problems in neuroscience, medical imaging, and causal ML require measuring **information itself** — entropy, mutual information, and neighbor statistics — which are **irregular** and **memory-bound** workloads that CPUs/GPUs struggle with.

**What becomes possible with fast information processing:**
- BCIs that adapt within milliseconds
- Medical imaging that registers scans in real time
- AI systems with live information-flow readouts
- Experiments that adjust on the fly to maximize information content

---

## Status (Sept 2025)

- ✅ Histogram MI (`method="hist"`), streaming/windowed helpers
- ✅ KSG MI (`method="ksg"`) using Chebyshev (∞) metric
- ✅ Smoke test + unit tests
- ✅ Benchmarks vs analytic Gaussian MI
- 🧪 WS3 stats (surrogates, permutation, FDR) landing
- 🧭 Hardware pathfinder next (same SDK API → FPGA/ASIC)

---

## Quickstart

Clone and set up:

```bash
# 1) Clone
git clone https://github.com/justindbilyeu/ITPU
cd ITPU

# 2) Create and activate a virtualenv
python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows (Git Bash or PowerShell)
source .venv/Scripts/activate

# 3) Install
python -m pip install --upgrade pip
pip install -e .  # or: pip install -e .[dev] if extras are defined

# 4) Run tests + smoke
pytest -q
python scripts/smoke_test.py


⸻

Minimal API

import numpy as np
from itpu.sdk import ITPU
from itpu.utils.windowed import windowed_mi

rng = np.random.default_rng(0)
x = rng.standard_normal(50_000)
y = 0.6*x + 0.4*rng.standard_normal(50_000)

itpu = ITPU(device="software")

# Histogram MI
mi_hist = itpu.mutual_info(x, y, method="hist", bins=64)

# Windowed MI
starts, mi_vals = windowed_mi(x, y, window_size=2000, hop_size=400, bins=64)

# KSG (non-parametric, Chebyshev metric)
mi_ksg = itpu.mutual_info(x[:20_000], y[:20_000], method="ksg", k=5)

print(mi_hist, mi_ksg, len(mi_vals))

Units: nats (divide by np.log(2) for bits).
Note: Short windows bias MI upward; use median + null-correction baseline.

⸻

Repository Layout

itpu/
  sdk.py                 # user-facing API
  utils/
    __init__.py
    windowed.py          # sliding/streaming MI
  kernels_sw/
    __init__.py
    ksg.py               # KSG internals (Chebyshev radius)
scripts/
  smoke_test.py
tests/
  test_mi_gaussian.py
  test_imports_and_nulls.py
benchmarks/
  mi_grid.py
docs/
  api.md
  kernels/
    HIST_BUILD.md
    KNN_QUERY.md
  ITPU_patch/
    code/phenomenology/coding_manual.md
examples/
  (optional demos)


⸻

Benchmarks

Run:

python benchmarks/mi_grid.py

This prints MI vs analytic (Gaussian) across correlation strengths and sample sizes.
	•	Histogram MI ≈ analytic MI
	•	KSG null MI < 0.05 at n=20k, k=5 (guarded by tests)

We’ll publish CSV + plots as artifacts in CI.

⸻

Roadmap
	•	R1 (now): correctness (tests), benchmarks, docs, WS3 stats (surrogates, permutation, FDR)
	•	R2: FPGA pathfinder spec + profiling (HIST_BUILD, KNN radius), PCIe dev card target
	•	R3: Partner pilots (BCI / EEG), streaming dashboards

⸻

Contributing

PRs/issues welcome. To develop:

pip install -e .  # or .[dev]
pytest -q
python scripts/smoke_test.py

Please keep public API stable (ITPU.mutual_info, windowed_mi) and add tests for new kernels.

⸻

License

Apache-2.0 — see LICENSE.

---
