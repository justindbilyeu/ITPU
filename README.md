# Information-Theoretic Processing Unit (ITPU)

**Accelerate entropy, mutual information (MI), and k-NN–based estimators — software today, designed for hardware tomorrow.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](https://github.com)

*Apache-2.0 • Pre-alpha • Software-first pivot (Sept 2025)*

---

## ITPU, in one line

Most computers are great at multiplying matrices—and bad at **measuring information**. ITPU flips that: it makes entropy, mutual information, and k-NN statistics fast and **streamable**, so you can see **information flow in real time**.

---

## Why This Matters

**Neuroscientists** wait hours to analyze brain recordings that should update in real time. A paralyzed patient testing a BCI can't get feedback fast enough to learn control efficiently.

**Medical researchers** miss patterns in high-dimensional data because finding information relationships across thousands of variables is computationally painful.

**AI teams** ship powerful models without tools to see how information actually flows between components as they reason.

**The core problem:** Today's computers excel at matrix math but struggle to *measure information itself*—entropy, mutual information (MI), and related statistics that reveal how systems work. These workloads are **irregular** and **memory-intensive**, so they don't map well to standard accelerators.

**Current tools often fall short:**
- CPUs are too slow for real-time use
- GPUs are often inefficient on branchy, irregular operations
- Existing libraries force trade-offs between speed, accuracy, and streaming
- Few options support true *real-time* information analysis

**What becomes possible with fast information processing:**
- BCIs that adapt within milliseconds
- Medical imaging that registers scans during surgery
- AI systems with live information-flow readouts
- Discovery workflows that surface correlations previously buried by compute limits
- Instruments that adjust experiments on the fly based on information content

---

## What is ITPU

Modern chips are great at matrix math (good for neural nets) but bad at measuring information itself. Many real problems—BCI/neuroscience, medical image registration, causal discovery—need **entropy/MI** and **k-NN statistics** fast and in **streaming** form.

**ITPU** is a coprocessor concept *and* an SDK: we're shipping a **software SDK now** (with the same API you'll use on a future FPGA/ASIC), so you can profile information flow today and drop in hardware later **without changing code**.

---

## Status: Software-First (Sept 2025)

- ✅ **Working now:** histogram-based MI (`method="hist"`), sliding-window / streaming helpers
- 🧪 **Experimental:** KSG MI (`method="ksg"`, k-NN estimator) and windowed KSG; benchmarking suite; EEG demo
- 🧭 **Road to hardware:** validate kernels + users in software → lift the exact API onto an FPGA pathfinder

---

## Quickstart

> **"Repo root"** = the folder that contains `README.md` and the `itpu/` directory.

```bash
# 1) Clone and enter the repo root
git clone https://github.com/justindbilyeu/ITPU
cd ITPU

# 2) (Recommended) Create & activate a virtual environment
python -m venv .venv

# macOS/Linux:
source .venv/bin/activate

# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3) Install minimal dependencies
pip install numpy scipy matplotlib

# 4) Run a smoke test
python scripts/smoke_test.py
# You should see a non-zero MI and a count of sliding-window results.

# 5) (Optional) Try the EEG streaming demo if present
# python examples/eeg_streaming_demo.py
```

-----

## Minimal API

```python
from itpu.sdk import ITPU
from itpu.utils.windowed import windowed_mi
import numpy as np

itpu = ITPU(device="software")  # same API will target FPGA later

# Example: point MI (histogram method)
x = np.random.randn(50_000)
y = 0.6*x + 0.4*np.random.randn(50_000)
mi = itpu.mutual_info(x, y, method="hist", bins=64)  # MI in nats

# Sliding-window MI (hist)
starts, mi_vals = windowed_mi(x, y, window_size=2000, hop_size=400, bins=64)

# KSG (k-NN) estimator (experimental; once merged)
# mi_ksg = itpu.mutual_info(x, y, method="ksg", k=5)  # MI in nats
```

**Methods:**

- **Histogram MI:** fast, discrete approximation (good for streaming & dashboards)
- **KSG MI:** non-parametric continuous estimator (Kraskov–Stögbauer–Grassberger, variant I)

**Units:** MI is reported in nats (divide by `np.log(2)` for bits).

-----

## What’s in This Repo Today

- `itpu/kernels_sw/hist.py` — software histogram & entropy primitives
- `itpu/kernels_sw/ksg.py` — experimental KSG estimator + windowed helper
- `itpu/sdk.py` — device-agnostic API (device=“software” now; “fpga” later)
- `itpu/utils/windowed.py` — sliding/streaming utilities
- `scripts/smoke_test.py` — quick correctness check
- `examples/eeg_streaming_demo.py` — EEG MI timeseries demo (local CSV or synthetic fallback)
- `benchmarks/` — apples-to-apples comparisons vs SciPy/scikit-learn (coming online)

-----

## Benchmarks (Tracking)

We’re adding benchmarks/ to compare ITPU software vs. SciPy/scikit-learn/JIDT on identical data:

- Throughput for histogram MI
- Latency for sliding windows
- Accuracy vs. known MI for synthetic distributions

**Early goalposts:** ≥2–5× faster histogram MI on common sizes and first-class streaming others lack.

-----

## Roadmap

- **R1 (now):** histogram MI + sliding windows, smoke tests, clean docs
- **R2 (next):** KSG MI (ksg_mi_estimate), comprehensive benchmarks, EEG streaming demo
- **R3:** convenience APIs (batched MI matrices, masks, categorical MI), optional CuPy acceleration
- **R4:** FPGA pathfinder spec + sizing; BCI partner pilots
- **R5:** ASIC decision (only after proven demand + performance)

-----

## Examples & Demos

- `scripts/smoke_test.py` — quick sanity check
- `examples/eeg_streaming_demo.py` — streaming MI on EEG (eyes open/closed), with synthetic fallback
- `benchmarks/compare_baselines.py` — apples-to-apples against popular libs (coming online)

-----

## FAQ

**Why not just use a GPU?**
GPUs crush dense matrices; MI/k-NN are irregular and memory-bound. We optimize dataflow for histograms and neighbor counts, offer true streaming, and then port exactly that to hardware.

**Do I need special hardware?**
No. Today is pure Python/NumPy/SciPy. The same code will target an FPGA card later.

**Bits or nats?**
We return nats. For bits, divide by `np.log(2)`.

-----

## Contributing

Issues and PRs welcome! High-leverage areas:

- KSG estimator internals and tests (`itpu/kernels_sw/ksg.py`)
- Streaming/windowing utilities
- Benchmarks and real-world datasets (EEG/BCI preferred for now)

**Dev tips:**

```bash
# From repo root
# If a dev requirements file exists:
# pip install -r requirements-dev.txt
pytest -q  # test suite (coming online)
```

-----

## License

Apache-2.0 — see <LICENSE>.

```

```
