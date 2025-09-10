# Information-Theoretic Processing Unit (ITPU)

**Accelerate entropy, mutual information (MI), and k-NN–based estimators—starting in software, designed for hardware.**  
Apache-2.0 • Pre-alpha

---

## What is ITPU (in plain English)

Modern chips are great at matrix math (good for neural nets) but bad at measuring information itself. Many real problems—BCI/neuroscience, medical image registration, causal discovery—need **entropy/MI** and **k-NN statistics** fast and in streaming form.  
**ITPU** is a coprocessor concept and an SDK: we’re shipping a **software SDK now** (same API you’ll use on a future FPGA/ASIC), so you can profile information flow today and drop in hardware later without changing code.

---

## Status: **Software-first pivot (Sept 2025)**

- ✅ **Working now:** histogram-based MI (`method="hist"`), sliding-window/streaming helpers
- 🧪 **In progress:** KSG MI (`method="ksg"`, k-NN estimator), benchmarking suite, EEG demo
- 🧭 **Road to hardware:** we’ll validate kernels + users in software, then lift the exact API onto an FPGA pathfinder

---
'''
## Quickstart (local, pre-alpha)

```bash
# 1) Clone and enter the repo root (folder that contains README.md and the itpu/ directory)
git clone https://github.com/justindbilyeu/ITPU
cd ITPU

# 2) (Recommended) Create a virtual env
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3) Minimal deps for the software path
pip install numpy scipy

# 4) Run the smoke test
python scripts/smoke_test.py

You should see non-zero MI printed and a count of sliding-window results.

⸻

Minimal API
- Histogram MI: fast, discrete approximation
- KSG MI: continuous estimator (Kraskov–Stögbauer–Grassberger, variant I)
  * Default metric = chebyshev (p = ∞ norm)
  * Returns MI in nats
from itpu.sdk import ITPU
from itpu.utils.windowed import windowed_mi
import numpy as np

itpu = ITPU(device="software")  # same API will target FPGA later

# Example: point MI (histogram method)
x = np.random.randn(50_000)
y = 0.6*x + 0.4*np.random.randn(50_000)
mi = itpu.mutual_info(x, y, method="hist", bins=64)  # returns MI in nats

# Sliding-window MI (hist)
starts, mi_vals = windowed_mi(x, y, window_size=2000, hop_size=400, bins=64)

KSG (k-NN) estimator is being wired up; you’ll call it with method="ksg", k=5 once merged.

⸻

Why software-first?
	•	Faster proof, lower risk. We validate accuracy, speed, and UX now.
	•	Same API later. When the FPGA card lands, you flip device="software" → device="fpga"; no code rewrite.
	•	Streaming is the wedge. Most MI libraries are batch-only; we’re prioritizing sliding windows + real-time.

⸻

Benchmarks (tracking)

We’re adding benchmarks/ to compare ITPU software vs. SciPy/scikit-learn/JIDT on identical data:
	•	Throughput for histogram MI
	•	Latency for sliding windows
	•	Accuracy vs. known MI for synthetic distributions

Early goalposts: ≥2–5× faster histogram MI on common sizes and first-class streaming others lack.

⸻

Roadmap (R1 → R3 software, R4 → R5 hardware)
	•	R1 (now): histogram MI + sliding windows, smoke tests, clean docs
	•	R2 (next): KSG MI (ksg_mi_estimate), comprehensive benchmarks, EEG streaming demo
	•	R3: convenience APIs (batched MI matrices, masks, categorical MI), optional CuPy acceleration
	•	R4: FPGA pathfinder spec + sizing; partner pilots (BCI first)
	•	R5: ASIC decision (only after proven demand + perf)

⸻

Example demos
	•	scripts/smoke_test.py – quick sanity check
	•	examples/eeg_streaming_demo.py – streaming MI on EEG (uses local CSV or synthetic fallback)
	•	benchmarks/compare_baselines.py – apples-to-apples against popular libs (coming online)

⸻

FAQ

Why not just use a GPU?
GPUs crush dense matrices; MI/k-NN are irregular and memory-bound. We optimize dataflow for histograms and neighbor counts, and offer true streaming—then port exactly that to hardware.

Do I need special hardware?
No. Today is pure Python/NumPy/SciPy. The same code will target an FPGA card later.

Units?
MI is reported in nats (divide by np.log(2) for bits).

⸻

Contributing

Issues and PRs welcome! High-leverage areas:
	•	KSG estimator internals and tests (itpu/kernels_sw/ksg.py)
	•	Streaming/windowing utilities
	•	Benchmarks and real-world datasets (EEG/BCI preferred for now)

Dev tips:

# From repo root
pip install -r requirements-dev.txt   # (if present; otherwise numpy/scipy/matplotlib)
pytest -q                             # test suite (coming online)


⸻

License

Apache-2.0. See LICENSE.
