Got it—here’s a tightened, software-first, research-aligned README you can paste over your current one. I emphasized our current star: the WS1→WS3 stack (phenomenology → information-theory pipelines), with IAAFT/block surrogates, nested CV decoding, and a clean hand-off to future hardware.

# Information-Theoretic Processing Unit (ITPU)

**Accelerate entropy, mutual information (MI), and k-NN–based estimators — software today, hardware tomorrow.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](https://github.com)

*Apache-2.0 • Pre-alpha • Software-first pivot (Sept 2025)*

---

## ITPU in one line
Most computers are great at multiplying matrices—and bad at **measuring information**. ITPU flips that: it makes entropy, MI, and k-NN statistics fast and **streamable**, so you can see **information flow in real time**.

---

## What’s new (focus of the project right now)

**The current star is WS1→WS3:** reliable phenomenology labels feeding rigorously controlled information-theory analysis.

- **WS1 (Phenomenology) starter kit**  
  `docs/ITPU_patch/code/phenomenology/coding_manual.md` — operational definitions (lattice/tunnel/spiral), boundary rules, κ/ICC procedures.  
  `docs/ITPU_patch/code/phenomenology/phenom_tools.py` — compute **Cohen’s κ** & **ICC**; optional `docs/ITPU_patch/notebooks/ws1_pilot.ipynb`.

- **WS3 (Information Theory) software-first**  
  - **Histogram MI** for streaming dashboards (stable).  
  - **KSG MI (k-NN)** experimental path (software baseline for later acceleration).  
  - Roadmapped now: **IAAFT & block surrogates**, **permutation tests**, **FDR**, and **nested CV decoding** to prevent leakage.

- **Hardware later, unchanged API**  
  We validate kernels & user needs in software, then lift the **same API** to FPGA/ASIC.

---

## Why this matters

- **Neuroscience & BCIs:** real-time feedback from information flow (not hours-later batch jobs).  
- **Clinical & scientific workflows:** detect structure in high-dimensional signals with correct nulls (surrogates) and multiple-comparison control.  
- **AI systems:** live info-flow readouts and causal sanity checks beyond matrix ops.

Today’s accelerators excel at dense GEMMs; **MI/entropy/k-NN** are irregular and memory-bound. ITPU optimizes the **dataflow** those workloads need.

---

## Status (Sept 2025)

- ✅ **Working now:** histogram-based MI (nats), sliding/windowed helpers, SDK entry points.  
- 🧪 **Experimental:** KSG MI (k-NN) + windowed variants; EEG demo.  
- 🚧 **In progress (starred):** WS1 κ pilot; WS3 surrogates (IAAFT/block) + nested CV decoding.  
- 🧭 **Next hardware step:** lift the exact API onto an FPGA pathfinder once WS3 passes prereg gates.

---

## Quickstart

> **Repo root** = folder with `README.md` (and `itpu/`).

```bash
# 1) Clone & enter
git clone https://github.com/justindbilyeu/ITPU
cd ITPU

# 2) (Recommended) venv
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3) Minimal deps
pip install numpy scipy matplotlib

# 4) Smoke test
python scripts/smoke_test.py
# Expect a non-zero MI and a count of sliding-window outputs.


⸻

Minimal API

from itpu.sdk import ITPU
from itpu.utils.windowed import windowed_mi
import numpy as np

itpu = ITPU(device="software")   # same API will target FPGA later

# Point MI (histogram method)
x = np.random.randn(50_000)
y = 0.6*x + 0.4*np.random.randn(50_000)
mi = itpu.mutual_info(x, y, method="hist", bins=64)   # nats

# Sliding-window MI (hist)
starts, mi_vals = windowed_mi(x, y, window_size=2000, hop_size=400, bins=64)

# KSG (experimental)
# mi_ksg = itpu.mutual_info(x, y, method="ksg", k=5)

Units: MI in nats (divide by np.log(2) for bits).

⸻

Research alignment (WS1–WS6)
	•	WS1 Phenomenology → labels with κ ≥ 0.6
Use the coding manual + tools in docs/ITPU_patch/... to run a κ pilot. No reliable labels → no WS3 Go.
	•	WS3 Information Theory (current star)
MI (hist/KSG), IAAFT/block surrogates, permutation tests, FDR, and nested CV decoding (AUC with CIs).
	•	WS5 TDA / WS4 Dynamics / WS6 Models
Out of scope for the current sprint, but the SDK is being shaped so topological/dynamical metrics and toy models can plug in next.

Go/No-Go gates (prereg discipline):
	•	κ ≥ 0.6 (WS1) and any of: AUC ≥ 0.70 (out-of-subject, nested CV), or O-info/PID class differences (q < 0.05 vs surrogates), or robust TDA differences.

⸻

What’s in this repo (today)
	•	itpu/kernels_sw/hist.py — histogram & entropy primitives
	•	itpu/kernels_sw/ksg.py — experimental KSG estimator + windowed helper
	•	itpu/sdk.py — device-agnostic API (device="software" now; "fpga" later)
	•	itpu/utils/windowed.py — sliding/streaming utilities
	•	scripts/smoke_test.py — quick correctness check
	•	examples/eeg_streaming_demo.py — EEG MI timeseries demo (if present)
	•	benchmarks/ — comparisons vs SciPy/scikit-learn (rolling online)

Docs (WS1 kit):
	•	docs/ITPU_patch/code/phenomenology/coding_manual.md
	•	docs/ITPU_patch/code/phenomenology/phenom_tools.py
	•	docs/ITPU_patch/notebooks/ws1_pilot.ipynb (optional training aid)

⸻

Roadmap
	•	R1 (now): histogram MI + sliding windows; WS1 κ pilot materials; clean docs
	•	R2 (next): KSG MI hardening; IAAFT/block surrogates; permutation+FDR; nested CV decoding
	•	R3: batched MI matrices, categorical MI, optional CuPy acceleration
	•	R4: FPGA pathfinder (same SDK API) with end-to-end streaming demo
	•	R5: ASIC decision contingent on verified demand + performance

⸻

Benchmarks (tracking)

We’re adding apples-to-apples comparisons vs SciPy/scikit-learn/JIDT:
	•	Throughput (histogram MI), sliding-window latency
	•	Accuracy vs analytic/synthetic ground truths
	•	Streaming stability under load

Early goalpost: ≥2–5× faster histogram MI on common sizes + first-class streaming others lack.

⸻

Contributing

PRs welcome—high-leverage areas right now:
	•	KSG estimator tests & correctness
	•	Streaming/windowing utilities
	•	Surrogates (IAAFT/block) + permutation/FDR scaffolding
	•	Nested CV decoding (leakage-safe PCA; AUC CIs)

Dev tips:

pytest -q          # test suite (as it comes online)
# please pin deps in any new modules and include minimal examples


⸻

License

Apache-2.0 — see LICENSE.

