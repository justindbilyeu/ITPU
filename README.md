Information-Theoretic Processing Unit (ITPU)

A coprocessor + SDK for entropy, mutual information, and k-NN statistics.
GPUs/TPUs are amazing at matrix math. ITPU is for measuring information flow itself—fast histograms/entropy, MI (plug-in/KSG/KDE), and streaming estimators—so you can watch and steer adaptive systems in real time.

⸻

Why ITPU?
	•	Pain today: Entropy/MI, histograms, and k-NN are branchy, memory-bound, and slow on CPUs/GPUs.
	•	What we do: Provide native kernels for information-theoretic ops, exposed via a small SDK, and designed to live beside your CPU/GPU as a coprocessor (FPGA now → ASIC later).
	•	Who needs it: Scientific ML, causal discovery, neuroscience/BCI, medical imaging (MI registration), quantum/QEC diagnostics, adaptive control.

⸻

Status

Pre-alpha. This repo starts with:
	•	A clean spec for kernels & dataflow.
	•	A software baseline (CPU/GPU) to run today.
	•	A tiny SDK stub (same API the card will expose).
	•	Example notebooks & tests.
FPGA pathfinder and partner pilots are next.

⸻

Core kernels (MVP)
	•	HIST_BUILD / HIST_REDUCE → fast joint/marginal histograms, entropy H, conditional entropy H(X|Y)
	•	REDUCE_MI → mutual information from histograms; supports plug-in & debiased forms
	•	KNN_QUERY → k-nearest-neighbor counts for KSG MI (metric-agnostic)
	•	LOG / EXP → range-reduced math with Kahan-style compensation
	•	WITNESS_FLUX_DERIV → streaming derivatives for MI/entropy time-series
	•	REDUNDANCY_THRESH → compute redundancy R^\delta (count of “witnesses” above a threshold)

Design goal: 100+ GB/s effective throughput on histogram/MI pipelines with on-chip SRAM tiling and DMA.

⸻

Quick start (software baseline)

# 1) Create env
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# 2) Install deps (baseline + examples)
pip install numpy scipy scikit-learn pandas matplotlib numba

# 3) Run a simple MI demo (software path)
python examples/mi_demo.py

Minimal example (examples/mi_demo.py):

import numpy as np
from itpu.sdk import ITPU  # software stub for now

# toy correlated data
rng = np.random.default_rng(0)
x = rng.normal(size=200_000)
y = 0.7*x + 0.3*rng.normal(size=x.size)

# instantiate (uses CPU/GPU baseline today)
itpu = ITPU(device="software")  # later: "fpga" or "card0"

# histogram MI
mi_hist = itpu.mutual_info(x, y, method="hist", bins=128)

# KSG MI (k-NN)
mi_ksg = itpu.mutual_info(x, y, method="ksg", k=5)

print(f"Histogram MI ~ {mi_hist:.3f}  |  KSG MI ~ {mi_ksg:.3f}")

The SDK stub uses vectorized NumPy/Numba (and optionally GPU via CuPy if available). When the hardware card is present, you’ll switch device="card0" and keep the same API.

⸻

Repo layout

itpu/
├─ README.md
├─ LICENSE
├─ NOTICE
├─ docs/
│  ├─ ITPU_overview.md
│  ├─ kernels.md           # specs for HIST/MI/KSG/LOG/EXP/DERIV
│  └─ roadmap.md
├─ src/itpu/
│  ├─ __init__.py
│  ├─ sdk.py               # user-facing API (software stub)
│  ├─ kernels_sw/          # software fallback implementations
│  └─ utils/
├─ examples/
│  ├─ mi_demo.py
│  ├─ streaming_entropy.ipynb
│  └─ ksg_tiled_demo.ipynb
├─ tests/
│  ├─ test_hist.py
│  ├─ test_mi_hist.py
│  └─ test_ksg.py
├─ hardware/
│  ├─ fpga_notes.md        # tiling, SRAM, DMA, timing
│  └─ isa_sketch.md        # dataflow diagrams
└─ results/


⸻

Architecture (at a glance)
	•	Tiled compute blocks with on-chip SRAM for joint histograms & neighbor lists.
	•	DMA engines feed tiles at high bandwidth; vector math units for log/exp.
	•	Two main paths:
	1.	Histogram path → HIST_BUILD → HIST_REDUCE → REDUCE_MI
	2.	k-NN path → KNN_QUERY → REDUCE_MI (KSG)
	•	Streaming operators for entropy/MI time-series & derivatives.

⸻

Integration targets
	•	Neuroscience/BCI: Real-time MI across channels; adaptive decoding with redundancy thresholds.
	•	Medical imaging: MI-based registration (local & global histograms).
	•	Causal/scientific ML: Dependency graphs, active sampling, representation probing.
	•	Quantum/QEC: Syndrome MI, anomaly detection, redundancy tracking.
	•	Model interpretability: Plug into hierarchical/agentic models to watch information flow between modules in real time.

Example: try it with the HRM (Hierarchical Reasoning Model) project to measure MI between planner ↔ worker at each reasoning step.

⸻

Roadmap (high level)
	•	R1 (software): finalize SDK, correctness tests, baselines vs NumPy/Scikit.
	•	R2 (FPGA pathfinder): histogram/MI pipeline, SDK backend, perf counters.
	•	R3 (k-NN/KSG tile): metric-agnostic distances, streaming k-NN stats.
	•	R4 (partner pilots): neuroscience, imaging, causal ML case studies.
	•	R5 (ASIC decision): spec a pathfinder card (PCIe/M.2), power & cost model.

⸻

FAQ (short)
	•	Is this an AI chip? No—it’s an information chip. We accelerate measuring information (entropy/MI/k-NN), not matrix multiplications.
	•	Why not just use a GPU? GPUs shine at dense linear algebra. MI/entropy/histograms are irregular & memory-bound. We make them first-class.
	•	Do I have to rewrite my code? Minimal. The SDK exposes 2–3 calls (build_hist, mutual_info, knn_stats). Same API for software & hardware.
	•	Privacy/security? All calculations run locally. No data leaves your machine unless you choose to share results.

⸻

Contributing

We welcome issues and PRs:
	•	Good first issues: docs fixes, example notebooks, unit tests, simple kernel optimizations.
	•	Partners: If you want to pilot the FPGA pathfinder or co-design kernels, open a “Partner Inquiry” issue with your use-case.

⸻

License

Apache-2.0 (see LICENSE).
Include at the top of new source files:

# SPDX-License-Identifier: Apache-2.0


⸻

Related work
	•	Geometric Plasticity (GP): the theory that motivated the ITPU—closed-loop adaptation driven by information flow (ringing, hysteresis, motif selection).
	•	Resonance Geometry (RG): broader program on how information flow sculpts structure.

⸻

Citation (placeholder)

If this helps your research/product, please cite:

@misc{itpu2025,
  title  = {Information-Theoretic Processing Unit (ITPU)},
  author = {Bilyeu, Justin and contributors},
  year   = {2025},
  note   = {Pre-alpha software baseline and hardware spec},
  url    = {https://github.com/<your-org>/itpu}
}


⸻

Contact

Open a GitHub issue or reach out: @example.com
Want to be a design-partner? Tell us your dataset, throughput need, and target form factor (PCIe, M.2, embedded).

⸻

TL;DR: ITPU makes information-theoretic primitives (entropy, MI, k-NN) fast and first-class, so you can see and control information flow—in labs, models, and real systems.
