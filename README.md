# Information-Theoretic Processing Unit (ITPU)

**A coprocessor + SDK for entropy, mutual information, and k-NN statistics.**
# ITPU: Making Information Flow as Fast as Matrix Math

You know how your phone gets hot when it's thinking really hard? Well, there's this whole category of math that computers are terrible at - measuring how information flows between things. Like figuring out which brain signals predict which thoughts, or which genes cause which diseases. Right now it takes forever and burns tons of energy. 

**We're building a chip that makes that math as fast as regular math.**

## Why This Matters

- **Neuroscientists** wait hours to analyze brain recordings that should update in real-time
- **Medical researchers** can't find disease patterns hidden in genetic data  
- **AI researchers** can't understand what their models are actually learning

**The problem**: GPUs are amazing at matrix multiplication, but terrible at measuring information flow (entropy, mutual information, k-nearest neighbors). These operations are branchy, memory-bound, and slow.

**Our solution**: Custom hardware that makes information-theoretic operations first-class citizens, not second-class citizens.

---

*Technical details below for developers...*


GPUs/TPUs are amazing at matrix math. ITPU is for measuring information flow itself—fast histograms/entropy, MI (plug-in/KSG/KDE), and streaming estimators—so you can watch and steer adaptive systems in real time.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](https://github.com)
![CI](https://github.com/justindbilyeu/ITPU/actions/workflows/ci.yml/badge.svg)

---

## What is the ITPU?

The Information-Theoretic Processing Unit (ITPU) is a coprocessor—and matching Python SDK—that makes it fast and easy to measure information, not just crunch matrices. Today's GPUs excel at linear algebra, but struggle with the irregular, memory-heavy tasks behind information science: entropy (how uncertain a signal is), mutual information (how much two signals share), and k-nearest-neighbor statistics used in high-quality MI estimators. The ITPU accelerates these kernels directly.

**Why it matters.** If you can compute these measures in real time, you can watch information flow through a system and adapt on the fly: stabilize a brain-computer interface, register medical images robustly, trace causal links in data, or monitor model internals while they reason.

**How it fits.** ITPU sits beside your CPU/GPU and exposes simple calls like `mutual_info(x, y)`. Use the software backend today; our FPGA pathfinder and, later, an ASIC target high throughput and low latency with far lower power than general-purpose hardware.

---

## Why ITPU?

- **Pain today**: Entropy/MI, histograms, and k-NN are branchy, memory-bound, and slow on CPUs/GPUs
- **What we do**: Provide native kernels for information-theoretic ops, exposed via a small SDK, and designed to live beside your CPU/GPU as a coprocessor (FPGA now → ASIC later)
- **Who needs it**: Scientific ML, causal discovery, neuroscience/BCI, medical imaging (MI registration), quantum/QEC diagnostics, adaptive control

---

## Status

**Pre-alpha.** This repo starts with:

- ✅ A clean spec for kernels & dataflow
- ✅ A software baseline (CPU/GPU) to run today
- ✅ A tiny SDK stub (same API the card will expose)
- ✅ Example notebooks & tests
- 🚧 FPGA pathfinder and partner pilots are next

---

## Core Kernels (MVP)

- **`HIST_BUILD / HIST_REDUCE`** → fast joint/marginal histograms, entropy H, conditional entropy H(X|Y)
- **`REDUCE_MI`** → mutual information from histograms; supports plug-in & debiased forms
- **`KNN_QUERY`** → k-nearest-neighbor counts for KSG MI (metric-agnostic)
- **`LOG / EXP`** → range-reduced math with Kahan-style compensation
- **`WITNESS_FLUX_DERIV`** → streaming derivatives for MI/entropy time-series
- **`REDUNDANCY_THRESH`** → compute redundancy R^δ (count of "witnesses" above a threshold)

**Design goal**: 100+ GB/s effective throughput on histogram/MI pipelines with on-chip SRAM tiling and DMA.

---

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# Install dependencies (baseline + examples)
pip install numpy scipy scikit-learn pandas matplotlib numba

# Run a simple MI demo (software path)
python examples/mi_demo.py
```

### Minimal Example

```python
import numpy as np
from itpu.sdk import ITPU  # software stub for now

# Generate toy correlated data
rng = np.random.default_rng(0)
x = rng.normal(size=200_000)
y = 0.7*x + 0.3*rng.normal(size=x.size)

# Instantiate (uses CPU/GPU baseline today)
itpu = ITPU(device="software")  # later: "fpga" or "card0"

# Histogram MI
mi_hist = itpu.mutual_info(x, y, method="hist", bins=128)

# KSG MI (k-NN)
mi_ksg = itpu.mutual_info(x, y, method="ksg", k=5)

print(f"Histogram MI ~ {mi_hist:.3f}  |  KSG MI ~ {mi_ksg:.3f}")
```

The SDK stub uses vectorized NumPy/Numba (and optionally GPU via CuPy if available). When the hardware card is present, you'll switch `device="card0"` and keep the same API.

---

## Repository Structure

```
itpu/
├── README.md
├── LICENSE
├── NOTICE
├── docs/
│   ├── ITPU_overview.md
│   ├── kernels.md           # specs for HIST/MI/KSG/LOG/EXP/DERIV
│   └── roadmap.md
├── src/itpu/
│   ├── __init__.py
│   ├── sdk.py               # user-facing API (software stub)
│   ├── kernels_sw/          # software fallback implementations
│   └── utils/
├── examples/
│   ├── mi_demo.py
│   ├── streaming_entropy.ipynb
│   └── ksg_tiled_demo.ipynb
├── tests/
│   ├── test_hist.py
│   ├── test_mi_hist.py
│   └── test_ksg.py
├── hardware/
│   ├── fpga_notes.md        # tiling, SRAM, DMA, timing
│   └── isa_sketch.md        # dataflow diagrams
└── results/
```

---

## Architecture Overview

- **Tiled compute blocks** with on-chip SRAM for joint histograms & neighbor lists
- **DMA engines** feed tiles at high bandwidth; vector math units for log/exp
- **Two main paths**:
  1. **Histogram path** → `HIST_BUILD` → `HIST_REDUCE` → `REDUCE_MI`
  2. **k-NN path** → `KNN_QUERY` → `REDUCE_MI` (KSG)
- **Streaming operators** for entropy/MI time-series & derivatives

---

## Integration Targets

| **Domain**                | **Use Case**                                                                     |
|---------------------------|----------------------------------------------------------------------------------|
| **Neuroscience/BCI**      | Real-time MI across channels; adaptive decoding with redundancy thresholds      |
| **Medical Imaging**       | MI-based registration (local & global histograms)                               |
| **Causal/Scientific ML**  | Dependency graphs, active sampling, representation probing                      |
| **Quantum/QEC**           | Syndrome MI, anomaly detection, redundancy tracking                             |
| **Model Interpretability**| Plug into hierarchical/agentic models to watch information flow between modules |

**Example**: Try it with the [HRM (Hierarchical Reasoning Model)](https://github.com/sapientinc/HRM) project to measure MI between planner ↔ worker at each reasoning step.

---

## Examples

### EEG Example

```bash
python examples/eeg_eye_state_demo.py
# If data/eeg_eye_state.csv exists, uses it; otherwise runs a synthetic fallback.
# Writes:
#   results/examples/eeg_mi_heatmap.png
#   results/examples/eeg_mi_to_label.png
```

**Acceptance criteria**: Running `python examples/eeg_eye_state_demo.py` produces two PNGs under `results/examples/` whether or not the CSV exists. No new dependencies; CI still green.

### MI Grid (Quick Sweep)

```bash
python examples/mi_grid.py
# -> writes results/examples/mi_grid.csv
#    (one row per bin setting; uses EEG CSV if present, otherwise synthetic fallback)
```

**Acceptance criteria**: `python examples/mi_grid.py` creates `results/examples/mi_grid.csv` and prints rows. Works whether or not `data/eeg_eye_state.csv` exists. No new heavy deps.

---

## Advanced Features

### Windowed MI Prototype

The SDK includes a windowed MI function for streaming-style analysis:

```python
from itpu.windowed import windowed_mi

# Sliding-window mutual information
mi_vals, centers = windowed_mi(
    x, y, 
    window_size=1000, 
    step=100, 
    method="hist"
)
```

### Debug-Friendly Output

Add an `output` flag to `mutual_info()` to request MI alone (default) or MI + entropies:

```python
# For method="hist", get detailed breakdown
result = itpu.mutual_info(x, y, method="hist", output="full")
# Returns: {"mi": float, "h_x": float, "h_y": float, "h_xy": float}
```

---

## Roadmap

- **R1 (Software)**: Finalize SDK, correctness tests, baselines vs NumPy/Scikit
- **R2 (FPGA Pathfinder)**: Histogram/MI pipeline, SDK backend, perf counters
- **R3 (k-NN/KSG Tile)**: Metric-agnostic distances, streaming k-NN stats
- **R4 (Partner Pilots)**: Neuroscience, imaging, causal ML case studies
- **R5 (ASIC Decision)**: Spec a pathfinder card (PCIe/M.2), power & cost model

---

## FAQ

**Is this an AI chip?**
No—it's an information chip. We accelerate measuring information (entropy/MI/k-NN), not matrix multiplications.

**Why not just use a GPU?**
GPUs shine at dense linear algebra. MI/entropy/histograms are irregular & memory-bound. We make them first-class.

**Do I have to rewrite my code?**
Minimal. The SDK exposes 2–3 calls (`build_hist`, `mutual_info`, `knn_stats`). Same API for software & hardware.

**Privacy/security?**
All calculations run locally. No data leaves your machine unless you choose to share results.

---

## Contributing

We welcome issues and PRs:

- **Good first issues**: docs fixes, example notebooks, unit tests, simple kernel optimizations
- **Partners**: If you want to pilot the FPGA pathfinder or co-design kernels, open a "Partner Inquiry" issue with your use-case

### Development Setup

```bash
git clone https://github.com/justindbilyeu/ITPU.git
cd itpu
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Include at the top of new source files:

```python
# SPDX-License-Identifier: Apache-2.0
```

---

## Related Work

- **[Geometric Plasticity (GP)](https://github.com/justindbilyeu/Geometric-Plasticity-)**: The theory that motivated the ITPU—closed-loop adaptation driven by information flow (ringing, hysteresis, motif selection)
- **Resonance Geometry (RG)**: Broader program on how information flow sculpts structure

---

## Citation

If this helps your research/product, please cite:

```bibtex
@misc{itpu2025,
  title  = {Information-Theoretic Processing Unit (ITPU)},
  author = {Bilyeu, Justin and contributors},
  year   = {2025},
  note   = {Pre-alpha software baseline and hardware spec},
  url    = {https://github.com/justindbilyeu/ITPU}
}
```

---

## Contact

- 📧 Open a [GitHub issue](https://github.com/justindbilyeu/ITPU/issues) or reach out: contact@example.com
- 🤝 Want to be a design-partner? Tell us your dataset, throughput need, and target form factor (PCIe, M.2, embedded)

---

> **TL;DR**: ITPU makes information-theoretic primitives (entropy, MI, k-NN) fast and first-class, so you can see and control information flow—in labs, models, and real systems.
