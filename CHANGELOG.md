# Changelog

All notable changes to ITPU will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### R1 Status — Surrogate Testing Framework: Complete (2026-04-20)

**Delivered:**
- `itpu/stats/` module: `surrogates.py`, `multiple_testing.py`, `surrogate_test.py`
- `shuffle_surrogate` and `block_bootstrap_surrogate` with full test coverage
- `benjamini_hochberg` with order-preservation correctness (12 tests)
- `surrogate_test()` wired end-to-end through `ITPU` SDK
- Locked calibration and power tests in `tests/test_surrogate_validation.py`

**KSG calibration gate result (locked, on record):**
- KS statistic: **0.0565** | KS p-value: **0.1497** — PASS (threshold > 0.05)
- 400 independent H₀ trials, n=1000, n_surrogates=999, surrogate_type="shuffle"

**Two bugs caught and fixed by the calibration gate (not the power test):**

1. *Wrong metric in SDK shadow KSG implementation* — `sdk.py` contained a
   duplicate `_mi_ksg` using Euclidean distance in the joint space instead of
   Chebyshev (KSG-1 requirement). Fixed by wiring `ITPU.mutual_info(method="ksg")`
   to delegate to `ksg_mi_estimate` in `kernels_sw/ksg.py` — one implementation,
   correct metric. Pre-fix KS: 0.0785 / p=0.0137 (FAIL).

2. *`max(mi, 0.0)` clipping breaking null distribution comparison* — Under H₀,
   ~50% of KSG estimates are naturally negative. Clipping both observed and null
   values to 0 caused p_value=1.0 whenever mi_observed clipped, producing a
   bimodal p-value distribution (KS: 0.4850 / p=0.0000). Fixed by adding
   `clip_zero=True` parameter to `ksg_mi_estimate` (default preserves all
   existing callers); SDK passes `clip_zero=False`.

Neither bug was catchable by the H₁ power test — ρ=0.6 is strong enough to
survive both. The calibration gate is the reason they were caught before shipping.

**R1 closed. CI: 40 passed, 1 deselected (slow calibration test).**

**R2 gate clear:** Software correctness established. Hardware pathfinder conversation
can begin after IAAFT issue is open (issue #13) and estimator selection guide has a draft (issue #15).

**Open fast-follow items for Phase 2 (tracked as GitHub issues):**
- #13 — IAAFT surrogate (preserves power spectrum; required for oscillatory/autocorrelated data)
- #14 — Batch FDR: `surrogate_test` accepts `fdr_alpha` but BH correction not yet applied
- #15 — MI Estimator Selection Guide: histogram bias formula, KSG metric behavior, hist shadow audit
- Adaptive k-selection for KSG (data-driven k rather than fixed k=5) — not yet filed

### Added

- Comprehensive benchmark suite comparing against SciPy/scikit-learn
- Real-time EEG mutual information dashboard demo
- CLI tools for benchmarking and demos (`itpu-benchmark`, `itpu-demo`)
- GitHub Actions CI/CD workflows
- Comprehensive development tooling (Makefile, pre-commit hooks)
- Issue templates for bug reports, features, and partnership inquiries
- Contributing guidelines and development documentation

### Changed

- Improved project structure with proper Python packaging
- Enhanced README with clearer positioning and installation instructions
- Software-first pivot messaging (September 2025)

### Fixed

- Repository structure and packaging configuration
- Development environment setup process

## [0.1.0-alpha] - 2025-09-14

### Added

- Initial software baseline implementation
- Core SDK with `ITPU` class and unified API
- Histogram-based mutual information estimation (`method="hist"`)
- Experimental KSG (k-nearest neighbor) MI estimation (`method="ksg"`)
- Sliding window utilities for streaming MI analysis
- Basic smoke test for core functionality
- Apache 2.0 license

### Software Components

- `itpu.sdk.ITPU` - Main user-facing API
- `itpu.kernels_sw.hist` - Histogram operations and entropy calculation
- `itpu.kernels_sw.ksg` - KSG mutual information estimator
- `itpu.utils.windowed` - Windowed and streaming utilities
- Basic examples and demo scripts

### Hardware Planning

- Kernel specifications for future FPGA implementation
- Design documents for hardware acceleration roadmap
- Performance targets for FPGA/ASIC development

### Performance Targets

- 100+ GB/s effective throughput goal for histogram/MI pipelines
- On-chip SRAM tiling and DMA design concepts
- Software baseline establishes correctness reference

## [Pre-alpha] - 2025-01-15 to 2025-09-13

### Research and Development Phase

- Information-theoretic processing unit concept development
- Algorithm research and prototype implementations
- Hardware architecture exploration
- Partnership discussions with potential early adopters

### Key Milestones

- Established core vision: accelerate entropy, MI, and k-NN statistics
- Identified target domains: neuroscience/BCI, medical imaging, causal ML
- Determined software-first development approach
- Created initial repository structure and documentation

-----

## Version History Summary

|Version    |Date        |Key Features                                     |
|-----------|------------|-------------------------------------------------|
|Pre-alpha  |Jan-Sep 2025|Concept development, algorithm research          |
|0.1.0-alpha|2025-09-14  |Initial software implementation, basic MI/entropy|
|Unreleased |TBD         |Benchmarks, demos, development tooling           |

## Future Roadmap

### R1 — Software SDK (complete)

- ✅ Histogram MI with sliding windows
- ✅ KSG MI (Chebyshev metric, calibrated)
- ✅ Surrogate testing: shuffle, block bootstrap, IAAFT
- ✅ Benjamini-Hochberg FDR correction
- ✅ Locked calibration gate: KS=0.0565, p=0.1497
- ✅ CONTRIBUTING.md and estimator guide

### R2 — FPGA Pathfinder (next)

- 📋 Profile histogram and KSG kernels on BCI workloads
- 📋 Spec a PCIe dev card
- 📋 Same SDK API — `device="fpga"` — no user code changes
- 📋 Batch FDR: wire `fdr_alpha` through to BH correction (#14)
- 📋 Adaptive k-selection for KSG

### R3 — Partner Pilots

- 📋 BCI/EEG real-time MI
- 📋 Medical imaging registration
- 📋 Causal ML validation

## Legend

- ✅ Completed
- 🧪 Experimental/In Progress
- 🚧 In Development
- 📋 Planned

## Breaking Changes

### Future Breaking Changes (Planned)

These changes are planned for future major versions:

#### v1.0.0 (Planned)

- Stable API freeze
- Potential changes to default parameters
- Hardware device enumeration changes

#### v0.2.0 (Planned)

- Possible changes to windowed utilities API
- Enhanced error handling (may change exception types)
- Performance optimization may change exact numerical results

## Migration Guides

### From Pre-alpha to 0.1.0-alpha

- Install via pip: `pip install itpu`
- Update imports: `from itpu.sdk import ITPU`
- New API: `itpu = ITPU(device="software")`
- Replace custom MI implementations with `itpu.mutual_info()`

### Future Migration Guides

Migration guides for breaking changes will be provided with each major release.

## Performance History

### 0.1.0-alpha Baseline

- Histogram MI: ~2-5x faster than naive SciPy implementations
- Memory efficiency: Optimized for streaming applications
- Streaming support: Real-time windowed MI analysis

*Detailed performance benchmarks available in `benchmark_results/` directory*

## Acknowledgments

### Contributors

- Justin Bilyeu - Project founder and lead developer
- [Future contributors will be listed here]

### Inspiration and Related Work

- Kraskov, Stögbauer, and Grassberger (2004) - KSG MI estimation
- scikit-learn mutual information implementations
- JIDT (Java Information Dynamics Toolkit)
- MNE-Python for neuroscience applications

### Funding and Support

*Grant acknowledgments and institutional support will be listed here as applicable*

-----

For the complete list of changes, see the [commit history](https://github.com/justindbilyeu/ITPU/commits/main).
