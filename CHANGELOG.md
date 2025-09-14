# Changelog

All notable changes to ITPU will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### R1 (Current - Software Foundation)

- ✅ Histogram MI with sliding windows
- 🧪 KSG MI (experimental)
- 🚧 Comprehensive benchmarks
- 🚧 Real-time demos

### R2 (Next - Enhanced Software)

- 📋 Stable KSG implementation
- 📋 GPU acceleration options (CuPy backend)
- 📋 Batch MI matrices (all pairwise channels)
- 📋 Advanced windowing strategies

### R3 (Future - Hardware Pathfinder)

- 📋 FPGA prototype development
- 📋 Hardware-software co-design
- 📋 Performance validation vs software
- 📋 Partner pilot programs

### R4 (Long-term - Production Hardware)

- 📋 ASIC feasibility and design
- 📋 Commercial partnerships
- 📋 Production deployment

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
