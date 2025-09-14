# Contributing to ITPU

Thank you for your interest in contributing to ITPU! This guide will help you get started with contributing to the Information-Theoretic Processing Unit project.

## Performance Considerations

### Benchmarking Requirements

When contributing performance improvements:

1. **Baseline measurements:**
   
   ```bash
   # Run benchmarks before your changes
   make benchmark-quick
   git add benchmark_results/baseline_*.csv
   ```
1. **Improvement measurements:**
   
   ```bash
   # After your changes
   make benchmark-quick
   # Compare results and include in PR
   ```
1. **Performance regression prevention:**
- New features shouldn’t slow down existing functionality
- Include performance tests for critical paths
- Document any trade-offs (speed vs accuracy, memory vs time)

### Algorithm Guidelines

- **Vectorization:** Use NumPy operations instead of Python loops
- **Memory efficiency:** Avoid unnecessary data copies
- **Numerical stability:** Handle edge cases (zeros, infinities, very small/large values)
- **Streaming support:** Design algorithms to work with windowed data

Example of good vectorized code:

```python
# Good: vectorized
def histogram_2d_vectorized(x, y, bins):
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    return hist

# Avoid: Python loops
def histogram_2d_slow(x, y, bins):
    hist = np.zeros((bins, bins))
    for i in range(len(x)):
        # ... slow bin assignment ...
    return hist
```

## Documentation

### API Documentation

- All public functions must have docstrings
- Include examples that users can copy-paste
- Document parameter types and shapes clearly
- Explain the algorithm briefly in the docstring

### Tutorials and Examples

- Create end-to-end examples for new features
- Include real-world use cases when possible
- Jupyter notebooks should run without errors
- Add to the `examples/` directory

### Hardware Documentation

For hardware-related contributions:

- Document FPGA resource usage
- Include timing constraints and performance targets
- Explain the mapping from software to hardware
- Provide synthesis and implementation notes

## Hardware Development

### FPGA Contributions

When contributing to hardware development:

1. **Follow the software API:**
- Hardware kernels must match software behavior
- Use identical test vectors for validation
- Maintain the same precision where possible
1. **Resource efficiency:**
- Minimize BRAM and DSP usage
- Document resource utilization
- Consider throughput vs area trade-offs
1. **Testing:**
- Include both software and hardware test benches
- Test with realistic data patterns
- Validate against software reference

### Hardware Directory Structure

```
hardware/
├── fpga/
│   ├── rtl/                 # RTL source files
│   ├── tb/                  # Test benches  
│   ├── constraints/         # Timing and pin constraints
│   └── scripts/             # Build scripts
├── docs/
│   ├── architecture.md     # Hardware architecture
│   ├── timing.md           # Timing analysis
│   └── resource_usage.md   # Resource utilization
└── verification/
    ├── test_vectors/        # Test data
    └── cosim/               # Co-simulation scripts
```

## Community

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests, design discussions
- **GitHub Discussions:** General questions, ideas, showcasing work
- **Pull Request Reviews:** Code-specific discussions

### Getting Help

- Check existing issues and documentation first
- Provide minimal reproducible examples
- Be specific about your environment and use case
- Be patient - maintainers are volunteers

### Becoming a Maintainer

Active contributors may be invited to become maintainers. This involves:

- Consistent high-quality contributions
- Helping other contributors
- Understanding the project vision
- Commitment to long-term involvement

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH` (e.g., `1.2.3`)
- Major: Breaking API changes
- Minor: New features, backward compatible
- Patch: Bug fixes, backward compatible
- Pre-release: `0.1.0-alpha`, `1.0.0-beta.1`

### Release Checklist

For maintainers preparing releases:

1. **Pre-release:**
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release branch
1. **Release:**
- [ ] Tag the release
- [ ] Build and test packages
- [ ] Deploy to PyPI
- [ ] Create GitHub release
- [ ] Update documentation site
1. **Post-release:**
- [ ] Announce on relevant channels
- [ ] Update example notebooks
- [ ] Plan next release

## Recognition

### Contributors

All contributors are recognized in:

- `CONTRIBUTORS.md` file
- GitHub contributors page
- Release notes for significant contributions
- Academic papers where appropriate

### Citation

If you use ITPU in academic work, please cite:

```bibtex
@misc{itpu2025,
  title = {Information-Theoretic Processing Unit (ITPU)},
  author = {Bilyeu, Justin and contributors},
  year = {2025},
  note = {Software for fast mutual information and entropy computation},
  url = {https://github.com/justindbilyeu/ITPU}
}
```

## FAQ

### Q: I’m new to information theory. Can I still contribute?

A: Yes! Many contributions don’t require deep information theory knowledge:

- Documentation improvements
- Code quality enhancements
- Testing and validation
- Performance optimizations
- Example applications

### Q: How do I contribute to hardware development without FPGA experience?

A: Start with:

- Algorithm specification and testing
- Software reference implementations
- Hardware-software interface design
- Performance modeling and analysis

### Q: My contribution is very domain-specific (e.g., neuroscience). Is it welcome?

A: Domain-specific contributions are valuable:

- Add to `examples/` with clear documentation
- Consider making features optional (extra dependencies)
- Ensure broad applicability where possible
- Help us understand your domain’s needs

### Q: How can I get my institution to adopt ITPU?

A: Consider:

- Running benchmarks on your data
- Comparing with your current tools
- Writing case studies
- Presenting at lab meetings
- Contributing domain-specific examples

### Q: I found a bug but don’t know how to fix it. What should I do?

A: That’s perfectly fine:

- Create a detailed bug report
- Include a minimal example that reproduces the issue
- Specify your environment
- Others will help debug and fix

### Q: Can I implement a completely different algorithm?

A: Major algorithmic additions should be discussed first:

- Create a feature request issue
- Explain the algorithm and its benefits
- Consider backward compatibility
- Plan for testing and validation

## Getting Started Checklist

Ready to contribute? Here’s your checklist:

- [ ] Read this contributing guide
- [ ] Set up development environment (`make setup-dev`)
- [ ] Run smoke test (`make smoke-test`)
- [ ] Browse existing issues for ideas
- [ ] Join GitHub Discussions for questions
- [ ] Make your first contribution (documentation is great!)
- [ ] Ask questions when you’re stuck

Welcome to the ITPU community! 🚀

-----

## License

By contributing to ITPU, you agree that your contributions will be licensed under the Apache License 2.0. Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Contribution Types](#contribution-types)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)
- [Hardware Development](#hardware-development)
- [Community](#community)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with mutual information, entropy, or signal processing concepts
- Basic understanding of NumPy and SciPy

### Development Setup

1. **Fork and clone the repository:**
   
   ```bash
   git clone https://github.com/yourusername/ITPU.git
   cd ITPU
   ```
1. **Create a virtual environment:**
   
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
1. **Install development dependencies:**
   
   ```bash
   make install-dev
   # or manually:
   pip install -e ".[dev]"
   ```
1. **Run the smoke test:**
   
   ```bash
   make smoke-test
   # or manually:
   python scripts/smoke_test.py
   ```
1. **Run the test suite:**
   
   ```bash
   make test
   # or manually:
   pytest tests/
   ```

## Contributing Process

### 1. Issue First

Before starting work, please:

- Check existing issues and pull requests
- Create an issue describing the bug, feature, or improvement
- Discuss the approach with maintainers
- Get approval for significant changes

### 2. Branch and Develop

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Test your changes
make test
make all-checks

# Commit with descriptive messages
git commit -m "Add: streaming MI for categorical data"
```

### 3. Pull Request

- Create a pull request against the `main` branch
- Fill out the PR template completely
- Ensure all CI checks pass
- Request review from maintainers
- Address feedback promptly

### 4. Review and Merge

- Maintainers will review your code
- Address any requested changes
- Once approved, your PR will be merged

## Contribution Types

### 🐛 Bug Fixes

- Fix incorrect calculations
- Resolve performance issues
- Address edge cases
- Improve error handling

**Good first issues:** Look for issues labeled `good-first-issue`

### ✨ New Features

- New MI/entropy estimation methods
- Streaming/windowing improvements
- Hardware acceleration features
- API enhancements

**Before implementing:** Discuss design in an issue first

### 📊 Performance Improvements

- Algorithm optimizations
- Memory usage reductions
- Parallelization
- GPU acceleration

**Requirements:** Include benchmarks showing improvement

### 📖 Documentation

- API documentation
- Tutorials and examples
- README improvements
- Docstring additions

**Style:** Follow NumPy docstring conventions

### 🧪 Testing

- Unit tests for new features
- Integration tests
- Performance benchmarks
- Edge case testing

**Coverage:** Aim for >90% test coverage

### 🔧 Tools and Infrastructure

- CI/CD improvements
- Development tools
- Build process enhancements
- Packaging improvements

## Code Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) with 88-character line limit
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting

**Check your code:**

```bash
make format      # Format code
make lint        # Check linting
make type-check  # Type checking
make all-checks  # Run all checks
```

### Code Organization

```
src/itpu/
├── __init__.py          # Main package
├── sdk.py               # User-facing API
├── kernels_sw/          # Software kernels
│   ├── hist.py          # Histogram operations
│   ├── ksg.py           # KSG estimators
│   └── utils.py         # Kernel utilities
├── utils/               # General utilities
│   ├── windowed.py      # Windowing functions
│   └── validation.py    # Input validation
├── benchmarks/          # Performance benchmarks
├── demos/               # Example applications
└── hardware/            # Hardware-related code
```

### Naming Conventions

- **Functions:** `snake_case` (e.g., `mutual_info`, `windowed_mi`)
- **Classes:** `PascalCase` (e.g., `MIEstimator`, `StreamingProcessor`)
- **Constants:** `UPPER_CASE` (e.g., `DEFAULT_BINS`, `MAX_CHANNELS`)
- **Private members:** Leading underscore (e.g., `_validate_input`)

### Documentation Style

Use NumPy-style docstrings:

```python
def mutual_info(x, y, method="hist", bins=64):
    """
    Compute mutual information between two variables.
    
    Parameters
    ----------
    x : array_like
        First variable, shape (n_samples,)
    y : array_like
        Second variable, shape (n_samples,)
    method : {"hist", "ksg"}, default="hist"
        Estimation method to use
    bins : int, default=64
        Number of histogram bins (for method="hist")
    
    Returns
    -------
    mi : float
        Mutual information in nats
    
    Examples
    --------
    >>> from itpu.sdk import ITPU
    >>> itpu = ITPU()
    >>> mi = itpu.mutual_info(x, y)
    """
```

## Testing

### Test Structure

```
tests/
├── test_sdk.py              # SDK interface tests
├── test_kernels_hist.py     # Histogram kernel tests
├── test_kernels_ksg.py      # KSG kernel tests
├── test_windowed.py         # Windowing tests
├── test_benchmarks.py       # Benchmark tests
├── integration/             # Integration tests
│   ├── test_eeg_demo.py     # EEG demo tests
│   └── test_streaming.py    # Streaming tests
└── performance/             # Performance tests
    ├── test_speed.py        # Speed benchmarks
    └── test_memory.py       # Memory usage tests
```

### Writing Tests

```python
import pytest
import numpy as np
from itpu.sdk import ITPU

class TestMutualInfo:
    def setup_method(self):
        """Setup for each test method."""
        self.itpu = ITPU(device="software")
        self.rng = np.random.default_rng(42)
    
    def test_independent_variables(self):
        """Test MI of independent variables approaches zero."""
        x = self.rng.normal(size=10000)
        y = self.rng.normal(size=10000)
        
        mi = self.itpu.mutual_info(x, y, method="hist", bins=64)
        
        # Independent variables should have low MI
        assert mi < 0.1
    
    @pytest.mark.slow
    def test_large_data(self):
        """Test with large datasets."""
        x = self.rng.normal(size=1_000_000)
        y = 0.8 * x + 0.2 * self.rng.normal(size=1_000_000)
        
        mi = self.itpu.mutual_info(x, y, method="hist", bins=128)
        
        # Highly correlated variables should have high MI
        assert mi > 0.5
```

### Running Tests

```bash
# All tests
make test

# Fast tests only (skip @pytest.mark.slow)
make test-fast

# With coverage
make test-cov

# Specific test file
pytest tests/test_sdk.py -v

# Specific test
pytest tests/test_sdk.py::TestMutualInfo::test_independent_variables -v
```

## 
