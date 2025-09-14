#!/bin/bash

# scripts/setup_dev.sh

# Development environment setup script

set -e  # Exit on any error

echo “🚀 Setting up ITPU development environment…”

# Check Python version

PYTHON_VERSION=$(python3 -c “import sys; print(f’{sys.version_info.major}.{sys.version_info.minor}’)”)
REQUIRED_VERSION=“3.8”

if python3 -c “import sys; exit(0 if sys.version_info >= (3, 8) else 1)”; then
echo “✅ Python $PYTHON_VERSION found (required: $REQUIRED_VERSION+)”
else
echo “❌ Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION”
exit 1
fi

# Create virtual environment if it doesn’t exist

if [ ! -d “.venv” ]; then
echo “📦 Creating virtual environment…”
python3 -m venv .venv
else
echo “📦 Virtual environment already exists”
fi

# Activate virtual environment

echo “🔌 Activating virtual environment…”
source .venv/bin/activate

# Upgrade pip

echo “⬆️  Upgrading pip…”
pip install –upgrade pip

# Install ITPU in development mode

echo “🔧 Installing ITPU in development mode…”
pip install -e “.[dev]”

# Install pre-commit hooks

echo “🪝 Setting up pre-commit hooks…”
if command -v pre-commit &> /dev/null; then
pre-commit install
else
echo “⚠️  pre-commit not found, skipping hooks setup”
fi

# Run smoke test

echo “🧪 Running smoke test…”
if python scripts/smoke_test.py; then
echo “✅ Smoke test passed!”
else
echo “❌ Smoke test failed!”
exit 1
fi

# Run quick test suite

echo “🧪 Running quick test suite…”
if pytest tests/ -x -q; then
echo “✅ Tests passed!”
else
echo “❌ Tests failed!”
exit 1
fi

echo “”
echo “🎉 Development environment setup complete!”
echo “”
echo “Next steps:”
echo “  1. Activate environment: source .venv/bin/activate”
echo “  2. Run full tests: make test”
echo “  3. Run benchmarks: make benchmark-quick”
echo “  4. Start coding!”
echo “”
echo “Available commands:”
echo “  make help          # Show all available commands”
echo “  make test          # Run test suite”
echo “  make format        # Format code”
echo “  make lint          # Check code quality”
echo “  make demo          # Run EEG demo”

-----

#!/bin/bash

# scripts/run_benchmarks.sh

# Comprehensive benchmark runner

set -e

BENCHMARK_DIR=“benchmark_results”
TIMESTAMP=$(date +”%Y%m%d_%H%M%S”)
RESULTS_DIR=”$BENCHMARK_DIR/$TIMESTAMP”

echo “🏃 Running ITPU benchmarks…”
echo “Results will be saved to: $RESULTS_DIR”

# Create results directory

mkdir -p “$RESULTS_DIR”

# Run different benchmark types

echo “”
echo “1️⃣ Quick performance benchmark…”
python -m itpu.cli benchmark –quick –save “$RESULTS_DIR/quick” –plot

echo “”
echo “2️⃣ Comprehensive benchmark…”
python -m itpu.cli benchmark –full –save “$RESULTS_DIR/full” –plot

echo “”
echo “3️⃣ Memory usage analysis…”
python -c “
import numpy as np
import psutil
import os
from itpu.sdk import ITPU

def benchmark_memory():
print(‘Memory usage analysis:’)
process = psutil.Process(os.getpid())

```
# Baseline memory
baseline = process.memory_info().rss / 1024 / 1024
print(f'  Baseline: {baseline:.1f} MB')

# Create ITPU instance
itpu = ITPU(device='software')
after_init = process.memory_info().rss / 1024 / 1024
print(f'  After ITPU init: {after_init:.1f} MB (+{after_init-baseline:.1f} MB)')

# Large dataset
sizes = [10_000, 100_000, 1_000_000]
for size in sizes:
    x = np.random.randn(size)
    y = 0.7*x + 0.3*np.random.randn(size)
    
    before = process.memory_info().rss / 1024 / 1024
    mi = itpu.mutual_info(x, y, method='hist', bins=64)
    after = process.memory_info().rss / 1024 / 1024
    
    print(f'  {size:>7,} samples: {after:.1f} MB (+{after-before:.1f} MB) MI={mi:.3f}')
```

benchmark_memory()
“ > “$RESULTS_DIR/memory_analysis.txt”

echo “”
echo “4️⃣ Comparison with alternatives…”
python -c “
import time
import numpy as np
from itpu.sdk import ITPU

try:
from scipy.stats import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
SCIPY_AVAILABLE = True
except ImportError:
SCIPY_AVAILABLE = False
print(‘SciPy/scikit-learn not available, skipping comparison’)

if SCIPY_AVAILABLE:
print(‘Comparison with scipy/sklearn:’)

```
itpu = ITPU(device='software')
np.random.seed(42)

sizes = [1000, 10000, 100000]
for size in sizes:
    x = np.random.randn(size)
    y = 0.8*x + 0.2*np.random.randn(size)
    
    # ITPU
    start = time.perf_counter()
    mi_itpu = itpu.mutual_info(x, y, method='hist', bins=64)
    time_itpu = time.perf_counter() - start
    
    # SciPy (discrete)
    x_disc = np.digitize(x, np.histogram_bin_edges(x, bins=64))
    y_disc = np.digitize(y, np.histogram_bin_edges(y, bins=64))
    start = time.perf_counter()
    mi_scipy = mutual_info_score(x_disc, y_disc)
    time_scipy = time.perf_counter() - start
    
    speedup = time_scipy / time_itpu
    print(f'  {size:>6,} samples: ITPU={time_itpu:.4f}s, SciPy={time_scipy:.4f}s, Speedup={speedup:.2f}x')
```

“ >> “$RESULTS_DIR/comparison.txt”

# Generate summary report

echo “”
echo “📊 Generating summary report…”
cat > “$RESULTS_DIR/README.md” << EOF

# ITPU Benchmark Results

**Generated:** $(date)
**ITPU Version:** $(python -c “import itpu; print(getattr(itpu, ‘**version**’, ‘unknown’))”)
**System:** $(uname -a)
**Python:** $(python –version)

## Files

- `quick/` - Quick benchmark results
- `full/` - Comprehensive benchmark results
- `memory_analysis.txt` - Memory usage analysis
- `comparison.txt` - Comparison with scipy/sklearn
- `performance_plots.png` - Performance visualization

## Key Metrics

$(if [ -f “$RESULTS_DIR/quick/point_mi_small.csv” ]; then
python -c “
import pandas as pd
import numpy as np
try:
df = pd.read_csv(’$RESULTS_DIR/quick/point_mi_small.csv’)
if ‘speedup_factor’ in df.columns:
speedups = df[‘speedup_factor’].dropna()
if len(speedups) > 0:
print(f’- Average speedup vs SciPy: {speedups.mean():.2f}x’)
print(f’- Best speedup: {speedups.max():.2f}x’)
else:
print(’- Speedup data not available’)
else:
print(’- Speedup data not available’)
except Exception as e:
print(f’- Could not analyze speedup: {e}’)
“
fi)

## Notes

Run with: `./scripts/run_benchmarks.sh`

For more details, see individual result files.
EOF

echo “✅ Benchmarks complete!”
echo “”
echo “📁 Results saved to: $RESULTS_DIR”
echo “📊 View summary: cat $RESULTS_DIR/README.md”
echo “🎨 Performance plots: $RESULTS_DIR/*/performance_plots.png”

-----

#!/bin/bash

# scripts/release_check.sh

# Pre-release validation script

set -e

echo “🔍 Running pre-release checks…”

# Version consistency check

echo “1️⃣ Checking version consistency…”
SETUP_VERSION=$(python setup.py –version 2>/dev/null || echo “unknown”)
PYPROJECT_VERSION=$(python -c “import tomllib; print(tomllib.load(open(‘pyproject.toml’, ‘rb’))[‘project’][‘version’])” 2>/dev/null || echo “unknown”)
INIT_VERSION=$(python -c “import sys; sys.path.insert(0, ‘src’); import itpu; print(getattr(itpu, ‘**version**’, ‘unknown’))” 2>/dev/null || echo “unknown”)

echo “  setup.py: $SETUP_VERSION”
echo “  pyproject.toml: $PYPROJECT_VERSION”
echo “  **init**.py: $INIT_VERSION”

if [ “$SETUP_VERSION” != “$PYPROJECT_VERSION” ] || [ “$PYPROJECT_VERSION” != “$INIT_VERSION” ]; then
echo “❌ Version mismatch detected!”
exit 1
else
echo “✅ Versions consistent: $SETUP_VERSION”
fi

# Code quality checks

echo “”
echo “2️⃣ Running code quality checks…”
make all-checks

# Test suite

echo “”
echo “3️⃣ Running full test suite…”
make test-cov

# Build test

echo “”
echo “4️⃣ Testing package build…”
python -m build
python -m twine check dist/*

# Import test

echo “”
echo “5️⃣ Testing package installation…”

# Create temporary virtual environment

TEMP_VENV=$(mktemp -d)
python -m venv “$TEMP_VENV”
source “$TEMP_VENV/bin/activate”

# Install from built package

pip install dist/*.whl

# Test basic functionality

python -c “
import itpu
from itpu.sdk import ITPU
import numpy as np

print(‘Testing basic functionality…’)
itpu_instance = ITPU(device=‘software’)
x = np.random.randn(1000)
y = 0.7*x + 0.3*np.random.randn(1000)
mi = itpu_instance.mutual_info(x, y, method=‘hist’, bins=32)
print(f’MI test result: {mi:.3f}’)
assert 0 < mi < 2, f’MI value {mi} seems unreasonable’
print(‘✅ Basic functionality test passed’)
“

# Cleanup

deactivate
rm -rf “$TEMP_VENV”

# Documentation check

echo “”
echo “6️⃣ Checking documentation…”
if [ -d “docs” ]; then
make docs
echo “✅ Documentation builds successfully”
else
echo “⚠️  No docs directory found”
fi

# CHANGELOG check

echo “”
echo “7️⃣ Checking CHANGELOG…”
if [ -f “CHANGELOG.md” ]; then
if grep -q “## [Unreleased]” CHANGELOG.md; then
echo “⚠️  CHANGELOG has unreleased changes - consider updating for release”
else
echo “✅ CHANGELOG looks ready for release”
fi
else
echo “⚠️  No CHANGELOG.md found”
fi

# Security check

echo “”
echo “8️⃣ Running security checks…”
if command -v bandit &> /dev/null; then
bandit -r src/ -ll
echo “✅ Security scan completed”
else
echo “⚠️  bandit not installed, skipping security scan”
fi

echo “”
echo “🎉 Pre-release checks completed!”
echo “”
echo “📋 Release checklist:”
echo “  ✅ Version consistency verified”
echo “  ✅ Code quality checks passed”  
echo “  ✅ Full test suite passed”
echo “  ✅ Package builds successfully”
echo “  ✅ Installation test passed”
echo “”
echo “🚀 Ready for release!”

-----

#!/bin/bash

# scripts/profile_performance.sh

# Performance profiling script

set -e

PROFILE_DIR=“profiling_results”
TIMESTAMP=$(date +”%Y%m%d_%H%M%S”)
RESULTS_DIR=”$PROFILE_DIR/$TIMESTAMP”

echo “🔬 Profiling ITPU performance…”
echo “Results will be saved to: $RESULTS_DIR”

mkdir -p “$RESULTS_DIR”

# CPU profiling

echo “”
echo “1️⃣ CPU profiling with cProfile…”
python -m cProfile -o “$RESULTS_DIR/profile.prof” -c “
import numpy as np
from itpu.sdk import ITPU

itpu = ITPU(device=‘software’)
np.random.seed(42)

# Profile different scenarios

print(‘Profiling small data…’)
for i in range(10):
x = np.random.randn(10000)
y = 0.7*x + 0.3*np.random.randn(10000)
mi = itpu.mutual_info(x, y, method=‘hist’, bins=64)

print(‘Profiling large data…’)
for i in range(3):
x = np.random.randn(100000)
y = 0.7*x + 0.3*np.random.randn(100000)
mi = itpu.mutual_info(x, y, method=‘hist’, bins=128)
“

# Generate profile report

python -c “
import pstats
p = pstats.Stats(’$RESULTS_DIR/profile.prof’)
p.sort_stats(‘cumulative’)
with open(’$RESULTS_DIR/profile_report.txt’, ‘w’) as f:
p.print_stats(20, file=f)
print(‘📊 Profile report saved to $RESULTS_DIR/profile_report.txt’)
“

# Memory profiling (if memory_profiler is available)

echo “”
echo “2️⃣ Memory profiling…”
if python -c “import memory_profiler” 2>/dev/null; then
python -m memory_profiler scripts/memory_profile_target.py > “$RESULTS_DIR/memory_profile.txt”
echo “📊 Memory profile saved to $RESULTS_DIR/memory_profile.txt”
else
echo “⚠️  memory_profiler not available, skipping memory profiling”
echo “   Install with: pip install memory_profiler”
fi

# Line profiling (if line_profiler is available)

echo “”
echo “3️⃣ Line-by-line profiling…”
if python -c “import line_profiler” 2>/dev/null; then
kernprof -l -v scripts/line_profile_target.py > “$RESULTS_DIR/line_profile.txt” 2>&1
echo “📊 Line profile saved to $RESULTS_DIR/line_profile.txt”
else
echo “⚠️  line_profiler not available, skipping line profiling”
echo “   Install with: pip install line_profiler”
fi

echo “”
echo “🎉 Performance profiling complete!”
echo “📁 Results in: $RESULTS_DIR”
