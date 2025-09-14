# ITPU Development Makefile

# Automates common development tasks

.PHONY: help install install-dev test test-fast test-cov lint format type-check clean build docs serve-docs benchmark demo smoke-test all

# Default target

help:
@echo “ITPU Development Commands:”
@echo “”
@echo “Setup:”
@echo “  install       Install ITPU in current environment”
@echo “  install-dev   Install with development dependencies”
@echo “  install-all   Install with all optional dependencies”
@echo “”
@echo “Testing:”
@echo “  test          Run full test suite”
@echo “  test-fast     Run tests excluding slow ones”
@echo “  test-cov      Run tests with coverage report”
@echo “  smoke-test    Quick smoke test (basic functionality)”
@echo “”
@echo “Code Quality:”
@echo “  lint          Run linting (flake8)”
@echo “  format        Format code (black + isort)”
@echo “  type-check    Run type checking (mypy)”
@echo “  all-checks    Run all code quality checks”
@echo “”
@echo “Documentation:”
@echo “  docs          Build documentation”
@echo “  serve-docs    Serve docs locally”
@echo “”
@echo “Demos & Benchmarks:”
@echo “  benchmark     Run performance benchmarks”
@echo “  demo          Run EEG real-time demo”
@echo “”
@echo “Build & Deploy:”
@echo “  build         Build distribution packages”
@echo “  clean         Clean build artifacts”
@echo “”

# Installation targets

install:
pip install -e .

install-dev:
pip install -e “.[dev]”

install-all:
pip install -e “.[all]”

# Testing targets

test:
pytest tests/ -v

test-fast:
pytest tests/ -v -m “not slow”

test-cov:
pytest tests/ –cov=src/itpu –cov-report=html –cov-report=term-missing

smoke-test:
python scripts/smoke_test.py

# Code quality targets

lint:
flake8 src/ tests/ examples/

format:
black src/ tests/ examples/ scripts/
isort src/ tests/ examples/ scripts/

type-check:
mypy src/itpu

all-checks: lint type-check
@echo “All code quality checks passed!”

# Documentation targets

docs:
cd docs && make html

serve-docs:
cd docs/_build/html && python -m http.server 8000

# Demo and benchmark targets

benchmark:
python -m itpu.cli benchmark –full –plot

benchmark-quick:
python -m itpu.cli benchmark –quick

demo:
python -m itpu.cli demo eeg

demo-synthetic:
python -m itpu.cli demo synthetic –save

# Build targets

build: clean
python -m build

clean:
rm -rf build/
rm -rf dist/
rm -rf src/*.egg-info/
rm -rf .pytest_cache/
rm -rf htmlcov/
rm -rf .coverage
rm -rf .mypy_cache/
find . -type d -name **pycache** -exec rm -rf {} +
find . -type f -name “*.pyc” -delete

# Development setup (run this first)

setup-dev: install-dev
@echo “Development environment setup complete!”
@echo “Run ‘make smoke-test’ to verify installation.”

# CI/CD simulation

ci: all-checks test-cov
@echo “CI checks passed!”

# Release preparation

release-check: ci build
@echo “Release checks passed!”
@echo “Ready for release.”
