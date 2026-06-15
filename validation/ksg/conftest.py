"""pytest configuration for the KSG validation suite.

Adds this directory to sys.path so that `import ksg`, `import ground_truth`,
and `import oracles` resolve correctly when pytest is invoked as:

    pytest validation/ksg/test_ksg.py

Also registers the custom markers used by the suite.
"""
import sys
import os

# Make the validation/ksg/ directory importable without installing anything.
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that run S=100 seeds at N=10000 (exclude with -m 'not slow')",
    )
    config.addinivalue_line(
        "markers",
        "diag: diagnostic-only tests — non-blocking, not counted in GATE exit code",
    )
