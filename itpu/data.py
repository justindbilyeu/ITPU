# itpu/data.py
"""
Lightweight data loaders (offline-first) for examples and tests.
No heavy deps (e.g., pandas). Works with plain CSV via numpy.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Tuple, Optional


def make_synthetic_eeg(n: int = 15_000, d: int = 14, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic 14-ch EEG-like matrix X and binary label y."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, d))
    X = 0.5 * z + np.sqrt(1 - 0.5**2) * noise
    y = (X[:, 0] + 0.2 * rng.standard_normal(n) > 0).astype(float)
    return X, y


def load_eeg_eye_state(path: str = "data/eeg_eye_state.csv") -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """
    Attempt to load UCI EEG Eye State CSV: 14 EEG channels + binary label (last col).
    Returns (X, y, used_fallback) where y may be None if not present.
    """
    if not os.path.exists(path):
        X, y = make_synthetic_eeg()
        return X, y, True

    try:
        raw = np.genfromtxt(path, delimiter=",", dtype=float, names=None)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError("CSV not wide enough")
        # If we have >=15 cols, assume last is label, first 14 are channels.
        if raw.shape[1] >= 15:
            X = raw[:, :14].astype(float)
            y = raw[:, 14].astype(float)
        else:
            # No label column — take all but last as channels, ignore label
            X = raw[:, :-1].astype(float)
            y = None
        return X, y, False
    except Exception:
        X, y = make_synthetic_eeg()
        return X, y, True
