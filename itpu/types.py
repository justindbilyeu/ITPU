from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


class EstimatorValue(float):
    """A float subclass tagged with estimator provenance.

    Behaves like a plain float for all standard operations. Raises TypeError
    on cross-estimator comparison (==) or addition to make mixing of histogram
    and KSG MI values an explicit, detectable error rather than a silent one.

    float(EstimatorValue(v, e)) strips the tag and returns a plain Python
    float, preserving backward compatibility with code that expects a number.
    """

    estimator: Literal["hist", "ksg"]

    def __new__(cls, value: float, estimator: Literal["hist", "ksg"]) -> "EstimatorValue":
        obj = super().__new__(cls, value)
        obj.estimator = estimator
        return obj

    # Required: Python sets __hash__ to None when __eq__ is overridden.
    __hash__ = float.__hash__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EstimatorValue) and other.estimator != self.estimator:
            raise TypeError(
                f"Cannot compare {self.estimator!r} MI ({float(self):.4f}) with "
                f"{other.estimator!r} MI ({float(other):.4f}). These estimators measure "
                f"different quantities. Use to_common_basis() for explicit conversion, "
                f"or ensure both values use the same estimator."
            )
        return super().__eq__(other)

    def __add__(self, other: object) -> "EstimatorValue":
        if isinstance(other, EstimatorValue) and other.estimator != self.estimator:
            raise TypeError(
                f"Cannot add {self.estimator!r} MI and {other.estimator!r} MI. "
                f"Cross-estimator arithmetic is undefined."
            )
        return EstimatorValue(super().__add__(other), self.estimator)

    def __radd__(self, other: object) -> "EstimatorValue":
        if isinstance(other, EstimatorValue) and other.estimator != self.estimator:
            raise TypeError(
                f"Cannot add {other.estimator!r} MI and {self.estimator!r} MI. "
                f"Cross-estimator arithmetic is undefined."
            )
        return EstimatorValue(super().__radd__(other), self.estimator)

    def __repr__(self) -> str:
        return f"EstimatorValue({float(self):.6f}, estimator='{self.estimator}')"

    def __str__(self) -> str:
        return f"{float(self):.6f} [{self.estimator}]"


@dataclass
class SurrogateResult:
    """Return type of surrogate_test().

    Carries the observed MI as a tagged EstimatorValue alongside the p-value,
    null distribution, and provenance metadata. __post_init__ enforces that
    mi.estimator matches the declared estimator field.
    """

    mi: EstimatorValue
    p_value: float
    n_surrogates: int
    estimator: Literal["hist", "ksg"]
    # Additional fields from the current surrogate_test() interface.
    null_distribution: np.ndarray = field(
        default_factory=lambda: np.empty(0), compare=False, repr=False
    )
    power_estimate: float = 0.0
    warnings: list = field(default_factory=list)
    # Populated by external calibration routines; nan when not computed.
    ks_stat: float = float("nan")

    def __post_init__(self) -> None:
        if self.mi.estimator != self.estimator:
            raise ValueError(
                f"SurrogateResult estimator mismatch: mi was computed with "
                f"'{self.mi.estimator}' but result claims '{self.estimator}'"
            )
