# SPDX-License-Identifier: Apache-2.0
"""
Information-Theoretic Processing Unit (ITPU)
Public package exports.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .sdk import ITPU
from .types import EstimatorValue, SurrogateResult
from .utils.windowed import windowed_mi

if TYPE_CHECKING:
    import numpy as np

__all__ = ["ITPU", "windowed_mi", "EstimatorValue", "SurrogateResult", "to_common_basis"]
__version__ = "0.1.0"


def to_common_basis(
    mi_value: EstimatorValue,
    target_estimator: Literal["hist", "ksg"],
    x: "np.ndarray",
    y: "np.ndarray",
    **kwargs,
) -> EstimatorValue:
    """Recompute MI using a different estimator on the original data.

    Intentionally requires re-passing x and y to signal that this is a
    non-trivial recomputation — not a unit conversion, but a different
    measurement construct applied to the same data.

    Parameters
    ----------
    mi_value:
        An EstimatorValue returned by ITPU.mutual_info(). Its numeric value
        is not used; only its type is validated.
    target_estimator:
        Estimator to use for recomputation. One of "hist" or "ksg".
    x, y:
        Original data arrays — must match those used to compute mi_value.
    **kwargs:
        Passed through to mutual_info() (e.g., bins=32, k=5).

    Returns
    -------
    EstimatorValue tagged with target_estimator.

    Raises
    ------
    TypeError
        If mi_value is not an EstimatorValue.
    """
    if not isinstance(mi_value, EstimatorValue):
        raise TypeError(
            f"Input must be an EstimatorValue from ITPU.mutual_info(), "
            f"got {type(mi_value).__name__}."
        )
    return ITPU(device="software").mutual_info(x, y, method=target_estimator, **kwargs)
