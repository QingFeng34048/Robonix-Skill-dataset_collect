from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def validate_action_vector(action: Sequence[float]) -> np.ndarray:
    """Return one finite model action [dq1..dq6, gripper]."""
    result = np.asarray(action, dtype=np.float64)
    if result.shape != (7,):
        raise ValueError(
            f"model action must have shape (7,), got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("model action contains NaN or Inf")
    return result


def normalized_gripper_to_metres(
    value: float,
    *,
    minimum_m: float,
    maximum_m: float,
) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("gripper action is NaN or Inf")
    clipped = min(max(float(value), 0.0), 1.0)
    return minimum_m + clipped * (maximum_m - minimum_m)
