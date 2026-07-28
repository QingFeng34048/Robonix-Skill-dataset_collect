from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .units import normalized_gripper_to_metres, validate_action_vector


@dataclass(frozen=True)
class SafetyLimits:
    joint_min_rad: tuple[float, ...]
    joint_max_rad: tuple[float, ...]
    default_max_delta_rad: float
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.08
    model_gripper_mode: str = "normalized_0_1"


class SafetyFilter:
    """Convert a delta-joint VLA action to a bounded absolute command."""

    def __init__(self, limits: SafetyLimits) -> None:
        self._limits = limits
        self._joint_min = np.asarray(
            limits.joint_min_rad, dtype=np.float64
        )
        self._joint_max = np.asarray(
            limits.joint_max_rad, dtype=np.float64
        )
        if self._joint_min.shape != (6,) or self._joint_max.shape != (6,):
            raise ValueError("joint limits must each contain six values")
        if not (
            np.all(np.isfinite(self._joint_min))
            and np.all(np.isfinite(self._joint_max))
        ):
            raise ValueError("joint limits contain NaN or Inf")
        if np.any(self._joint_min >= self._joint_max):
            raise ValueError("each joint limit must satisfy min < max")
        if limits.default_max_delta_rad <= 0:
            raise ValueError("default_max_delta_rad must be > 0")
        if limits.gripper_min_m >= limits.gripper_max_m:
            raise ValueError("gripper_min_m must be < gripper_max_m")
        if limits.model_gripper_mode not in {
            "normalized_0_1",
            "absolute_m",
        }:
            raise ValueError("unsupported model_gripper_mode")

    def apply(
        self,
        *,
        current_joints: Sequence[float],
        action: Sequence[float],
        task_max_delta: float | None = None,
    ) -> tuple[tuple[float, ...], float]:
        current = np.asarray(current_joints, dtype=np.float64)
        if current.shape != (6,):
            raise ValueError(
                f"current_joints must have shape (6,), got {current.shape}"
            )
        if not np.all(np.isfinite(current)):
            raise ValueError("current_joints contains NaN or Inf")

        raw = validate_action_vector(action)
        delta_limit = float(
            self._limits.default_max_delta_rad
            if task_max_delta is None
            else task_max_delta
        )
        if not np.isfinite(delta_limit) or delta_limit <= 0:
            raise ValueError("task_max_delta must be finite and > 0")

        clipped_delta = np.clip(raw[:6], -delta_limit, delta_limit)
        target = np.clip(
            current + clipped_delta,
            self._joint_min,
            self._joint_max,
        )

        if self._limits.model_gripper_mode == "normalized_0_1":
            gripper = normalized_gripper_to_metres(
                float(raw[6]),
                minimum_m=self._limits.gripper_min_m,
                maximum_m=self._limits.gripper_max_m,
            )
        else:
            gripper = float(
                np.clip(
                    raw[6],
                    self._limits.gripper_min_m,
                    self._limits.gripper_max_m,
                )
            )

        return tuple(float(value) for value in target), gripper
