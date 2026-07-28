from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


def _float_tuple(
    value: Any,
    *,
    name: str,
    length: int,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list with {length} numbers")
    result = tuple(float(item) for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _string_tuple(
    value: Any,
    *,
    name: str,
    length: int,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list with {length} strings")
    result = tuple(str(item).strip() for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if any(not item for item in result):
        raise ValueError(f"{name} contains an empty value")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate values")
    return result


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated runtime-only configuration passed by Robonix.

    Robot-specific values belong in the deployment manifest, not in the
    reusable package manifest.
    """

    vla_server_url: str = "http://127.0.0.1:8777/act"
    camera_provider_id: str = "front_camera"
    arm_provider_id: str = "piper_arm"
    task_config_path: Path = Path("configs/runtime/tasks.yaml")

    control_hz: float = 10.0
    request_timeout_s: float = 10.0
    task_timeout_s: float = 120.0
    dependency_wait_timeout_s: float = 5.0
    max_image_age_s: float = 1.0
    max_state_age_s: float = 0.5
    jpeg_quality: int = 90
    max_concurrent_runs: int = 1
    require_vla_healthcheck: bool = False

    enable_safety_filter: bool = True
    joint_min_rad: tuple[float, ...] = field(default_factory=tuple)
    joint_max_rad: tuple[float, ...] = field(default_factory=tuple)
    default_max_delta_rad: float = 0.04
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.08
    model_gripper_mode: str = "normalized_0_1"

    joint_names: tuple[str, ...] = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )
    gripper_name: str = "gripper"
    piper_command_metadata: bool = True
    joint_velocity_pct: float = 30.0
    gripper_effort: float = 1.0
    hold_position_on_cancel: bool = True

    package_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1],
        repr=False,
        compare=False,
    )

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        package_root: str | Path | None = None,
    ) -> "RuntimeConfig":
        data: dict[str, Any] = dict(raw or {})
        # Accept either the direct Robonix config mapping or a copied
        # ``config: {...}`` block.
        if set(data) == {"config"} and isinstance(data["config"], Mapping):
            data = dict(data["config"])

        root_value = (
            package_root
            or os.environ.get("RBNX_PACKAGE_ROOT")
            or Path(__file__).resolve().parents[1]
        )
        root = Path(root_value).expanduser().resolve()

        allowed = {
            field_name
            for field_name in cls.__dataclass_fields__
            if field_name != "package_root"
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                "unknown runtime config key(s): " + ", ".join(unknown)
            )

        values: dict[str, Any] = dict(data)
        task_path = Path(
            values.get("task_config_path", "configs/runtime/tasks.yaml")
        ).expanduser()
        if not task_path.is_absolute():
            task_path = root / task_path
        values["task_config_path"] = task_path.resolve()

        if "joint_min_rad" in values:
            values["joint_min_rad"] = _float_tuple(
                values["joint_min_rad"],
                name="joint_min_rad",
                length=6,
            )
        if "joint_max_rad" in values:
            values["joint_max_rad"] = _float_tuple(
                values["joint_max_rad"],
                name="joint_max_rad",
                length=6,
            )
        if "joint_names" in values:
            values["joint_names"] = _string_tuple(
                values["joint_names"],
                name="joint_names",
                length=6,
            )

        values["package_root"] = root
        config = cls(**values)
        return config

    def validate(self) -> None:
        parsed = urlparse(self.vla_server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "vla_server_url must be an absolute http(s) URL"
            )
        if not self.camera_provider_id.strip():
            raise ValueError("camera_provider_id must not be empty")
        if not self.arm_provider_id.strip():
            raise ValueError("arm_provider_id must not be empty")
        if not self.gripper_name.strip():
            raise ValueError("gripper_name must not be empty")
        if self.gripper_name in self.joint_names:
            raise ValueError("gripper_name duplicates a joint name")
        if not self.task_config_path.is_file():
            raise ValueError(
                f"task_config_path does not exist: {self.task_config_path}"
            )

        positive_fields = {
            "control_hz": self.control_hz,
            "request_timeout_s": self.request_timeout_s,
            "task_timeout_s": self.task_timeout_s,
            "dependency_wait_timeout_s": self.dependency_wait_timeout_s,
            "max_image_age_s": self.max_image_age_s,
            "max_state_age_s": self.max_state_age_s,
            "default_max_delta_rad": self.default_max_delta_rad,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and > 0")

        if not 1 <= int(self.jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if int(self.max_concurrent_runs) < 1:
            raise ValueError("max_concurrent_runs must be >= 1")
        if not 1.0 <= float(self.joint_velocity_pct) <= 100.0:
            raise ValueError("joint_velocity_pct must be in [1, 100]")
        if not 0.5 <= float(self.gripper_effort) <= 3.0:
            raise ValueError("gripper_effort must be in [0.5, 3.0]")

        if self.model_gripper_mode not in {
            "normalized_0_1",
            "absolute_m",
        }:
            raise ValueError(
                "model_gripper_mode must be normalized_0_1 or absolute_m"
            )
        if not (
            math.isfinite(self.gripper_min_m)
            and math.isfinite(self.gripper_max_m)
            and self.gripper_min_m < self.gripper_max_m
        ):
            raise ValueError(
                "gripper_min_m and gripper_max_m must be finite, "
                "with min < max"
            )

        if self.enable_safety_filter:
            if len(self.joint_min_rad) != 6 or len(self.joint_max_rad) != 6:
                raise ValueError(
                    "enable_safety_filter=true requires six hardware-specific "
                    "joint_min_rad and joint_max_rad values"
                )
            for index, (lower, upper) in enumerate(
                zip(self.joint_min_rad, self.joint_max_rad, strict=True)
            ):
                if lower >= upper:
                    raise ValueError(
                        f"joint limit {index} must satisfy min < max"
                    )
