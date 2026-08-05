from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes", "on"}:
            return True

        if normalized in {"false", "0", "no", "off"}:
            return False

    raise ValueError(f"{name} must be a boolean")


def _joint_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("joint_names must be a list")

    result = tuple(str(item).strip() for item in value)

    if len(result) != 6:
        raise ValueError("joint_names must contain exactly six names")

    if any(not item for item in result):
        raise ValueError("joint_names contains an empty name")

    if len(set(result)) != len(result):
        raise ValueError("joint_names contains duplicate names")

    return result


@dataclass(frozen=True)
class RuntimeConfig:
    camera_provider_id: str
    arm_provider_id: str

    workspace_root: Path
    output_subdir: str
    dataset_name: str

    fps: float
    model_width: int
    model_height: int

    max_image_age_s: float
    max_state_age_s: float
    dependency_wait_timeout_s: float
    default_max_duration_s: float

    max_episode_frames: int
    minimum_frames: int

    gripper_state_mode: str
    gripper_open_threshold_m: float

    save_failed_episodes: bool
    gzip_compression_level: int

    joint_names: tuple[str, ...]
    gripper_name: str

    piper_command_metadata: bool
    joint_velocity_pct: float
    gripper_effort: float

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        base_dir: str | Path | None = None,
    ) -> "RuntimeConfig":
        data = dict(raw or {})

        # 允许直接传 config，也允许传 {"config": {...}}。
        if set(data) == {"config"} and isinstance(
            data["config"],
            Mapping,
        ):
            data = dict(data["config"])

        allowed = {
            "camera_provider_id",
            "arm_provider_id",
            "workspace_root",
            "output_subdir",
            "dataset_name",
            "fps",
            "model_width",
            "model_height",
            "max_image_age_s",
            "max_state_age_s",
            "dependency_wait_timeout_s",
            "default_max_duration_s",
            "max_episode_frames",
            "minimum_frames",
            "gripper_state_mode",
            "gripper_open_threshold_m",
            "save_failed_episodes",
            "gzip_compression_level",
            "joint_names",
            "gripper_name",
            "piper_command_metadata",
            "joint_velocity_pct",
            "gripper_effort",
        }

        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                "unknown runtime config key(s): "
                + ", ".join(unknown)
            )

        default_base = (
            base_dir
            or os.environ.get("RBNX_INVOCATION_CWD")
            or os.environ.get("RBNX_PACKAGE_ROOT")
            or Path.cwd()
        )
        resolved_base = Path(default_base).expanduser().resolve()

        workspace_value = Path(
            str(data.get("workspace_root", "workspace"))
        ).expanduser()

        if not workspace_value.is_absolute():
            workspace_value = resolved_base / workspace_value

        workspace_root = workspace_value.resolve()

        config = cls(
            camera_provider_id=str(
                data.get("camera_provider_id", "front_camera")
            ).strip(),
            arm_provider_id=str(
                data.get("arm_provider_id", "piper_arm")
            ).strip(),
            workspace_root=workspace_root,
            output_subdir=str(
                data.get("output_subdir", "datasets_hdf5")
            ).strip(),
            dataset_name=str(
                data.get("dataset_name", "piper_multitask")
            ).strip(),
            fps=float(data.get("fps", 10.0)),
            model_width=int(data.get("model_width", 224)),
            model_height=int(data.get("model_height", 224)),
            max_image_age_s=float(
                data.get("max_image_age_s", 1.0)
            ),
            max_state_age_s=float(
                data.get("max_state_age_s", 0.5)
            ),
            dependency_wait_timeout_s=float(
                data.get("dependency_wait_timeout_s", 5.0)
            ),
            default_max_duration_s=float(
                data.get("default_max_duration_s", 120.0)
            ),
            max_episode_frames=int(
                data.get("max_episode_frames", 3600)
            ),
            minimum_frames=int(data.get("minimum_frames", 2)),
            gripper_state_mode=str(
                data.get(
                    "gripper_state_mode",
                    "binary_0_1",
                )
            ).strip(),
            gripper_open_threshold_m=float(
                data.get("gripper_open_threshold_m", 0.02)
            ),
            save_failed_episodes=_as_bool(
                data.get("save_failed_episodes", True),
                name="save_failed_episodes",
            ),
            gzip_compression_level=int(
                data.get("gzip_compression_level", 4)
            ),
            joint_names=_joint_names(
                data.get(
                    "joint_names",
                    [
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                )
            ),
            gripper_name=str(
                data.get("gripper_name", "gripper")
            ).strip(),
            piper_command_metadata=_as_bool(
                data.get("piper_command_metadata", True),
                name="piper_command_metadata",
            ),
            joint_velocity_pct=float(
                data.get("joint_velocity_pct", 30.0)
            ),
            gripper_effort=float(
                data.get("gripper_effort", 1.0)
            ),
        )

        config.validate()
        return config

    @property
    def output_root(self) -> Path:
        relative = Path(self.output_subdir)

        if relative.is_absolute():
            raise ValueError(
                "output_subdir must be relative to workspace_root"
            )

        output = (self.workspace_root / relative).resolve()

        try:
            output.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(
                "output_subdir escapes workspace_root"
            ) from exc

        return output

    def validate(self) -> None:
        if not self.camera_provider_id:
            raise ValueError(
                "camera_provider_id must not be empty"
            )

        if not self.arm_provider_id:
            raise ValueError("arm_provider_id must not be empty")

        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty")

        if not self.output_subdir:
            raise ValueError("output_subdir must not be empty")

        positive_numbers = {
            "fps": self.fps,
            "max_image_age_s": self.max_image_age_s,
            "max_state_age_s": self.max_state_age_s,
            "dependency_wait_timeout_s": (
                self.dependency_wait_timeout_s
            ),
            "default_max_duration_s": (
                self.default_max_duration_s
            ),
        }

        for name, value in positive_numbers.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and > 0"
                )

        if self.model_width < 1 or self.model_height < 1:
            raise ValueError(
                "model_width and model_height must be > 0"
            )

        if self.max_episode_frames < 2:
            raise ValueError(
                "max_episode_frames must be >= 2"
            )

        if self.minimum_frames < 2:
            raise ValueError("minimum_frames must be >= 2")

        if self.minimum_frames > self.max_episode_frames:
            raise ValueError(
                "minimum_frames cannot exceed "
                "max_episode_frames"
            )

        if self.gripper_state_mode not in {
            "binary_0_1",
            "absolute_m",
        }:
            raise ValueError(
                "gripper_state_mode must be "
                "binary_0_1 or absolute_m"
            )

        if (
            not math.isfinite(self.gripper_open_threshold_m)
            or self.gripper_open_threshold_m < 0
        ):
            raise ValueError(
                "gripper_open_threshold_m must be finite "
                "and >= 0"
            )

        if not 0 <= self.gzip_compression_level <= 9:
            raise ValueError(
                "gzip_compression_level must be in [0, 9]"
            )

        if not self.gripper_name:
            raise ValueError("gripper_name must not be empty")

        if self.gripper_name in self.joint_names:
            raise ValueError(
                "gripper_name duplicates a joint name"
            )

        if not 1.0 <= self.joint_velocity_pct <= 100.0:
            raise ValueError(
                "joint_velocity_pct must be in [1, 100]"
            )

        if not 0.5 <= self.gripper_effort <= 3.0:
            raise ValueError(
                "gripper_effort must be in [0.5, 3.0]"
            )

        self.workspace_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

