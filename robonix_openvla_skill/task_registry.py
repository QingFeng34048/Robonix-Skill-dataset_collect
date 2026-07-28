from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    max_steps: int = 120
    max_delta_rad: float = 0.04
    execute_chunk_steps: int = 2
    enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TaskSpec":
        allowed = {
            "task_id",
            "instruction",
            "max_steps",
            "max_delta_rad",
            "max_delta",  # compatibility with the training config
            "execute_chunk_steps",
            "enabled",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                "unknown task config key(s): " + ", ".join(unknown)
            )
        values = dict(raw)
        if "max_delta" in values and "max_delta_rad" not in values:
            values["max_delta_rad"] = values.pop("max_delta")
        else:
            values.pop("max_delta", None)
        task = cls(**values)
        task.validate()
        return task

    def validate(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.instruction.strip():
            raise ValueError(
                f"task {self.task_id!r}: instruction must not be empty"
            )
        if int(self.max_steps) < 1:
            raise ValueError(
                f"task {self.task_id!r}: max_steps must be >= 1"
            )
        if (
            not math.isfinite(float(self.max_delta_rad))
            or float(self.max_delta_rad) <= 0
        ):
            raise ValueError(
                f"task {self.task_id!r}: max_delta_rad must be > 0"
            )
        if int(self.execute_chunk_steps) < 1:
            raise ValueError(
                f"task {self.task_id!r}: execute_chunk_steps must be >= 1"
            )


class TaskRegistry:
    def __init__(self, tasks: list[TaskSpec]) -> None:
        if not tasks:
            raise ValueError("task registry is empty")
        ids = [task.task_id for task in tasks]
        duplicates = sorted(
            task_id for task_id in set(ids) if ids.count(task_id) > 1
        )
        if duplicates:
            raise ValueError(
                "duplicate task_id(s): " + ", ".join(duplicates)
            )
        self._tasks = {task.task_id: task for task in tasks}
        if not any(task.enabled for task in tasks):
            raise ValueError("task registry has no enabled task")

    @classmethod
    def load(cls, path: str | Path) -> "TaskRegistry":
        source = Path(path)
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            raise ValueError(
                f"cannot read task registry {source}: {exc}"
            ) from exc
        except yaml.YAMLError as exc:
            raise ValueError(
                f"invalid task registry YAML {source}: {exc}"
            ) from exc

        if not isinstance(raw, Mapping):
            raise ValueError("task registry root must be a mapping")
        items = raw.get("tasks")
        if not isinstance(items, list):
            raise ValueError("task registry must contain a tasks list")
        tasks: list[TaskSpec] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"tasks[{index}] must be a mapping")
            tasks.append(TaskSpec.from_mapping(item))
        return cls(tasks)

    def get(self, task_id: str) -> TaskSpec:
        key = str(task_id).strip()
        if not key:
            raise RuntimeError("task_id must not be empty")
        task = self._tasks.get(key)
        if task is None:
            raise RuntimeError(f"unknown task_id: {key}")
        if not task.enabled:
            raise RuntimeError(f"task is disabled: {key}")
        return task

    def list_enabled(self) -> tuple[TaskSpec, ...]:
        return tuple(task for task in self._tasks.values() if task.enabled)
