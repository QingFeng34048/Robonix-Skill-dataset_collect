from pathlib import Path

import pytest

from robonix_openvla_skill.task_registry import TaskRegistry


def test_load_and_get(tmp_path: Path) -> None:
    path = tmp_path / "tasks.yaml"
    path.write_text(
        """
tasks:
  - task_id: pick
    instruction: pick object
    max_steps: 5
    max_delta_rad: 0.02
    execute_chunk_steps: 1
    enabled: true
""",
        encoding="utf-8",
    )
    registry = TaskRegistry.load(path)
    task = registry.get("pick")
    assert task.max_steps == 5
    assert task.max_delta_rad == 0.02


def test_duplicate_task_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tasks.yaml"
    path.write_text(
        """
tasks:
  - {task_id: pick, instruction: one}
  - {task_id: pick, instruction: two}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        TaskRegistry.load(path)
