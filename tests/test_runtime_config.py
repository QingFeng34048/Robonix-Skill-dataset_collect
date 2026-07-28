from pathlib import Path

import pytest

from robonix_openvla_skill.runtime_config import RuntimeConfig


def _root(tmp_path: Path) -> Path:
    path = tmp_path / "configs/runtime"
    path.mkdir(parents=True)
    (path / "tasks.yaml").write_text(
        "tasks:\\n  - task_id: demo\\n"
        "    instruction: demo\\n",
        encoding="utf-8",
    )
    return tmp_path


def test_from_dict_resolves_package_relative_path(tmp_path: Path) -> None:
    root = _root(tmp_path)
    config = RuntimeConfig.from_dict(
        {
            "task_config_path": "configs/runtime/tasks.yaml",
            "joint_min_rad": [-2.0] * 6,
            "joint_max_rad": [2.0] * 6,
        },
        package_root=root,
    )
    config.validate()
    assert config.task_config_path == (
        root / "configs/runtime/tasks.yaml"
    ).resolve()


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path)
    with pytest.raises(ValueError, match="unknown runtime config"):
        RuntimeConfig.from_dict(
            {"task_config_path": "configs/runtime/tasks.yaml", "typo": 1},
            package_root=root,
        )


def test_real_joint_limits_are_required(tmp_path: Path) -> None:
    root = _root(tmp_path)
    config = RuntimeConfig.from_dict(
        {"task_config_path": "configs/runtime/tasks.yaml"},
        package_root=root,
    )
    with pytest.raises(ValueError, match="hardware-specific"):
        config.validate()
