from pathlib import Path

import pytest

from dataset_collect_skill.runtime_config import (
    RuntimeConfig,
)


def test_resolves_workspace_and_output(
    tmp_path: Path,
) -> None:
    config = RuntimeConfig.from_dict(
        {
            "workspace_root": "workspace",
            "output_subdir": "hdf5",
        },
        base_dir=tmp_path,
    )

    assert config.workspace_root == (
        tmp_path / "workspace"
    ).resolve()

    assert config.output_root == (
        tmp_path / "workspace" / "hdf5"
    ).resolve()

    assert config.output_root.is_dir()


def test_rejects_output_path_escape(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="escapes workspace_root",
    ):
        RuntimeConfig.from_dict(
            {
                "workspace_root": "workspace",
                "output_subdir": "../outside",
            },
            base_dir=tmp_path,
        )


def test_rejects_unknown_config_key(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="unknown runtime config",
    ):
        RuntimeConfig.from_dict(
            {
                "workspace_root": "workspace",
                "unknown_option": True,
            },
            base_dir=tmp_path,
        )


def test_rejects_invalid_gripper_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="gripper_state_mode",
    ):
        RuntimeConfig.from_dict(
            {
                "workspace_root": "workspace",
                "gripper_state_mode": "degrees",
            },
            base_dir=tmp_path,
        )
