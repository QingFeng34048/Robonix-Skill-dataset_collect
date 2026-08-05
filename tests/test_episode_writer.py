from pathlib import Path

import h5py
import numpy as np
import pytest

from dataset_collect_skill.episode_writer import (
    EpisodeSample,
    save_episode,
)


def _sample(
    joint_1: float,
    gripper: float,
    timestamp: float,
) -> EpisodeSample:
    return EpisodeSample(
        image_rgb=np.zeros(
            (8, 8, 3),
            dtype=np.uint8,
        ),
        state=np.asarray(
            [
                joint_1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                gripper,
            ],
            dtype=np.float32,
        ),
        timestamp_s=timestamp,
    )


def test_writes_compatible_hdf5_episode(
    tmp_path: Path,
) -> None:
    result = save_episode(
        output_root=tmp_path,
        run_id="1234567890abcdef",
        task_id="pick_banana",
        instruction="pick up the banana",
        dataset_name="piper_multitask",
        fps=10.0,
        samples=[
            _sample(0.0, 0.0, 1.0),
            _sample(0.1, 1.0, 1.1),
            _sample(0.15, 1.0, 1.2),
        ],
        success=True,
        note="good trajectory",
        minimum_frames=2,
        gzip_compression_level=4,
        gripper_state_mode="binary_0_1",
    )

    assert result.path.is_file()
    assert result.frames == 3

    with h5py.File(result.path, "r") as file:
        assert file.attrs["task_id"] == "pick_banana"
        assert bool(file.attrs["success"]) is True

        states = file["observations/state"][:]
        actions = file["action"][:]
        images = file["observations/images"][:]
        timestamps = file[
            "observations/timestamp"
        ][:]

        assert states.shape == (3, 7)
        assert actions.shape == (3, 7)
        assert images.shape == (3, 8, 8, 3)
        assert timestamps.shape == (3,)

        assert actions[0, 0] == pytest.approx(0.1)
        assert actions[1, 0] == pytest.approx(0.05)
        assert actions[0, 6] == pytest.approx(1.0)
        assert actions[-1, 0] == pytest.approx(0.0)


def test_rejects_short_episode(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        save_episode(
            output_root=tmp_path,
            run_id="run",
            task_id="pick",
            instruction="pick",
            dataset_name="dataset",
            fps=10.0,
            samples=[_sample(0.0, 0.0, 1.0)],
            success=True,
            note="",
            minimum_frames=2,
            gzip_compression_level=0,
            gripper_state_mode="binary_0_1",
        )
