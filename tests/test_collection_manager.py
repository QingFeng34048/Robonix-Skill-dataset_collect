from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_collect_skill.collection_manager import (
    CollectionManager,
)
from dataset_collect_skill.runtime_config import (
    RuntimeConfig,
)


@dataclass(frozen=True)
class FakeArmState:
    joints_rad: tuple[float, ...]
    gripper_m: float


class FakeCamera:
    def __init__(self) -> None:
        self.connected = False

    def connect(
        self,
        *,
        wait_timeout_s: float,
    ) -> None:
        del wait_timeout_s
        self.connected = True

    def latest_rgb(
        self,
        *,
        max_age_s: float,
    ) -> np.ndarray:
        del max_age_s

        return np.zeros(
            (16, 16, 3),
            dtype=np.uint8,
        )

    def close(self) -> None:
        self.connected = False


class FakeArm:
    def __init__(self) -> None:
        self.connected = False
        self.index = 0

    def connect(
        self,
        *,
        wait_timeout_s: float,
    ) -> None:
        del wait_timeout_s
        self.connected = True

    def latest_state(
        self,
        *,
        max_age_s: float,
    ) -> FakeArmState:
        del max_age_s

        self.index += 1

        return FakeArmState(
            joints_rad=(
                self.index * 0.001,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            gripper_m=0.04,
        )

    def close(self) -> None:
        self.connected = False


def _settings(
    tmp_path: Path,
) -> RuntimeConfig:
    return RuntimeConfig.from_dict(
        {
            "workspace_root": str(tmp_path),
            "output_subdir": "hdf5",
            "fps": 100.0,
            "model_width": 8,
            "model_height": 8,
            "minimum_frames": 2,
            "max_episode_frames": 100,
            "default_max_duration_s": 1.0,
        },
        base_dir=tmp_path,
    )


def test_records_and_saves_episode(
    tmp_path: Path,
) -> None:
    manager = CollectionManager(
        skill=None,
        settings=_settings(tmp_path),
        camera=FakeCamera(),
        arm=FakeArm(),
    )

    manager.connect_dependencies()

    started = manager.start_episode(
        task_id="pick_banana",
        instruction="pick up the banana",
        max_duration_s=1.0,
    )

    assert started.accepted

    deadline = time.time() + 1.0

    while time.time() < deadline:
        status = manager.status(started.run_id)

        if status.frames >= 3:
            break

        time.sleep(0.01)

    finished = manager.finish_episode(
        run_id=started.run_id,
        success=True,
        note="test",
    )

    assert finished.accepted
    assert finished.frames >= 2
    assert finished.output_uri.startswith("file:")

    final_status = manager.status(started.run_id)

    assert final_status.state == "SUCCEEDED"
    assert final_status.progress == 1.0

    manager.close()


def test_cancel_discards_samples(
    tmp_path: Path,
) -> None:
    manager = CollectionManager(
        skill=None,
        settings=_settings(tmp_path),
        camera=FakeCamera(),
        arm=FakeArm(),
    )

    manager.connect_dependencies()

    started = manager.start_episode(
        task_id="pick_banana",
        instruction="pick up the banana",
        max_duration_s=1.0,
    )

    time.sleep(0.03)

    accepted, detail = manager.cancel(
        started.run_id
    )

    assert accepted
    assert "discarded" in detail
    assert (
        manager.status(started.run_id).state
        == "CANCELED"
    )

    manager.close()


def test_rejects_second_active_episode(
    tmp_path: Path,
) -> None:
    manager = CollectionManager(
        skill=None,
        settings=_settings(tmp_path),
        camera=FakeCamera(),
        arm=FakeArm(),
    )

    manager.connect_dependencies()

    first = manager.start_episode(
        task_id="pick_banana",
        instruction="pick up the banana",
        max_duration_s=1.0,
    )

    second = manager.start_episode(
        task_id="put_banana",
        instruction="put down the banana",
        max_duration_s=1.0,
    )

    assert first.accepted
    assert second.accepted is False
    assert second.run_id == ""

    manager.cancel(first.run_id)
    manager.close()
