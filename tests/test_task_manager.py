import time
from dataclasses import dataclass
from pathlib import Path

from robonix_openvla_skill.runtime_config import RuntimeConfig
from robonix_openvla_skill.task_manager import TaskManager
from robonix_openvla_skill.vla_client import VLAResult


@dataclass
class FakeState:
    joints_rad: tuple[float, ...] = (0.0,) * 6
    gripper_m: float = 0.04

    def model_vector(self):
        return [*self.joints_rad, self.gripper_m]


class FakeCamera:
    def connect(self, **kwargs):
        pass

    def latest_jpeg(self, **kwargs):
        return b"jpeg"

    def close(self):
        pass


class FakeArm:
    def __init__(self):
        self.state = FakeState()
        self.commands = []
        self.held = False

    def connect(self, **kwargs):
        pass

    def latest_state(self, **kwargs):
        return self.state

    def command_joint_target(self, *, joints_rad, gripper_m):
        self.commands.append((tuple(joints_rad), gripper_m))
        self.state = FakeState(tuple(joints_rad), gripper_m)

    def hold_current_position(self, **kwargs):
        self.held = True

    def close(self):
        pass


class FakeVLA:
    def predict(self, **kwargs):
        return VLAResult(
            actions=((0.01, 0, 0, 0, 0, 0, 0.5),),
            done=True,
            success=True,
            detail="done",
        )

    def close(self):
        pass


def _settings(tmp_path: Path) -> RuntimeConfig:
    task_path = tmp_path / "tasks.yaml"
    task_path.write_text(
        """
tasks:
  - task_id: demo
    instruction: demo task
    max_steps: 4
    max_delta_rad: 0.02
    execute_chunk_steps: 1
    enabled: true
""",
        encoding="utf-8",
    )
    config = RuntimeConfig.from_dict(
        {
            "task_config_path": str(task_path),
            "control_hz": 100.0,
            "joint_min_rad": [-1.0] * 6,
            "joint_max_rad": [1.0] * 6,
        },
        package_root=tmp_path,
    )
    config.validate()
    return config


def test_async_run_reaches_success(tmp_path: Path) -> None:
    arm = FakeArm()
    manager = TaskManager(
        skill=None,
        settings=_settings(tmp_path),
        camera=FakeCamera(),
        arm=arm,
        vla_client=FakeVLA(),
    )
    manager.connect_dependencies()
    started = manager.start(
        task_id="demo",
        instruction="",
        timeout_s=1.0,
    )
    assert started.accepted
    deadline = time.time() + 1.0
    while time.time() < deadline:
        status = manager.status(started.run_id)
        if status.state == "SUCCEEDED":
            break
        time.sleep(0.01)
    assert status.state == "SUCCEEDED"
    assert status.progress == 1.0
    assert arm.commands
    manager.close()


def test_unknown_run_id_raises(tmp_path: Path) -> None:
    manager = TaskManager(
        skill=None,
        settings=_settings(tmp_path),
        camera=FakeCamera(),
        arm=FakeArm(),
        vla_client=FakeVLA(),
    )
    manager.connect_dependencies()
    try:
        manager.status("missing")
    except RuntimeError as exc:
        assert "unknown run_id" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    manager.close()


def test_deactivate_then_reactivate(tmp_path: Path) -> None:
    manager = TaskManager(
        skill=None,
        settings=_settings(tmp_path),
        camera=FakeCamera(),
        arm=FakeArm(),
        vla_client=FakeVLA(),
    )
    manager.connect_dependencies()
    manager.disconnect_dependencies()
    manager.connect_dependencies()
    started = manager.start(
        task_id="demo",
        instruction="",
        timeout_s=1.0,
    )
    assert started.accepted
    manager.close()
