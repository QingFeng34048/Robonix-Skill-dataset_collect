import pytest

from robonix_openvla_skill.safety import SafetyFilter, SafetyLimits


def _filter() -> SafetyFilter:
    return SafetyFilter(
        SafetyLimits(
            joint_min_rad=(-1.0,) * 6,
            joint_max_rad=(1.0,) * 6,
            default_max_delta_rad=0.1,
            gripper_min_m=0.0,
            gripper_max_m=0.08,
            model_gripper_mode="normalized_0_1",
        )
    )


def test_clips_delta_absolute_limits_and_gripper() -> None:
    joints, gripper = _filter().apply(
        current_joints=(0.95, 0, 0, 0, 0, 0),
        action=(0.5, -0.5, 0, 0, 0, 0, 1.5),
        task_max_delta=0.05,
    )
    assert joints[0] == pytest.approx(1.0)
    assert joints[1] == pytest.approx(-0.05)
    assert gripper == pytest.approx(0.08)


def test_rejects_non_finite_action() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        _filter().apply(
            current_joints=(0,) * 6,
            action=(0, 0, 0, 0, 0, float("nan"), 0),
        )
