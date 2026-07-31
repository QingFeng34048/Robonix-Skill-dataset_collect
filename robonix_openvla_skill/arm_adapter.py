from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sensor_msgs.msg import JointState

from robonix_api import (
    ATLAS,
    Kind,
    Ros2Params,
    Skill,
    Transport,
)
from robonix_api.ros import RosBackend


log = logging.getLogger(__name__)

JOINT_STATES_CONTRACT = "robonix/primitive/arm/joint_states"
JOINT_COMMAND_CONTRACT = "robonix/primitive/arm/joint_command"


class ArmAdapterError(RuntimeError):
    """ArmAdapter 基础错误。"""


class ArmNotReadyError(ArmAdapterError):
    """机械臂尚未收到有效反馈。"""


class StaleArmStateError(ArmAdapterError):
    """机械臂反馈已经过期。"""


@dataclass(frozen=True)
class ArmState:
    """按配置顺序整理后的机械臂状态。"""

    joints_rad: tuple[float, ...]
    gripper_m: float

    joint_velocities_rad_s: tuple[float, ...] | None
    gripper_velocity_m_s: float | None

    received_monotonic: float
    source_stamp_ns: int

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.received_monotonic

    def model_vector(self) -> list[float]:
        """返回 [j1...j6, gripper]。"""

        return [*self.joints_rad, self.gripper_m]


def _channel_qos(channel, fallback: str) -> str:
    params = channel.params

    if isinstance(params, Ros2Params):
        qos = params.qos_profile.strip()
        if qos:
            return qos

    return fallback


class ArmAdapter:
   

    def __init__(
        self,
        *,
        skill: Skill,
        provider_id: str,
        joint_names: Sequence[str] = (
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ),
        gripper_name: str = "gripper",
        piper_command_metadata: bool = True,
        joint_velocity_pct: float = 30.0,
        gripper_effort: float = 1.0,
    ) -> None:
        if not provider_id.strip():
            raise ValueError("arm provider_id must not be empty")

        resolved_names = tuple(str(name).strip() for name in joint_names)

        if not resolved_names or any(not name for name in resolved_names):
            raise ValueError("joint_names must contain non-empty names")

        if len(set(resolved_names)) != len(resolved_names):
            raise ValueError("joint_names must not contain duplicates")

        if not gripper_name.strip():
            raise ValueError("gripper_name must not be empty")

        if gripper_name in resolved_names:
            raise ValueError(
                "gripper_name must not duplicate an arm joint name"
            )

        if not 1.0 <= joint_velocity_pct <= 100.0:
            raise ValueError("joint_velocity_pct must be in [1, 100]")

        if not 0.5 <= gripper_effort <= 3.0:
            raise ValueError("gripper_effort must be in [0.5, 3.0]")

        self._skill = skill
        self._provider_id = provider_id.strip()

        self._joint_names = resolved_names
        self._gripper_name = gripper_name.strip()

       
        self._piper_command_metadata = piper_command_metadata
        self._joint_velocity_pct = float(joint_velocity_pct)
        self._gripper_effort = float(gripper_effort)

        self._state_channel = None
        self._command_channel = None

        self._state_subscription = None
        self._command_publisher = None

        self._state_condition = threading.Condition()
        self._latest_state: ArmState | None = None
        self._first_state_event = threading.Event()

        self._last_state_error_log = 0.0

    @property
    def connected(self) -> bool:
        return (
            self._state_channel is not None
            and self._command_channel is not None
            and self._state_subscription is not None
            and self._command_publisher is not None
        )

    def connect(self, *, wait_timeout_s: float = 5.0) -> None:
        """连接同一个 Arm Primitive 的反馈和命令能力。"""

        if wait_timeout_s <= 0:
            raise ValueError("wait_timeout_s must be > 0")

        self.close()

        state_capability = ATLAS.find_unique_capability(
            contract_id=JOINT_STATES_CONTRACT,
            transport=Transport.ROS2,
            provider_kind=Kind.PRIMITIVE,
            provider_id=self._provider_id,
        )

        command_capability = ATLAS.find_unique_capability(
            contract_id=JOINT_COMMAND_CONTRACT,
            transport=Transport.ROS2,
            provider_kind=Kind.PRIMITIVE,
            provider_id=self._provider_id,
        )

        state_channel = self._skill.connect_capability(
            state_capability,
            JOINT_STATES_CONTRACT,
            Transport.ROS2,
        )

        try:
            command_channel = self._skill.connect_capability(
                command_capability,
                JOINT_COMMAND_CONTRACT,
                Transport.ROS2,
            )
        except Exception:
            state_channel.close()
            raise

        state_qos = _channel_qos(
            state_channel,
            fallback="reliable",
        )
        command_qos = _channel_qos(
            command_channel,
            fallback="reliable",
        )

        try:
            state_subscription = self._skill.create_subscription(
                contract_id=JOINT_STATES_CONTRACT,
                topic=state_channel.endpoint,
                msg_type=JointState,
                callback=self._on_joint_state,
                qos=state_qos,
                declare=False,
            )

            
            command_publisher = self._skill.create_publisher(
                contract_id=JOINT_COMMAND_CONTRACT,
                topic=command_channel.endpoint,
                msg_type=JointState,
                qos=command_qos,
                declare=False,
            )
        except Exception:
            state_channel.close()
            command_channel.close()
            raise

        self._state_channel = state_channel
        self._command_channel = command_channel
        self._state_subscription = state_subscription
        self._command_publisher = command_publisher

        log.info(
            "connected arm provider=%s states=%s command=%s",
            self._provider_id,
            state_channel.endpoint,
            command_channel.endpoint,
        )

        if not self._first_state_event.wait(wait_timeout_s):
            state_endpoint = state_channel.endpoint
            self.close()
            raise ArmNotReadyError(
                f"no JointState received from {self._provider_id!r} "
                f"on {state_endpoint!r} within {wait_timeout_s:.1f}s"
            )

    def _on_joint_state(self, msg: JointState) -> None:
        """把任意顺序的 JointState 整理成固定模型顺序。"""

        try:
            names = [str(name) for name in msg.name]
            positions = [float(value) for value in msg.position]

            if not names:
                raise ValueError("JointState.name is empty")

            if len(names) != len(positions):
                raise ValueError(
                    f"JointState name/position size mismatch: "
                    f"{len(names)} != {len(positions)}"
                )

            if len(set(names)) != len(names):
                raise ValueError("JointState contains duplicate names")

            index_by_name = {
                name: index
                for index, name in enumerate(names)
            }

            required = (*self._joint_names, self._gripper_name)
            missing = [
                name
                for name in required
                if name not in index_by_name
            ]

            if missing:
                raise ValueError(
                    f"JointState missing required joints: {missing}; "
                    f"received names={names}"
                )

            joints = tuple(
                positions[index_by_name[name]]
                for name in self._joint_names
            )
            gripper = positions[index_by_name[self._gripper_name]]

            all_positions = (*joints, gripper)
            if not all(math.isfinite(value) for value in all_positions):
                raise ValueError("JointState contains NaN or Inf")

            velocities: tuple[float, ...] | None = None
            gripper_velocity: float | None = None

            if len(msg.velocity) == len(names):
                velocities = tuple(
                    float(msg.velocity[index_by_name[name]])
                    for name in self._joint_names
                )
                gripper_velocity = float(
                    msg.velocity[index_by_name[self._gripper_name]]
                )

            source_stamp_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )

            state = ArmState(
                joints_rad=joints,
                gripper_m=gripper,
                joint_velocities_rad_s=velocities,
                gripper_velocity_m_s=gripper_velocity,
                received_monotonic=time.monotonic(),
                source_stamp_ns=source_stamp_ns,
            )

            with self._state_condition:
                self._latest_state = state
                self._state_condition.notify_all()

            self._first_state_event.set()

        except Exception as exc:
            now = time.monotonic()

            if now - self._last_state_error_log >= 1.0:
                log.warning("invalid arm JointState: %s", exc)
                self._last_state_error_log = now

    def latest_state(self, *, max_age_s: float) -> ArmState:
        """取得最新机械臂反馈，并拒绝过期状态。"""

        if max_age_s <= 0:
            raise ValueError("max_age_s must be > 0")

        with self._state_condition:
            state = self._latest_state

        if state is None:
            raise ArmNotReadyError(
                "arm is connected but no valid JointState is available"
            )

        age_s = state.age_s
        if age_s > max_age_s:
            raise StaleArmStateError(
                f"latest arm state is stale: "
                f"age={age_s:.3f}s, limit={max_age_s:.3f}s"
            )

        
        return state

    def command_joint_target(
        self,
        *,
        joints_rad: Sequence[float],
        gripper_m: float,
    ) -> None:
        """发布绝对关节位置目标。

        输入必须已经过 SafetyFilter：
        - joints_rad：旋转关节弧度；
        - gripper_m：夹爪开口，米。
        """

        publisher = self._command_publisher

        if publisher is None:
            raise ArmNotReadyError("arm command publisher is not connected")

        joints = tuple(float(value) for value in joints_rad)
        gripper = float(gripper_m)

        if len(joints) != len(self._joint_names):
            raise ValueError(
                f"expected {len(self._joint_names)} arm joints, "
                f"got {len(joints)}"
            )

        if not all(math.isfinite(value) for value in (*joints, gripper)):
            raise ValueError("joint command contains NaN or Inf")

        msg = JointState()

        node = RosBackend.get().node
        msg.header.stamp = node.get_clock().now().to_msg()

        msg.name = [
            *self._joint_names,
            self._gripper_name,
        ]

        msg.position = [
            *joints,
            gripper,
        ]

        if self._piper_command_metadata:
            # Piper 当前驱动从 velocity[6] 读取所有关节速度百分比，
            # 从 effort[6] 读取夹爪力度。
            msg.velocity = [
                *([0.0] * len(self._joint_names)),
                self._joint_velocity_pct,
            ]

            msg.effort = [
                *([0.0] * len(self._joint_names)),
                self._gripper_effort,
            ]

        publisher.publish(msg)

    def hold_current_position(
        self,
        *,
        max_state_age_s: float,
    ) -> None:
       

        state = self.latest_state(max_age_s=max_state_age_s)

        self.command_joint_target(
            joints_rad=state.joints_rad,
            gripper_m=state.gripper_m,
        )

    def wait_until_reached(
        self,
        *,
        target_joints_rad: Sequence[float],
        target_gripper_m: float | None = None,
        joint_tolerance_rad: float = 0.03,
        gripper_tolerance_m: float = 0.005,
        timeout_s: float = 5.0,
        max_state_age_s: float = 0.5,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """基于真实反馈等待目标到位，而不是固定 sleep。"""

        target = np.asarray(
            target_joints_rad,
            dtype=np.float64,
        )

        if target.shape != (len(self._joint_names),):
            raise ValueError(
                f"target joint shape must be "
                f"({len(self._joint_names)},), got {target.shape}"
            )

        if not np.all(np.isfinite(target)):
            raise ValueError("target joints contain NaN or Inf")

        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

        if joint_tolerance_rad <= 0:
            raise ValueError("joint_tolerance_rad must be > 0")

        deadline = time.monotonic() + timeout_s

        with self._state_condition:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return False

                state = self._latest_state

                if (
                    state is not None
                    and state.age_s <= max_state_age_s
                ):
                    current = np.asarray(
                        state.joints_rad,
                        dtype=np.float64,
                    )

                    joint_ok = bool(
                        np.max(np.abs(current - target))
                        <= joint_tolerance_rad
                    )

                    gripper_ok = True
                    if target_gripper_m is not None:
                        gripper_ok = (
                            abs(
                                state.gripper_m
                                - float(target_gripper_m)
                            )
                            <= gripper_tolerance_m
                        )

                    if joint_ok and gripper_ok:
                        return True

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False

                self._state_condition.wait(
                    timeout=min(remaining, 0.05)
                )

    def close(self) -> None:
        """释放 ROS 实体和 Atlas Channels。"""

        subscription = self._state_subscription
        publisher = self._command_publisher

        state_channel = self._state_channel
        command_channel = self._command_channel

        self._state_subscription = None
        self._command_publisher = None

        self._state_channel = None
        self._command_channel = None

        try:
            node = RosBackend.get().node

            if subscription is not None:
                node.destroy_subscription(subscription)

            if publisher is not None:
                node.destroy_publisher(publisher)

        except Exception as exc:
            log.debug("destroy arm ROS entities failed: %s", exc)

        if state_channel is not None:
            state_channel.close()

        if command_channel is not None:
            command_channel.close()

        with self._state_condition:
            self._latest_state = None
            self._state_condition.notify_all()

        self._first_state_event.clear()

