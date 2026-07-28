from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from .runtime_config import RuntimeConfig
from .safety import SafetyFilter, SafetyLimits
from .task_registry import TaskRegistry, TaskSpec
from .units import normalized_gripper_to_metres, validate_action_vector
from .vla_client import VLAClient

log = logging.getLogger(__name__)

TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELED", "TIMEOUT"}


class DependencyUnavailable(RuntimeError):
    """A runtime dependency is absent or not ready yet."""


@dataclass(frozen=True)
class StartResult:
    accepted: bool
    run_id: str
    detail: str


@dataclass(frozen=True)
class TaskStatus:
    state: str
    detail: str
    progress: float
    current_step: int
    max_steps: int


@dataclass
class _RunRecord:
    task: TaskSpec
    instruction: str
    timeout_s: float
    cancel_event: threading.Event
    status: TaskStatus
    lock: threading.Lock


class TaskManager:
    def __init__(
        self,
        *,
        skill: Any,
        settings: RuntimeConfig,
        camera: Any | None = None,
        arm: Any | None = None,
        vla_client: VLAClient | Any | None = None,
    ) -> None:
        self._skill = skill
        self._settings = settings
        self._registry = TaskRegistry.load(settings.task_config_path)
        self._camera = camera
        self._arm = arm
        self._vla = vla_client or VLAClient(
            server_url=settings.vla_server_url,
            timeout_s=settings.request_timeout_s,
        )
        self._safety = (
            SafetyFilter(
                SafetyLimits(
                    joint_min_rad=settings.joint_min_rad,
                    joint_max_rad=settings.joint_max_rad,
                    default_max_delta_rad=settings.default_max_delta_rad,
                    gripper_min_m=settings.gripper_min_m,
                    gripper_max_m=settings.gripper_max_m,
                    model_gripper_mode=settings.model_gripper_mode,
                )
            )
            if settings.enable_safety_filter
            else None
        )
        self._runs: dict[str, _RunRecord] = {}
        self._runs_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_runs,
            thread_name_prefix="openvla-oft",
        )
        self._connected = False
        self._closed = False

    def connect_dependencies(self) -> None:
        if self._closed:
            raise RuntimeError("task manager is closed")
        if self._connected:
            return
        try:
            if self._camera is None:
                if self._skill is None:
                    raise RuntimeError(
                        "skill is required to create CameraAdapter"
                    )
                from .camera_adapter import CameraAdapter

                self._camera = CameraAdapter(
                    skill=self._skill,
                    provider_id=self._settings.camera_provider_id,
                )
            if self._arm is None:
                if self._skill is None:
                    raise RuntimeError(
                        "skill is required to create ArmAdapter"
                    )
                from .arm_adapter import ArmAdapter

                self._arm = ArmAdapter(
                    skill=self._skill,
                    provider_id=self._settings.arm_provider_id,
                    joint_names=self._settings.joint_names,
                    gripper_name=self._settings.gripper_name,
                    piper_command_metadata=(
                        self._settings.piper_command_metadata
                    ),
                    joint_velocity_pct=self._settings.joint_velocity_pct,
                    gripper_effort=self._settings.gripper_effort,
                )

            self._camera.connect(
                wait_timeout_s=self._settings.dependency_wait_timeout_s
            )
            try:
                self._arm.connect(
                    wait_timeout_s=self._settings.dependency_wait_timeout_s
                )
            except Exception:
                self._camera.close()
                raise
            if self._settings.require_vla_healthcheck:
                self._vla.healthcheck()
            self._connected = True
        except Exception as exc:
            self._safe_close_adapter(self._camera)
            self._safe_close_adapter(self._arm)
            raise DependencyUnavailable(str(exc)) from exc

    def start(
        self,
        *,
        task_id: str,
        instruction: str,
        timeout_s: float,
    ) -> StartResult:
        if self._closed:
            raise RuntimeError("task manager is closed")
        if not self._connected:
            raise RuntimeError("skill is inactive; dependencies are not ready")

        task = self._registry.get(task_id)
        resolved_instruction = str(instruction).strip() or task.instruction
        resolved_timeout = (
            float(timeout_s)
            if float(timeout_s) > 0
            else float(self._settings.task_timeout_s)
        )
        if resolved_timeout <= 0:
            raise RuntimeError("timeout_s must be > 0")

        with self._runs_lock:
            active = sum(
                1
                for record in self._runs.values()
                if self._read_status(record).state not in TERMINAL_STATES
            )
            if active >= self._settings.max_concurrent_runs:
                return StartResult(
                    accepted=False,
                    run_id="",
                    detail=(
                        "maximum concurrent runs reached: "
                        f"{self._settings.max_concurrent_runs}"
                    ),
                )

            run_id = uuid.uuid4().hex
            record = _RunRecord(
                task=task,
                instruction=resolved_instruction,
                timeout_s=resolved_timeout,
                cancel_event=threading.Event(),
                status=TaskStatus(
                    state="PENDING",
                    detail="queued",
                    progress=0.0,
                    current_step=0,
                    max_steps=task.max_steps,
                ),
                lock=threading.Lock(),
            )
            self._runs[run_id] = record
            self._executor.submit(self._run_task, run_id, record)

        return StartResult(
            accepted=True,
            run_id=run_id,
            detail="accepted",
        )

    def status(self, run_id: str) -> TaskStatus:
        record = self._get_run(run_id)
        return self._read_status(record)

    def cancel(self, run_id: str) -> tuple[bool, str]:
        record = self._get_run(run_id)
        status = self._read_status(record)
        if status.state in TERMINAL_STATES:
            return False, f"run already finished with {status.state}"
        record.cancel_event.set()
        self._update_status(record, detail="cancellation requested")
        return True, "cancellation requested"

    def cancel_all(self) -> None:
        with self._runs_lock:
            records = list(self._runs.values())
        for record in records:
            if self._read_status(record).state not in TERMINAL_STATES:
                record.cancel_event.set()

    def _run_task(self, run_id: str, record: _RunRecord) -> None:
        started = time.monotonic()
        self._update_status(record, state="RUNNING", detail="running")
        try:
            while True:
                if record.cancel_event.is_set():
                    self._finish_canceled(record)
                    return
                if time.monotonic() - started >= record.timeout_s:
                    self._finish(
                        record,
                        state="TIMEOUT",
                        detail=f"task timed out after {record.timeout_s:.1f}s",
                    )
                    return

                status = self._read_status(record)
                if status.current_step >= record.task.max_steps:
                    self._finish(
                        record,
                        state="FAILED",
                        detail="maximum control steps reached",
                    )
                    return

                frame = self._camera.latest_jpeg(
                    max_age_s=self._settings.max_image_age_s,
                    quality=self._settings.jpeg_quality,
                )
                state = self._arm.latest_state(
                    max_age_s=self._settings.max_state_age_s
                )
                result = self._vla.predict(
                    task_id=record.task.task_id,
                    instruction=record.instruction,
                    image_jpeg=frame,
                    state=state.model_vector(),
                )

                current_joints = tuple(state.joints_rad)
                actions = result.actions[
                    : record.task.execute_chunk_steps
                ]
                for action in actions:
                    if record.cancel_event.is_set():
                        self._finish_canceled(record)
                        return
                    if time.monotonic() - started >= record.timeout_s:
                        self._finish(
                            record,
                            state="TIMEOUT",
                            detail=(
                                f"task timed out after "
                                f"{record.timeout_s:.1f}s"
                            ),
                        )
                        return

                    target_joints, target_gripper = self._convert_action(
                        current_joints=current_joints,
                        action=action,
                        task_max_delta=record.task.max_delta_rad,
                    )
                    self._arm.command_joint_target(
                        joints_rad=target_joints,
                        gripper_m=target_gripper,
                    )
                    current_joints = target_joints

                    step = self._read_status(record).current_step + 1
                    progress = min(
                        1.0, step / float(record.task.max_steps)
                    )
                    self._update_status(
                        record,
                        current_step=step,
                        progress=progress,
                        detail=result.detail or "running",
                    )
                    time.sleep(1.0 / self._settings.control_hz)

                if result.done:
                    if result.success:
                        self._finish(
                            record,
                            state="SUCCEEDED",
                            detail=result.detail or "task completed",
                        )
                    else:
                        self._finish(
                            record,
                            state="FAILED",
                            detail=result.detail or "model reported failure",
                        )
                    return
        except Exception as exc:
            log.exception("run %s failed", run_id)
            self._finish(
                record,
                state="FAILED",
                detail=f"{type(exc).__name__}: {exc}",
            )

    def _convert_action(
        self,
        *,
        current_joints: tuple[float, ...],
        action: tuple[float, ...],
        task_max_delta: float,
    ) -> tuple[tuple[float, ...], float]:
        if self._safety is not None:
            return self._safety.apply(
                current_joints=current_joints,
                action=action,
                task_max_delta=task_max_delta,
            )

        # Simulation-only escape hatch. Shape/finiteness and gripper units are
        # still validated, but joint deltas and absolute limits are not clipped.
        raw = validate_action_vector(action)
        target = tuple(
            float(current + delta)
            for current, delta in zip(
                current_joints, raw[:6], strict=True
            )
        )
        if self._settings.model_gripper_mode == "normalized_0_1":
            gripper = normalized_gripper_to_metres(
                float(raw[6]),
                minimum_m=self._settings.gripper_min_m,
                maximum_m=self._settings.gripper_max_m,
            )
        else:
            gripper = min(
                max(float(raw[6]), self._settings.gripper_min_m),
                self._settings.gripper_max_m,
            )
        return target, gripper

    def _finish_canceled(self, record: _RunRecord) -> None:
        detail = "canceled"
        if self._settings.hold_position_on_cancel:
            try:
                self._arm.hold_current_position(
                    max_state_age_s=self._settings.max_state_age_s
                )
                detail = "canceled; current position held"
            except Exception as exc:
                detail = f"canceled; hold command failed: {exc}"
        self._finish(record, state="CANCELED", detail=detail)

    def _finish(
        self,
        record: _RunRecord,
        *,
        state: str,
        detail: str,
    ) -> None:
        status = self._read_status(record)
        progress = 1.0 if state == "SUCCEEDED" else status.progress
        self._update_status(
            record,
            state=state,
            detail=detail,
            progress=progress,
        )

    def _get_run(self, run_id: str) -> _RunRecord:
        key = str(run_id).strip()
        if not key:
            raise RuntimeError("run_id must not be empty")
        with self._runs_lock:
            record = self._runs.get(key)
        if record is None:
            raise RuntimeError(f"unknown run_id: {key}")
        return record

    @staticmethod
    def _read_status(record: _RunRecord) -> TaskStatus:
        with record.lock:
            return record.status

    @staticmethod
    def _update_status(
        record: _RunRecord,
        **changes: Any,
    ) -> None:
        with record.lock:
            record.status = replace(record.status, **changes)

    @staticmethod
    def _safe_close_adapter(adapter: Any | None) -> None:
        if adapter is None:
            return
        try:
            adapter.close()
        except Exception:
            log.debug("adapter close failed", exc_info=True)

    def wait_for_idle(self, *, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            with self._runs_lock:
                records = list(self._runs.values())
            if all(
                self._read_status(record).state in TERMINAL_STATES
                for record in records
            ):
                return True
            time.sleep(0.02)
        return False

    def disconnect_dependencies(self) -> None:
        """Release hot robot resources while keeping the skill reactivatable."""
        if not self._connected:
            return
        self.cancel_all()
        self.wait_for_idle(timeout_s=2.0)
        self._safe_close_adapter(self._camera)
        self._safe_close_adapter(self._arm)
        self._connected = False

    def close(self) -> None:
        """Final shutdown. Unlike deactivate, this object cannot be reused."""
        if self._closed:
            return
        self.disconnect_dependencies()
        self._executor.shutdown(wait=False, cancel_futures=True)
        try:
            self._vla.close()
        except Exception:
            log.debug("VLA client close failed", exc_info=True)
        self._closed = True
