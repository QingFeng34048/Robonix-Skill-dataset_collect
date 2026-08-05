from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .episode_writer import save_episode


@dataclass(frozen=True)
class CollectionStatus:
    state: str
    run_id: str
    task_id: str
    frames: int
    output_uri: str
    detail: str


class CollectionManager:
    def __init__(
        self,
        *,
        camera,
        arm,
        output_root: Path,
        dataset_name: str,
        fps: int,
        model_resolution: tuple[int, int],
        gripper_open_threshold_m: float,
        max_image_age_s: float,
        max_state_age_s: float,
    ) -> None:
        self.camera = camera
        self.arm = arm
        self.output_root = output_root
        self.dataset_name = dataset_name
        self.fps = fps
        self.model_resolution = model_resolution
        self.gripper_open_threshold_m = gripper_open_threshold_m
        self.max_image_age_s = max_image_age_s
        self.max_state_age_s = max_state_age_s

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict] = []

        self._status = CollectionStatus(
            state="IDLE",
            run_id="",
            task_id="",
            frames=0,
            output_uri="",
            detail="ready",
        )
        self._instruction = ""

    def start_episode(
        self,
        *,
        task_id: str,
        instruction: str,
    ) -> str:
        with self._lock:
            if self._status.state == "RECORDING":
                raise RuntimeError("another episode is already recording")

            run_id = uuid.uuid4().hex
            self._samples = []
            self._instruction = instruction
            self._stop_event.clear()

            self._status = CollectionStatus(
                state="RECORDING",
                run_id=run_id,
                task_id=task_id,
                frames=0,
                output_uri="",
                detail="recording",
            )

            self._thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
            )
            self._thread.start()
            return run_id

    def _capture_loop(self) -> None:
        interval = 1.0 / self.fps

        try:
            while not self._stop_event.is_set():
                started = time.monotonic()

                rgb = self.camera.latest_rgb(
                    max_age_s=self.max_image_age_s
                )
                state = self.arm.latest_state(
                    max_age_s=self.max_state_age_s
                )

                resized = cv2.resize(
                    rgb,
                    self.model_resolution,
                )

                # 保持与你原数据一致：夹爪状态为 0/1。
                gripper_binary = float(
                    state.gripper_m >= self.gripper_open_threshold_m
                )

                vector = np.asarray(
                    [*state.joints_rad, gripper_binary],
                    dtype=np.float32,
                )

                with self._lock:
                    self._samples.append(
                        {
                            "image": resized,
                            "state": vector,
                            "timestamp": time.time(),
                        }
                    )
                    self._status = CollectionStatus(
                        **{
                            **self._status.__dict__,
                            "frames": len(self._samples),
                        }
                    )

                elapsed = time.monotonic() - started
                self._stop_event.wait(max(0.0, interval - elapsed))

        except Exception as exc:
            with self._lock:
                self._status = CollectionStatus(
                    **{
                        **self._status.__dict__,
                        "state": "FAILED",
                        "detail": str(exc),
                    }
                )

    def finish_episode(
        self,
        *,
        run_id: str,
        success: bool,
    ) -> Path:
        self._require_run(run_id)
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)

        with self._lock:
            samples = list(self._samples)
            status = self._status

        output = save_episode(
            output_root=self.output_root,
            task_id=status.task_id,
            instruction=self._instruction,
            dataset_name=self.dataset_name,
            fps=self.fps,
            samples=samples,
            success=success,
        )

        with self._lock:
            self._status = CollectionStatus(
                state="SUCCEEDED",
                run_id=run_id,
                task_id=status.task_id,
                frames=len(samples),
                output_uri=output.as_uri(),
                detail="episode saved",
            )

        return output

    def cancel(self, run_id: str) -> None:
        self._require_run(run_id)
        self._stop_event.set()

        with self._lock:
            self._samples = []
            self._status = CollectionStatus(
                **{
                    **self._status.__dict__,
                    "state": "CANCELED",
                    "detail": "episode discarded",
                }
            )

    def status(self, run_id: str) -> CollectionStatus:
        self._require_run(run_id)
        with self._lock:
            return self._status

    def _require_run(self, run_id: str) -> None:
        with self._lock:
            if self._status.run_id != run_id:
                raise RuntimeError(f"unknown run_id: {run_id}")

