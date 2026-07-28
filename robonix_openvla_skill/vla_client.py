from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

import requests


@dataclass(frozen=True)
class VLAResult:
    actions: tuple[tuple[float, ...], ...]
    done: bool = False
    success: bool = True
    detail: str = ""


class VLAClient:
    def __init__(
        self,
        *,
        server_url: str,
        timeout_s: float,
        session: requests.Session | Any | None = None,
    ) -> None:
        parsed = urlsplit(server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("server_url must be an absolute http(s) URL")
        if timeout_s <= 0 or not math.isfinite(float(timeout_s)):
            raise ValueError("timeout_s must be finite and > 0")
        self._server_url = server_url
        self._timeout_s = float(timeout_s)
        self._session = session or requests.Session()
        self._owns_session = session is None

    def _health_url(self) -> str:
        parsed = urlsplit(self._server_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/act"):
            path = path[:-4]
        return urlunsplit(
            (parsed.scheme, parsed.netloc, f"{path}/health", "", "")
        )

    def healthcheck(self) -> None:
        try:
            response = self._session.get(
                self._health_url(),
                timeout=self._timeout_s,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"VLA healthcheck failed: {exc}") from exc

    def predict(
        self,
        *,
        task_id: str,
        instruction: str,
        image_jpeg: bytes,
        state: Sequence[float],
    ) -> VLAResult:
        if not str(task_id).strip():
            raise ValueError("task_id must not be empty")
        if not str(instruction).strip():
            raise ValueError("instruction must not be empty")
        if not image_jpeg:
            raise ValueError("image_jpeg must not be empty")

        state_values = [float(value) for value in state]
        if len(state_values) != 7:
            raise ValueError("state must contain [j1..j6, gripper]")
        if not all(math.isfinite(value) for value in state_values):
            raise ValueError("state contains NaN or Inf")

        payload = {
            "task_id": str(task_id).strip(),
            "instruction": str(instruction).strip(),
            "state": state_values,
        }
        files = {
            "image": ("frame.jpg", image_jpeg, "image/jpeg"),
        }
        try:
            response = self._session.post(
                self._server_url,
                data={"payload": json.dumps(payload)},
                files=files,
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"VLA inference request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("VLA server returned invalid JSON") from exc

        if not isinstance(body, dict):
            raise RuntimeError("VLA response must be a JSON object")
        raw_actions = body.get("action_chunk")
        if raw_actions is None and "action" in body:
            raw_actions = [body["action"]]
        if not isinstance(raw_actions, list) or not raw_actions:
            raise RuntimeError(
                "VLA response must contain action or non-empty action_chunk"
            )
        if len(raw_actions) > 128:
            raise RuntimeError("VLA action_chunk is unexpectedly large")

        actions: list[tuple[float, ...]] = []
        for index, raw in enumerate(raw_actions):
            if not isinstance(raw, (list, tuple)) or len(raw) != 7:
                raise RuntimeError(
                    f"VLA action {index} must contain seven numbers"
                )
            action = tuple(float(value) for value in raw)
            if not all(math.isfinite(value) for value in action):
                raise RuntimeError(
                    f"VLA action {index} contains NaN or Inf"
                )
            actions.append(action)

        return VLAResult(
            actions=tuple(actions),
            done=bool(body.get("done", False)),
            success=bool(body.get("success", True)),
            detail=str(body.get("detail", "")),
        )

    def close(self) -> None:
        if self._owns_session:
            self._session.close()
