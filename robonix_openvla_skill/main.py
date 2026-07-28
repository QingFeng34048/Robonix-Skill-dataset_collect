from __future__ import annotations

import logging
import os
from typing import Any

from robonix_api import Deferred, Err, Ok, Skill
from openvla_oft_mcp import (
    CancelTask_Request,
    CancelTask_Response,
    ExecuteTask_Request,
    ExecuteTask_Response,
    GetTaskStatus_Request,
    GetTaskStatus_Response,
)

from .runtime_config import RuntimeConfig
from .task_manager import DependencyUnavailable, TaskManager

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

skill = Skill(
    id=os.environ.get("RBNX_INSTANCE_NAME", "openvla_oft"),
    namespace="robonix/skill/openvla_oft",
)

_settings: RuntimeConfig | None = None
_manager: TaskManager | None = None


def _require_manager() -> TaskManager:
    if _manager is None:
        raise RuntimeError("skill is not initialized")
    return _manager


@skill.on_init
def on_init(config: dict[str, Any]):
    global _settings, _manager
    try:
        if _manager is not None:
            _manager.close()
        settings = RuntimeConfig.from_dict(config)
        settings.validate()
        _settings = settings
        _manager = TaskManager(skill=skill, settings=settings)
        return Ok()
    except Exception as exc:
        log.exception("skill initialization failed")
        return Err(str(exc))


@skill.on_activate
def on_activate():
    try:
        _require_manager().connect_dependencies()
        return Ok()
    except DependencyUnavailable as exc:
        log.warning("dependencies are not ready: %s", exc)
        return Deferred(str(exc))
    except Exception as exc:
        log.exception("skill activation failed")
        return Err(str(exc))


@skill.on_deactivate
def on_deactivate():
    if _manager is not None:
        _manager.disconnect_dependencies()
    return Ok()


@skill.on_shutdown
def on_shutdown():
    global _manager
    if _manager is not None:
        _manager.close()
        _manager = None
    return Ok()


@skill.mcp(
    "robonix/skill/openvla_oft/execute",
    description="Start an asynchronous OpenVLA-OFT manipulation task.",
)
def execute(request: ExecuteTask_Request) -> ExecuteTask_Response:
    result = _require_manager().start(
        task_id=request.task_id,
        instruction=request.instruction,
        timeout_s=float(request.timeout_s),
    )
    return ExecuteTask_Response(
        accepted=result.accepted,
        run_id=result.run_id,
        detail=result.detail,
    )


@skill.mcp("robonix/skill/openvla_oft/execute/status")
def status(request: GetTaskStatus_Request) -> GetTaskStatus_Response:
    current = _require_manager().status(request.run_id)
    return GetTaskStatus_Response(
        state=current.state,
        detail=current.detail,
        progress=float(current.progress),
        current_step=int(current.current_step),
        max_steps=int(current.max_steps),
    )


@skill.mcp("robonix/skill/openvla_oft/execute/cancel")
def cancel(request: CancelTask_Request) -> CancelTask_Response:
    accepted, detail = _require_manager().cancel(request.run_id)
    return CancelTask_Response(
        accepted=accepted,
        detail=detail,
    )


if __name__ == "__main__":
    skill.run()
