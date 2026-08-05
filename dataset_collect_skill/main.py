from __future__ import annotations

import logging
import os
from typing import Any

from robonix_api import Deferred, Err, Ok, Skill

from dataset_collect_mcp import (
    CancelCollection_Request,
    CancelCollection_Response,
    FinishEpisode_Request,
    FinishEpisode_Response,
    GetStatus_Request,
    GetStatus_Response,
    StartEpisode_Request,
    StartEpisode_Response,
)

from .collection_manager import (
    CollectionManager,
    DependencyUnavailable,
)
from .runtime_config import RuntimeConfig


logging.basicConfig(
    level=os.environ.get(
        "LOG_LEVEL",
        "INFO",
    ).upper(),
    format=(
        "%(asctime)s %(levelname)s "
        "%(name)s: %(message)s"
    ),
)

log = logging.getLogger(__name__)


skill = Skill(
    id=os.environ.get(
        "RBNX_INSTANCE_NAME",
        "dataset_collect",
    ),
    namespace="robonix/skill/dataset_collect",
)


_settings: RuntimeConfig | None = None
_manager: CollectionManager | None = None


def _require_manager() -> CollectionManager:
    if _manager is None:
        raise RuntimeError(
            "skill is not initialized"
        )

    return _manager


@skill.on_init
def on_init(
    config: dict[str, Any],
):
    global _settings, _manager

    try:
        if _manager is not None:
            _manager.close()

        settings = RuntimeConfig.from_dict(config)

        _settings = settings
        _manager = CollectionManager(
            skill=skill,
            settings=settings,
        )

        log.info(
            "dataset collection skill initialized: "
            "output_root=%s, dataset=%s",
            settings.output_root,
            settings.dataset_name,
        )

        return Ok()

    except Exception as exc:
        log.exception(
            "dataset collection initialization failed"
        )
        return Err(str(exc))


@skill.on_activate
def on_activate():
    try:
        _require_manager().connect_dependencies()

        log.info(
            "dataset collection dependencies connected"
        )
        return Ok()

    except DependencyUnavailable as exc:
        log.warning(
            "dataset collection dependencies "
            "are not ready: %s",
            exc,
        )
        return Deferred(str(exc))

    except Exception as exc:
        log.exception(
            "dataset collection activation failed"
        )
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
    "robonix/skill/dataset_collect/episode/start",
    description=(
        "Start recording a robot demonstration for "
        "the requested task. The robot must already "
        "be in teaching or drag mode."
    ),
)
def start_episode(
    request: StartEpisode_Request,
) -> StartEpisode_Response:
    result = _require_manager().start_episode(
        task_id=request.task_id,
        instruction=request.instruction,
        max_duration_s=float(
            request.max_duration_s
        ),
    )

    return StartEpisode_Response(
        accepted=result.accepted,
        run_id=result.run_id,
        detail=result.detail,
    )


@skill.mcp(
    "robonix/skill/dataset_collect/episode/finish",
    description=(
        "Stop the current recording, label it as "
        "success or failure and save the HDF5 episode."
    ),
)
def finish_episode(
    request: FinishEpisode_Request,
) -> FinishEpisode_Response:
    result = _require_manager().finish_episode(
        run_id=request.run_id,
        success=bool(request.success),
        note=request.note,
    )

    return FinishEpisode_Response(
        accepted=result.accepted,
        output_uri=result.output_uri,
        frames=int(result.frames),
        detail=result.detail,
    )


@skill.mcp(
    "robonix/skill/dataset_collect/status",
    description=(
        "Query frame count, progress and state of "
        "a dataset collection run."
    ),
)
def get_status(
    request: GetStatus_Request,
) -> GetStatus_Response:
    current = _require_manager().status(
        request.run_id
    )

    return GetStatus_Response(
        state=current.state,
        detail=current.detail,
        progress=float(current.progress),
        frames=int(current.frames),
        elapsed_s=float(current.elapsed_s),
        max_duration_s=float(
            current.max_duration_s
        ),
        output_uri=current.output_uri,
    )


@skill.mcp(
    "robonix/skill/dataset_collect/cancel",
    description=(
        "Cancel a dataset collection run and "
        "discard all unsaved samples."
    ),
)
def cancel_collection(
    request: CancelCollection_Request,
) -> CancelCollection_Response:
    accepted, detail = _require_manager().cancel(
        request.run_id
    )

    return CancelCollection_Response(
        accepted=accepted,
        detail=detail,
    )


if __name__ == "__main__":
    skill.run()
