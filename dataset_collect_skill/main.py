from robonix_api import Deferred, Err, Ok, Skill

from dataset_collect_mcp import (
    StartEpisode_Request,
    StartEpisode_Response,
    FinishEpisode_Request,
    FinishEpisode_Response,
)

skill = Skill(
    id="dataset_collect",
    namespace="robonix/skill/dataset_collect",
)

_manager = None


@skill.on_init
def on_init(config: dict):
    global _manager
    # 这里只解析配置，不连接硬件。
    _manager = create_manager_config_only(config)
    return Ok()


@skill.on_activate
def on_activate():
    try:
        _manager.connect_dependencies(skill)
        return Ok()
    except Exception as exc:
        return Deferred(str(exc))


@skill.on_deactivate
def on_deactivate():
    _manager.disconnect_dependencies()
    return Ok()


@skill.mcp("robonix/skill/dataset_collect/episode/start")
def start_episode(
    request: StartEpisode_Request,
) -> StartEpisode_Response:
    run_id = _manager.start_episode(
        task_id=request.task_id,
        instruction=request.instruction,
    )
    return StartEpisode_Response(
        accepted=True,
        run_id=run_id,
        detail="recording started",
    )


@skill.mcp("robonix/skill/dataset_collect/episode/finish")
def finish_episode(
    request: FinishEpisode_Request,
) -> FinishEpisode_Response:
    output = _manager.finish_episode(
        run_id=request.run_id,
        success=request.success,
    )
    return FinishEpisode_Response(
        accepted=True,
        output_uri=output.as_uri(),
        detail="saved",
    )

