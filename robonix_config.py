from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class TaskConfig:
    task_id: str
    instruction: str
    init_pose: list[float]
    max_steps: int = 120
    max_delta: float = 0.05
    execute_chunk_steps: int = 2
    enabled: bool = True


@dataclass
class RobotConfig:
    can_port: str = "can0"
    rad_to_sdk_int: float = 57295.7795
    piper_raw_to_degree: float = 0.001
    gripper_open_sdk: int = 80000
    gripper_close_sdk: int = 0
    gripper_threshold_raw: int = 3000
    gripper_speed: int = 1000
    move_speed: int = 80
    teaching_speed: int = 30


@dataclass
class CameraConfig:
    camera_id: int = 1
    capture_res: list[int] = field(default_factory=lambda: [640, 480])
    model_res: list[int] = field(default_factory=lambda: [224, 224])
    flip_code: int = -1


@dataclass
class DatasetConfig:
    name: str = "piper_multitask"
    hdf5_root: str = "outputs/dataset_hdf5"
    rlds_root: str = "outputs/data_rlds"
    version: str = "1.0.0"
    exclude_fail: bool = True
    min_action_norm: float = 0.0
    validation_ratio: float = 0.1


@dataclass
class CollectConfig:
    fps: int = 10
    default_task_id: str = "pick_banana"
    return_to_init_after_episode: bool = True


@dataclass
class ConvertConfig:
    merge_tasks: bool = True
    preserve_gripper_changes: bool = True
    random_seed: int = 42


@dataclass
class TrainConfig:
    vla_path: str = "openvla/openvla-7b"
    run_root_dir: str = "outputs/checkpoints/piper_multitask"
    shuffle_buffer_size: int = 100_000

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = True

    batch_size: int = 4
    learning_rate: float = 5e-5
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 10_000
    grad_accumulation_steps: int = 1
    max_steps: int = 30_000
    use_val_set: bool = True
    val_freq: int = 2_000
    val_time_limit: int = 180
    save_freq: int = 1_000
    save_latest_checkpoint_only: bool = True
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    diffusion_sample_freq: int = 50

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = False

    wandb_entity: str = ""
    wandb_project: str = "openvla-oft"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10


@dataclass
class ServerConfig:
    checkpoint_path: str = "outputs/checkpoints/piper_multitask/YOUR_RUN_DIR"
    host: str = "0.0.0.0"
    port: int = 8001
    save_images: bool = False
    save_dir: str = "outputs/inference_images"
    center_crop: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    return_first_action_for_legacy: bool = True
    num_diffusion_steps_inference: int = 50


@dataclass
class ClientConfig:
    server_url: str = "http://localhost:8001/act"
    default_task_id: str = "pick_banana"
    control_freq: int = 10
    request_timeout: float = 10.0
    use_action_chunk: bool = True


@dataclass
class ExperimentConfig:
    project_name: str = "piper_multitask"
    tasks: list[TaskConfig] = field(default_factory=list)
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    convert: ConvertConfig = field(default_factory=ConvertConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)


def expand_value(value: str | Path) -> str:
    return os.path.expandvars(os.path.expanduser(str(value)))


def resolve_path(value: str | Path) -> Path:
    path = Path(expand_value(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _section(data: dict, key: str) -> dict:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be a mapping.")
    return value


def load_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    tasks_raw = raw.get("tasks", []) or []
    tasks = [TaskConfig(**item) for item in tasks_raw]

    cfg = ExperimentConfig(
        project_name=raw.get("project_name", "piper_multitask"),
        tasks=tasks,
        robot=RobotConfig(**_section(raw, "robot")),
        camera=CameraConfig(**_section(raw, "camera")),
        dataset=DatasetConfig(**_section(raw, "dataset")),
        collect=CollectConfig(**_section(raw, "collect")),
        convert=ConvertConfig(**_section(raw, "convert")),
        train=TrainConfig(**_section(raw, "train")),
        server=ServerConfig(**_section(raw, "server")),
        client=ClientConfig(**_section(raw, "client")),
    )
    validate_config(cfg)
    return cfg


def get_task_map(cfg: ExperimentConfig) -> dict[str, TaskConfig]:
    return {task.task_id: task for task in cfg.tasks if task.enabled}


def validate_config(cfg: ExperimentConfig) -> None:
    if not cfg.tasks:
        raise ValueError("No tasks configured.")

    task_ids = [task.task_id for task in cfg.tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task_id found in config.")

    task_map = get_task_map(cfg)
    if not task_map:
        raise ValueError("No enabled tasks configured.")

    for default_id, field_name in (
        (cfg.collect.default_task_id, "collect.default_task_id"),
        (cfg.client.default_task_id, "client.default_task_id"),
    ):
        if default_id not in task_map:
            raise ValueError(f"{field_name}={default_id!r} is not an enabled task.")

    for task in task_map.values():
        if len(task.init_pose) != 7:
            raise ValueError(f"Task {task.task_id!r}: init_pose must contain 7 values.")
        if not task.instruction.strip():
            raise ValueError(f"Task {task.task_id!r}: instruction cannot be empty.")
        if task.execute_chunk_steps < 1:
            raise ValueError(f"Task {task.task_id!r}: execute_chunk_steps must be >= 1.")
        if task.max_steps < 1:
            raise ValueError(f"Task {task.task_id!r}: max_steps must be >= 1.")
        if task.max_delta <= 0:
            raise ValueError(f"Task {task.task_id!r}: max_delta must be > 0.")

    if len(cfg.camera.capture_res) != 2 or len(cfg.camera.model_res) != 2:
        raise ValueError("camera.capture_res and camera.model_res must be [width, height].")

    if cfg.train.use_l1_regression and cfg.train.use_diffusion:
        raise ValueError("use_l1_regression and use_diffusion cannot both be true.")
    if not cfg.train.use_l1_regression and not cfg.train.use_diffusion:
        raise ValueError("Enable either use_l1_regression or use_diffusion.")

    if not 0.0 <= cfg.dataset.validation_ratio < 1.0:
        raise ValueError("dataset.validation_ratio must be in [0, 1).")
