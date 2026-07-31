"""HDF5 -> RLDS/TFDS，多任务可合并为一个 dataset.name。"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robonix_config import ExperimentConfig, get_task_map, load_config, resolve_path


def _snake_to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part)


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_episode_metadata(path: Path) -> tuple[str, str]:
    with h5py.File(path, "r") as f:
        task_id = _decode_attr(f.attrs.get("task_id", path.parent.name))
        instruction = _decode_attr(f.attrs.get("language_instruction", ""))
    return task_id, instruction


def _load_steps(path: Path, min_action_norm: float, preserve_gripper_changes: bool) -> list[dict]:
    with h5py.File(path, "r") as f:
        actions = f["action"][:].astype(np.float32)
        images = f["observations"]["images"][:].astype(np.uint8)
        states = f["observations"]["state"][:].astype(np.float32)
        instruction = _decode_attr(f.attrs.get("language_instruction", ""))
        reward = float(f.attrs.get("reward", 0.0))

    if not (len(actions) == len(images) == len(states)):
        raise ValueError(f"Length mismatch in {path}")

    if min_action_norm > 0 and len(actions) > 1:
        joint_motion = np.linalg.norm(actions[:, :6], axis=1) >= min_action_norm
        keep = joint_motion.copy()

        if preserve_gripper_changes:
            changed = np.nonzero(states[1:, 6] != states[:-1, 6])[0]
            keep[changed] = True
            keep[np.minimum(changed + 1, len(keep) - 1)] = True

        keep[0] = True
        keep[-1] = True
        keep_indices = np.nonzero(keep)[0]
        if keep_indices.size < 2:
            return []

        states = states[keep_indices]
        images = images[keep_indices]
        actions = np.zeros_like(states, dtype=np.float32)
        actions[:-1, :6] = states[1:, :6] - states[:-1, :6]
        actions[:-1, 6] = states[1:, 6]
        actions[-1, :6] = 0.0
        actions[-1, 6] = states[-1, 6]

    steps = []
    for i in range(len(actions)):
        steps.append(
            {
                "observation": {"image": images[i], "state": states[i]},
                "action": actions[i],
                "reward": 0.0,
                "discount": 1.0,
                "is_first": i == 0,
                "is_last": i == len(actions) - 1,
                "is_terminal": i == len(actions) - 1,
                "language_instruction": instruction,
            }
        )

    if steps:
        steps[-1]["reward"] = reward
        steps[-1]["discount"] = 0.0
    return steps


def _make_builder(
    dataset_name: str,
    version: str,
    train_paths: list[Path],
    val_paths: list[Path],
    min_action_norm: float,
    preserve_gripper_changes: bool,
    image_shape: tuple[int, int, int],
):
    class_name = _snake_to_camel(dataset_name)
    step_features = tfds.features.FeaturesDict(
        {
            "observation": tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Tensor(shape=image_shape, dtype=tf.uint8),
                    "state": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                }
            ),
            "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            "reward": tfds.features.Scalar(dtype=tf.float32),
            "discount": tfds.features.Scalar(dtype=tf.float32),
            "is_first": tfds.features.Scalar(dtype=tf.bool),
            "is_last": tfds.features.Scalar(dtype=tf.bool),
            "is_terminal": tfds.features.Scalar(dtype=tf.bool),
            "language_instruction": tfds.features.Text(),
        }
    )
    features = tfds.features.FeaturesDict({"steps": tfds.features.Dataset(step_features)})

    def _info(self):
        return tfds.core.DatasetInfo(builder=self, features=features)

    def _split_generators(self, dl_manager):
        del dl_manager
        splits = [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"paths": train_paths})
        ]
        if val_paths:
            splits.append(
                tfds.core.SplitGenerator(name=tfds.Split.VALIDATION, gen_kwargs={"paths": val_paths})
            )
        return splits

    def _generate_examples(self, paths):
        for path in paths:
            steps = _load_steps(path, min_action_norm, preserve_gripper_changes)
            if not steps:
                continue
            task_id, _ = _read_episode_metadata(path)
            episode_key = f"{task_id}__{path.stem}"
            yield episode_key, {"steps": steps}

    return type(
        class_name,
        (tfds.core.GeneratorBasedBuilder,),
        {
            "VERSION": tfds.core.Version(version),
            "_info": _info,
            "_split_generators": _split_generators,
            "_generate_examples": _generate_examples,
        },
    )


def _collect_task_paths(cfg: ExperimentConfig) -> dict[str, list[Path]]:
    input_root = resolve_path(cfg.dataset.hdf5_root)
    task_map = get_task_map(cfg)
    task_to_paths: dict[str, list[Path]] = {}

    for task_id, task in task_map.items():
        task_dir = input_root / task_id
        if not task_dir.is_dir():
            print(f"Warning: task directory missing: {task_dir}")
            continue

        paths = sorted(task_dir.glob("*.hdf5"))
        if cfg.dataset.exclude_fail:
            paths = [p for p in paths if "FAIL" not in p.name]

        checked = []
        for path in paths:
            file_task_id, instruction = _read_episode_metadata(path)
            if file_task_id != task_id:
                raise ValueError(f"{path}: task_id={file_task_id!r}, expected {task_id!r}")
            if instruction != task.instruction:
                raise ValueError(
                    f"{path}: instruction mismatch; expected={task.instruction!r}, got={instruction!r}"
                )
            checked.append(path)

        print(f"{task_id}: {len(checked)} episodes")
        if checked:
            task_to_paths[task_id] = checked

    if not task_to_paths:
        raise SystemExit("No HDF5 episodes found for enabled tasks.")
    return task_to_paths


def _split_task_paths(
    task_to_paths: dict[str, list[Path]], validation_ratio: float, random_seed: int
) -> tuple[list[Path], list[Path]]:
    train_paths: list[Path] = []
    val_paths: list[Path] = []

    for task_id, paths in sorted(task_to_paths.items()):
        shuffled = list(paths)
        seed = random_seed + zlib.crc32(task_id.encode("utf-8"))
        rng = np.random.default_rng(seed)
        rng.shuffle(shuffled)

        n_val = 0
        if validation_ratio > 0 and len(shuffled) >= 2:
            n_val = max(1, int(round(len(shuffled) * validation_ratio)))
            n_val = min(n_val, len(shuffled) - 1)

        val_paths.extend(shuffled[:n_val])
        train_paths.extend(shuffled[n_val:])
        print(f"  split {task_id}: train={len(shuffled) - n_val}, validation={n_val}")

    return train_paths, val_paths


def _build_one_dataset(
    cfg: ExperimentConfig,
    dataset_name: str,
    task_to_paths: dict[str, list[Path]],
) -> None:
    train_paths, val_paths = _split_task_paths(
        task_to_paths,
        cfg.dataset.validation_ratio,
        cfg.convert.random_seed,
    )
    if not train_paths:
        raise RuntimeError(f"Dataset {dataset_name!r} has no training episodes.")

    width, height = cfg.camera.model_res
    builder_cls = _make_builder(
        dataset_name=dataset_name,
        version=cfg.dataset.version,
        train_paths=train_paths,
        val_paths=val_paths,
        min_action_norm=cfg.dataset.min_action_norm,
        preserve_gripper_changes=cfg.convert.preserve_gripper_changes,
        image_shape=(height, width, 3),
    )
    output_root = resolve_path(cfg.dataset.rlds_root)
    output_root.mkdir(parents=True, exist_ok=True)
    builder = builder_cls(data_dir=output_root)
    builder.download_and_prepare()
    print(f"Built {dataset_name}: {builder.data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config_path)

    task_to_paths = _collect_task_paths(cfg)
    if cfg.convert.merge_tasks:
        _build_one_dataset(cfg, cfg.dataset.name, task_to_paths)
    else:
        for task_id, paths in task_to_paths.items():
            _build_one_dataset(cfg, task_id, {task_id: paths})


if __name__ == "__main__":
    main()
