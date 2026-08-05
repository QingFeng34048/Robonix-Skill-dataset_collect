from __future__ import annotations

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


def save_episode(
    *,
    output_root: Path,
    task_id: str,
    instruction: str,
    dataset_name: str,
    fps: int,
    samples: list[dict],
    success: bool,
) -> Path:
    if len(samples) < 2:
        raise ValueError("episode must contain at least two samples")

    states = np.asarray(
        [sample["state"] for sample in samples],
        dtype=np.float32,
    )
    images = np.asarray(
        [sample["image"] for sample in samples],
        dtype=np.uint8,
    )

    if states.shape[1:] != (7,):
        raise ValueError(f"expected state shape [T, 7], got {states.shape}")

    actions = np.zeros_like(states, dtype=np.float32)
    actions[:-1, :6] = states[1:, :6] - states[:-1, :6]
    actions[:-1, 6] = states[1:, 6]
    actions[-1, 6] = states[-1, 6]

    status = "SUCCESS" if success else "FAIL"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    task_dir = output_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    output = task_dir / (
        f"ep_{status}_{timestamp}_{len(samples)}.hdf5"
    )

    with h5py.File(output, "w") as file:
        file.attrs["task_id"] = task_id
        file.attrs["language_instruction"] = instruction
        file.attrs["dataset_name"] = dataset_name
        file.attrs["reward"] = 1.0 if success else -1.0
        file.attrs["success"] = success
        file.attrs["fps"] = fps
        file.attrs["sim"] = False

        file.create_dataset("action", data=actions)

        observations = file.create_group("observations")
        observations.create_dataset(
            "images",
            data=images,
            compression="gzip",
        )
        observations.create_dataset("state", data=states)

    return output

