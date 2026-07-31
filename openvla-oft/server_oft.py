"""OpenVLA-OFT 多任务推理服务端：配置统一、task_id 校验、共享 dataset unnorm_key。"""

from __future__ import annotations

import argparse
import cgi
import io
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robonix_config import ExperimentConfig, expand_value, get_task_map, load_config, resolve_path

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


def _build_cfg(exp: ExperimentConfig) -> SimpleNamespace:
    return SimpleNamespace(
        base_model=expand_value(exp.train.vla_path),
        pretrained_checkpoint=str(resolve_path(exp.server.checkpoint_path)),
        use_l1_regression=exp.train.use_l1_regression,
        use_diffusion=exp.train.use_diffusion,
        use_film=exp.train.use_film,
        num_images_in_input=exp.train.num_images_in_input,
        use_proprio=exp.train.use_proprio,
        load_in_8bit=exp.server.load_in_8bit,
        load_in_4bit=exp.server.load_in_4bit,
        center_crop=exp.server.center_crop,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        # 多任务统一数据集只使用一个统计键，避免客户端自由文本决定反归一化。
        unnorm_key=exp.dataset.name,
        lora_rank=exp.train.lora_rank,
        num_diffusion_steps_train=exp.train.num_diffusion_steps_train,
        num_diffusion_steps_inference=exp.server.num_diffusion_steps_inference,
    )


def _validate_oft_checkpoint(path: str) -> None:
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"OFT checkpoint path does not exist: {checkpoint}")
    if not (checkpoint / "dataset_statistics.json").exists():
        raise FileNotFoundError(
            f"Missing dataset_statistics.json in {checkpoint}. Use the OFT checkpoint root directory."
        )
    action_heads = list(checkpoint.glob("action_head--*checkpoint.pt"))
    if len(action_heads) != 1:
        raise FileNotFoundError(
            f"Expected exactly one action_head--*checkpoint.pt in {checkpoint}, found {len(action_heads)}."
        )


def _image_from_bytes(image_bytes: bytes, form: cgi.FieldStorage) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        width = form.getvalue("width")
        height = form.getvalue("height")
        if width is None or height is None:
            raise
        width, height = int(width), int(height)
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        if arr.size != width * height * 3:
            raise ValueError("raw image size mismatch")
        return Image.fromarray(arr.reshape((height, width, 3)), mode="RGB")


def _to_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class VLARequestHandler(BaseHTTPRequestHandler):
    cfg = None
    task_map = None
    model = None
    processor = None
    action_head = None
    proprio_projector = None
    save_images = False
    save_dir = None
    return_first_action_for_legacy = True

    step = 0
    cond = threading.Condition()
    busy = False

    def _send_json(self, status_code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _resolve_task(self, form: cgi.FieldStorage):
        task_id = form.getvalue("task_id")
        instruction = form.getvalue("instruction")

        if task_id:
            task = self.task_map.get(str(task_id))
            if task is None:
                raise ValueError(f"unknown task_id: {task_id}")
            return task.task_id, task.instruction

        # 兼容旧客户端：仅在 instruction 与某个 canonical instruction 完全匹配时允许。
        if instruction:
            matches = [task for task in self.task_map.values() if task.instruction == str(instruction)]
            if len(matches) == 1:
                task = matches[0]
                return task.task_id, task.instruction

        raise ValueError("missing or invalid task_id")

    def do_POST(self) -> None:
        with VLARequestHandler.cond:
            while VLARequestHandler.busy:
                VLARequestHandler.cond.wait()
            VLARequestHandler.busy = True

        try:
            if self.path != "/act":
                self._send_json(404, {"error": "not found"})
                return

            content_type = self.headers.get("Content-Type")
            if not content_type:
                self._send_json(400, {"error": "missing Content-Type"})
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
                keep_blank_values=True,
            )

            try:
                task_id, instruction = self._resolve_task(form)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            image_field = form["image"] if "image" in form else None
            if image_field is None:
                self._send_json(400, {"error": "missing image"})
                return

            image_bytes = image_field.file.read() if hasattr(image_field, "file") else image_field.value
            try:
                image = _image_from_bytes(image_bytes, form)
            except Exception:
                self._send_json(400, {"error": "invalid image"})
                return

            VLARequestHandler.step += 1
            np_image = np.asarray(image, dtype=np.uint8)
            if self.save_images and self.save_dir is not None:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(self.save_dir / f"{VLARequestHandler.step:06d}.png", np_image)

            obs = {"full_image": np_image, "task_description": instruction}

            state_raw = form.getvalue("state")
            state_list = None
            if state_raw is not None:
                try:
                    state_list = json.loads(state_raw)
                except Exception:
                    self._send_json(400, {"error": "state must be JSON list"})
                    return

            if self.cfg.use_proprio:
                if state_list is None:
                    self._send_json(400, {"error": "missing state; use_proprio=True"})
                    return
                state = np.asarray(state_list, dtype=np.float32)
                if state.shape != (PROPRIO_DIM,):
                    self._send_json(400, {"error": f"state shape must be ({PROPRIO_DIM},), got {state.shape}"})
                    return
                if not np.all(np.isfinite(state)):
                    self._send_json(400, {"error": "state contains NaN or Inf"})
                    return
                obs["state"] = state

            try:
                actions = get_vla_action(
                    self.cfg,
                    self.model,
                    self.processor,
                    obs,
                    instruction,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    noisy_action_projector=None,
                    use_film=self.cfg.use_film,
                )
                action_chunk = [_to_list(action) for action in actions]
                if not action_chunk:
                    raise RuntimeError("model returned empty action chunk")
                first_action = action_chunk[0]

                print(
                    f"step={VLARequestHandler.step}, task_id={task_id}, "
                    f"instruction={instruction!r}, state={state_list}, "
                    f"first_action={first_action}, chunk_len={len(action_chunk)}"
                )

                self._send_json(
                    200,
                    {
                        "task_id": task_id,
                        "instruction": instruction,
                        "action": first_action if self.return_first_action_for_legacy else action_chunk,
                        "action_chunk": action_chunk,
                        "chunk_len": len(action_chunk),
                    },
                )
            except Exception as exc:
                print(traceback.format_exc())
                self._send_json(500, {"error": str(exc)})
        finally:
            with VLARequestHandler.cond:
                VLARequestHandler.busy = False
                VLARequestHandler.cond.notify()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    exp = load_config(args.config_path)

    cfg = _build_cfg(exp)
    _validate_oft_checkpoint(cfg.pretrained_checkpoint)

    model = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=model.llm_dim)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=PROPRIO_DIM)

    VLARequestHandler.cfg = cfg
    VLARequestHandler.task_map = get_task_map(exp)
    VLARequestHandler.model = model
    VLARequestHandler.processor = processor
    VLARequestHandler.action_head = action_head
    VLARequestHandler.proprio_projector = proprio_projector
    VLARequestHandler.save_images = exp.server.save_images
    VLARequestHandler.save_dir = resolve_path(exp.server.save_dir) if exp.server.save_images else None
    VLARequestHandler.return_first_action_for_legacy = exp.server.return_first_action_for_legacy

    print(f"Loaded OFT checkpoint: {cfg.pretrained_checkpoint}")
    print(f"unnorm_key: {cfg.unnorm_key}")
    print(f"use_proprio: {cfg.use_proprio}")
    print(f"Serving on http://{exp.server.host}:{exp.server.port}/act")
    server = ThreadingHTTPServer((exp.server.host, exp.server.port), VLARequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
