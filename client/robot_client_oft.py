"""Piper OpenVLA-OFT 多任务客户端：统一配置、task_id、短 chunk 闭环执行。"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from piper_sdk import C_PiperInterface_V2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robonix_config import ExperimentConfig, TaskConfig, get_task_map, load_config


class RobotClient:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.task_map = get_task_map(cfg)
        self.task_id = cfg.client.default_task_id
        self.task_cfg = self.task_map[self.task_id]
        self.step_count = 0
        self.task_step_count = 0
        self.last_target_joints = None

        print(f"正在打开摄像头 (ID={cfg.camera.camera_id})...")
        self.cap = cv2.VideoCapture(cfg.camera.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.capture_res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.capture_res[1])
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        self.latest_frame = None
        self.camera_lock = threading.Lock()
        self.is_running = True
        self.camera_thread = threading.Thread(target=self._update_camera_frame, daemon=True)
        self.camera_thread.start()
        while self.latest_frame is None:
            time.sleep(0.01)

        print(f"正在连接机械臂 ({cfg.robot.can_port})...")
        self.piper = C_PiperInterface_V2(cfg.robot.can_port)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)

        self.move_to_task_init(self.task_cfg)
        print(">>> 系统初始化完成 <<<")

    def _update_camera_frame(self) -> None:
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.camera_lock:
                    self.latest_frame = frame.copy()
            else:
                time.sleep(0.01)

    def move_to_task_init(self, task: TaskConfig) -> None:
        position = task.init_pose
        factor = self.cfg.robot.rad_to_sdk_int
        joints = [round(position[i] * factor) for i in range(6)]
        gripper = (
            self.cfg.robot.gripper_open_sdk
            if position[6] > 0.5
            else self.cfg.robot.gripper_close_sdk
        )

        print(f"移动到任务初始位置: {task.task_id}")
        self.piper.ModeCtrl(0x01, 0x01, self.cfg.robot.move_speed, 0x00)
        self.piper.JointCtrl(*joints)
        self.piper.GripperCtrl(gripper, self.cfg.robot.gripper_speed, 0x01, 0)
        time.sleep(3.0)
        self.last_target_joints = None
        self.task_step_count = 0

    def select_task(self) -> None:
        tasks = list(self.task_map.values())
        print("\n可用推理任务：")
        for index, task in enumerate(tasks):
            print(f"  [{index}] {task.task_id}: {task.instruction}")
        try:
            selected = int(input("任务编号: ").strip())
            task = tasks[selected]
        except (ValueError, IndexError):
            print("无效任务编号，保持当前任务。")
            return

        self.task_id = task.task_id
        self.task_cfg = task
        self.last_target_joints = None
        self.task_step_count = 0
        self.move_to_task_init(task)
        print(f"当前任务: {self.task_id} | {self.task_cfg.instruction}")

    def get_robot_state_rad(self) -> tuple[list[float], float]:
        j_msg = self.piper.GetArmJointMsgs().joint_state
        g_msg = self.piper.GetArmGripperMsgs().gripper_state
        factor = self.cfg.robot.rad_to_sdk_int
        joints = [
            j_msg.joint_1 / factor,
            j_msg.joint_2 / factor,
            j_msg.joint_3 / factor,
            j_msg.joint_4 / factor,
            j_msg.joint_5 / factor,
            j_msg.joint_6 / factor,
        ]
        gripper_state = float(
            g_msg.grippers_angle > self.cfg.robot.gripper_threshold_raw
        )
        return joints, gripper_state

    def capture_image_bytes(self):
        with self.camera_lock:
            if self.latest_frame is None:
                return None, None
            frame = self.latest_frame.copy()

        frame = cv2.flip(frame, self.cfg.camera.flip_code)
        img_resized = cv2.resize(frame, tuple(self.cfg.camera.model_res))
        ok, img_encoded = cv2.imencode(".jpg", img_resized)
        if not ok:
            return None, img_resized
        return img_encoded.tobytes(), img_resized

    def execute_action(self, base_joints: list[float], action_pred: list[float]) -> list[float]:
        action = np.asarray(action_pred, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")

        delta_joints = np.clip(
            action[:6],
            -self.task_cfg.max_delta,
            self.task_cfg.max_delta,
        )
        target_joints_rad = np.asarray(base_joints, dtype=np.float32) + delta_joints
        target_gripper_state = float(action[6])

        cmd_joints = [int(round(j * self.cfg.robot.rad_to_sdk_int)) for j in target_joints_rad]
        cmd_gripper = (
            self.cfg.robot.gripper_open_sdk
            if target_gripper_state > 0.5
            else self.cfg.robot.gripper_close_sdk
        )

        self.piper.JointCtrl(*cmd_joints)
        self.piper.GripperCtrl(cmd_gripper, self.cfg.robot.gripper_speed, 0x01, 0)
        return target_joints_rad.tolist()

    def display_frame(self, display_img, extra_text: str = "") -> str | None:
        if display_img is None:
            return None
        info = f"Task: {self.task_id} | Step: {self.task_step_count}"
        if extra_text:
            info += f" | {extra_text}"
        cv2.putText(display_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display_img, "Q: quit | T: switch task", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow("Robot View", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit"
        if key == ord("t"):
            return "switch"
        return None

    def request_action_chunk(self, img_bytes: bytes, state_list: list[float]) -> list[list[float]]:
        files = {"image": ("obs.jpg", img_bytes, "image/jpeg")}
        data = {
            "task_id": self.task_id,
            "instruction": self.task_cfg.instruction,
            "state": json.dumps(state_list),
        }
        response = requests.post(
            self.cfg.client.server_url,
            files=files,
            data=data,
            timeout=self.cfg.client.request_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Server {response.status_code}: {response.text}")

        result = response.json()
        if self.cfg.client.use_action_chunk and "action_chunk" in result:
            action_chunk = result["action_chunk"]
        else:
            raw_action = result["action"]
            action_chunk = raw_action if raw_action and isinstance(raw_action[0], list) else [raw_action]

        if not action_chunk:
            raise RuntimeError("Server returned empty action chunk")
        return action_chunk

    def run(self) -> None:
        print("\n" + "=" * 64)
        print("OpenVLA-OFT 多任务远程推理客户端")
        print(f"Server: {self.cfg.client.server_url}")
        print("按 T 可切换任务；每次只执行任务配置中的前 K 个 action，然后重新观测。")
        print("=" * 64)
        self.select_task()
        input(">>> 按回车开始推理...")

        interval = 1.0 / self.cfg.client.control_freq
        try:
            while True:
                if self.task_step_count >= self.task_cfg.max_steps:
                    print(f"任务 {self.task_id} 已达到 max_steps={self.task_cfg.max_steps}，请选择下一任务。")
                    self.select_task()

                img_bytes, display_img = self.capture_image_bytes()
                event = self.display_frame(display_img, "requesting...")
                if event == "quit":
                    break
                if event == "switch":
                    self.select_task()
                    continue
                if img_bytes is None:
                    time.sleep(0.01)
                    continue

                curr_joints, curr_gripper = self.get_robot_state_rad()
                state_list = curr_joints + [curr_gripper]

                try:
                    action_chunk = self.request_action_chunk(img_bytes, state_list)
                except Exception as exc:
                    print(f"Req Failed: {exc}")
                    time.sleep(0.05)
                    continue

                execute_steps = min(self.task_cfg.execute_chunk_steps, len(action_chunk))
                print(
                    f"[OFT] chunk_len={len(action_chunk)}, execute={execute_steps}, "
                    f"first={action_chunk[0][:3]}..."
                )

                switch_requested = False
                for idx, action in enumerate(action_chunk[:execute_steps]):
                    step_start = time.time()
                    self.step_count += 1
                    self.task_step_count += 1

                    base_joints = self.last_target_joints if self.last_target_joints is not None else curr_joints
                    self.last_target_joints = self.execute_action(base_joints, action)

                    _, disp = self.capture_image_bytes()
                    event = self.display_frame(disp, f"chunk {idx + 1}/{execute_steps}")
                    if event == "quit":
                        raise KeyboardInterrupt
                    if event == "switch":
                        switch_requested = True
                        break

                    elapsed = time.time() - step_start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)

                if switch_requested:
                    self.select_task()
        except KeyboardInterrupt:
            print("\n停止运行...")
        finally:
            self.is_running = False
            self.cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config_path)
    RobotClient(cfg).run()


if __name__ == "__main__":
    main()
