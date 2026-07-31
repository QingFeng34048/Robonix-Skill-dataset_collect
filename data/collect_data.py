"""Piper 多任务数据采集：统一 YAML 配置，每个 episode 只对应一个 task_id。"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
from piper_sdk import C_PiperInterface_V2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robonix_config import ExperimentConfig, TaskConfig, get_task_map, load_config, resolve_path

DEG_TO_RAD = np.pi / 180.0


class DataCollector:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.task_map = get_task_map(cfg)
        self.recording = False
        self.exit_flag = False
        self.buffer: list[dict] = []

        self.save_root = resolve_path(cfg.dataset.hdf5_root)
        self.save_root.mkdir(parents=True, exist_ok=True)

        self.task_id = cfg.collect.default_task_id
        self.task_cfg = self.task_map[self.task_id]
        self.instruction = self.task_cfg.instruction

        print(f"摄像头 (ID={cfg.camera.camera_id})")
        self.cap = cv2.VideoCapture(cfg.camera.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.capture_res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.capture_res[1])
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 ID={cfg.camera.camera_id}")

        print(f"连接机械臂 ({cfg.robot.can_port})")
        self.piper = C_PiperInterface_V2(cfg.robot.can_port)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)

        self.move_to_task_init(self.task_cfg)

    def move_to_task_init(self, task: TaskConfig) -> None:
        position = task.init_pose
        factor = self.cfg.robot.rad_to_sdk_int
        joints = [round(position[i] * factor) for i in range(6)]
        gripper = (
            self.cfg.robot.gripper_open_sdk
            if position[6] > 0.5
            else self.cfg.robot.gripper_close_sdk
        )

        print(f"移动到任务初始位姿: {task.task_id}")
        self.piper.ModeCtrl(0x01, 0x01, self.cfg.robot.teaching_speed, 0x00)
        self.piper.JointCtrl(*joints)
        self.piper.GripperCtrl(abs(gripper), self.cfg.robot.gripper_speed, 0x01, 0)
        time.sleep(3.0)

        # 进入示教模式，允许人工拖动采集轨迹。
        self.piper.MotionCtrl_2(0x02, 0x00, 0x00)
        time.sleep(1.0)

    def select_task(self) -> None:
        if self.recording:
            print("录制过程中不能切换任务")
            return

        tasks = list(self.task_map.values())
        print("\n可用任务：")
        for index, task in enumerate(tasks):
            print(f"  [{index}] {task.task_id}: {task.instruction}")

        try:
            selected = int(input("任务编号: ").strip())
            task = tasks[selected]
        except (ValueError, IndexError):
            print("无效任务编号")
            return

        self.task_id = task.task_id
        self.task_cfg = task
        self.instruction = task.instruction
        self.buffer = []
        self.move_to_task_init(task)
        print(f"当前任务: {self.task_id} | {self.instruction}")

    def get_robot_state(self) -> np.ndarray:
        j_msg = self.piper.GetArmJointMsgs().joint_state
        g_msg = self.piper.GetArmGripperMsgs().gripper_state

        raw_to_rad = self.cfg.robot.piper_raw_to_degree * DEG_TO_RAD
        joints = [
            j_msg.joint_1 * raw_to_rad,
            j_msg.joint_2 * raw_to_rad,
            j_msg.joint_3 * raw_to_rad,
            j_msg.joint_4 * raw_to_rad,
            j_msg.joint_5 * raw_to_rad,
            j_msg.joint_6 * raw_to_rad,
        ]

        # 与推理客户端完全一致：1=open, 0=closed。
        gripper_binary = float(
            g_msg.grippers_angle > self.cfg.robot.gripper_threshold_raw
        )
        return np.asarray(joints + [gripper_binary], dtype=np.float32)

    def capture_step(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame = cv2.flip(frame, self.cfg.camera.flip_code)
        img_model = cv2.resize(frame, tuple(self.cfg.camera.model_res))
        img_rgb = cv2.cvtColor(img_model, cv2.COLOR_BGR2RGB)
        state_vector = self.get_robot_state()

        return {
            "image": img_rgb,
            "state": state_vector,
            "state_ref": state_vector.copy(),
            "timestamp": time.time(),
        }, frame

    def save_episode(self, is_success: bool) -> None:
        if len(self.buffer) < 2:
            print("轨迹少于 2 帧，丢弃。")
            self.buffer = []
            return

        final_reward = 1.0 if is_success else -1.0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        status_str = "SUCCESS" if is_success else "FAIL"

        task_dir = self.save_root / self.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        filename = task_dir / f"ep_{status_str}_{timestamp}_{len(self.buffer)}.hdf5"

        all_states = np.asarray([x["state_ref"] for x in self.buffer], dtype=np.float32)
        all_images = np.asarray([x["image"] for x in self.buffer], dtype=np.uint8)
        actions = np.zeros_like(all_states, dtype=np.float32)
        actions[:-1, :6] = all_states[1:, :6] - all_states[:-1, :6]
        actions[:-1, 6] = all_states[1:, 6]
        actions[-1, :6] = 0.0
        actions[-1, 6] = all_states[-1, 6]

        with h5py.File(filename, "w") as f:
            f.attrs["task_id"] = self.task_id
            f.attrs["language_instruction"] = self.instruction
            f.attrs["dataset_name"] = self.cfg.dataset.name
            f.attrs["reward"] = final_reward
            f.attrs["success"] = bool(is_success)
            f.attrs["fps"] = int(self.cfg.collect.fps)
            f.attrs["sim"] = False
            f.create_dataset("action", data=actions)
            obs_group = f.create_group("observations")
            obs_group.create_dataset("images", data=all_images, compression="gzip")
            obs_group.create_dataset("state", data=all_states)

        print(f"\n[{status_str}] 已保存: {filename}")
        self.buffer = []

        if self.cfg.collect.return_to_init_after_episode:
            self.move_to_task_init(self.task_cfg)

    def run(self) -> None:
        print("\n" + "=" * 64)
        print("Piper 多任务数据采集")
        print("  [S] 开始录制")
        print("  [Y/N] 结束录制（成功/失败）")
        print("  [T] 空闲时切换任务")
        print("  [Q] 退出")
        print("=" * 64)
        print(f"当前任务: {self.task_id} | {self.instruction}")

        interval = 1.0 / self.cfg.collect.fps
        try:
            while not self.exit_flag:
                start_time = time.time()
                step_data, display_frame = self.capture_step()
                if step_data is None:
                    time.sleep(0.01)
                    continue

                curr_g_val = step_data["state"][6]
                if curr_g_val > 0.5:
                    g_text, g_color = "Gripper: OPEN (1.0)", (0, 255, 0)
                else:
                    g_text, g_color = "Gripper: CLOSED (0.0)", (0, 0, 255)

                cv2.putText(display_frame, g_text, (30, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_color, 2)
                cv2.putText(
                    display_frame,
                    f"Task: {self.task_id}",
                    (30, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                if self.recording:
                    self.buffer.append(step_data)
                    cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(display_frame, f"REC {len(self.buffer)}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "IDLE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Collector", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    self.exit_flag = True
                elif key == ord("t") and not self.recording:
                    self.select_task()
                elif key == ord("s") and not self.recording:
                    print(f"\n>>> 开始录制: {self.task_id}")
                    self.recording = True
                    self.buffer = []
                elif key == ord("y") and self.recording:
                    print("<<< 成功 (+1)")
                    self.recording = False
                    self.save_episode(True)
                elif key == ord("n") and self.recording:
                    print("<<< 失败 (-1)")
                    self.recording = False
                    self.save_episode(False)

                elapsed = time.time() - start_time
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config_path)
    DataCollector(cfg).run()


if __name__ == "__main__":
    main()
