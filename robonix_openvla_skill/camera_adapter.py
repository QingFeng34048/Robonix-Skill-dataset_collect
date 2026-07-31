from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from sensor_msgs.msg import Image

from robonix_api import (
    ATLAS,
    Kind,
    Ros2Params,
    Skill,
    Transport,
)
from robonix_api.ros import RosBackend


log = logging.getLogger(__name__)

RGB_CONTRACT = "robonix/primitive/camera/rgb"


class CameraAdapterError(RuntimeError):
    """CameraAdapter 基础错误。"""


class CameraNotReadyError(CameraAdapterError):
    """相机尚未收到有效帧。"""


class StaleCameraFrameError(CameraAdapterError):
    """相机帧已经过期。"""


@dataclass(frozen=True)
class CameraFrame:
    """一帧已经转换成 RGB 排列的图像。"""

    rgb: np.ndarray
    received_monotonic: float
    source_stamp_ns: int
    frame_id: str

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.received_monotonic


def _channel_qos(channel, fallback: str) -> str:
    """优先使用 Atlas 中能力提供方声明的 QoS。"""

    params = channel.params
    if isinstance(params, Ros2Params):
        qos = params.qos_profile.strip()
        if qos:
            return qos

    return fallback


class CameraAdapter:
    """通过 Atlas 连接 RGB Camera Primitive，并缓存最新图像。

    该类只负责：

    1. 发现相机能力；
    2. 订阅 Image；
    3. 转成 RGB ndarray；
    4. 检查数据是否过期；
    5. 必要时编码为 JPEG。

    它不负责裁剪、模型归一化和推理请求。
    """

    def __init__(
        self,
        *,
        skill: Skill,
        provider_id: str,
        contract_id: str = RGB_CONTRACT,
    ) -> None:
        if not provider_id.strip():
            raise ValueError("camera provider_id must not be empty")

        self._skill = skill
        self._provider_id = provider_id.strip()
        self._contract_id = contract_id

        self._channel = None
        self._subscription = None

        self._frame_lock = threading.Lock()
        self._latest_frame: CameraFrame | None = None
        self._first_frame_event = threading.Event()

        self._last_callback_error_log = 0.0

    @property
    def connected(self) -> bool:
        return self._channel is not None and self._subscription is not None

    @property
    def endpoint(self) -> str | None:
        return self._channel.endpoint if self._channel is not None else None

    def connect(self, *, wait_timeout_s: float = 5.0) -> None:
        """查找并连接相机 Primitive。

        该方法设计为可重复调用。重复连接前会先关闭旧资源。
        """

        if wait_timeout_s <= 0:
            raise ValueError("wait_timeout_s must be > 0")

        self.close()

        capability = ATLAS.find_unique_capability(
            contract_id=self._contract_id,
            transport=Transport.ROS2,
            provider_kind=Kind.PRIMITIVE,
            provider_id=self._provider_id,
        )

        channel = self._skill.connect_capability(
            capability,
            self._contract_id,
            Transport.ROS2,
        )

        qos = _channel_qos(channel, fallback="best_effort")

        try:
            # 使用 Atlas 返回的 endpoint，而不是硬编码 ROS topic。
            #
            # declare=False 很重要：
            # 我们是这个能力的消费者，不是新的相机能力提供者。
            subscription = self._skill.create_subscription(
                contract_id=self._contract_id,
                topic=channel.endpoint,
                msg_type=Image,
                callback=self._on_image,
                qos=qos,
                declare=False,
            )
        except Exception:
            channel.close()
            raise

        self._channel = channel
        self._subscription = subscription

        log.info(
            "connected camera provider=%s endpoint=%s qos=%s",
            self._provider_id,
            channel.endpoint,
            qos,
        )

        if not self._first_frame_event.wait(wait_timeout_s):
            endpoint = channel.endpoint
            self.close()
            raise CameraNotReadyError(
                f"no RGB image received from {self._provider_id!r} "
                f"on {endpoint!r} within {wait_timeout_s:.1f}s"
            )

    def _on_image(self, msg: Image) -> None:
        """ROS 回调。

        回调中只进行必要的消息转换和缓存，不调用模型服务器。
        """

        try:
            rgb = self._decode_image(msg)

            source_stamp_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )

            frame = CameraFrame(
                rgb=rgb,
                received_monotonic=time.monotonic(),
                source_stamp_ns=source_stamp_ns,
                frame_id=str(msg.header.frame_id),
            )

            with self._frame_lock:
                self._latest_frame = frame

            self._first_frame_event.set()

        except Exception as exc:
            # 相机频率可能很高，限制错误日志频率，避免每帧刷屏。
            now = time.monotonic()
            if now - self._last_callback_error_log >= 1.0:
                log.warning("invalid camera image: %s", exc)
                self._last_callback_error_log = now

    @staticmethod
    def _decode_image(msg: Image) -> np.ndarray:
        """把 sensor_msgs/Image 转成拥有独立内存的 RGB uint8 数组。

        注意：
        - 必须考虑 msg.step，不能直接假设每行没有 padding；
        - 返回值必须 copy，不能继续引用 DDS 消息缓冲区；
        - 不对未知 encoding 猜测颜色顺序。
        """

        height = int(msg.height)
        width = int(msg.width)

        if height <= 0 or width <= 0:
            raise ValueError(
                f"invalid image size: width={width}, height={height}"
            )

        encoding = str(msg.encoding).strip().lower()

        if encoding in {"rgb8", "bgr8"}:
            packed = CameraAdapter._packed_rows(
                msg,
                bytes_per_pixel=3,
            )

            if encoding == "rgb8":
                rgb = packed
            else:
                rgb = packed[:, :, ::-1]

        elif encoding in {"rgba8", "bgra8"}:
            packed = CameraAdapter._packed_rows(
                msg,
                bytes_per_pixel=4,
            )

            if encoding == "rgba8":
                rgb = packed[:, :, :3]
            else:
                rgb = packed[:, :, 2::-1]

        elif encoding == "mono8":
            grey = CameraAdapter._packed_rows(
                msg,
                bytes_per_pixel=1,
            )[:, :, 0]

            rgb = np.repeat(grey[:, :, None], repeats=3, axis=2)

        elif encoding in {
            "yuyv",
            "yuy2",
            "yuv422_yuy2",
        }:
            packed = CameraAdapter._packed_rows(
                msg,
                bytes_per_pixel=2,
            )
            rgb = cv2.cvtColor(packed, cv2.COLOR_YUV2RGB_YUY2)

        else:
            raise ValueError(
                f"unsupported Image encoding {msg.encoding!r}; "
                "configure the Camera Primitive to publish rgb8, bgr8, "
                "rgba8, bgra8, mono8 or YUYV"
            )

        if rgb.shape != (height, width, 3):
            raise ValueError(
                f"decoded image has invalid shape {rgb.shape}, "
                f"expected {(height, width, 3)}"
            )

        # 连续、独立的 uint8 内存，避免 DDS 缓冲生命周期问题。
        return np.ascontiguousarray(rgb, dtype=np.uint8).copy()

    @staticmethod
    def _packed_rows(
        msg: Image,
        *,
        bytes_per_pixel: int,
    ) -> np.ndarray:
        """按 msg.step 移除每行末尾的 padding。"""

        height = int(msg.height)
        width = int(msg.width)

        packed_row_bytes = width * bytes_per_pixel
        step = int(msg.step) or packed_row_bytes

        if step < packed_row_bytes:
            raise ValueError(
                f"Image.step={step} is smaller than required "
                f"{packed_row_bytes}"
            )

        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        required = height * step

        if raw.size < required:
            raise ValueError(
                f"Image.data has {raw.size} bytes, expected at least "
                f"{required}"
            )

        rows = raw[:required].reshape(height, step)
        pixels = rows[:, :packed_row_bytes]

        return pixels.reshape(
            height,
            width,
            bytes_per_pixel,
        )

    def latest_frame(self, *, max_age_s: float) -> CameraFrame:
        """取得最新帧，并拒绝过期数据。"""

        if max_age_s <= 0:
            raise ValueError("max_age_s must be > 0")

        with self._frame_lock:
            frame = self._latest_frame

            if frame is None:
                raise CameraNotReadyError(
                    "camera is connected but no valid frame is available"
                )

            # 对 ndarray 做 copy，防止调用者修改缓存帧。
            result = CameraFrame(
                rgb=frame.rgb.copy(),
                received_monotonic=frame.received_monotonic,
                source_stamp_ns=frame.source_stamp_ns,
                frame_id=frame.frame_id,
            )

        age_s = result.age_s
        if age_s > max_age_s:
            raise StaleCameraFrameError(
                f"latest camera frame is stale: "
                f"age={age_s:.3f}s, limit={max_age_s:.3f}s"
            )

        return result

    def latest_rgb(self, *, max_age_s: float) -> np.ndarray:
        return self.latest_frame(max_age_s=max_age_s).rgb

    def latest_jpeg(
        self,
        *,
        max_age_s: float,
        quality: int = 90,
    ) -> bytes:
        """取得最新图像并编码为 JPEG。"""

        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")

        rgb = self.latest_rgb(max_age_s=max_age_s)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )

        if not ok:
            raise CameraAdapterError("failed to encode camera frame as JPEG")

        return encoded.tobytes()

    def close(self) -> None:
        """释放 ROS Subscription 和 Atlas Channel。"""

        subscription = self._subscription
        channel = self._channel

        self._subscription = None
        self._channel = None

        if subscription is not None:
            try:
                RosBackend.get().node.destroy_subscription(subscription)
            except Exception as exc:
                log.debug("destroy camera subscription failed: %s", exc)

        if channel is not None:
            # Channel.close() 是幂等的。
            channel.close()

        with self._frame_lock:
            self._latest_frame = None

        self._first_frame_event.clear()

