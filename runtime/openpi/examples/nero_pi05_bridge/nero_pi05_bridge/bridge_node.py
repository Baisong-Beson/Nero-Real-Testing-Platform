"""ROS 2 shadow bridge for observing NERO with an OpenPI pi0.5 policy."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from nero_pi05_bridge.observation import LEFT_GRIPPER_MAX_M
from nero_pi05_bridge.observation import LEFT_GRIPPER_MIN_M
from nero_pi05_bridge.observation import RIGHT_GRIPPER_MAX_M
from nero_pi05_bridge.observation import RIGHT_GRIPPER_MIN_M
from nero_pi05_bridge.observation import action_summary
from nero_pi05_bridge.observation import decode_camera_message
from nero_pi05_bridge.observation import parse_joint_state
from nero_pi05_bridge.observation import resize_with_pad
from nero_pi05_bridge.observation import split_dual_actions
from nero_pi05_bridge.observation import topic_is_compressed
from nero_pi05_bridge.observation import validate_actions
from nero_pi05_bridge.policy_client import PolicyClient


class NeroPi05ShadowBridge(Node):
    """Collect observations and log policy output without any control interfaces."""

    def __init__(self) -> None:
        super().__init__("nero_pi05_bridge")

        self.declare_parameter("arm", "left")
        arm = str(self.get_parameter("arm").value).lower()
        if arm not in ("left", "right", "both"):
            raise ValueError(f"arm must be 'left', 'right', or 'both', got {arm!r}")

        self.declare_parameter("shadow", value=True)
        if not bool(self.get_parameter("shadow").value):
            raise RuntimeError("Phase 1 is shadow-only; shadow=false is intentionally rejected")

        if arm == "both":
            default_external = "/zed_m/left/image_raw/compressed"
            default_wrist = "/right_wrist/color/image_raw/compressed"
            default_joint = "/right_arm/feedback/joint_states"
            default_left_wrist = "/left_wrist/color/image_raw/compressed"
            default_left_joint = "/left_arm/feedback/joint_states"
            default_grip_min = RIGHT_GRIPPER_MIN_M
            default_grip_max = RIGHT_GRIPPER_MAX_M
        else:
            default_external = "/external/zed_node/rgb/color/rect/image"
            default_wrist = f"/gripper/camera_{arm[0]}/color/image_raw"
            default_joint = f"/{arm}_arm/feedback/joint_states"
            default_left_wrist = ""
            default_left_joint = ""
            default_grip_min = 0.030
            default_grip_max = 0.099

        self.declare_parameter("external_image_topic", default_external)
        self.declare_parameter("wrist_image_topic", default_wrist)
        self.declare_parameter("joint_state_topic", default_joint)
        self.declare_parameter("left_wrist_image_topic", default_left_wrist)
        self.declare_parameter("left_joint_state_topic", default_left_joint)
        self.declare_parameter("policy_host", "127.0.0.1")
        self.declare_parameter("policy_port", 8000)
        self.declare_parameter("api_key", "")
        self.declare_parameter("prompt", "pick up the object")
        self.declare_parameter("inference_rate_hz", 5.0)
        self.declare_parameter("image_size", 224)
        self.declare_parameter("gripper_min_width_m", default_grip_min)
        self.declare_parameter("gripper_max_width_m", default_grip_max)
        self.declare_parameter("left_gripper_min_width_m", LEFT_GRIPPER_MIN_M)
        self.declare_parameter("left_gripper_max_width_m", LEFT_GRIPPER_MAX_M)
        self.declare_parameter("max_observation_age_sec", 0.5)
        self.declare_parameter("max_observation_skew_sec", 0.25)
        self.declare_parameter("connect_timeout_sec", 5.0)
        self.declare_parameter("inference_timeout_sec", 30.0)
        self.declare_parameter("connection_retry_sec", 2.0)
        self.declare_parameter("log_dir", "/tmp/nero_pi05_bridge")

        self._arm = arm
        self._bimanual = arm == "both"
        self._action_dim = 16 if self._bimanual else 8
        self._external_topic = str(self.get_parameter("external_image_topic").value)
        self._wrist_topic = str(self.get_parameter("wrist_image_topic").value)
        self._joint_topic = str(self.get_parameter("joint_state_topic").value)
        self._left_wrist_topic = str(self.get_parameter("left_wrist_image_topic").value)
        self._left_joint_topic = str(self.get_parameter("left_joint_state_topic").value)
        self._prompt = str(self.get_parameter("prompt").value)
        self._inference_rate_hz = float(self.get_parameter("inference_rate_hz").value)
        self._image_size = int(self.get_parameter("image_size").value)
        self._gripper_min_width_m = float(self.get_parameter("gripper_min_width_m").value)
        self._gripper_max_width_m = float(self.get_parameter("gripper_max_width_m").value)
        self._left_gripper_min_width_m = float(self.get_parameter("left_gripper_min_width_m").value)
        self._left_gripper_max_width_m = float(self.get_parameter("left_gripper_max_width_m").value)
        self._max_age_sec = float(self.get_parameter("max_observation_age_sec").value)
        self._max_skew_sec = float(self.get_parameter("max_observation_skew_sec").value)
        retry_sec = float(self.get_parameter("connection_retry_sec").value)

        if self._bimanual and (not self._left_wrist_topic or not self._left_joint_topic):
            raise ValueError("arm=both requires left_wrist_image_topic and left_joint_state_topic")
        if self._inference_rate_hz <= 0:
            raise ValueError("inference_rate_hz must be positive")
        if self._max_age_sec <= 0 or self._max_skew_sec < 0:
            raise ValueError("Observation age/skew limits are invalid")
        if retry_sec <= 0:
            raise ValueError("connection_retry_sec must be positive")

        self._policy_client = PolicyClient(
            str(self.get_parameter("policy_host").value),
            int(self.get_parameter("policy_port").value),
            api_key=str(self.get_parameter("api_key").value),
            connect_timeout_sec=float(self.get_parameter("connect_timeout_sec").value),
            inference_timeout_sec=float(self.get_parameter("inference_timeout_sec").value),
        )
        self._connection_retry_sec = retry_sec

        self._lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._latest: dict[str, dict[str, Any]] = {}
        self._counts = {
            "external_images": 0,
            "wrist_images": 0,
            "left_wrist_images": 0,
            "joint_states": 0,
            "left_joint_states": 0,
            "inference_ok": 0,
            "inference_error": 0,
            "not_ready": 0,
        }
        self._last_error = ""
        self._connected = False
        self._stop_event = threading.Event()
        self._destroyed = False

        log_dir = Path(str(self.get_parameter("log_dir").value)).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = log_dir / f"shadow_{arm}_{timestamp}_{os.getpid()}.jsonl"
        self._log_file = self._log_path.open("a", encoding="utf-8", buffering=1)

        self._status_pub = self.create_publisher(String, "~/status", 10)
        self._inference_pub = self.create_publisher(String, "~/last_inference", 10)
        self._subscribe_camera(self._external_topic, self._external_image_callback)
        self._subscribe_camera(self._wrist_topic, self._wrist_image_callback)
        self.create_subscription(
            JointState,
            self._joint_topic,
            self._joint_state_callback,
            qos_profile_sensor_data,
        )
        if self._bimanual:
            self._subscribe_camera(self._left_wrist_topic, self._left_wrist_image_callback)
            self.create_subscription(
                JointState,
                self._left_joint_topic,
                self._left_joint_state_callback,
                qos_profile_sensor_data,
            )
        self.create_timer(1.0, self._publish_status)

        self._worker = threading.Thread(
            target=self._worker_main,
            name="openpi-shadow-inference",
            daemon=True,
        )
        self._worker.start()

        topics = {
            "external_image": self._external_topic,
            "wrist_image": self._wrist_topic,
            "joint_state": self._joint_topic,
        }
        if self._bimanual:
            topics["left_wrist_image"] = self._left_wrist_topic
            topics["left_joint_state"] = self._left_joint_topic
        gripper_calibration: dict[str, Any] = {
            "right": {
                "min_width_m": self._gripper_min_width_m,
                "max_width_m": self._gripper_max_width_m,
            },
            "encoding": "absolute_closedness_0_open_1_closed",
        }
        if self._bimanual:
            gripper_calibration["left"] = {
                "min_width_m": self._left_gripper_min_width_m,
                "max_width_m": self._left_gripper_max_width_m,
            }
        else:
            gripper_calibration["min_width_m"] = self._gripper_min_width_m
            gripper_calibration["max_width_m"] = self._gripper_max_width_m

        self._write_log(
            {
                "event": "start",
                "mode": "shadow",
                "arm": self._arm,
                "action_dim": self._action_dim,
                "topics": topics,
                "policy_uri": self._policy_client.uri,
                "prompt": self._prompt,
                "inference_rate_hz": self._inference_rate_hz,
                "image_size": self._image_size,
                "gripper_calibration": gripper_calibration,
            }
        )
        self.get_logger().info("Phase 1 SHADOW mode is active; no robot control publisher or service client exists")
        self.get_logger().info(f"Target arm: {self._arm}; log: {self._log_path}")

    def _subscribe_camera(self, topic: str, callback) -> None:
        msg_type = CompressedImage if topic_is_compressed(topic) else Image
        self.create_subscription(msg_type, topic, callback, qos_profile_sensor_data)

    @staticmethod
    def _stamp_to_float(message) -> float:
        return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9

    def _image_to_rgb(self, message) -> np.ndarray:
        image = decode_camera_message(message)
        return resize_with_pad(np.asarray(image, dtype=np.uint8), self._image_size)

    def _external_image_callback(self, message) -> None:
        self._store_image("external", message, "external_images")

    def _wrist_image_callback(self, message) -> None:
        self._store_image("wrist", message, "wrist_images")

    def _left_wrist_image_callback(self, message) -> None:
        self._store_image("left_wrist", message, "left_wrist_images")

    def _store_image(self, key: str, message, counter: str) -> None:
        try:
            image = self._image_to_rgb(message)
        except Exception as exc:  # camera encodings must fail closed
            self._set_error(f"{key} image conversion failed: {exc}")
            return
        if hasattr(message, "encoding"):
            source_encoding = message.encoding
            source_shape = [int(message.height), int(message.width)]
        else:
            source_encoding = str(getattr(message, "format", "compressed"))
            source_shape = list(image.shape[:2])
        entry = {
            "value": image,
            "received_monotonic": time.monotonic(),
            "header_stamp": self._stamp_to_float(message),
            "source_encoding": source_encoding,
            "source_shape": source_shape,
        }
        with self._lock:
            self._latest[key] = entry
            self._counts[counter] += 1

    def _store_joints(
        self,
        key: str,
        counter: str,
        message: JointState,
        *,
        gripper_min_width_m: float,
        gripper_max_width_m: float,
    ) -> None:
        try:
            joints, gripper = parse_joint_state(
                message.name,
                message.position,
                gripper_min_width_m=gripper_min_width_m,
                gripper_max_width_m=gripper_max_width_m,
            )
        except Exception as exc:
            self._set_error(f"{key} joint state rejected: {exc}")
            return
        entry = {
            "value": (joints, gripper),
            "received_monotonic": time.monotonic(),
            "header_stamp": self._stamp_to_float(message),
        }
        with self._lock:
            self._latest[key] = entry
            self._counts[counter] += 1

    def _joint_state_callback(self, message: JointState) -> None:
        self._store_joints(
            "joint",
            "joint_states",
            message,
            gripper_min_width_m=self._gripper_min_width_m,
            gripper_max_width_m=self._gripper_max_width_m,
        )

    def _left_joint_state_callback(self, message: JointState) -> None:
        self._store_joints(
            "left_joint",
            "left_joint_states",
            message,
            gripper_min_width_m=self._left_gripper_min_width_m,
            gripper_max_width_m=self._left_gripper_max_width_m,
        )

    def _stream_keys(self) -> tuple[str, ...]:
        if self._bimanual:
            return ("external", "wrist", "left_wrist", "joint", "left_joint")
        return ("external", "wrist", "joint")

    def _snapshot(self) -> tuple[dict | None, dict]:
        now = time.monotonic()
        keys = self._stream_keys()
        with self._lock:
            entries = {key: self._latest.get(key) for key in keys}

        missing = [key for key, entry in entries.items() if entry is None]
        if missing:
            return None, {"reason": "missing", "streams": missing}

        ages = {key: now - entry["received_monotonic"] for key, entry in entries.items()}
        stale = {key: age for key, age in ages.items() if age > self._max_age_sec}
        if stale:
            return None, {"reason": "stale", "ages_sec": stale}

        receive_times = [entry["received_monotonic"] for entry in entries.values()]
        skew = max(receive_times) - min(receive_times)
        if skew > self._max_skew_sec:
            return None, {"reason": "skew", "skew_sec": skew, "ages_sec": ages}

        if self._bimanual:
            right_joints, right_gripper = entries["joint"]["value"]
            left_joints, left_gripper = entries["left_joint"]["value"]
            state = np.concatenate((right_joints, right_gripper, left_joints, left_gripper))
            observation = {
                "observation/exterior_image_1_left": entries["external"]["value"],
                "observation/wrist_image_left": entries["wrist"]["value"],
                "observation/wrist_image_right": entries["left_wrist"]["value"],
                "observation/state": state.astype(np.float32),
                "prompt": self._prompt,
            }
            metadata = {
                "ages_sec": ages,
                "skew_sec": skew,
                "header_stamps": {key: entry["header_stamp"] for key, entry in entries.items()},
                "source_encodings": {
                    key: entries[key]["source_encoding"]
                    for key in ("external", "wrist", "left_wrist")
                },
                "source_shapes": {
                    key: entries[key]["source_shape"] for key in ("external", "wrist", "left_wrist")
                },
                "right_joint_position": right_joints.tolist(),
                "right_gripper_closedness": right_gripper.tolist(),
                "left_joint_position": left_joints.tolist(),
                "left_gripper_closedness": left_gripper.tolist(),
                "state": state.tolist(),
            }
            return observation, metadata

        joints, gripper = entries["joint"]["value"]
        observation = {
            "observation/exterior_image_1_left": entries["external"]["value"],
            "observation/wrist_image_left": entries["wrist"]["value"],
            "observation/joint_position": joints,
            "observation/gripper_position": gripper,
            "prompt": self._prompt,
        }
        metadata = {
            "ages_sec": ages,
            "skew_sec": skew,
            "header_stamps": {key: entry["header_stamp"] for key, entry in entries.items()},
            "source_encodings": {
                "external": entries["external"]["source_encoding"],
                "wrist": entries["wrist"]["source_encoding"],
            },
            "source_shapes": {
                "external": entries["external"]["source_shape"],
                "wrist": entries["wrist"]["source_shape"],
            },
            "joint_position": joints.tolist(),
            "gripper_closedness": gripper.tolist(),
        }
        return observation, metadata

    def _worker_main(self) -> None:
        asyncio.run(self._inference_loop())

    async def _inference_loop(self) -> None:
        period = 1.0 / self._inference_rate_hz
        try:
            while rclpy.ok() and not self._stop_event.is_set():
                loop_start = time.monotonic()
                observation, observation_metadata = self._snapshot()
                if observation is None:
                    with self._lock:
                        self._counts["not_ready"] += 1
                    await asyncio.sleep(min(period, 0.1))
                    continue

                try:
                    if not self._connected:
                        metadata = await self._policy_client.connect()
                        self._connected = True
                        self._write_log({"event": "connected", "server_metadata": metadata})
                        self.get_logger().info(f"Connected to OpenPI at {self._policy_client.uri}")

                    inference_start = time.monotonic()
                    response = await self._policy_client.infer(observation)
                    inference_ms = (time.monotonic() - inference_start) * 1000.0
                    actions = validate_actions(response, action_dim=self._action_dim)
                    summary = action_summary(actions)
                    if self._bimanual:
                        right_actions, left_actions = split_dual_actions(actions)
                        summary["right_first_action"] = right_actions[0].tolist()
                        summary["left_first_action"] = left_actions[0].tolist()
                    record = {
                        "event": "inference",
                        "ok": True,
                        "inference_ms": inference_ms,
                        "observation": observation_metadata,
                        "actions": summary,
                        "server_timing": response.get("server_timing", {}),
                        "policy_timing": response.get("policy_timing", {}),
                    }
                    self._write_log(record)
                    with self._lock:
                        self._counts["inference_ok"] += 1
                        self._last_error = ""
                    self._inference_pub.publish(String(data=json.dumps(record, separators=(",", ":"))))
                except Exception as exc:
                    self._connected = False
                    await self._policy_client.close()
                    error = f"{type(exc).__name__}: {exc}"
                    self._set_error(error)
                    self._write_log({"event": "inference", "ok": False, "error": error})
                    with self._lock:
                        self._counts["inference_error"] += 1
                    await asyncio.sleep(self._connection_retry_sec)

                elapsed = time.monotonic() - loop_start
                await asyncio.sleep(max(0.0, period - elapsed))
        finally:
            await self._policy_client.close()

    def _set_error(self, error: str) -> None:
        with self._lock:
            self._last_error = error

    def _write_log(self, record: dict) -> None:
        record = {"time_unix": time.time(), **record}
        line = json.dumps(
            record,
            separators=(",", ":"),
            allow_nan=False,
            default=self._json_default,
        )
        with self._log_lock:
            self._log_file.write(line + "\n")

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")

    def _publish_status(self) -> None:
        with self._lock:
            counts = dict(self._counts)
            last_error = self._last_error
            streams = sorted(self._latest)
        status = {
            "mode": "shadow",
            "arm": self._arm,
            "action_dim": self._action_dim,
            "connected": self._connected,
            "streams_ready": streams,
            "counts": counts,
            "last_error": last_error,
            "log_path": str(self._log_path),
        }
        self._status_pub.publish(String(data=json.dumps(status, separators=(",", ":"))))

    def stop(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=7.0)
        self._write_log({"event": "stop", "counts": self._counts})
        self._log_file.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NeroPi05ShadowBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
