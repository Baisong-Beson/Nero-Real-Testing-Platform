"""ROS 2 runtime for the dual-arm Gate 3 executor (audit / execute).

Single-arm Gate 3 is unchanged in ``task_executor_ros.py``. This node
subscribes to five observation streams, infers a 16D action chunk, and
publishes ``[0:8]`` to the right arm and ``[8:16]`` to the left arm.
Any per-arm safety violation hard-stops both arms.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import signal
import threading
import time
from typing import Any

import numpy as np

from nero_pi05_bridge.dual_task_executor import DualGate3Plan
from nero_pi05_bridge.dual_task_executor import ExecutionAuthorization
from nero_pi05_bridge.dual_task_executor import arm_gripper_width
from nero_pi05_bridge.dual_task_executor import assert_both_near_home
from nero_pi05_bridge.dual_task_executor import authorize_execution
from nero_pi05_bridge.dual_task_executor import evaluate_dual_step
from nero_pi05_bridge.dual_task_executor import joint_names
from nero_pi05_bridge.dual_task_executor import load_plan
from nero_pi05_bridge.dual_task_executor import locked_gripper_command
from nero_pi05_bridge.observation import decode_camera_message
from nero_pi05_bridge.observation import parse_joint_state
from nero_pi05_bridge.observation import resize_with_pad
from nero_pi05_bridge.observation import split_dual_actions
from nero_pi05_bridge.observation import topic_is_compressed
from nero_pi05_bridge.observation import validate_actions
from nero_pi05_bridge.policy_client import PolicyClient
from nero_pi05_bridge.task_executor import StageMachine

DEFAULT_URDF = str(Path(__file__).resolve().parents[5] / "runtime" / "robot" / "nero.urdf")
EXECUTE_SETTLE_S = 1.5
DEADMAN_WAIT_TIMEOUT_S = 120.0
DEADMAN_WAIT_MIN_MSGS = 5
DEFAULT_EXTERNAL_TOPIC = "/zed_m/left/image_raw/compressed"
DEFAULT_RIGHT_WRIST_TOPIC = "/right_wrist/color/image_raw/compressed"
DEFAULT_LEFT_WRIST_TOPIC = "/left_wrist/color/image_raw/compressed"


class ResultWriter:
    def __init__(self, output_root: Path, *, mode: str, plan: DualGate3Plan):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_suffix = f"{time.time_ns() % 1_000_000_000:09d}"
        self.directory = output_root / f"{timestamp}_{run_suffix}_{mode}_{plan.fingerprint[:8]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._events = (self.directory / "events.jsonl").open("w")

    def event(self, name: str, **values: Any) -> None:
        record = {
            "monotonic_s": time.monotonic(),
            "wall_time_s": time.time(),
            "event": name,
            **values,
        }
        self._events.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        self._events.flush()

    def finish(self, result: dict[str, Any]) -> Path:
        self._events.close()
        destination = self.directory / "summary.json"
        temporary = self.directory / "summary.json.partial"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n")
        temporary.replace(destination)
        return destination


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def run_executor(
    plan: DualGate3Plan,
    *,
    mode: str,
    output_dir: Path,
    authorization: ExecutionAuthorization | None,
    urdf: Path,
    external_image_topic: str | None = None,
    right_wrist_image_topic: str | None = None,
    left_wrist_image_topic: str | None = None,
) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty as EmptyMsg
    from std_srvs.srv import Empty as EmptySrv
    from std_srvs.srv import SetBool

    from nero_pi05_bridge.nero_fk import NeroFK

    if mode not in {"audit", "execute"}:
        raise ValueError("mode must be audit or execute")
    if mode == "execute" and authorization is None:
        raise PermissionError("execute mode requires authorization")

    right_ns = plan.right_arm_namespace
    left_ns = plan.left_arm_namespace
    right_command_topic = f"{right_ns}/control/joint_states"
    left_command_topic = f"{left_ns}/control/joint_states"
    deadman_topic = plan.deadman_topic
    stage_advance_topic = f"{right_ns}/gate3_stage_advance"
    right_joint_topic = f"{right_ns}/feedback/joint_states"
    left_joint_topic = f"{left_ns}/feedback/joint_states"
    external_topic = external_image_topic or DEFAULT_EXTERNAL_TOPIC
    right_wrist_topic = right_wrist_image_topic or DEFAULT_RIGHT_WRIST_TOPIC
    left_wrist_topic = left_wrist_image_topic or DEFAULT_LEFT_WRIST_TOPIC

    writer = ResultWriter(output_dir, mode=mode, plan=plan)
    result: dict[str, Any] = {
        "mode": mode,
        "plan": plan.summary(),
        "right_command_topic": right_command_topic,
        "left_command_topic": left_command_topic,
        "deadman_topic": deadman_topic,
        "stage_advance_topic": stage_advance_topic,
        "external_image_topic": external_topic,
        "right_wrist_image_topic": right_wrist_topic,
        "left_wrist_image_topic": left_wrist_topic,
        "ok": False,
    }
    right_fk = NeroFK(urdf)
    left_fk = NeroFK(urdf)
    abort = {"requested": False}

    class DualGate3Node(Node):
        def __init__(self) -> None:
            super().__init__("nero_gate3_dual_task_executor")
            self.lock = threading.Lock()
            self.latest: dict[str, dict[str, Any]] = {}
            self.deadman_monotonic: float | None = None
            self.deadman_count = 0
            self.stage_advance_requested = False
            self.right_command_publisher = None
            self.left_command_publisher = None
            self.right_gate_client = None
            self.left_gate_client = None
            self.right_enable_client = None
            self.left_enable_client = None
            self.right_estop_client = None
            self.left_estop_client = None
            self._subscribe_camera(external_topic, self._on_external)
            self._subscribe_camera(right_wrist_topic, self._on_right_wrist)
            self._subscribe_camera(left_wrist_topic, self._on_left_wrist)
            self.create_subscription(JointState, right_joint_topic, self._on_right_joints, 10)
            self.create_subscription(JointState, left_joint_topic, self._on_left_joints, 10)
            self.create_subscription(EmptyMsg, deadman_topic, self._on_deadman, 10)
            self.create_subscription(EmptyMsg, stage_advance_topic, self._on_stage_advance, 10)

        def _subscribe_camera(self, topic: str, callback) -> None:
            msg_type = CompressedImage if topic_is_compressed(topic) else Image
            self.create_subscription(msg_type, topic, callback, 1)

        def _on_deadman(self, _message: EmptyMsg) -> None:
            with self.lock:
                self.deadman_monotonic = time.monotonic()
                self.deadman_count += 1

        def _on_stage_advance(self, _message: EmptyMsg) -> None:
            with self.lock:
                self.stage_advance_requested = True

        def consume_stage_advance(self) -> bool:
            with self.lock:
                requested = self.stage_advance_requested
                self.stage_advance_requested = False
                return requested

        def _on_external(self, message) -> None:
            self._store_image("external", message)

        def _on_right_wrist(self, message) -> None:
            self._store_image("wrist", message)

        def _on_left_wrist(self, message) -> None:
            self._store_image("left_wrist", message)

        def _store_image(self, key: str, message) -> None:
            try:
                image = decode_camera_message(message)
                image = resize_with_pad(np.asarray(image, dtype=np.uint8), 224)
            except Exception as exc:
                writer.event("image_error", stream=key, error=str(exc))
                return
            with self.lock:
                self.latest[key] = {
                    "value": image,
                    "received_monotonic": time.monotonic(),
                }

        def _store_joints(
            self,
            key: str,
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
                writer.event("joint_error", stream=key, error=str(exc))
                return
            with self.lock:
                self.latest[key] = {
                    "value": (joints, gripper),
                    "received_monotonic": time.monotonic(),
                    "names": list(message.name),
                }

        def _on_right_joints(self, message: JointState) -> None:
            self._store_joints(
                "right_joint",
                message,
                gripper_min_width_m=plan.right_gripper_min_width_m,
                gripper_max_width_m=plan.right_gripper_max_width_m,
            )

        def _on_left_joints(self, message: JointState) -> None:
            self._store_joints(
                "left_joint",
                message,
                gripper_min_width_m=plan.left_gripper_min_width_m,
                gripper_max_width_m=plan.left_gripper_max_width_m,
            )

        def snapshot(self, prompt: str) -> tuple[dict | None, dict]:
            now = time.monotonic()
            keys = ("external", "wrist", "left_wrist", "right_joint", "left_joint")
            with self.lock:
                entries = {key: self.latest.get(key) for key in keys}
                deadman_age = None if self.deadman_monotonic is None else now - self.deadman_monotonic
            missing = [key for key, entry in entries.items() if entry is None]
            if missing:
                return None, {"reason": "missing", "streams": missing}
            ages = {key: now - entry["received_monotonic"] for key, entry in entries.items()}
            stale = {key: age for key, age in ages.items() if age > plan.feedback_max_age_s}
            if stale:
                return None, {"reason": "stale", "ages_sec": stale}
            right_joints, right_gripper = entries["right_joint"]["value"]
            left_joints, left_gripper = entries["left_joint"]["value"]
            state = np.concatenate((right_joints, right_gripper, left_joints, left_gripper))
            observation = {
                "observation/exterior_image_1_left": entries["external"]["value"],
                "observation/wrist_image_left": entries["wrist"]["value"],
                "observation/wrist_image_right": entries["left_wrist"]["value"],
                "observation/state": state.astype(np.float32),
                "prompt": prompt,
            }
            metadata = {
                "ages_sec": ages,
                "deadman_age_sec": deadman_age,
                "right_joint_position": right_joints.tolist(),
                "right_gripper_closedness": right_gripper.tolist(),
                "left_joint_position": left_joints.tolist(),
                "left_gripper_closedness": left_gripper.tolist(),
                "state": state.tolist(),
                "prompt": prompt,
            }
            return observation, metadata

        def deadman_ok(self) -> bool:
            with self.lock:
                if self.deadman_monotonic is None:
                    return False
                return (time.monotonic() - self.deadman_monotonic) <= plan.deadman_timeout_s

        def wait_for_deadman(self, *, timeout_s: float, min_msgs: int) -> int:
            """Block until SPACE is held. Does not open the control gate."""
            print(
                f"Hold SPACE on the deadman terminal now.\n"
                f"  Waiting for {min_msgs} heartbeats on {deadman_topic} "
                f"(timeout {timeout_s:.0f}s). Execute starts automatically — "
                f"do not press Enter here.",
                flush=True,
            )
            with self.lock:
                start_count = self.deadman_count
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline and not abort["requested"]:
                rclpy.spin_once(self, timeout_sec=0.05)
                with self.lock:
                    got = self.deadman_count - start_count
                if got >= min_msgs and self.deadman_ok():
                    print(f"deadman live ({got} msgs); starting execute", flush=True)
                    return got
            with self.lock:
                got = self.deadman_count - start_count
            raise RuntimeError(
                f"timed out waiting for deadman on {deadman_topic} "
                f"(saw {got} msgs, need {min_msgs})"
            )

        def enable_execution_interfaces(self) -> None:
            self.right_gate_client = self.create_client(SetBool, f"{right_ns}/control_enable")
            self.left_gate_client = self.create_client(SetBool, f"{left_ns}/control_enable")
            self.right_enable_client = self.create_client(SetBool, f"{right_ns}/enable_agx_arm")
            self.left_enable_client = self.create_client(SetBool, f"{left_ns}/enable_agx_arm")
            self.right_estop_client = self.create_client(EmptySrv, f"{right_ns}/emergency_stop")
            self.left_estop_client = self.create_client(EmptySrv, f"{left_ns}/emergency_stop")

        def create_command_publishers(self) -> None:
            if self.right_command_publisher is not None or self.left_command_publisher is not None:
                raise RuntimeError("command publishers already created")
            self.right_command_publisher = self.create_publisher(JointState, right_command_topic, 1)
            self.left_command_publisher = self.create_publisher(JointState, left_command_topic, 1)

        def call_set_bool(self, client, *, value: bool, label: str) -> None:
            if client is None or not client.wait_for_service(timeout_sec=3.0):
                raise RuntimeError(f"{label} service unavailable")
            request = SetBool.Request()
            request.data = value
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=6.0)
            if not future.done() or future.result() is None or not future.result().success:
                raise RuntimeError(f"{label} failed")

        def call_estop(self) -> None:
            for label, client in (
                ("right", self.right_estop_client),
                ("left", self.left_estop_client),
            ):
                if client is None:
                    continue
                if not client.wait_for_service(timeout_sec=1.0):
                    writer.event("estop_unavailable", arm=label)
                    continue
                future = client.call_async(EmptySrv.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                writer.event("estop_called", arm=label)

        def publish_arm_command(
            self,
            arm: str,
            joints7: np.ndarray,
            gripper_width_m: float,
            *,
            gripper_effort_n: float,
        ) -> None:
            publisher = (
                self.right_command_publisher if arm == "right" else self.left_command_publisher
            )
            if publisher is None:
                raise RuntimeError(f"{arm} command publisher missing")
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = list(joint_names()) + ["gripper"]
            message.position = [float(v) for v in joints7] + [float(gripper_width_m)]
            message.effort = [0.0] * 7 + [float(gripper_effort_n)]
            publisher.publish(message)

        def publish_both(
            self,
            right_joints: np.ndarray,
            right_width: float,
            left_joints: np.ndarray,
            left_width: float,
        ) -> None:
            self.publish_arm_command(
                "right",
                right_joints,
                right_width,
                gripper_effort_n=plan.gripper_effort_n,
            )
            self.publish_arm_command(
                "left",
                left_joints,
                left_width,
                gripper_effort_n=plan.gripper_effort_n,
            )

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = DualGate3Node()

    def _handle_signal(_signum, _frame) -> None:
        abort["requested"] = True
        writer.event("operator_interrupt")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    policy = PolicyClient(
        plan.policy_host,
        plan.policy_port,
        inference_timeout_sec=plan.inference_timeout_s,
    )
    gate_open = False
    arms_enabled = False
    commanded_chunks: list[list[list[float]]] = []
    hard_stops: list[dict[str, Any]] = []
    stage_history: list[dict[str, Any]] = []

    async def _loop() -> None:
        nonlocal gate_open, arms_enabled
        period = plan.control_period_sec
        writer.event("start", mode=mode, urdf=str(urdf), gripper_control=plan.gripper_control)

        deadline = time.monotonic() + 15.0
        observation = None
        metadata: dict[str, Any] = {}
        while time.monotonic() < deadline and not abort["requested"]:
            rclpy.spin_once(node, timeout_sec=0.05)
            observation, metadata = node.snapshot(plan.stages[0].prompt)
            if observation is not None:
                break
        if observation is None:
            raise RuntimeError(f"timed out waiting for observations: {metadata}")

        right_joints = np.asarray(metadata["right_joint_position"], dtype=np.float64)
        left_joints = np.asarray(metadata["left_joint_position"], dtype=np.float64)
        assert_both_near_home(plan, right_joints, left_joints)
        writer.event(
            "home_ok",
            right_current_joints=right_joints.tolist(),
            left_current_joints=left_joints.tolist(),
            right_home_joints=plan.right_home_joints_rad.tolist(),
            left_home_joints=plan.left_home_joints_rad.tolist(),
            tolerance_rad=plan.home_tolerance_rad,
        )

        right_pub_count = node.count_publishers(right_command_topic)
        left_pub_count = node.count_publishers(left_command_topic)
        result["preflight_right_command_publisher_count"] = right_pub_count
        result["preflight_left_command_publisher_count"] = left_pub_count
        if mode == "execute" and (right_pub_count != 0 or left_pub_count != 0):
            raise RuntimeError(
                f"command topics already have publishers: right={right_pub_count} left={left_pub_count}"
            )
        if mode == "audit" and (right_pub_count != 0 or left_pub_count != 0):
            writer.event(
                "audit_warn_existing_publishers",
                right=right_pub_count,
                left=left_pub_count,
            )

        await policy.connect()
        writer.event("policy_connected", uri=policy.uri)

        right_view = plan.arm_view("right")
        left_view = plan.arm_view("left")
        previous_right_width = locked_gripper_command(right_view)
        previous_left_width = locked_gripper_command(left_view)
        total_deadline = time.monotonic() + plan.max_total_duration_s
        stages = StageMachine(stages=plan.stages).start(time.monotonic())
        writer.event("stage_started", stage=stages.current.to_dict(), index=stages.index)
        stage_history.append({"event": "start", "index": stages.index, **stages.current.to_dict()})

        if mode == "execute":
            node.wait_for_deadman(
                timeout_s=DEADMAN_WAIT_TIMEOUT_S,
                min_msgs=DEADMAN_WAIT_MIN_MSGS,
            )
            writer.event("deadman_ready")
            node.enable_execution_interfaces()
            node.create_command_publishers()
            if node.count_publishers(right_command_topic) != 1 or node.count_publishers(left_command_topic) != 1:
                raise RuntimeError("dual executor must be the unique command publisher on both arms")
            node.call_set_bool(node.right_enable_client, value=True, label="right_enable_agx_arm")
            node.call_set_bool(node.left_enable_client, value=True, label="left_enable_agx_arm")
            arms_enabled = True
            node.call_set_bool(node.right_gate_client, value=True, label="right_control_enable")
            node.call_set_bool(node.left_gate_client, value=True, label="left_control_enable")
            gate_open = True
            writer.event("gate_opened")

            drain_deadline = time.monotonic() + 0.1
            while time.monotonic() < drain_deadline:
                rclpy.spin_once(node, timeout_sec=0.0)
                await asyncio.sleep(0.001)

            settle_deadline = time.monotonic() + EXECUTE_SETTLE_S
            settle_commands = 0
            next_settle_tick = time.monotonic()
            while time.monotonic() < settle_deadline and not abort["requested"] and not hard_stops:
                hold, hold_metadata = node.snapshot(stages.current.prompt)
                if hold is None:
                    hard_stops.append({"code": "observation", "detail": hold_metadata})
                    writer.event("hard_stop", code="observation", detail=hold_metadata)
                    break
                if not node.deadman_ok():
                    hard_stops.append({"code": "deadman", "detail": hold_metadata})
                    writer.event("hard_stop", code="deadman", detail=hold_metadata)
                    break
                node.publish_both(
                    np.asarray(hold_metadata["right_joint_position"], dtype=np.float64),
                    arm_gripper_width(
                        right_view,
                        float(np.asarray(hold_metadata["right_gripper_closedness"], dtype=np.float64)[0]),
                    ),
                    np.asarray(hold_metadata["left_joint_position"], dtype=np.float64),
                    arm_gripper_width(
                        left_view,
                        float(np.asarray(hold_metadata["left_gripper_closedness"], dtype=np.float64)[0]),
                    ),
                )
                settle_commands += 1
                next_settle_tick += period
                while time.monotonic() < next_settle_tick:
                    rclpy.spin_once(node, timeout_sec=0.0)
                    await asyncio.sleep(0.001)
            writer.event("servo_settled", seconds=EXECUTE_SETTLE_S, commands=settle_commands)

        while (
            time.monotonic() < total_deadline
            and not abort["requested"]
            and not stages.done
            and not hard_stops
        ):
            rclpy.spin_once(node, timeout_sec=0.0)

            advanced_request = node.consume_stage_advance()
            stages, changed = stages.maybe_advance(time.monotonic(), requested=advanced_request)
            if changed:
                if stages.done:
                    writer.event("stages_complete")
                    stage_history.append({"event": "complete"})
                    break
                writer.event("stage_changed", stage=stages.current.to_dict(), index=stages.index)
                stage_history.append(
                    {"event": "changed", "index": stages.index, **stages.current.to_dict()}
                )

            observation, metadata = node.snapshot(stages.current.prompt)
            if observation is None:
                hard_stops.append({"code": "observation", "detail": metadata})
                writer.event("hard_stop", code="observation", detail=metadata)
                break
            if mode == "execute" and not node.deadman_ok():
                hard_stops.append({"code": "deadman", "detail": metadata})
                writer.event("hard_stop", code="deadman", detail=metadata)
                break

            try:
                infer_task = asyncio.create_task(policy.infer(observation))
                while not infer_task.done():
                    rclpy.spin_once(node, timeout_sec=0.0)
                    await asyncio.sleep(0.001)
                response = await infer_task
                actions = validate_actions(response, action_dim=16)
            except Exception as exc:
                hard_stops.append({"code": "inference", "error": str(exc) or type(exc).__name__})
                writer.event("hard_stop", code="inference", error=str(exc) or type(exc).__name__)
                break

            commanded_chunks.append(actions.tolist())
            right_chunk, left_chunk = split_dual_actions(actions)
            chunk_steps = min(plan.chunk_steps_per_inference, int(actions.shape[0]))
            next_tick = time.monotonic()
            stop_loop = False
            previous_right_joints = None
            previous_left_joints = None
            previous_right_velocity = None
            previous_left_velocity = None
            previous_right_tcp = None
            previous_left_tcp = None

            for step_index in range(chunk_steps):
                if abort["requested"] or time.monotonic() >= total_deadline:
                    stop_loop = True
                    break
                if step_index > 0:
                    rclpy.spin_once(node, timeout_sec=0.0)
                    observation, metadata = node.snapshot(stages.current.prompt)
                    if observation is None:
                        hard_stops.append({"code": "observation", "detail": metadata})
                        writer.event("hard_stop", code="observation", detail=metadata)
                        stop_loop = True
                        break
                    if mode == "execute" and not node.deadman_ok():
                        hard_stops.append({"code": "deadman", "detail": metadata})
                        writer.event("hard_stop", code="deadman", detail=metadata)
                        stop_loop = True
                        break

                current_right = np.asarray(metadata["right_joint_position"], dtype=np.float64)
                current_left = np.asarray(metadata["left_joint_position"], dtype=np.float64)
                requested_right = np.asarray(right_chunk[step_index, :7], dtype=np.float64)
                requested_left = np.asarray(left_chunk[step_index, :7], dtype=np.float64)
                right_closedness = float(right_chunk[step_index, 7])
                left_closedness = float(left_chunk[step_index, 7])
                if plan.gripper_control == "locked":
                    right_width = locked_gripper_command(right_view)
                    left_width = locked_gripper_command(left_view)
                else:
                    right_width = arm_gripper_width(right_view, right_closedness)
                    left_width = arm_gripper_width(left_view, left_closedness)

                right_current_tcp = right_fk.tcp_xyz(current_right, previous_right_width)
                left_current_tcp = left_fk.tcp_xyz(current_left, previous_left_width)
                right_requested_tcp = right_fk.tcp_xyz(requested_right, right_width)
                left_requested_tcp = left_fk.tcp_xyz(requested_left, left_width)

                violations = evaluate_dual_step(
                    plan,
                    right_current_joints=previous_right_joints if previous_right_joints is not None else current_right,
                    left_current_joints=previous_left_joints if previous_left_joints is not None else current_left,
                    right_requested_joints=requested_right,
                    left_requested_joints=requested_left,
                    right_requested_closedness=right_closedness,
                    left_requested_closedness=left_closedness,
                    right_current_tcp=previous_right_tcp if previous_right_tcp is not None else right_current_tcp,
                    left_current_tcp=previous_left_tcp if previous_left_tcp is not None else left_current_tcp,
                    right_requested_tcp=right_requested_tcp,
                    left_requested_tcp=left_requested_tcp,
                    right_previous_joints=previous_right_joints,
                    left_previous_joints=previous_left_joints,
                    right_previous_velocity=previous_right_velocity,
                    left_previous_velocity=previous_left_velocity,
                    right_previous_tcp=previous_right_tcp,
                    left_previous_tcp=previous_left_tcp,
                    right_previous_gripper_width_m=previous_right_width,
                    left_previous_gripper_width_m=previous_left_width,
                    is_first_command=step_index == 0,
                    check_command_lead=mode == "execute",
                )
                record = {
                    "stage": stages.current.name,
                    "stage_index": stages.index,
                    "prompt": stages.current.prompt,
                    "chunk_step": step_index,
                    "chunk_steps": chunk_steps,
                    "right_current_joints": current_right.tolist(),
                    "left_current_joints": current_left.tolist(),
                    "right_requested_joints": requested_right.tolist(),
                    "left_requested_joints": requested_left.tolist(),
                    "right_current_tcp": right_current_tcp.tolist(),
                    "left_current_tcp": left_current_tcp.tolist(),
                    "right_requested_tcp": right_requested_tcp.tolist(),
                    "left_requested_tcp": left_requested_tcp.tolist(),
                    "gripper_control": plan.gripper_control,
                    "right_requested_closedness": right_closedness,
                    "left_requested_closedness": left_closedness,
                    "right_gripper_width_m": right_width,
                    "left_gripper_width_m": left_width,
                    "violations": [{"code": v.code, "message": v.message} for v in violations],
                    "observation": metadata,
                }
                if step_index == 0:
                    record["action_chunk"] = actions.tolist()
                writer.event("candidate_command", **record)

                if violations:
                    hard_stops.append(
                        {
                            "code": "safety",
                            "stage": stages.current.name,
                            "chunk_step": step_index,
                            "violations": record["violations"],
                        }
                    )
                    writer.event("hard_stop", code="safety", violations=record["violations"])
                    stop_loop = True
                    break

                if mode == "execute":
                    node.publish_both(requested_right, right_width, requested_left, left_width)
                    writer.event(
                        "command_published",
                        stage=stages.current.name,
                        chunk_step=step_index,
                        right_joints=requested_right.tolist(),
                        left_joints=requested_left.tolist(),
                        right_gripper_width_m=right_width,
                        left_gripper_width_m=left_width,
                        gripper_effort_n=plan.gripper_effort_n,
                    )

                right_baseline = previous_right_joints if previous_right_joints is not None else current_right
                left_baseline = previous_left_joints if previous_left_joints is not None else current_left
                previous_right_velocity = (requested_right - right_baseline) / period
                previous_left_velocity = (requested_left - left_baseline) / period
                previous_right_joints = requested_right
                previous_left_joints = requested_left
                previous_right_tcp = right_requested_tcp
                previous_left_tcp = left_requested_tcp
                previous_right_width = right_width
                previous_left_width = left_width

                next_tick += period
                while time.monotonic() < next_tick:
                    rclpy.spin_once(node, timeout_sec=0.0)
                    await asyncio.sleep(0.001)

            if stop_loop:
                break

        result["commanded_chunk_count"] = len(commanded_chunks)
        result["hard_stops"] = hard_stops
        result["stage_history"] = stage_history
        result["stages_completed"] = stages.done or (
            stages.index == len(plan.stages) - 1
            and not hard_stops
            and not abort["requested"]
            and time.monotonic() >= (stages.stage_started_monotonic or 0) + stages.current.max_duration_s
        )
        if not stages.done and not hard_stops and not abort["requested"]:
            stages, changed = stages.maybe_advance(time.monotonic() + 1e9, requested=False)
            if changed and stages.done:
                writer.event("stages_complete_by_total_budget")
                stage_history.append({"event": "complete_by_total_budget"})
        result["ok"] = len(hard_stops) == 0 and not abort["requested"]
        np.savez_compressed(
            writer.directory / "audit_chunks.npz",
            chunks=np.asarray(commanded_chunks, dtype=np.float64)
            if commanded_chunks
            else np.zeros((0, 0, 16)),
        )

    try:
        asyncio.run(_loop())
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        writer.event("fatal", error=result["error"])
    finally:
        if mode == "execute":
            if not result.get("ok"):
                try:
                    node.call_estop()
                    writer.event("estop_after_failure")
                except Exception as exc:
                    writer.event("estop_error", error=str(exc))
            if gate_open:
                for label, client in (
                    ("right", node.right_gate_client),
                    ("left", node.left_gate_client),
                ):
                    if client is None:
                        continue
                    try:
                        node.call_set_bool(client, value=False, label=f"{label}_control_enable_close")
                        writer.event("gate_closed", arm=label)
                    except Exception as exc:
                        writer.event("gate_close_error", arm=label, error=str(exc))
            if arms_enabled:
                writer.event("arms_left_enabled", ok=bool(result.get("ok")))
                result["arms_left_enabled"] = True
        try:
            asyncio.run(policy.close())
        except Exception:
            pass
        summary_path = writer.finish(result)
        node.destroy_node()
        rclpy.shutdown()
        print(json.dumps({"summary": str(summary_path), "ok": result.get("ok")}, indent=2))
    return 0 if result.get("ok") else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NERO dual-arm Gate 3 staged task executor")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "audit", "execute"), default="plan")
    parser.add_argument("--approval-token")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/nero_gate3_dual_task_executor"))
    parser.add_argument("--urdf", type=Path, default=Path(DEFAULT_URDF))
    parser.add_argument("--external-image-topic")
    parser.add_argument("--right-wrist-image-topic")
    parser.add_argument("--left-wrist-image-topic")
    parser.add_argument("--prompt", help="Override the single-stage prompt for this run")
    args = parser.parse_args(argv)

    plan = load_plan(args.plan, prompt_override=args.prompt)
    print(json.dumps(plan.summary(), indent=2))
    if args.mode == "plan":
        return 0
    token = args.approval_token.strip() if args.approval_token else None
    if token == "":
        token = None
    authorization = None
    if args.mode == "execute":
        authorization = authorize_execution(plan, token)
    elif token is not None:
        parser.error("--approval-token is accepted only in execute mode")
    return run_executor(
        plan,
        mode=args.mode,
        output_dir=args.output_dir.expanduser().resolve(),
        authorization=authorization,
        urdf=args.urdf.expanduser().resolve(),
        external_image_topic=args.external_image_topic,
        right_wrist_image_topic=args.right_wrist_image_topic,
        left_wrist_image_topic=args.left_wrist_image_topic,
    )


if __name__ == "__main__":
    raise SystemExit(main())
