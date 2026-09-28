"""ROS 2 runtime for Gate 2 NERO arm executor (audit / execute).

Default is fail-closed: no control publisher is created in audit mode.
Execute mode requires plan.execution_enabled=true, matching approval token,
fresh deadman heartbeat, and hard-stop safety checks on every command.
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

from nero_pi05_bridge.arm_executor import ArmExecutorPlan
from nero_pi05_bridge.arm_executor import ExecutionAuthorization
from nero_pi05_bridge.arm_executor import authorize_execution
from nero_pi05_bridge.arm_executor import joint_names
from nero_pi05_bridge.arm_executor import load_plan
from nero_pi05_bridge.arm_executor import locked_gripper_command
from nero_pi05_bridge.arm_executor import validate_joint_target
from nero_pi05_bridge.observation import parse_joint_state
from nero_pi05_bridge.observation import resize_with_pad
from nero_pi05_bridge.observation import validate_actions
from nero_pi05_bridge.policy_client import PolicyClient

DEFAULT_URDF = str(Path(__file__).resolve().parents[5] / "runtime" / "robot" / "nero.urdf")


class ResultWriter:
    def __init__(self, output_root: Path, *, mode: str, plan: ArmExecutorPlan):
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


def _image_msg_to_rgb8(message) -> np.ndarray:
    """Decode sensor_msgs/Image to HWC RGB uint8 without cv_bridge.

    Avoids ROS Humble cv_bridge + NumPy 2 ABI crashes under conda env pika.
    """
    encoding = str(message.encoding).lower()
    height = int(message.height)
    width = int(message.width)
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image shape {height}x{width}")
    if encoding in {"rgb8", "bgr8"}:
        array = np.frombuffer(message.data, dtype=np.uint8)
        expected = height * width * 3
        if array.size < expected:
            raise ValueError(f"image buffer too small for {encoding}: {array.size} < {expected}")
        image = array[:expected].reshape((height, width, 3))
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image)
    if encoding in {"rgba8", "bgra8"}:
        array = np.frombuffer(message.data, dtype=np.uint8)
        expected = height * width * 4
        if array.size < expected:
            raise ValueError(f"image buffer too small for {encoding}: {array.size} < {expected}")
        image = array[:expected].reshape((height, width, 4))[:, :, :3]
        if encoding == "bgra8":
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image)
    raise ValueError(f"unsupported image encoding for Gate 2: {message.encoding}")


def run_executor(
    plan: ArmExecutorPlan,
    *,
    mode: str,
    output_dir: Path,
    authorization: ExecutionAuthorization | None,
    urdf: Path,
    external_image_topic: str | None = None,
    wrist_image_topic: str | None = None,
) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty as EmptyMsg
    from std_srvs.srv import SetBool
    from std_srvs.srv import Empty as EmptySrv

    from nero_pi05_bridge.nero_fk import NeroFK

    if mode not in {"audit", "execute"}:
        raise ValueError("mode must be audit or execute")
    if mode == "execute" and authorization is None:
        raise PermissionError("execute mode requires authorization")

    namespace = plan.arm_namespace
    command_topic = f"{namespace}/control/joint_states"
    deadman_topic = f"{namespace}/gate2_deadman"
    joint_topic = f"{namespace}/feedback/joint_states"
    # Match bridge_node / shadow defaults: ZED external + right wrist D405.
    external_topic = external_image_topic or "/external/zed_node/rgb/color/rect/image"
    wrist_topic = wrist_image_topic or "/gripper/camera_r/color/image_raw"

    writer = ResultWriter(output_dir, mode=mode, plan=plan)
    result: dict[str, Any] = {
        "mode": mode,
        "plan": plan.summary(),
        "command_topic": command_topic,
        "deadman_topic": deadman_topic,
        "external_image_topic": external_topic,
        "wrist_image_topic": wrist_topic,
        "ok": False,
    }
    fk = NeroFK(urdf)
    abort = {"requested": False}

    class Gate2Node(Node):
        def __init__(self) -> None:
            super().__init__("nero_gate2_arm_executor")
            self.lock = threading.Lock()
            self.latest: dict[str, dict[str, Any]] = {}
            self.deadman_monotonic: float | None = None
            self.command_publisher = None
            self.gate_client = None
            self.enable_client = None
            self.estop_client = None
            self.create_subscription(Image, external_topic, self._on_external, 1)
            self.create_subscription(Image, wrist_topic, self._on_wrist, 1)
            self.create_subscription(JointState, joint_topic, self._on_joints, 10)
            self.create_subscription(EmptyMsg, deadman_topic, self._on_deadman, 10)

        def _on_deadman(self, _message: EmptyMsg) -> None:
            with self.lock:
                self.deadman_monotonic = time.monotonic()

        def _on_external(self, message: Image) -> None:
            self._store_image("external", message)

        def _on_wrist(self, message: Image) -> None:
            self._store_image("wrist", message)

        def _store_image(self, key: str, message: Image) -> None:
            try:
                image = _image_msg_to_rgb8(message)
                image = resize_with_pad(np.asarray(image, dtype=np.uint8), 224)
            except Exception as exc:
                writer.event("image_error", stream=key, error=str(exc))
                return
            with self.lock:
                self.latest[key] = {
                    "value": image,
                    "received_monotonic": time.monotonic(),
                }

        def _on_joints(self, message: JointState) -> None:
            try:
                joints, gripper = parse_joint_state(
                    message.name,
                    message.position,
                    gripper_min_width_m=0.030,
                    gripper_max_width_m=0.099,
                )
            except Exception as exc:
                writer.event("joint_error", error=str(exc))
                return
            with self.lock:
                self.latest["joint"] = {
                    "value": (joints, gripper),
                    "received_monotonic": time.monotonic(),
                    "names": list(message.name),
                }

        def snapshot(self) -> tuple[dict | None, dict]:
            now = time.monotonic()
            with self.lock:
                entries = {key: self.latest.get(key) for key in ("external", "wrist", "joint")}
                deadman_age = None if self.deadman_monotonic is None else now - self.deadman_monotonic
            missing = [key for key, entry in entries.items() if entry is None]
            if missing:
                return None, {"reason": "missing", "streams": missing}
            ages = {key: now - entry["received_monotonic"] for key, entry in entries.items()}
            stale = {key: age for key, age in ages.items() if age > plan.feedback_max_age_s}
            if stale:
                return None, {"reason": "stale", "ages_sec": stale}
            joints, gripper = entries["joint"]["value"]
            observation = {
                "observation/exterior_image_1_left": entries["external"]["value"],
                "observation/wrist_image_left": entries["wrist"]["value"],
                "observation/joint_position": joints,
                "observation/gripper_position": gripper,
                "prompt": plan.prompt,
            }
            metadata = {
                "ages_sec": ages,
                "deadman_age_sec": deadman_age,
                "joint_position": joints.tolist(),
                "gripper_closedness": gripper.tolist(),
            }
            return observation, metadata

        def deadman_ok(self) -> bool:
            with self.lock:
                if self.deadman_monotonic is None:
                    return False
                return (time.monotonic() - self.deadman_monotonic) <= plan.deadman_timeout_s

        def enable_execution_interfaces(self) -> None:
            self.gate_client = self.create_client(SetBool, f"{namespace}/control_enable")
            self.enable_client = self.create_client(SetBool, f"{namespace}/enable_agx_arm")
            self.estop_client = self.create_client(EmptySrv, f"{namespace}/emergency_stop")

        def create_command_publisher(self) -> None:
            if self.command_publisher is not None:
                raise RuntimeError("command publisher already created")
            self.command_publisher = self.create_publisher(JointState, command_topic, 1)

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
            if self.estop_client is None:
                return
            if not self.estop_client.wait_for_service(timeout_sec=1.0):
                writer.event("estop_unavailable")
                return
            future = self.estop_client.call_async(EmptySrv.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            writer.event("estop_called")

        def publish_command(self, joints7: np.ndarray, gripper_width_m: float) -> None:
            if self.command_publisher is None:
                raise RuntimeError("command publisher missing")
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = list(joint_names()) + ["gripper"]
            message.position = [float(v) for v in joints7] + [float(gripper_width_m)]
            self.command_publisher.publish(message)

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Gate2Node()

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
    arm_enabled = False
    commanded_chunks: list[list[list[float]]] = []
    hard_stops: list[dict[str, Any]] = []

    async def _loop() -> None:
        nonlocal gate_open, arm_enabled
        period = plan.control_period_sec
        writer.event("start", mode=mode, urdf=str(urdf))

        # Preflight: wait for first observation.
        deadline = time.monotonic() + 15.0
        observation = None
        metadata = {}
        while time.monotonic() < deadline and not abort["requested"]:
            rclpy.spin_once(node, timeout_sec=0.05)
            observation, metadata = node.snapshot()
            if observation is not None:
                break
        if observation is None:
            raise RuntimeError(f"timed out waiting for observations: {metadata}")

        publisher_count = node.count_publishers(command_topic)
        result["preflight_command_publisher_count"] = publisher_count
        if mode == "execute" and publisher_count != 0:
            raise RuntimeError(f"command topic already has {publisher_count} publisher(s)")
        if mode == "audit" and publisher_count != 0:
            writer.event("audit_warn_existing_publishers", count=publisher_count)

        await policy.connect()
        writer.event("policy_connected", uri=policy.uri)

        previous_joints = None
        previous_velocity = None
        previous_tcp = None
        first_command = True
        window_deadline = time.monotonic() + plan.test_window_s

        if mode == "execute":
            node.enable_execution_interfaces()
            node.create_command_publisher()
            # Unique publisher check after creation.
            if node.count_publishers(command_topic) != 1:
                raise RuntimeError("Gate 2 executor must be the unique command publisher")
            node.call_set_bool(node.enable_client, value=True, label="enable_agx_arm")
            arm_enabled = True
            node.call_set_bool(node.gate_client, value=True, label="control_enable")
            gate_open = True
            writer.event("gate_opened")

        while time.monotonic() < window_deadline and not abort["requested"]:
            loop_start = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.0)
            observation, metadata = node.snapshot()
            if observation is None:
                hard_stops.append({"code": "observation", "detail": metadata})
                writer.event("hard_stop", code="observation", detail=metadata)
                break
            if mode == "execute" and not node.deadman_ok():
                hard_stops.append({"code": "deadman", "detail": metadata})
                writer.event("hard_stop", code="deadman", detail=metadata)
                break

            try:
                # Keep ROS callbacks flowing during inference; otherwise camera /
                # joint ages expire while awaiting the policy websocket.
                infer_task = asyncio.create_task(policy.infer(observation))
                while not infer_task.done():
                    rclpy.spin_once(node, timeout_sec=0.0)
                    await asyncio.sleep(0.001)
                response = await infer_task
                actions = validate_actions(response)
            except Exception as exc:
                hard_stops.append({"code": "inference", "error": str(exc) or type(exc).__name__})
                writer.event("hard_stop", code="inference", error=str(exc) or type(exc).__name__)
                break

            current_joints = np.asarray(observation["observation/joint_position"], dtype=np.float64)
            locked_width = locked_gripper_command(plan)
            current_tcp = fk.tcp_xyz(current_joints, locked_width)
            # Use first action step as the immediate absolute target.
            requested = np.asarray(actions[0, :7], dtype=np.float64)
            requested_tcp = fk.tcp_xyz(requested, locked_width)
            # In audit the arm does not move, so velocity / TCP-step checks must be
            # relative to the previous candidate command, not the frozen feedback.
            baseline_joints = (
                previous_joints
                if mode == "audit" and previous_joints is not None
                else current_joints
            )
            baseline_tcp = (
                previous_tcp if mode == "audit" and previous_tcp is not None else current_tcp
            )
            violations = validate_joint_target(
                plan,
                current_joints=baseline_joints,
                previous_joints=previous_joints,
                previous_velocity=previous_velocity,
                requested_joints=requested,
                current_tcp=baseline_tcp,
                requested_tcp=requested_tcp,
                previous_tcp=previous_tcp,
                is_first_command=first_command,
            )
            record = {
                "current_joints": current_joints.tolist(),
                "requested_joints": requested.tolist(),
                "current_tcp": current_tcp.tolist(),
                "requested_tcp": requested_tcp.tolist(),
                "action_chunk": actions.tolist(),
                "violations": [{"code": v.code, "message": v.message} for v in violations],
                "observation": metadata,
            }
            writer.event("candidate_command", **record)
            commanded_chunks.append(actions.tolist())

            if violations:
                hard_stops.append(
                    {
                        "code": "safety",
                        "violations": [{"code": v.code, "message": v.message} for v in violations],
                    }
                )
                writer.event("hard_stop", code="safety", violations=record["violations"])
                break

            if mode == "execute":
                node.publish_command(requested, locked_width)
                writer.event("command_published", joints=requested.tolist(), gripper_width_m=locked_width)

            period_delta = requested - baseline_joints
            previous_velocity = period_delta / plan.control_period_sec
            previous_joints = requested
            previous_tcp = requested_tcp
            first_command = False

            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0.0, period - elapsed))

        result["commanded_chunk_count"] = len(commanded_chunks)
        result["hard_stops"] = hard_stops
        result["ok"] = len(hard_stops) == 0 and not abort["requested"]
        # Persist chunks for offline native_replay / TCP checks.
        np.savez_compressed(
            writer.directory / "audit_chunks.npz",
            chunks=np.asarray(commanded_chunks, dtype=np.float64) if commanded_chunks else np.zeros((0, 0, 8)),
        )

    try:
        asyncio.run(_loop())
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        writer.event("fatal", error=result["error"])
    finally:
        if mode == "execute":
            try:
                node.call_estop()
            except Exception as exc:
                writer.event("estop_error", error=str(exc))
            if gate_open and node.gate_client is not None:
                try:
                    node.call_set_bool(node.gate_client, value=False, label="control_enable_close")
                    writer.event("gate_closed")
                except Exception as exc:
                    writer.event("gate_close_error", error=str(exc))
            if arm_enabled and node.enable_client is not None:
                try:
                    node.call_set_bool(node.enable_client, value=False, label="enable_agx_arm_disable")
                    writer.event("arm_disabled")
                except Exception as exc:
                    writer.event("arm_disable_error", error=str(exc))
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
    parser = argparse.ArgumentParser(description="NERO Gate 2 arm executor")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "audit", "execute"), default="plan")
    parser.add_argument("--approval-token")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/nero_gate2_arm_executor"))
    parser.add_argument("--urdf", type=Path, default=Path(DEFAULT_URDF))
    parser.add_argument("--external-image-topic")
    parser.add_argument("--wrist-image-topic")
    args = parser.parse_args(argv)

    plan = load_plan(args.plan)
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
        wrist_image_topic=args.wrist_image_topic,
    )


if __name__ == "__main__":
    raise SystemExit(main())
