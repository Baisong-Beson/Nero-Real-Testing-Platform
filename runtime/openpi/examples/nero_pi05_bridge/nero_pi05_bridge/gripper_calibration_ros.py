"""ROS 2 runtime for fail-closed NERO Gate 1 gripper calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path
import signal
import statistics
import time
from typing import Any

from agx_arm_msgs.msg import AgxArmStatus
from agx_arm_msgs.msg import GripperStatus
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

from nero_pi05_bridge.gripper_calibration import CalibrationPlan
from nero_pi05_bridge.gripper_calibration import ExecutionAuthorization
from nero_pi05_bridge.gripper_calibration import check_feedback_freshness
from nero_pi05_bridge.gripper_calibration import check_feedback_health
from nero_pi05_bridge.gripper_calibration import max_abs_delta
from nero_pi05_bridge.gripper_calibration import max_width_speed
from nero_pi05_bridge.gripper_calibration import measured_step_limit_m

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
GRIPPER_FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "driver_error_status",
)


class ResultWriter:
    def __init__(self, output_root: Path, *, mode: str, plan: CalibrationPlan):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_suffix = f"{time.time_ns() % 1_000_000_000:09d}"
        self.directory = output_root / (f"{timestamp}_{run_suffix}_{mode}_{plan.fingerprint[:8]}")
        self.directory.mkdir(parents=True, exist_ok=False)
        self._events = (self.directory / "events.jsonl").open("w")

    def event(self, name: str, **values: Any) -> None:
        record = {
            "monotonic_s": time.monotonic(),
            "wall_time_s": time.time(),
            "event": name,
            **values,
        }
        self._events.write(json.dumps(record, sort_keys=True) + "\n")
        self._events.flush()

    def finish(self, result: dict[str, Any]) -> Path:
        self._events.close()
        destination = self.directory / "summary.json"
        temporary = self.directory / "summary.json.partial"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)
        return destination


class CalibrationNode(Node):
    def __init__(self, plan: CalibrationPlan):
        super().__init__("nero_gate1_gripper_calibration")
        namespace = plan.arm_namespace
        self.command_topic = f"{namespace}/control/joint_states"
        self.joints: JointState | None = None
        self.tcp: PoseStamped | None = None
        self.gripper: GripperStatus | None = None
        self.arm_status: AgxArmStatus | None = None
        self.feedback_times: dict[str, float] = {}
        self.gripper_samples: list[tuple[float, float, float]] = []
        self.abort_requested = False
        self.gate_client = None
        self.enable_client = None
        self.command_publisher = None
        self.create_subscription(
            JointState,
            f"{namespace}/feedback/joint_states",
            self._on_joints,
            10,
        )
        self.create_subscription(
            PoseStamped,
            f"{namespace}/feedback/tcp_pose",
            self._on_tcp,
            10,
        )
        self.create_subscription(
            GripperStatus,
            f"{namespace}/feedback/gripper_status",
            self._on_gripper,
            10,
        )
        self.create_subscription(
            AgxArmStatus,
            f"{namespace}/feedback/arm_status",
            self._on_arm_status,
            10,
        )

    def _on_joints(self, message: JointState) -> None:
        self.joints = message
        self.feedback_times["joints"] = time.monotonic()

    def _on_tcp(self, message: PoseStamped) -> None:
        self.tcp = message
        self.feedback_times["tcp"] = time.monotonic()

    def _on_gripper(self, message: GripperStatus) -> None:
        now = time.monotonic()
        self.gripper = message
        self.feedback_times["gripper"] = now
        self.gripper_samples.append((now, float(message.width), float(message.force)))

    def _on_arm_status(self, message: AgxArmStatus) -> None:
        self.arm_status = message
        self.feedback_times["arm_status"] = time.monotonic()

    def wait_for_feedback(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.check_abort()
            rclpy.spin_once(self, timeout_sec=0.05)
            if all(value is not None for value in (self.joints, self.tcp, self.gripper, self.arm_status)):
                self.assert_fresh()
                return
        raise RuntimeError("timed out waiting for complete feedback")

    def wait_for_fresh_feedback(self, timeout_s: float, *, maximum_age_s: float = 0.5) -> None:
        """Spin until every required topic updates after this call starts.

        enable_agx_arm / control_enable can stall driver feedback for >0.5s while
        spin_until_future_complete is blocked on the service response. Cached
        snapshots then look stale even though the node is healthy again.
        """
        started = time.monotonic()
        deadline = started + timeout_s
        required = ("joints", "tcp", "gripper", "arm_status")
        while time.monotonic() < deadline:
            self.check_abort()
            rclpy.spin_once(self, timeout_sec=0.05)
            if all(
                (timestamp := self.feedback_times.get(name)) is not None and timestamp >= started
                for name in required
            ):
                self.assert_fresh(maximum_age_s)
                return
        raise RuntimeError("timed out waiting for fresh feedback after service/stall")

    def assert_fresh(self, maximum_age_s: float = 0.5) -> None:
        check_feedback_freshness(
            self.feedback_times,
            now=time.monotonic(),
            required={"joints", "tcp", "gripper", "arm_status"},
            maximum_age_s=maximum_age_s,
        )

    def check_abort(self) -> None:
        if self.abort_requested:
            raise KeyboardInterrupt("operator interrupt requested")

    def enable_execution_interfaces(self, plan: CalibrationPlan) -> None:
        if self.gate_client is not None or self.enable_client is not None:
            raise RuntimeError("execution interfaces already created")
        namespace = plan.arm_namespace
        self.gate_client = self.create_client(SetBool, f"{namespace}/control_enable")
        self.enable_client = self.create_client(SetBool, f"{namespace}/enable_agx_arm")

    def create_command_publisher(self) -> None:
        if self.command_publisher is not None:
            raise RuntimeError("command publisher already created")
        self.command_publisher = self.create_publisher(JointState, self.command_topic, 1)

    def call_set_bool(
        self,
        client,
        *,
        value: bool,
        label: str,
        honor_abort: bool = True,
    ) -> str:
        if honor_abort:
            self.check_abort()
        if client is None or not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f"{label} service unavailable")
        request = SetBool.Request()
        request.data = value
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=6.0)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"{label} service timed out")
        response = future.result()
        if not response.success:
            raise RuntimeError(f"{label} failed: {response.message}")
        return response.message


def arm_vector(message: JointState | None) -> list[float]:
    if message is None or len(message.name) != len(message.position):
        raise RuntimeError("invalid arm JointState feedback")
    positions = dict(zip(message.name, message.position, strict=True))
    missing = [name for name in ARM_JOINT_NAMES if name not in positions]
    if missing:
        raise RuntimeError(f"arm JointState is missing {missing}")
    result = [float(positions[name]) for name in ARM_JOINT_NAMES]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("arm JointState contains NaN or Inf")
    return result


def max_arm_velocity(message: JointState | None) -> float:
    if message is None or not message.velocity:
        return 0.0
    if len(message.name) != len(message.velocity):
        raise RuntimeError("arm JointState velocity shape mismatch")
    velocities = dict(zip(message.name, message.velocity, strict=True))
    values = [abs(float(velocities.get(name, 0.0))) for name in ARM_JOINT_NAMES]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("arm velocity contains NaN or Inf")
    return max(values)


def tcp_vector(message: PoseStamped | None) -> list[float]:
    if message is None:
        raise RuntimeError("missing TCP feedback")
    position = message.pose.position
    result = [float(position.x), float(position.y), float(position.z)]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("TCP feedback contains NaN or Inf")
    return result


def gripper_faults(message: GripperStatus | None) -> list[str]:
    if message is None:
        raise RuntimeError("missing gripper feedback")
    return [field for field in GRIPPER_FAULT_FIELDS if bool(getattr(message, field))]


def feedback_snapshot(node: CalibrationNode) -> dict[str, Any]:
    if node.gripper is None or node.arm_status is None:
        raise RuntimeError("feedback is incomplete")
    gripper_width = float(node.gripper.width)
    gripper_force = float(node.gripper.force)
    if not math.isfinite(gripper_width) or not math.isfinite(gripper_force):
        raise RuntimeError("gripper feedback contains NaN or Inf")
    if not 0.0 <= gripper_width <= 0.1:
        raise RuntimeError(f"gripper feedback width is outside [0.0, 0.1]: {gripper_width}")
    return {
        "joints_rad": arm_vector(node.joints),
        "tcp_xyz_m": tcp_vector(node.tcp),
        "gripper_width_m": gripper_width,
        "gripper_force_n": gripper_force,
        "gripper_faults": gripper_faults(node.gripper),
        "gripper_driver_enabled": bool(node.gripper.driver_enable_status),
        "arm_motion_status": int(node.arm_status.motion_status),
        "arm_err_status": int(node.arm_status.err_status),
    }


def assert_healthy(node: CalibrationNode) -> None:
    node.assert_fresh()
    check_feedback_health(feedback_snapshot(node))


def new_motion_metrics() -> dict[str, float]:
    return {
        "max_joint_delta_rad": 0.0,
        "max_tcp_delta_m": 0.0,
        "max_arm_velocity_abs_rad_s": 0.0,
    }


def observe(
    node: CalibrationNode,
    duration_s: float,
    baseline_joints: list[float],
    baseline_tcp: list[float],
    metrics: dict[str, float],
) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        node.check_abort()
        rclpy.spin_once(node, timeout_sec=min(0.02, max(0.0, deadline - time.monotonic())))
        assert_healthy(node)
        metrics["max_joint_delta_rad"] = max(
            metrics["max_joint_delta_rad"],
            max_abs_delta(arm_vector(node.joints), baseline_joints),
        )
        metrics["max_tcp_delta_m"] = max(
            metrics["max_tcp_delta_m"],
            max_abs_delta(tcp_vector(node.tcp), baseline_tcp),
        )
        metrics["max_arm_velocity_abs_rad_s"] = max(
            metrics["max_arm_velocity_abs_rad_s"],
            max_arm_velocity(node.joints),
        )


def enforce_static(plan: CalibrationPlan, metrics: dict[str, float]) -> None:
    if metrics["max_joint_delta_rad"] > plan.safety.joint_static_tolerance_rad:
        raise RuntimeError(f"arm joint movement exceeded threshold: {metrics['max_joint_delta_rad']:.9f} rad")
    if metrics["max_tcp_delta_m"] > plan.safety.tcp_static_tolerance_m:
        raise RuntimeError(f"TCP movement exceeded threshold: {metrics['max_tcp_delta_m']:.9f} m")


def run_audit(
    node: CalibrationNode,
    plan: CalibrationPlan,
    writer: ResultWriter,
    result: dict[str, Any],
) -> None:
    publisher_count = node.count_publishers(node.command_topic)
    result["preflight_command_publisher_count"] = publisher_count
    if publisher_count != 0:
        raise RuntimeError(f"command topic already has {publisher_count} publisher(s)")
    node.wait_for_feedback(plan.safety.feedback_timeout_s)
    assert_healthy(node)
    result["initial_feedback"] = feedback_snapshot(node)
    baseline_joints = arm_vector(node.joints)
    baseline_tcp = tcp_vector(node.tcp)
    metrics = new_motion_metrics()
    writer.event("audit_baseline_started", duration_s=plan.safety.baseline_duration_s)
    observe(
        node,
        plan.safety.baseline_duration_s,
        baseline_joints,
        baseline_tcp,
        metrics,
    )
    enforce_static(plan, metrics)
    result["audit_motion_metrics"] = metrics
    result["audit_feedback"] = feedback_snapshot(node)
    result["audit_passed"] = True
    writer.event("audit_passed", **metrics)


def publish_one_target(
    node: CalibrationNode,
    plan: CalibrationPlan,
    writer: ResultWriter,
    baseline_joints: list[float],
    baseline_tcp: list[float],
    metrics: dict[str, float],
    *,
    target_index: int,
) -> None:
    target = plan.targets[target_index]
    writer.event(
        "command_gate_open_requested",
        target_index=target_index,
        width_m=target.width_m,
        effort_n=target.effort_n,
    )
    node.call_set_bool(
        node.gate_client,
        value=True,
        label=f"open control gate for target {target_index}",
    )
    try:
        command = JointState()
        command.header.stamp = node.get_clock().now().to_msg()
        command.name = ["gripper"]
        command.position = [target.width_m]
        command.effort = [target.effort_n]
        node.command_publisher.publish(command)
        writer.event(
            "single_gripper_command_published",
            target_index=target_index,
            width_m=target.width_m,
            effort_n=target.effort_n,
        )
        observe(
            node,
            plan.safety.gate_open_duration_s,
            baseline_joints,
            baseline_tcp,
            metrics,
        )
        enforce_static(plan, metrics)
    finally:
        node.call_set_bool(
            node.gate_client,
            value=False,
            label=f"close control gate for target {target_index}",
            honor_abort=False,
        )
        writer.event("command_gate_closed", target_index=target_index)


def run_execute(
    node: CalibrationNode,
    plan: CalibrationPlan,
    writer: ResultWriter,
    result: dict[str, Any],
) -> None:
    run_audit(node, plan, writer, result)
    initial_width = float(node.gripper.width)
    step_limit_m = measured_step_limit_m(plan.safety)
    if abs(plan.targets[0].width_m - initial_width) > step_limit_m + 1e-12:
        raise RuntimeError(
            "current width to first target exceeds measured step limit "
            f"({abs(plan.targets[0].width_m - initial_width):.6f} m > {step_limit_m:.6f} m)"
        )

    baseline_joints = arm_vector(node.joints)
    baseline_tcp = tcp_vector(node.tcp)
    metrics = new_motion_metrics()
    node.enable_execution_interfaces(plan)
    writer.event("execution_interfaces_created")
    node.call_set_bool(node.gate_client, value=False, label="preflight close control gate")
    result["arm_enable_message"] = node.call_set_bool(
        node.enable_client,
        value=True,
        label="enable arm for Gate 1",
    )
    result["arm_was_enabled_by_program"] = True
    writer.event("arm_enabled_for_gate1")
    # Driver often pauses feedback during enable; reacquire before static checks.
    node.wait_for_fresh_feedback(plan.safety.feedback_timeout_s)
    writer.event("feedback_reacquired_after_enable")
    observe(
        node,
        plan.safety.baseline_duration_s,
        baseline_joints,
        baseline_tcp,
        metrics,
    )
    enforce_static(plan, metrics)

    node.create_command_publisher()
    deadline = time.monotonic() + plan.safety.feedback_timeout_s
    while node.count_subscribers(node.command_topic) < 1 and time.monotonic() < deadline:
        node.check_abort()
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.count_subscribers(node.command_topic) < 1:
        raise RuntimeError("gripper command topic has no subscriber")
    if node.count_publishers(node.command_topic) != 1:
        raise RuntimeError("unexpected additional command publisher appeared")

    result["steps"] = []
    for target_index, target in enumerate(plan.targets):
        current_width = float(node.gripper.width)
        step_limit_m = measured_step_limit_m(plan.safety)
        if abs(target.width_m - current_width) > step_limit_m + 1e-12:
            raise RuntimeError(
                f"current width to target {target_index} exceeds measured step limit "
                f"({abs(target.width_m - current_width):.6f} m > {step_limit_m:.6f} m)"
            )
        sample_start = len(node.gripper_samples)
        step_started = time.monotonic()
        publish_one_target(
            node,
            plan,
            writer,
            baseline_joints,
            baseline_tcp,
            metrics,
            target_index=target_index,
        )
        reached = False
        while time.monotonic() - step_started < plan.safety.step_timeout_s:
            observe(node, 0.05, baseline_joints, baseline_tcp, metrics)
            enforce_static(plan, metrics)
            if abs(float(node.gripper.width) - target.width_m) <= plan.safety.gripper_tolerance_m:
                reached = True
                break
        if not reached:
            raise RuntimeError(f"target {target_index} timed out at width={node.gripper.width:.6f} m")
        arrival_s = time.monotonic() - step_started
        settle_start = len(node.gripper_samples)
        observe(
            node,
            plan.safety.settle_duration_s,
            baseline_joints,
            baseline_tcp,
            metrics,
        )
        enforce_static(plan, metrics)
        settled = node.gripper_samples[settle_start:]
        if len(settled) < 2:
            raise RuntimeError("too few settled gripper samples")
        widths = [sample[1] for sample in settled]
        forces = [sample[2] for sample in settled]
        step_samples = node.gripper_samples[sample_start:]
        step_result = {
            "target_index": target_index,
            "target_width_m": target.width_m,
            "target_effort_n": target.effort_n,
            "arrival_s": arrival_s,
            "settled_mean_m": statistics.fmean(widths),
            "settled_std_m": statistics.pstdev(widths),
            "settled_error_m": statistics.fmean(widths) - target.width_m,
            "settled_force_mean_n": statistics.fmean(forces),
            "max_width_speed_m_s": max_width_speed(step_samples),
            "sample_count": len(step_samples),
        }
        result["steps"].append(step_result)
        writer.event("target_reached_and_settled", **step_result)
    result["execution_motion_metrics"] = metrics
    result["sequence_complete"] = True


def run_calibration(
    plan: CalibrationPlan,
    *,
    mode: str,
    output_dir: Path,
    authorization: ExecutionAuthorization | None,
    keep_arm_enabled: bool = False,
) -> int:
    if mode not in {"audit", "execute"}:
        raise ValueError(f"unsupported ROS calibration mode: {mode}")
    if mode == "execute" and (authorization is None or authorization.plan_fingerprint != plan.fingerprint):
        raise PermissionError("ROS execution requires authorization for this exact plan")
    if mode == "audit" and authorization is not None:
        raise PermissionError("audit mode must not receive execution authorization")
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CalibrationNode(plan)
    writer = ResultWriter(output_dir, mode=mode, plan=plan)
    result: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "plan": plan.summary(),
        "commandable": mode == "execute",
        "subscriptions_only": mode == "audit",
        "audit_passed": False,
        "sequence_complete": False,
        "cleanup_gate_closed": False,
        "cleanup_arm_disabled": False,
        "arm_was_enabled_by_program": False,
    }
    previous_handlers = {}

    def request_abort(signum, _frame):
        node.abort_requested = True
        writer.event("operator_interrupt", signal=signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_abort)

    exit_code = 1
    try:
        writer.event("run_started", mode=mode, plan_fingerprint=plan.fingerprint)
        if mode == "audit":
            run_audit(node, plan, writer, result)
        else:
            run_execute(node, plan, writer, result)
        exit_code = 0
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        writer.event("run_failed", error_type=type(error).__name__, error=str(error))
    finally:
        if mode == "execute" and node.gate_client is not None:
            try:
                node.call_set_bool(
                    node.gate_client,
                    value=False,
                    label="cleanup close gate",
                    honor_abort=False,
                )
                result["cleanup_gate_closed"] = True
                writer.event("cleanup_gate_closed")
            except BaseException as error:
                result["cleanup_gate_error"] = str(error)
                exit_code = 1
        if mode == "execute" and node.enable_client is not None:
            if keep_arm_enabled:
                # Experiment loop (and any follow-on home/park) needs the arm to
                # keep holding pose. Disabling here drops joint4 before the next
                # move_j can re-enable.
                result["cleanup_arm_disabled"] = False
                result["arm_left_enabled"] = True
                writer.event("arm_left_enabled")
            else:
                try:
                    node.call_set_bool(
                        node.enable_client,
                        value=False,
                        label="cleanup disable arm",
                        honor_abort=False,
                    )
                    result["cleanup_arm_disabled"] = True
                    writer.event("cleanup_arm_disabled")
                except BaseException as error:
                    result["cleanup_disable_error"] = str(error)
                    exit_code = 1
        try:
            node.wait_for_fresh_feedback(min(2.0, plan.safety.feedback_timeout_s))
            result["final_feedback"] = feedback_snapshot(node)
        except BaseException as error:
            result["final_feedback_error"] = str(error)
            if mode == "execute":
                exit_code = 1
        result["success"] = exit_code == 0
        summary_path = writer.finish(result)
        print(json.dumps({"summary": str(summary_path), "success": result["success"]}))
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        node.destroy_node()
        rclpy.shutdown()
    return exit_code
