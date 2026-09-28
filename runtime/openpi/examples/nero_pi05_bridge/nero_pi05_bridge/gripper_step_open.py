"""Open a NERO gripper from any measured width in bounded steps.

Gate 1 open plans start at 0.06 m. A policy-closed gripper at ~0.001 m is
~0.059 m away, which exceeds ``max_step_delta_m`` so the calibration CLI
refuses to publish. This command walks ``width += 0.03`` up to the driver
ceiling (0.1 m) and is meant for experiment-loop recovery, not Gate 1 audit.
"""

from __future__ import annotations

import argparse
import json
import time

DRIVER_WIDTH_MAX_M = 0.1
DEFAULT_STEP_M = 0.03
DEFAULT_OPEN_MIN_M = 0.09
DEFAULT_RIGHT_TARGET_M = 0.0996
DEFAULT_LEFT_TARGET_M = 0.1


def default_target_m(namespace: str) -> float:
    if namespace.rstrip("/").endswith("left_arm"):
        return DEFAULT_LEFT_TARGET_M
    return DEFAULT_RIGHT_TARGET_M


def step_targets(
    current_m: float,
    target_m: float,
    *,
    step_m: float = DEFAULT_STEP_M,
) -> list[float]:
    """Inclusive increasing steps from ``current_m`` to ``target_m``."""
    if step_m <= 0:
        raise ValueError("step_m must be positive")
    target = min(float(target_m), DRIVER_WIDTH_MAX_M)
    current = float(current_m)
    if current + 1e-12 >= target:
        return []
    out: list[float] = []
    width = current
    while width + 1e-12 < target:
        width = min(target, width + step_m)
        out.append(width)
    return out


def run_open(
    *,
    namespace: str,
    target_m: float,
    step_m: float,
    open_min_m: float,
    timeout_s: float,
) -> dict:
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_srvs.srv import SetBool

    target_m = min(float(target_m), DRIVER_WIDTH_MAX_M)
    command_topic = f"{namespace}/control/joint_states"
    summary: dict = {
        "arm_namespace": namespace,
        "target_m": target_m,
        "step_m": step_m,
        "ok": False,
    }

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    class OpenNode(Node):
        def __init__(self) -> None:
            super().__init__("nero_gripper_step_open")
            self.latest: JointState | None = None
            self.create_subscription(
                JointState, f"{namespace}/feedback/joint_states", self._on_joints, 10
            )
            self.gate_client = self.create_client(SetBool, f"{namespace}/control_enable")
            self.enable_client = self.create_client(SetBool, f"{namespace}/enable_agx_arm")
            self.command_pub = None

        def _on_joints(self, message: JointState) -> None:
            self.latest = message

        def width(self, timeout: float = 5.0) -> float:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.latest is None:
                    continue
                mapping = dict(zip(self.latest.name, self.latest.position))
                if "gripper" in mapping:
                    return float(mapping["gripper"])
            raise RuntimeError(f"{namespace} no gripper feedback")

        def call_set_bool(self, client, *, value: bool, label: str) -> None:
            if not client.wait_for_service(timeout_sec=3.0):
                raise RuntimeError(f"{label} unavailable")
            request = SetBool.Request()
            request.data = value
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=6.0)
            if not future.done() or future.result() is None or not future.result().success:
                raise RuntimeError(f"{label} failed")

        def publish_width(self, width_m: float) -> None:
            if self.command_pub is None:
                if self.count_publishers(command_topic) != 0:
                    raise RuntimeError(f"{command_topic} has publishers")
                self.command_pub = self.create_publisher(JointState, command_topic, 1)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and self.count_subscribers(command_topic) < 1:
                    rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.2)
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = ["gripper"]
            message.position = [min(width_m, DRIVER_WIDTH_MAX_M)]
            message.effort = [1.0]
            self.command_pub.publish(message)

    node = OpenNode()
    gate_open = False
    try:
        before = node.width()
        summary["before_m"] = before
        if before >= open_min_m:
            summary["ok"] = True
            summary["skipped"] = True
            summary["after_m"] = before
            print(json.dumps(summary, indent=2))
            return summary

        targets = step_targets(before, target_m, step_m=step_m)
        summary["targets_m"] = targets
        node.call_set_bool(node.enable_client, value=True, label="enable_agx_arm")
        node.call_set_bool(node.gate_client, value=True, label="control_enable")
        gate_open = True

        after = before
        deadline = time.monotonic() + timeout_s
        for width_m in targets:
            while time.monotonic() < deadline:
                node.publish_width(width_m)
                rclpy.spin_once(node, timeout_sec=0.1)
                after = node.width(timeout=1.0)
                if after + 1e-12 >= min(width_m, open_min_m) - 0.005:
                    break
            if after >= open_min_m:
                break
        if after < open_min_m:
            raise RuntimeError(
                f"{namespace} still closed after stepped open "
                f"(after={after:.4f} m, need >= {open_min_m:.3f} m)"
            )
        summary["ok"] = True
        summary["after_m"] = after
        print(json.dumps(summary, indent=2))
        return summary
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(summary, indent=2))
        raise
    finally:
        if gate_open:
            try:
                node.call_set_bool(node.gate_client, value=False, label="control_enable_close")
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open a NERO gripper from any width in 3 cm steps"
    )
    parser.add_argument("--arm-namespace", required=True)
    parser.add_argument("--target-m", type=float)
    parser.add_argument("--step-m", type=float, default=DEFAULT_STEP_M)
    parser.add_argument("--open-min-m", type=float, default=DEFAULT_OPEN_MIN_M)
    parser.add_argument("--timeout-s", type=float, default=8.0)
    args = parser.parse_args(argv)

    namespace = args.arm_namespace
    if not namespace.startswith("/") or namespace.endswith("/") or "//" in namespace:
        parser.error("arm-namespace must look like /right_arm")
    target = (
        float(args.target_m) if args.target_m is not None else default_target_m(namespace)
    )
    if not (0 < target <= DRIVER_WIDTH_MAX_M):
        parser.error(f"target-m must be in (0, {DRIVER_WIDTH_MAX_M}]")
    try:
        result = run_open(
            namespace=namespace,
            target_m=target,
            step_m=float(args.step_m),
            open_min_m=float(args.open_min_m),
            timeout_s=float(args.timeout_s),
        )
    except Exception:
        return 2
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
