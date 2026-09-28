"""CLI to park and disable the NERO arm after an inference run.

Runs a fixed two-waypoint sequence through ``{ns}/control/move_j`` and then drops
the arm enable. The second waypoint puts joint4 (wrist pitch) at 0 rad on purpose:
disabling the arm removes the holding torque, and from a raised wrist the arm falls
80 deg or more, whereas from a flat wrist there is nothing left to fall.

The two waypoints are deliberately not a single move. Leg 1 yaws the base clear of
the workspace with the wrist still raised, so lowering the wrist in leg 2 cannot
sweep the gripper across whatever the arm was just working on.

This is a separate command from the Gate 3 executor: inference and shutdown are run
independently, so a failed run never leaves parking half-done and parking never
needs a policy server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from nero_pi05_bridge.nero_home_cli import DEFAULT_URDF
from nero_pi05_bridge.nero_home_cli import _parse_joints
from nero_pi05_bridge.nero_home_cli import validate_home_target
from nero_pi05_bridge.offline_adapter import NERO_JOINT_NAMES

# Leg 1 yaws the base away from the table with the wrist held up; leg 2 returns the
# base to zero and lays the wrist flat so the disable in step 3 causes no drop.
DEFAULT_PARK_WAYPOINTS_DEG = (
    [-20.0, 90.0, 90.0, 90.0, 0.0, 0.0, 0.0],
    [0.0, 90.0, 90.0, 0.0, 0.0, 0.0, 0.0],
)
# move_j holds ~2.6 deg (0.046 rad) short of a commanded joint4 angle while fighting
# gravity, so a tighter tolerance than this reports a timeout on a correct park.
DEFAULT_TOLERANCE_RAD = 0.05
DEFAULT_SOFT_MARGIN_RAD = 0.05
DEFAULT_LEG_TIMEOUT_S = 60.0


def _waypoints_rad(degrees: list[list[float]] | None = None) -> list[np.ndarray]:
    source = [list(w) for w in DEFAULT_PARK_WAYPOINTS_DEG] if degrees is None else degrees
    return [np.deg2rad(np.asarray(w, dtype=np.float64)) for w in source]


def park_fingerprint(
    *, namespace: str, waypoints_rad: list[np.ndarray], tolerance_rad: float
) -> str:
    payload = {
        "arm_namespace": namespace,
        "disable_after": True,
        "tolerance_rad": round(float(tolerance_rad), 9),
        "topic": "control/move_j",
        "waypoints_rad": [[round(float(v), 9) for v in w] for w in waypoints_rad],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def approval_token(
    *, namespace: str, waypoints_rad: list[np.ndarray], tolerance_rad: float
) -> str:
    digest = park_fingerprint(
        namespace=namespace, waypoints_rad=waypoints_rad, tolerance_rad=tolerance_rad
    )
    return f"NERO_PARK_MOVE_J_{digest[:16]}"


def validate_park_sequence(
    waypoints_rad: list[np.ndarray],
    *,
    soft_margin_rad: float = DEFAULT_SOFT_MARGIN_RAD,
) -> list[dict[str, Any]]:
    """Validate every waypoint before any motion is commanded."""
    if len(waypoints_rad) < 1:
        raise ValueError("park sequence needs at least one waypoint")
    return [
        validate_home_target(waypoint, soft_margin_rad=soft_margin_rad)
        for waypoint in waypoints_rad
    ]


def run_park(
    *,
    mode: str,
    namespace: str,
    waypoints_rad: list[np.ndarray],
    tolerance_rad: float,
    leg_timeout_s: float,
    soft_margin_rad: float,
    approval: str | None,
) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Empty as EmptySrv
    from std_srvs.srv import SetBool

    token = approval_token(
        namespace=namespace, waypoints_rad=waypoints_rad, tolerance_rad=tolerance_rad
    )
    targets = validate_park_sequence(waypoints_rad, soft_margin_rad=soft_margin_rad)
    summary: dict[str, Any] = {
        "mode": mode,
        "arm_namespace": namespace,
        "move_j_topic": f"{namespace}/control/move_j",
        "tolerance_rad": tolerance_rad,
        "required_approval_token": token,
        "waypoints_deg": [t["home_joints_deg"] for t in targets],
        "legs": [],
    }

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    class ParkNode(Node):
        def __init__(self) -> None:
            super().__init__("nero_park_cli")
            self.latest: JointState | None = None
            self.create_subscription(
                JointState, f"{namespace}/feedback/joint_states", self._on_joints, 10
            )
            self.move_j_pub = None
            self.gate_client = self.create_client(SetBool, f"{namespace}/control_enable")
            self.enable_client = self.create_client(SetBool, f"{namespace}/enable_agx_arm")
            self.estop_client = self.create_client(EmptySrv, f"{namespace}/emergency_stop")

        def _on_joints(self, message: JointState) -> None:
            self.latest = message

        def read_joints(self, timeout_s: float = 10.0) -> np.ndarray:
            deadline = time.monotonic() + timeout_s
            self.latest = None
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.latest is not None:
                    return _parse_joints(list(self.latest.name), self.latest.position)
            raise RuntimeError("timed out waiting for joint feedback")

        def call_set_bool(self, client, *, value: bool, label: str) -> None:
            if not client.wait_for_service(timeout_sec=3.0):
                raise RuntimeError(f"{label} service unavailable")
            request = SetBool.Request()
            request.data = value
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=6.0)
            if not future.done() or future.result() is None or not future.result().success:
                raise RuntimeError(f"{label} failed")

        def call_estop(self) -> None:
            if not self.estop_client.wait_for_service(timeout_sec=1.0):
                return
            future = self.estop_client.call_async(EmptySrv.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        def publish_move_j(self, joints: np.ndarray) -> None:
            if self.move_j_pub is None:
                self.move_j_pub = self.create_publisher(
                    JointState, f"{namespace}/control/move_j", 1
                )
                # Wait for DDS discovery so the first command is not dropped.
                deadline = time.monotonic() + 2.0
                while (
                    time.monotonic() < deadline
                    and self.count_subscribers(f"{namespace}/control/move_j") < 1
                ):
                    rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.2)
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            # Must include all 7 joints; driver fills missing names with 0.0.
            message.name = list(NERO_JOINT_NAMES)
            message.position = [float(v) for v in joints]
            self.move_j_pub.publish(message)

        def move_to(self, target: np.ndarray, *, timeout_s: float, tolerance: float):
            self.publish_move_j(target)
            last_pub = time.monotonic()
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                # Re-send move_j periodically; single-shot publish can be dropped.
                if time.monotonic() - last_pub > 1.0:
                    self.publish_move_j(target)
                    last_pub = time.monotonic()
                if self.latest is None:
                    continue
                current = _parse_joints(list(self.latest.name), self.latest.position)
                if float(np.max(np.abs(current - target))) > tolerance:
                    continue
                # Require two consecutive samples inside tolerance to avoid false settle.
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.latest is None:
                    continue
                confirm = _parse_joints(list(self.latest.name), self.latest.position)
                if float(np.max(np.abs(confirm - target))) <= tolerance:
                    return confirm
            return None

    node = ParkNode()
    gate_open = False
    arm_enabled = False
    parked = False
    try:
        current = node.read_joints()
        summary["start_joints_deg"] = np.rad2deg(current).tolist()
        if mode == "check":
            print(json.dumps(summary, indent=2))
            return 0

        if approval != token:
            raise PermissionError("approval token does not match this park sequence fingerprint")

        node.call_set_bool(node.enable_client, value=True, label="enable_agx_arm")
        arm_enabled = True
        node.call_set_bool(node.gate_client, value=True, label="control_enable")
        gate_open = True

        for index, target in enumerate(waypoints_rad, start=1):
            settled = node.move_to(target, timeout_s=leg_timeout_s, tolerance=tolerance_rad)
            leg: dict[str, Any] = {
                "leg": index,
                "target_deg": np.rad2deg(target).tolist(),
                "ok": settled is not None,
            }
            if settled is None:
                leg["error"] = f"leg {index} timeout after {leg_timeout_s:.1f}s"
                summary["legs"].append(leg)
                node.call_estop()
                summary["ok"] = False
                # Leave the arm enabled: dropping torque at an unknown pose is worse
                # than holding it for the operator.
                summary["arm_left_enabled"] = True
                summary["error"] = leg["error"]
                print(json.dumps(summary, indent=2))
                return 2
            leg["settled_deg"] = np.rad2deg(settled).tolist()
            leg["max_abs_delta_rad"] = float(np.max(np.abs(settled - target)))
            summary["legs"].append(leg)

        parked = True
        node.call_set_bool(node.gate_client, value=False, label="control_enable_close")
        gate_open = False
        node.call_set_bool(node.enable_client, value=False, label="enable_agx_arm_disable")
        arm_enabled = False

        # Report where the arm actually comes to rest once holding torque is gone.
        settle_deadline = time.monotonic() + 3.0
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        resting = node.read_joints()
        summary["ok"] = True
        summary["arm_left_enabled"] = False
        summary["resting_joints_deg"] = np.rad2deg(resting).tolist()
        summary["drop_from_last_waypoint_deg"] = float(
            np.max(np.abs(np.rad2deg(resting - waypoints_rad[-1])))
        )
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(summary, indent=2))
        try:
            node.call_estop()
        except Exception:
            pass
        return 2
    finally:
        if gate_open:
            try:
                node.call_set_bool(node.gate_client, value=False, label="control_enable_close")
            except Exception:
                pass
        # A park that never reached its last waypoint keeps the arm enabled on purpose.
        if arm_enabled and parked:
            try:
                node.call_set_bool(node.enable_client, value=False, label="enable_agx_arm_disable")
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Park the NERO arm through two waypoints and disable it"
    )
    parser.add_argument("--mode", choices=("check", "park", "plan"), default="check")
    parser.add_argument("--arm-namespace", default="/right_arm")
    parser.add_argument("--approval-token")
    parser.add_argument("--tolerance-rad", type=float, default=DEFAULT_TOLERANCE_RAD)
    parser.add_argument("--leg-timeout-s", type=float, default=DEFAULT_LEG_TIMEOUT_S)
    parser.add_argument("--soft-limit-margin-rad", type=float, default=DEFAULT_SOFT_MARGIN_RAD)
    parser.add_argument("--urdf", type=Path, default=Path(DEFAULT_URDF))
    parser.add_argument(
        "--leg1-deg",
        type=float,
        nargs=7,
        default=list(DEFAULT_PARK_WAYPOINTS_DEG[0]),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    parser.add_argument(
        "--leg2-deg",
        type=float,
        nargs=7,
        default=list(DEFAULT_PARK_WAYPOINTS_DEG[1]),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    args = parser.parse_args(argv)

    namespace = args.arm_namespace
    if not namespace.startswith("/") or namespace.endswith("/") or "//" in namespace:
        parser.error("arm-namespace must look like /right_arm")
    waypoints = _waypoints_rad([list(args.leg1_deg), list(args.leg2_deg)])
    if any(not math.isfinite(v) for w in waypoints for v in w):
        parser.error("park waypoints must be finite")

    token = approval_token(
        namespace=namespace, waypoints_rad=waypoints, tolerance_rad=args.tolerance_rad
    )
    if args.mode == "plan":
        try:
            validate_park_sequence(waypoints, soft_margin_rad=args.soft_limit_margin_rad)
            within_limits = True
            limit_error = None
        except ValueError as exc:
            within_limits = False
            limit_error = str(exc)
        print(
            json.dumps(
                {
                    "arm_namespace": namespace,
                    "waypoints_deg": [list(args.leg1_deg), list(args.leg2_deg)],
                    "waypoints_rad": [w.tolist() for w in waypoints],
                    "tolerance_rad": args.tolerance_rad,
                    "move_j_topic": f"{namespace}/control/move_j",
                    "disable_after": True,
                    "within_soft_limits": within_limits,
                    "limit_error": limit_error,
                    "required_approval_token": token,
                },
                indent=2,
            )
        )
        return 0 if within_limits else 1

    if args.mode == "park" and not args.approval_token:
        parser.error("--approval-token is required for --mode park")
    if args.mode == "check" and args.approval_token:
        parser.error("--approval-token is accepted only in park mode")

    return run_park(
        mode=args.mode,
        namespace=namespace,
        waypoints_rad=waypoints,
        tolerance_rad=args.tolerance_rad,
        leg_timeout_s=args.leg_timeout_s,
        soft_margin_rad=args.soft_limit_margin_rad,
        approval=args.approval_token.strip() if args.approval_token else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
