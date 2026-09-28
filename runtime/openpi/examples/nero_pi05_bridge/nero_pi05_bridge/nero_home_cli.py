"""CLI to check / home the NERO arm to the Gate 3 approach start pose via move_j.

Publishes a complete 7-joint JointState to ``{ns}/control/move_j``. Missing joints
are treated as 0 rad by the driver, so this tool always fills all seven names.
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

from nero_pi05_bridge.offline_adapter import NERO_JOINT_NAMES
from nero_pi05_bridge.offline_adapter import NERO_POSITION_LOWER
from nero_pi05_bridge.offline_adapter import NERO_POSITION_UPPER

DEFAULT_URDF = str(Path(__file__).resolve().parents[5] / "runtime" / "robot" / "nero.urdf")
DEFAULT_HOME_DEG = [0.0, 90.0, 90.0, 90.0, 0.0, 0.0, 0.0]
DEFAULT_TOLERANCE_RAD = 0.05
DEFAULT_SOFT_MARGIN_RAD = 0.05


def _home_rad(degrees: list[float] | None = None) -> np.ndarray:
    deg = DEFAULT_HOME_DEG if degrees is None else degrees
    return np.deg2rad(np.asarray(deg, dtype=np.float64))


def home_fingerprint(*, namespace: str, joints_rad: np.ndarray, tolerance_rad: float) -> str:
    payload = {
        "arm_namespace": namespace,
        "home_joints_rad": [round(float(v), 9) for v in joints_rad],
        "tolerance_rad": round(float(tolerance_rad), 9),
        "topic": "control/move_j",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def approval_token(*, namespace: str, joints_rad: np.ndarray, tolerance_rad: float) -> str:
    return f"NERO_HOME_MOVE_J_{home_fingerprint(namespace=namespace, joints_rad=joints_rad, tolerance_rad=tolerance_rad)[:16]}"


def validate_home_target(
    joints_rad: np.ndarray,
    *,
    soft_margin_rad: float = DEFAULT_SOFT_MARGIN_RAD,
    workspace_aabb: dict[str, float] | None = None,
    urdf: Path | None = None,
) -> dict[str, Any]:
    joints = np.asarray(joints_rad, dtype=np.float64)
    if joints.shape != (7,):
        raise ValueError("home joints must have shape (7,)")
    if not np.all(np.isfinite(joints)):
        raise ValueError("home joints contain NaN or Inf")
    soft_lower = NERO_POSITION_LOWER + soft_margin_rad
    soft_upper = NERO_POSITION_UPPER - soft_margin_rad
    if np.any(joints < soft_lower) or np.any(joints > soft_upper):
        raise ValueError("home joints outside soft-limit envelope")
    result: dict[str, Any] = {
        "home_joints_rad": joints.tolist(),
        "home_joints_deg": np.rad2deg(joints).tolist(),
        "soft_lower": soft_lower.tolist(),
        "soft_upper": soft_upper.tolist(),
        "within_soft_limits": True,
    }
    if workspace_aabb is not None and urdf is not None:
        from nero_pi05_bridge.arm_executor import WorkspaceAABB
        from nero_pi05_bridge.nero_fk import NeroFK

        aabb = WorkspaceAABB(**{k: float(workspace_aabb[k]) for k in workspace_aabb})
        tcp = NeroFK(urdf).tcp_xyz(joints, 0.099)
        result["tcp_xyz"] = tcp.tolist()
        result["tcp_inside_aabb"] = aabb.contains(tcp)
        if not result["tcp_inside_aabb"]:
            raise ValueError(f"home TCP {tcp.tolist()} outside workspace AABB")
    return result


def _parse_joints(message_names: list[str], positions) -> np.ndarray:
    mapping = {name: float(positions[idx]) for idx, name in enumerate(message_names)}
    missing = [name for name in NERO_JOINT_NAMES if name not in mapping]
    if missing:
        raise RuntimeError(f"feedback missing joints: {missing}")
    return np.asarray([mapping[name] for name in NERO_JOINT_NAMES], dtype=np.float64)


def run_home(
    *,
    mode: str,
    namespace: str,
    joints_rad: np.ndarray,
    tolerance_rad: float,
    timeout_s: float,
    soft_margin_rad: float,
    workspace_aabb: dict[str, float] | None,
    urdf: Path | None,
    approval: str | None,
) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Empty as EmptySrv
    from std_srvs.srv import SetBool

    token = approval_token(namespace=namespace, joints_rad=joints_rad, tolerance_rad=tolerance_rad)
    target_info = validate_home_target(
        joints_rad,
        soft_margin_rad=soft_margin_rad,
        workspace_aabb=workspace_aabb,
        urdf=urdf,
    )
    summary = {
        "mode": mode,
        "arm_namespace": namespace,
        "move_j_topic": f"{namespace}/control/move_j",
        "tolerance_rad": tolerance_rad,
        "required_approval_token": token,
        **target_info,
    }

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    class HomeNode(Node):
        def __init__(self) -> None:
            super().__init__("nero_home_cli")
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

        def wait_feedback(self, timeout_s: float = 10.0) -> np.ndarray:
            deadline = time.monotonic() + timeout_s
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
                self.move_j_pub = self.create_publisher(JointState, f"{namespace}/control/move_j", 1)
                # Wait for DDS discovery so the first command is not dropped.
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and self.count_subscribers(
                    f"{namespace}/control/move_j"
                ) < 1:
                    rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.2)
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            # Must include all 7 joints; driver fills missing names with 0.0.
            message.name = list(NERO_JOINT_NAMES)
            message.position = [float(v) for v in joints]
            self.move_j_pub.publish(message)

    node = HomeNode()
    gate_open = False
    arm_enabled = False
    ok = False
    try:
        current = node.wait_feedback()
        delta = np.abs(current - joints_rad)
        summary["current_joints_rad"] = current.tolist()
        summary["abs_delta_rad"] = delta.tolist()
        summary["max_abs_delta_rad"] = float(delta.max())
        summary["within_tolerance"] = bool(float(delta.max()) <= tolerance_rad)
        print(json.dumps(summary, indent=2))
        if mode == "check":
            return 0 if summary["within_tolerance"] else 1

        if approval != token:
            raise PermissionError("approval token does not match this home target fingerprint")

        # Already home: still report success without commanding motion.
        if summary["within_tolerance"]:
            ok = True
            return 0

        node.call_set_bool(node.enable_client, value=True, label="enable_agx_arm")
        arm_enabled = True
        keep_arm_enabled = False
        node.call_set_bool(node.gate_client, value=True, label="control_enable")
        gate_open = True
        node.publish_move_j(joints_rad)
        last_pub = time.monotonic()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            # Re-send move_j periodically; single-shot publish can be dropped.
            if time.monotonic() - last_pub > 1.0:
                node.publish_move_j(joints_rad)
                last_pub = time.monotonic()
            if node.latest is None:
                continue
            current = _parse_joints(list(node.latest.name), node.latest.position)
            if float(np.max(np.abs(current - joints_rad))) <= tolerance_rad:
                # Require two consecutive samples inside tolerance to avoid false settle.
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.latest is None:
                    continue
                current2 = _parse_joints(list(node.latest.name), node.latest.position)
                if float(np.max(np.abs(current2 - joints_rad))) <= tolerance_rad:
                    ok = True
                    keep_arm_enabled = True  # hold pose for Gate 3; do not disable
                    summary["final_joints_rad"] = current2.tolist()
                    summary["final_max_abs_delta_rad"] = float(
                        np.max(np.abs(current2 - joints_rad))
                    )
                    break
        if not ok:
            node.call_estop()
            summary["error"] = f"home timeout after {timeout_s:.1f}s; emergency_stop called"
            print(json.dumps(summary, indent=2))
            return 2
        print(
            json.dumps(
                {
                    "ok": True,
                    "final_max_abs_delta_rad": summary["final_max_abs_delta_rad"],
                    "final_joints_rad": summary["final_joints_rad"],
                    "arm_left_enabled": True,
                },
                indent=2,
            )
        )
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
        # Only disable on failure. Successful home keeps enable so joints hold.
        if arm_enabled and not locals().get("keep_arm_enabled", False):
            try:
                node.call_set_bool(node.enable_client, value=False, label="enable_agx_arm_disable")
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NERO home via control/move_j")
    parser.add_argument("--mode", choices=("check", "home", "plan"), default="check")
    parser.add_argument("--arm-namespace", default="/right_arm")
    parser.add_argument("--approval-token")
    parser.add_argument("--tolerance-rad", type=float, default=DEFAULT_TOLERANCE_RAD)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--soft-limit-margin-rad", type=float, default=DEFAULT_SOFT_MARGIN_RAD)
    parser.add_argument("--urdf", type=Path, default=Path(DEFAULT_URDF))
    parser.add_argument(
        "--workspace-aabb-json",
        type=Path,
        help="Optional AABB JSON (Gate 2/3 workspace) to validate home TCP",
    )
    parser.add_argument(
        "--home-deg",
        type=float,
        nargs=7,
        default=DEFAULT_HOME_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    args = parser.parse_args(argv)

    namespace = args.arm_namespace
    if not namespace.startswith("/") or namespace.endswith("/") or "//" in namespace:
        parser.error("arm-namespace must look like /right_arm")
    joints = _home_rad(list(args.home_deg))
    if any(not math.isfinite(v) for v in joints):
        parser.error("home joints must be finite")
    token = approval_token(
        namespace=namespace, joints_rad=joints, tolerance_rad=args.tolerance_rad
    )
    if args.mode == "plan":
        print(
            json.dumps(
                {
                    "arm_namespace": namespace,
                    "home_joints_rad": joints.tolist(),
                    "home_joints_deg": list(args.home_deg),
                    "tolerance_rad": args.tolerance_rad,
                    "move_j_topic": f"{namespace}/control/move_j",
                    "required_approval_token": token,
                },
                indent=2,
            )
        )
        return 0

    aabb = None
    urdf = None
    if args.workspace_aabb_json is not None:
        payload = json.loads(args.workspace_aabb_json.expanduser().resolve().read_text())
        aabb = payload.get("workspace_aabb", payload)
        urdf = args.urdf.expanduser().resolve()

    if args.mode == "home" and not args.approval_token:
        parser.error("--approval-token is required for --mode home")
    if args.mode == "check" and args.approval_token:
        parser.error("--approval-token is accepted only in home mode")

    return run_home(
        mode=args.mode,
        namespace=namespace,
        joints_rad=joints,
        tolerance_rad=args.tolerance_rad,
        timeout_s=args.timeout_s,
        soft_margin_rad=args.soft_limit_margin_rad,
        workspace_aabb=aabb,
        urdf=urdf,
        approval=args.approval_token.strip() if args.approval_token else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
