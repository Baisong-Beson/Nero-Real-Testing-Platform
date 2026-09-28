"""ROS-free Gate 2 arm executor: authorization + hard-stop safety envelope.

Unlike ``NeroNativeShadowAdapter`` (clip-and-flag for ghost viz), this module
rejects any command that would violate soft limits, joint rate limits, TCP jump
limits, or the workspace AABB. It never publishes ROS commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from nero_pi05_bridge.offline_adapter import NERO_JOINT_NAMES
from nero_pi05_bridge.offline_adapter import NERO_POSITION_LOWER
from nero_pi05_bridge.offline_adapter import NERO_POSITION_UPPER

PLAN_KEYS = {
    "schema_version",
    "name",
    "arm_namespace",
    "execution_enabled",
    "control_frequency_hz",
    "test_window_s",
    "driver_speed_percent",
    "locked_gripper_width_m",
    "deadman_timeout_s",
    "feedback_max_age_s",
    "inference_timeout_s",
    "max_abs_velocity_rad_s",
    "max_abs_acceleration_rad_s2",
    "soft_limit_margin_rad",
    "tcp_jump_threshold_m",
    "workspace_aabb",
    "prompt",
    "policy_host",
    "policy_port",
    "notes",
}


@dataclass(frozen=True)
class WorkspaceAABB:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, xyz: np.ndarray) -> bool:
        point = np.asarray(xyz, dtype=np.float64).reshape(3)
        return bool(
            self.x_min <= point[0] <= self.x_max
            and self.y_min <= point[1] <= self.y_max
            and self.z_min <= point[2] <= self.z_max
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }


@dataclass(frozen=True)
class ArmExecutorPlan:
    schema_version: int
    name: str
    arm_namespace: str
    execution_enabled: bool
    control_frequency_hz: float
    test_window_s: float
    driver_speed_percent: int
    locked_gripper_width_m: float
    deadman_timeout_s: float
    feedback_max_age_s: float
    inference_timeout_s: float
    max_abs_velocity_rad_s: np.ndarray
    max_abs_acceleration_rad_s2: np.ndarray
    soft_limit_margin_rad: float
    tcp_jump_threshold_m: float
    workspace_aabb: WorkspaceAABB
    prompt: str
    policy_host: str
    policy_port: int
    notes: tuple[str, ...]
    canonical_json: str

    @property
    def control_period_sec(self) -> float:
        return 1.0 / self.control_frequency_hz

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @property
    def approval_token(self) -> str:
        return f"NERO_GATE2_EXECUTE_{self.fingerprint[:16]}"

    @property
    def soft_lower(self) -> np.ndarray:
        return NERO_POSITION_LOWER + self.soft_limit_margin_rad

    @property
    def soft_upper(self) -> np.ndarray:
        return NERO_POSITION_UPPER - self.soft_limit_margin_rad

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arm_namespace": self.arm_namespace,
            "execution_enabled": self.execution_enabled,
            "control_frequency_hz": self.control_frequency_hz,
            "test_window_s": self.test_window_s,
            "driver_speed_percent": self.driver_speed_percent,
            "tcp_jump_threshold_m": self.tcp_jump_threshold_m,
            "workspace_aabb": self.workspace_aabb.to_dict(),
            "prompt": self.prompt,
            "fingerprint_sha256": self.fingerprint,
            "required_approval_token": self.approval_token,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ExecutionAuthorization:
    plan_fingerprint: str


@dataclass(frozen=True)
class SafetyViolation:
    code: str
    message: str


def _require_exact_keys(data: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - data.keys())
    extra = sorted(data.keys() - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _finite_float(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _array7(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (7,):
        raise ValueError(f"{label} must have shape (7,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf")
    if np.any(arr <= 0):
        raise ValueError(f"{label} must be strictly positive")
    return arr


def load_plan(path: Path) -> ArmExecutorPlan:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("arm executor plan must be a JSON object")
    _require_exact_keys(raw, PLAN_KEYS, "plan")
    if int(raw["schema_version"]) != 1:
        raise ValueError("only Gate 2 schema_version=1 is supported")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ValueError("plan name must be a non-empty string")
    namespace = raw["arm_namespace"]
    if not isinstance(namespace, str) or not namespace.startswith("/"):
        raise ValueError("arm_namespace must be an absolute ROS namespace")
    if namespace.endswith("/") or "//" in namespace:
        raise ValueError("arm_namespace must not end in / or contain //")
    if not isinstance(raw["execution_enabled"], bool):
        raise ValueError("execution_enabled must be boolean")

    frequency = _finite_float(raw["control_frequency_hz"], "control_frequency_hz")
    window = _finite_float(raw["test_window_s"], "test_window_s")
    speed = int(raw["driver_speed_percent"])
    locked_width = _finite_float(raw["locked_gripper_width_m"], "locked_gripper_width_m")
    deadman = _finite_float(raw["deadman_timeout_s"], "deadman_timeout_s")
    feedback_age = _finite_float(raw["feedback_max_age_s"], "feedback_max_age_s")
    inference_timeout = _finite_float(raw["inference_timeout_s"], "inference_timeout_s")
    velocity = _array7(raw["max_abs_velocity_rad_s"], "max_abs_velocity_rad_s")
    acceleration = _array7(raw["max_abs_acceleration_rad_s2"], "max_abs_acceleration_rad_s2")
    margin = _finite_float(raw["soft_limit_margin_rad"], "soft_limit_margin_rad")
    tcp_jump = _finite_float(raw["tcp_jump_threshold_m"], "tcp_jump_threshold_m")

    if frequency <= 0 or window <= 0 or deadman <= 0 or feedback_age <= 0 or inference_timeout <= 0:
        raise ValueError("timing limits must be positive")
    if not 1 <= speed <= 20:
        raise ValueError("driver_speed_percent must be in [1, 20] for Gate 2")
    if not 0.030 <= locked_width <= 0.099:
        raise ValueError("locked_gripper_width_m must be within [0.030, 0.099]")
    if margin <= 0 or np.any(NERO_POSITION_LOWER + margin >= NERO_POSITION_UPPER - margin):
        raise ValueError("soft_limit_margin_rad is invalid")
    if tcp_jump <= 0 or tcp_jump > 0.2:
        raise ValueError("tcp_jump_threshold_m must be in (0, 0.2]")
    if window > 2.0:
        raise ValueError("test_window_s must not exceed 2.0 s for Gate 2")

    aabb_raw = raw["workspace_aabb"]
    if not isinstance(aabb_raw, dict):
        raise ValueError("workspace_aabb must be an object")
    aabb_keys = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
    _require_exact_keys(aabb_raw, aabb_keys, "workspace_aabb")
    aabb_vals = {key: _finite_float(aabb_raw[key], f"workspace_aabb.{key}") for key in aabb_keys}
    if not (
        aabb_vals["x_min"] < aabb_vals["x_max"]
        and aabb_vals["y_min"] < aabb_vals["y_max"]
        and aabb_vals["z_min"] < aabb_vals["z_max"]
    ):
        raise ValueError("workspace_aabb min/max ordering is invalid")
    workspace = WorkspaceAABB(**aabb_vals)

    if not isinstance(raw["prompt"], str) or not raw["prompt"].strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(raw["policy_host"], str) or not raw["policy_host"].strip():
        raise ValueError("policy_host must be a non-empty string")
    port = int(raw["policy_port"])
    if not 1 <= port <= 65535:
        raise ValueError("policy_port is out of range")
    notes = raw["notes"]
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        raise ValueError("notes must be a list of strings")

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return ArmExecutorPlan(
        schema_version=1,
        name=raw["name"],
        arm_namespace=namespace,
        execution_enabled=raw["execution_enabled"],
        control_frequency_hz=frequency,
        test_window_s=window,
        driver_speed_percent=speed,
        locked_gripper_width_m=locked_width,
        deadman_timeout_s=deadman,
        feedback_max_age_s=feedback_age,
        inference_timeout_s=inference_timeout,
        max_abs_velocity_rad_s=velocity,
        max_abs_acceleration_rad_s2=acceleration,
        soft_limit_margin_rad=margin,
        tcp_jump_threshold_m=tcp_jump,
        workspace_aabb=workspace,
        prompt=raw["prompt"],
        policy_host=raw["policy_host"],
        policy_port=port,
        notes=tuple(notes),
        canonical_json=canonical,
    )


def authorize_execution(plan: ArmExecutorPlan, token: str | None) -> ExecutionAuthorization:
    if not plan.execution_enabled:
        raise PermissionError("plan has execution_enabled=false")
    if token != plan.approval_token:
        raise PermissionError("approval token does not match this exact plan fingerprint")
    return ExecutionAuthorization(plan_fingerprint=plan.fingerprint)


def validate_joint_target(
    plan: ArmExecutorPlan,
    *,
    current_joints: np.ndarray,
    previous_joints: np.ndarray | None,
    previous_velocity: np.ndarray | None,
    requested_joints: np.ndarray,
    current_tcp: np.ndarray | None = None,
    requested_tcp: np.ndarray | None = None,
    previous_tcp: np.ndarray | None = None,
    is_first_command: bool = False,
) -> list[SafetyViolation]:
    """Return hard-stop violations for one absolute joint command (no clipping)."""
    current = np.asarray(current_joints, dtype=np.float64)
    requested = np.asarray(requested_joints, dtype=np.float64)
    if current.shape != (7,) or requested.shape != (7,):
        raise ValueError("joint vectors must have shape (7,)")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(requested)):
        raise ValueError("joint vectors contain NaN or Inf")

    violations: list[SafetyViolation] = []
    period = plan.control_period_sec

    if np.any(current < plan.soft_lower) or np.any(current > plan.soft_upper):
        violations.append(
            SafetyViolation("current_soft_limit", "current joints are outside soft-limit envelope")
        )
    if np.any(requested < plan.soft_lower) or np.any(requested > plan.soft_upper):
        violations.append(
            SafetyViolation("requested_soft_limit", "requested joints are outside soft-limit envelope")
        )

    delta = requested - current
    max_step = plan.max_abs_velocity_rad_s * period
    if np.any(np.abs(delta) > max_step + 1e-12):
        violations.append(
            SafetyViolation(
                "joint_velocity",
                f"requested joint step exceeds velocity envelope (period={period:.4f}s)",
            )
        )

    velocity = delta / period
    if previous_velocity is not None:
        prev_v = np.asarray(previous_velocity, dtype=np.float64)
        if prev_v.shape != (7,):
            raise ValueError("previous_velocity must have shape (7,)")
        accel = (velocity - prev_v) / period
        if np.any(np.abs(accel) > plan.max_abs_acceleration_rad_s2 + 1e-12):
            violations.append(
                SafetyViolation("joint_acceleration", "requested joint acceleration exceeds envelope")
            )

    if previous_joints is not None:
        prev = np.asarray(previous_joints, dtype=np.float64)
        if prev.shape != (7,):
            raise ValueError("previous_joints must have shape (7,)")

    if requested_tcp is not None:
        tcp = np.asarray(requested_tcp, dtype=np.float64).reshape(3)
        if not plan.workspace_aabb.contains(tcp):
            violations.append(SafetyViolation("workspace", "requested TCP is outside workspace AABB"))

    if current_tcp is not None and requested_tcp is not None:
        cur_tcp = np.asarray(current_tcp, dtype=np.float64).reshape(3)
        req_tcp = np.asarray(requested_tcp, dtype=np.float64).reshape(3)
        jump = float(np.linalg.norm(req_tcp - cur_tcp))
        if is_first_command and jump > plan.tcp_jump_threshold_m:
            violations.append(
                SafetyViolation(
                    "tcp_first_step_jump",
                    f"first-step TCP jump {jump:.4f} m exceeds {plan.tcp_jump_threshold_m:.4f} m",
                )
            )
        if not is_first_command and jump > plan.tcp_jump_threshold_m:
            violations.append(
                SafetyViolation(
                    "tcp_intra_step_jump",
                    f"TCP step jump {jump:.4f} m exceeds {plan.tcp_jump_threshold_m:.4f} m",
                )
            )

    if previous_tcp is not None and requested_tcp is not None:
        prev_tcp = np.asarray(previous_tcp, dtype=np.float64).reshape(3)
        req_tcp = np.asarray(requested_tcp, dtype=np.float64).reshape(3)
        boundary = float(np.linalg.norm(req_tcp - prev_tcp))
        if boundary > plan.tcp_jump_threshold_m:
            violations.append(
                SafetyViolation(
                    "tcp_boundary_jump",
                    f"TCP boundary jump {boundary:.4f} m exceeds {plan.tcp_jump_threshold_m:.4f} m",
                )
            )

    return violations


def locked_gripper_command(plan: ArmExecutorPlan) -> float:
    """Return the fixed gripper width Gate 2 is allowed to publish."""
    return float(plan.locked_gripper_width_m)


def joint_names() -> tuple[str, ...]:
    return NERO_JOINT_NAMES
