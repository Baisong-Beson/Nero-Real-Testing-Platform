"""ROS-free dual-arm Gate 3 plan, authorization, and per-step safety.

Single-arm Gate 3 stays in ``task_executor.py``. This module only handles the
16D ``[right_8, left_8]`` layout. Joint envelopes reuse
``arm_executor.validate_joint_target`` via a duck-typed per-arm view.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from nero_pi05_bridge.arm_executor import SafetyViolation
from nero_pi05_bridge.arm_executor import WorkspaceAABB
from nero_pi05_bridge.arm_executor import validate_joint_target
from nero_pi05_bridge.offline_adapter import NERO_JOINT_NAMES
from nero_pi05_bridge.offline_adapter import NERO_POSITION_LOWER
from nero_pi05_bridge.offline_adapter import NERO_POSITION_UPPER
from nero_pi05_bridge.task_executor import GRIPPER_CLOSEDNESS_CLIP_TOLERANCE
from nero_pi05_bridge.task_executor import MAX_CHUNK_STEPS_PER_INFERENCE
from nero_pi05_bridge.task_executor import MAX_DRIVER_SPEED_PERCENT
from nero_pi05_bridge.task_executor import TaskStage
from nero_pi05_bridge.task_executor import assert_near_home
from nero_pi05_bridge.task_executor import gripper_width_from_closedness
from nero_pi05_bridge.task_executor import validate_command_lead

MAX_TOTAL_DURATION_S = 40.0

PLAN_KEYS = {
    "schema_version",
    "name",
    "right_arm_namespace",
    "left_arm_namespace",
    "deadman_topic",
    "execution_enabled",
    "control_frequency_hz",
    "max_total_duration_s",
    "chunk_steps_per_inference",
    "max_command_lead_rad",
    "driver_speed_percent",
    "right_gripper_min_width_m",
    "right_gripper_max_width_m",
    "left_gripper_min_width_m",
    "left_gripper_max_width_m",
    "locked_right_gripper_width_m",
    "locked_left_gripper_width_m",
    "gripper_control",
    "max_gripper_speed_m_s",
    "gripper_effort_n",
    "deadman_timeout_s",
    "feedback_max_age_s",
    "inference_timeout_s",
    "max_abs_velocity_rad_s",
    "max_abs_acceleration_rad_s2",
    "soft_limit_margin_rad",
    "tcp_jump_threshold_m",
    "workspace_aabb",
    "right_home_joints_rad",
    "left_home_joints_rad",
    "home_tolerance_rad",
    "stages",
    "policy_host",
    "policy_port",
    "notes",
}

STAGE_KEYS = {"name", "prompt", "max_duration_s"}


@dataclass(frozen=True)
class ArmSafetyView:
    """Duck-typed plan slice consumed by Gate 2/3 safety helpers."""

    control_frequency_hz: float
    max_abs_velocity_rad_s: np.ndarray
    max_abs_acceleration_rad_s2: np.ndarray
    soft_limit_margin_rad: float
    tcp_jump_threshold_m: float
    workspace_aabb: WorkspaceAABB
    max_command_lead_rad: float
    max_gripper_speed_m_s: float
    home_joints_rad: np.ndarray
    home_tolerance_rad: float
    gripper_min_width_m: float
    gripper_max_width_m: float
    locked_gripper_width_m: float

    @property
    def control_period_sec(self) -> float:
        return 1.0 / self.control_frequency_hz

    @property
    def soft_lower(self) -> np.ndarray:
        return NERO_POSITION_LOWER + self.soft_limit_margin_rad

    @property
    def soft_upper(self) -> np.ndarray:
        return NERO_POSITION_UPPER - self.soft_limit_margin_rad


@dataclass(frozen=True)
class DualGate3Plan:
    schema_version: int
    name: str
    right_arm_namespace: str
    left_arm_namespace: str
    deadman_topic: str
    execution_enabled: bool
    control_frequency_hz: float
    max_total_duration_s: float
    chunk_steps_per_inference: int
    max_command_lead_rad: float
    driver_speed_percent: int
    right_gripper_min_width_m: float
    right_gripper_max_width_m: float
    left_gripper_min_width_m: float
    left_gripper_max_width_m: float
    locked_right_gripper_width_m: float
    locked_left_gripper_width_m: float
    gripper_control: str
    max_gripper_speed_m_s: float
    gripper_effort_n: float
    deadman_timeout_s: float
    feedback_max_age_s: float
    inference_timeout_s: float
    max_abs_velocity_rad_s: np.ndarray
    max_abs_acceleration_rad_s2: np.ndarray
    soft_limit_margin_rad: float
    tcp_jump_threshold_m: float
    workspace_aabb: WorkspaceAABB
    right_home_joints_rad: np.ndarray
    left_home_joints_rad: np.ndarray
    home_tolerance_rad: float
    stages: tuple[TaskStage, ...]
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
        return f"NERO_DUAL_GATE3_EXECUTE_{self.fingerprint[:16]}"

    def arm_view(self, arm: str) -> ArmSafetyView:
        if arm == "right":
            return ArmSafetyView(
                control_frequency_hz=self.control_frequency_hz,
                max_abs_velocity_rad_s=self.max_abs_velocity_rad_s,
                max_abs_acceleration_rad_s2=self.max_abs_acceleration_rad_s2,
                soft_limit_margin_rad=self.soft_limit_margin_rad,
                tcp_jump_threshold_m=self.tcp_jump_threshold_m,
                workspace_aabb=self.workspace_aabb,
                max_command_lead_rad=self.max_command_lead_rad,
                max_gripper_speed_m_s=self.max_gripper_speed_m_s,
                home_joints_rad=self.right_home_joints_rad,
                home_tolerance_rad=self.home_tolerance_rad,
                gripper_min_width_m=self.right_gripper_min_width_m,
                gripper_max_width_m=self.right_gripper_max_width_m,
                locked_gripper_width_m=self.locked_right_gripper_width_m,
            )
        if arm == "left":
            return ArmSafetyView(
                control_frequency_hz=self.control_frequency_hz,
                max_abs_velocity_rad_s=self.max_abs_velocity_rad_s,
                max_abs_acceleration_rad_s2=self.max_abs_acceleration_rad_s2,
                soft_limit_margin_rad=self.soft_limit_margin_rad,
                tcp_jump_threshold_m=self.tcp_jump_threshold_m,
                workspace_aabb=self.workspace_aabb,
                max_command_lead_rad=self.max_command_lead_rad,
                max_gripper_speed_m_s=self.max_gripper_speed_m_s,
                home_joints_rad=self.left_home_joints_rad,
                home_tolerance_rad=self.home_tolerance_rad,
                gripper_min_width_m=self.left_gripper_min_width_m,
                gripper_max_width_m=self.left_gripper_max_width_m,
                locked_gripper_width_m=self.locked_left_gripper_width_m,
            )
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "right_arm_namespace": self.right_arm_namespace,
            "left_arm_namespace": self.left_arm_namespace,
            "deadman_topic": self.deadman_topic,
            "execution_enabled": self.execution_enabled,
            "control_frequency_hz": self.control_frequency_hz,
            "max_total_duration_s": self.max_total_duration_s,
            "chunk_steps_per_inference": self.chunk_steps_per_inference,
            "max_command_lead_rad": self.max_command_lead_rad,
            "driver_speed_percent": self.driver_speed_percent,
            "gripper_control": self.gripper_control,
            "tcp_jump_threshold_m": self.tcp_jump_threshold_m,
            "workspace_aabb": self.workspace_aabb.to_dict(),
            "right_home_joints_rad": self.right_home_joints_rad.tolist(),
            "left_home_joints_rad": self.left_home_joints_rad.tolist(),
            "home_tolerance_rad": self.home_tolerance_rad,
            "stages": [stage.to_dict() for stage in self.stages],
            "fingerprint_sha256": self.fingerprint,
            "required_approval_token": self.approval_token,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ExecutionAuthorization:
    plan_fingerprint: str


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


def _array7(value: Any, label: str, *, positive: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (7,):
        raise ValueError(f"{label} must have shape (7,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf")
    if positive and np.any(arr <= 0):
        raise ValueError(f"{label} must be strictly positive")
    return arr


def _absolute_topic(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{label} must be an absolute ROS name")
    if value.endswith("/") or "//" in value:
        raise ValueError(f"{label} must not end in / or contain //")
    return value


def _gripper_interval(minimum: float, maximum: float, label: str) -> tuple[float, float]:
    if minimum < 0 or maximum <= minimum:
        raise ValueError(f"{label} gripper interval is invalid")
    return minimum, maximum


def load_plan(path: Path, *, prompt_override: str | None = None) -> DualGate3Plan:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("dual gate3 plan must be a JSON object")
    _require_exact_keys(raw, PLAN_KEYS, "plan")
    if int(raw["schema_version"]) != 1:
        raise ValueError("only dual Gate 3 schema_version=1 is supported")
    if prompt_override is not None:
        if not isinstance(prompt_override, str):
            raise ValueError("prompt override must be a non-empty string")
        prompt_override = prompt_override.strip()
        if not prompt_override:
            raise ValueError("prompt override must be a non-empty string")
        if not isinstance(raw["stages"], list) or len(raw["stages"]) != 1:
            raise ValueError("prompt override requires exactly one stage")
        if not isinstance(raw["stages"][0], dict):
            raise ValueError("stages[0] must be an object")
        raw["stages"][0]["prompt"] = prompt_override
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ValueError("plan name must be a non-empty string")
    right_ns = _absolute_topic(raw["right_arm_namespace"], "right_arm_namespace")
    left_ns = _absolute_topic(raw["left_arm_namespace"], "left_arm_namespace")
    if right_ns == left_ns:
        raise ValueError("right_arm_namespace and left_arm_namespace must differ")
    deadman_topic = _absolute_topic(raw["deadman_topic"], "deadman_topic")
    if not isinstance(raw["execution_enabled"], bool):
        raise ValueError("execution_enabled must be boolean")

    frequency = _finite_float(raw["control_frequency_hz"], "control_frequency_hz")
    total = _finite_float(raw["max_total_duration_s"], "max_total_duration_s")
    chunk_steps = int(raw["chunk_steps_per_inference"])
    command_lead = _finite_float(raw["max_command_lead_rad"], "max_command_lead_rad")
    speed = int(raw["driver_speed_percent"])
    right_min, right_max = _gripper_interval(
        _finite_float(raw["right_gripper_min_width_m"], "right_gripper_min_width_m"),
        _finite_float(raw["right_gripper_max_width_m"], "right_gripper_max_width_m"),
        "right",
    )
    left_min, left_max = _gripper_interval(
        _finite_float(raw["left_gripper_min_width_m"], "left_gripper_min_width_m"),
        _finite_float(raw["left_gripper_max_width_m"], "left_gripper_max_width_m"),
        "left",
    )
    locked_right = _finite_float(raw["locked_right_gripper_width_m"], "locked_right_gripper_width_m")
    locked_left = _finite_float(raw["locked_left_gripper_width_m"], "locked_left_gripper_width_m")
    gripper_control = raw["gripper_control"]
    if gripper_control not in {"locked", "policy"}:
        raise ValueError("gripper_control must be 'locked' or 'policy'")
    gripper_speed = _finite_float(raw["max_gripper_speed_m_s"], "max_gripper_speed_m_s")
    gripper_effort = _finite_float(raw["gripper_effort_n"], "gripper_effort_n")
    deadman = _finite_float(raw["deadman_timeout_s"], "deadman_timeout_s")
    feedback_age = _finite_float(raw["feedback_max_age_s"], "feedback_max_age_s")
    inference_timeout = _finite_float(raw["inference_timeout_s"], "inference_timeout_s")
    velocity = _array7(raw["max_abs_velocity_rad_s"], "max_abs_velocity_rad_s")
    acceleration = _array7(raw["max_abs_acceleration_rad_s2"], "max_abs_acceleration_rad_s2")
    margin = _finite_float(raw["soft_limit_margin_rad"], "soft_limit_margin_rad")
    tcp_jump = _finite_float(raw["tcp_jump_threshold_m"], "tcp_jump_threshold_m")
    right_home = _array7(raw["right_home_joints_rad"], "right_home_joints_rad", positive=False)
    left_home = _array7(raw["left_home_joints_rad"], "left_home_joints_rad", positive=False)
    home_tol = _finite_float(raw["home_tolerance_rad"], "home_tolerance_rad")

    if frequency <= 0 or total <= 0 or deadman <= 0 or feedback_age <= 0 or inference_timeout <= 0:
        raise ValueError("timing limits must be positive")
    if total > MAX_TOTAL_DURATION_S:
        raise ValueError(f"max_total_duration_s must not exceed {MAX_TOTAL_DURATION_S} s")
    if not 1 <= chunk_steps <= MAX_CHUNK_STEPS_PER_INFERENCE:
        raise ValueError(
            f"chunk_steps_per_inference must be in [1, {MAX_CHUNK_STEPS_PER_INFERENCE}]"
        )
    if not 0.0 < command_lead <= 0.5:
        raise ValueError("max_command_lead_rad must be in (0, 0.5]")
    if not 1 <= speed <= MAX_DRIVER_SPEED_PERCENT:
        raise ValueError(
            f"driver_speed_percent must be in [1, {MAX_DRIVER_SPEED_PERCENT}] for Gate 3"
        )
    if not right_min <= locked_right <= right_max:
        raise ValueError("locked_right_gripper_width_m is outside the right gripper interval")
    if not left_min <= locked_left <= left_max:
        raise ValueError("locked_left_gripper_width_m is outside the left gripper interval")
    if gripper_speed <= 0:
        raise ValueError("max_gripper_speed_m_s must be positive")
    if not 0.0 < gripper_effort <= 1.0:
        raise ValueError("gripper_effort_n must be in (0, 1.0]")
    if margin < 0 or np.any(NERO_POSITION_LOWER + margin >= NERO_POSITION_UPPER - margin):
        raise ValueError("soft_limit_margin_rad is invalid")
    if tcp_jump <= 0 or tcp_jump > 0.2:
        raise ValueError("tcp_jump_threshold_m must be in (0, 0.2]")
    if home_tol <= 0:
        raise ValueError("home_tolerance_rad must be positive")
    soft_lower = NERO_POSITION_LOWER + margin
    soft_upper = NERO_POSITION_UPPER - margin
    if np.any(right_home < soft_lower) or np.any(right_home > soft_upper):
        raise ValueError("right_home_joints_rad must lie inside soft-limit envelope")
    if np.any(left_home < soft_lower) or np.any(left_home > soft_upper):
        raise ValueError("left_home_joints_rad must lie inside soft-limit envelope")

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

    stages_raw = raw["stages"]
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError("stages must be a non-empty list")
    stages: list[TaskStage] = []
    stage_budget = 0.0
    for idx, item in enumerate(stages_raw):
        if not isinstance(item, dict):
            raise ValueError(f"stages[{idx}] must be an object")
        _require_exact_keys(item, STAGE_KEYS, f"stages[{idx}]")
        name = item["name"]
        prompt = item["prompt"]
        duration = _finite_float(item["max_duration_s"], f"stages[{idx}].max_duration_s")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"stages[{idx}].name must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"stages[{idx}].prompt must be a non-empty string")
        if duration <= 0:
            raise ValueError(f"stages[{idx}].max_duration_s must be positive")
        stage_budget += duration
        stages.append(TaskStage(name=name, prompt=prompt, max_duration_s=duration))
    if stage_budget > total + 1e-12:
        raise ValueError("sum of stage max_duration_s exceeds max_total_duration_s")

    if not isinstance(raw["policy_host"], str) or not raw["policy_host"].strip():
        raise ValueError("policy_host must be a non-empty string")
    port = int(raw["policy_port"])
    if not 1 <= port <= 65535:
        raise ValueError("policy_port is out of range")
    notes = raw["notes"]
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        raise ValueError("notes must be a list of strings")

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return DualGate3Plan(
        schema_version=1,
        name=raw["name"],
        right_arm_namespace=right_ns,
        left_arm_namespace=left_ns,
        deadman_topic=deadman_topic,
        execution_enabled=raw["execution_enabled"],
        control_frequency_hz=frequency,
        max_total_duration_s=total,
        chunk_steps_per_inference=chunk_steps,
        max_command_lead_rad=command_lead,
        driver_speed_percent=speed,
        right_gripper_min_width_m=right_min,
        right_gripper_max_width_m=right_max,
        left_gripper_min_width_m=left_min,
        left_gripper_max_width_m=left_max,
        locked_right_gripper_width_m=locked_right,
        locked_left_gripper_width_m=locked_left,
        gripper_control=gripper_control,
        max_gripper_speed_m_s=gripper_speed,
        gripper_effort_n=gripper_effort,
        deadman_timeout_s=deadman,
        feedback_max_age_s=feedback_age,
        inference_timeout_s=inference_timeout,
        max_abs_velocity_rad_s=velocity,
        max_abs_acceleration_rad_s2=acceleration,
        soft_limit_margin_rad=margin,
        tcp_jump_threshold_m=tcp_jump,
        workspace_aabb=workspace,
        right_home_joints_rad=right_home,
        left_home_joints_rad=left_home,
        home_tolerance_rad=home_tol,
        stages=tuple(stages),
        policy_host=raw["policy_host"],
        policy_port=port,
        notes=tuple(notes),
        canonical_json=canonical,
    )


def authorize_execution(plan: DualGate3Plan, token: str | None) -> ExecutionAuthorization:
    if not plan.execution_enabled:
        raise PermissionError("plan has execution_enabled=false")
    if token != plan.approval_token:
        raise PermissionError("approval token does not match this exact plan fingerprint")
    return ExecutionAuthorization(plan_fingerprint=plan.fingerprint)


def locked_gripper_command(view: ArmSafetyView) -> float:
    return float(view.locked_gripper_width_m)


def arm_gripper_width(view: ArmSafetyView, closedness: float) -> float:
    return gripper_width_from_closedness(
        closedness,
        gripper_min_width_m=view.gripper_min_width_m,
        gripper_max_width_m=view.gripper_max_width_m,
    )


def validate_arm_gripper_command(
    view: ArmSafetyView,
    *,
    previous_width_m: float | None,
    requested_closedness: float,
) -> list[SafetyViolation]:
    violations: list[SafetyViolation] = []
    if not math.isfinite(requested_closedness):
        violations.append(SafetyViolation("gripper_nonfinite", "gripper closedness is not finite"))
        return violations
    if (
        requested_closedness < -GRIPPER_CLOSEDNESS_CLIP_TOLERANCE
        or requested_closedness > 1.0 + GRIPPER_CLOSEDNESS_CLIP_TOLERANCE
    ):
        violations.append(
            SafetyViolation(
                "gripper_closedness_range",
                f"gripper closedness {requested_closedness:.4f} outside "
                f"[-{GRIPPER_CLOSEDNESS_CLIP_TOLERANCE}, {1.0 + GRIPPER_CLOSEDNESS_CLIP_TOLERANCE}]",
            )
        )
    width = arm_gripper_width(view, requested_closedness)
    if previous_width_m is not None:
        max_step = view.max_gripper_speed_m_s * view.control_period_sec
        delta = abs(width - float(previous_width_m))
        if delta > max_step + 1e-12:
            violations.append(
                SafetyViolation(
                    "gripper_speed",
                    f"gripper width step {delta:.4f} m exceeds {max_step:.4f} m",
                )
            )
    return violations


def _prefix_violations(arm: str, violations: list[SafetyViolation]) -> list[SafetyViolation]:
    return [
        SafetyViolation(code=f"{arm}_{item.code}", message=f"{arm}: {item.message}")
        for item in violations
    ]


def evaluate_dual_step(
    plan: DualGate3Plan,
    *,
    right_current_joints: np.ndarray,
    left_current_joints: np.ndarray,
    right_requested_joints: np.ndarray,
    left_requested_joints: np.ndarray,
    right_requested_closedness: float,
    left_requested_closedness: float,
    right_current_tcp: np.ndarray,
    left_current_tcp: np.ndarray,
    right_requested_tcp: np.ndarray,
    left_requested_tcp: np.ndarray,
    right_previous_joints: np.ndarray | None,
    left_previous_joints: np.ndarray | None,
    right_previous_velocity: np.ndarray | None,
    left_previous_velocity: np.ndarray | None,
    right_previous_tcp: np.ndarray | None,
    left_previous_tcp: np.ndarray | None,
    right_previous_gripper_width_m: float | None,
    left_previous_gripper_width_m: float | None,
    is_first_command: bool,
    check_command_lead: bool,
) -> list[SafetyViolation]:
    """Validate one synchronized dual-arm command. Any arm failure is returned."""
    combined: list[SafetyViolation] = []
    for arm, current, requested, current_tcp, requested_tcp, prev_j, prev_v, prev_tcp, prev_g, closed in (
        (
            "right",
            right_current_joints,
            right_requested_joints,
            right_current_tcp,
            right_requested_tcp,
            right_previous_joints,
            right_previous_velocity,
            right_previous_tcp,
            right_previous_gripper_width_m,
            right_requested_closedness,
        ),
        (
            "left",
            left_current_joints,
            left_requested_joints,
            left_current_tcp,
            left_requested_tcp,
            left_previous_joints,
            left_previous_velocity,
            left_previous_tcp,
            left_previous_gripper_width_m,
            left_requested_closedness,
        ),
    ):
        view = plan.arm_view(arm)
        violations = validate_joint_target(
            view,
            current_joints=current,
            previous_joints=prev_j,
            previous_velocity=prev_v,
            requested_joints=requested,
            current_tcp=current_tcp,
            requested_tcp=requested_tcp,
            previous_tcp=prev_tcp,
            is_first_command=is_first_command,
        )
        if plan.gripper_control == "policy":
            violations.extend(
                validate_arm_gripper_command(
                    view,
                    previous_width_m=prev_g,
                    requested_closedness=closed,
                )
            )
        if check_command_lead:
            violations.extend(
                validate_command_lead(
                    view,
                    requested_joints=requested,
                    measured_joints=current,
                )
            )
        combined.extend(_prefix_violations(arm, violations))
    return combined


def assert_both_near_home(plan: DualGate3Plan, right_joints: np.ndarray, left_joints: np.ndarray) -> None:
    assert_near_home(plan.arm_view("right"), right_joints)
    assert_near_home(plan.arm_view("left"), left_joints)


def joint_names() -> tuple[str, ...]:
    return NERO_JOINT_NAMES
