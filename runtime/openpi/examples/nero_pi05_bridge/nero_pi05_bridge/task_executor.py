"""ROS-free Gate 3 task executor: staged prompts, gripper unlock, hard-stop envelope.

Reuses ``validate_joint_target`` from Gate 2 via duck-typed plan attributes so the
joint safety envelope stays a single source of truth. Does not publish ROS commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from typing import Protocol

import numpy as np

from nero_pi05_bridge.arm_executor import SafetyViolation
from nero_pi05_bridge.arm_executor import WorkspaceAABB
from nero_pi05_bridge.offline_adapter import NERO_JOINT_NAMES
from nero_pi05_bridge.offline_adapter import NERO_POSITION_LOWER
from nero_pi05_bridge.offline_adapter import NERO_POSITION_UPPER

GRIPPER_MIN_WIDTH_M = 0.030
GRIPPER_MAX_WIDTH_M = 0.099

# pi0.5 emits a 16-step action horizon; replaying more than that is impossible.
MAX_CHUNK_STEPS_PER_INFERENCE = 16

# Gate 3 has to reproduce the speeds the demonstrations were recorded at, otherwise the
# policy's commands outrun the arm and the closed loop never converges: at 5 percent the
# arm managed 0.056 rad/s on joint4 against a training median of 0.386 rad/s. 35 percent
# tracks that median. Gate 2 keeps its own stricter cap of 20 because it only ever runs a
# 2 s window. This value is recorded in the plan for the audit trail; the speed itself is
# set by the agx_arm_ctrl launch argument.
MAX_DRIVER_SPEED_PERCENT = 35

PLAN_KEYS = {
    "schema_version",
    "name",
    "arm_namespace",
    "execution_enabled",
    "control_frequency_hz",
    "max_total_duration_s",
    "chunk_steps_per_inference",
    "max_command_lead_rad",
    "driver_speed_percent",
    "locked_gripper_width_m",
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
    "home_joints_rad",
    "home_tolerance_rad",
    "stages",
    "policy_host",
    "policy_port",
    "notes",
}

STAGE_KEYS = {"name", "prompt", "max_duration_s"}


class JointSafetyPlan(Protocol):
    """Minimal contract expected by ``arm_executor.validate_joint_target``."""

    @property
    def control_period_sec(self) -> float: ...

    @property
    def soft_lower(self) -> np.ndarray: ...

    @property
    def soft_upper(self) -> np.ndarray: ...

    max_abs_velocity_rad_s: np.ndarray
    max_abs_acceleration_rad_s2: np.ndarray
    tcp_jump_threshold_m: float
    workspace_aabb: WorkspaceAABB


@dataclass(frozen=True)
class TaskStage:
    name: str
    prompt: str
    max_duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "max_duration_s": self.max_duration_s,
        }


@dataclass(frozen=True)
class Gate3Plan:
    schema_version: int
    name: str
    arm_namespace: str
    execution_enabled: bool
    control_frequency_hz: float
    max_total_duration_s: float
    chunk_steps_per_inference: int
    max_command_lead_rad: float
    driver_speed_percent: int
    locked_gripper_width_m: float
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
    home_joints_rad: np.ndarray
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
        return f"NERO_GATE3_EXECUTE_{self.fingerprint[:16]}"

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
            "max_total_duration_s": self.max_total_duration_s,
            "chunk_steps_per_inference": self.chunk_steps_per_inference,
            "max_command_lead_rad": self.max_command_lead_rad,
            "driver_speed_percent": self.driver_speed_percent,
            "gripper_control": self.gripper_control,
            "tcp_jump_threshold_m": self.tcp_jump_threshold_m,
            "workspace_aabb": self.workspace_aabb.to_dict(),
            "home_joints_rad": self.home_joints_rad.tolist(),
            "home_tolerance_rad": self.home_tolerance_rad,
            "stages": [stage.to_dict() for stage in self.stages],
            "fingerprint_sha256": self.fingerprint,
            "required_approval_token": self.approval_token,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ExecutionAuthorization:
    plan_fingerprint: str


@dataclass(frozen=True)
class StageMachine:
    """Advances through plan.stages by timeout or explicit advance requests."""

    stages: tuple[TaskStage, ...]
    index: int = 0
    stage_started_monotonic: float | None = None

    @property
    def current(self) -> TaskStage:
        return self.stages[self.index]

    @property
    def done(self) -> bool:
        return self.index >= len(self.stages)

    def start(self, now: float) -> StageMachine:
        if not self.stages:
            raise ValueError("stages must be non-empty")
        return StageMachine(stages=self.stages, index=0, stage_started_monotonic=now)

    def maybe_advance(self, now: float, *, requested: bool) -> tuple[StageMachine, bool]:
        if self.done or self.stage_started_monotonic is None:
            return self, False
        elapsed = now - self.stage_started_monotonic
        if requested or elapsed >= self.current.max_duration_s:
            nxt = self.index + 1
            if nxt >= len(self.stages):
                return StageMachine(stages=self.stages, index=nxt, stage_started_monotonic=None), True
            return StageMachine(stages=self.stages, index=nxt, stage_started_monotonic=now), True
        return self, False


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


def gripper_width_from_closedness(
    closedness: float,
    *,
    gripper_min_width_m: float = GRIPPER_MIN_WIDTH_M,
    gripper_max_width_m: float = GRIPPER_MAX_WIDTH_M,
) -> float:
    if not math.isfinite(closedness):
        raise ValueError("closedness must be finite")
    closed = float(np.clip(closedness, 0.0, 1.0))
    return gripper_max_width_m - closed * (gripper_max_width_m - gripper_min_width_m)


# Policy outputs occasionally undershoot 0 / overshoot 1 by a few thousandths.
# Clip those; only hard-stop on larger excursions that indicate a real fault.
GRIPPER_CLOSEDNESS_CLIP_TOLERANCE = 0.05


def validate_gripper_command(
    plan: Gate3Plan,
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
    width = gripper_width_from_closedness(requested_closedness)
    if previous_width_m is not None:
        max_step = plan.max_gripper_speed_m_s * plan.control_period_sec
        delta = abs(width - float(previous_width_m))
        if delta > max_step + 1e-12:
            violations.append(
                SafetyViolation(
                    "gripper_speed",
                    f"gripper width step {delta:.4f} m exceeds {max_step:.4f} m",
                )
            )
    return violations


def validate_command_lead(
    plan: Gate3Plan,
    *,
    requested_joints: np.ndarray,
    measured_joints: np.ndarray,
) -> list[SafetyViolation]:
    """Bound how far a replayed chunk target may run ahead of measured joints.

    Replaying several chunk steps per inference is open loop, so the per-step
    velocity envelope is checked against the previous *command*. This check is the
    counterpart that still catches a servo that has stopped following.
    """
    requested = np.asarray(requested_joints, dtype=np.float64)
    measured = np.asarray(measured_joints, dtype=np.float64)
    if requested.shape != (7,) or measured.shape != (7,):
        raise ValueError("requested_joints and measured_joints must have shape (7,)")
    lead = np.abs(requested - measured)
    index = int(np.argmax(lead))
    if float(lead[index]) > plan.max_command_lead_rad:
        return [
            SafetyViolation(
                "command_lead",
                f"joint{index + 1} command leads measured position by "
                f"{float(lead[index]):.4f} rad (limit={plan.max_command_lead_rad:.4f})",
            )
        ]
    return []


def assert_near_home(plan: Gate3Plan, current_joints: np.ndarray) -> None:
    current = np.asarray(current_joints, dtype=np.float64)
    if current.shape != (7,):
        raise ValueError("current_joints must have shape (7,)")
    delta = np.abs(current - plan.home_joints_rad)
    if np.any(delta > plan.home_tolerance_rad + 1e-12):
        raise RuntimeError(
            "current joints are outside home tolerance: "
            f"max_abs_delta={float(delta.max()):.4f} rad "
            f"(limit={plan.home_tolerance_rad:.4f})"
        )


def load_plan(path: Path) -> Gate3Plan:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("gate3 plan must be a JSON object")
    _require_exact_keys(raw, PLAN_KEYS, "plan")
    if int(raw["schema_version"]) != 1:
        raise ValueError("only Gate 3 schema_version=1 is supported")
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
    total = _finite_float(raw["max_total_duration_s"], "max_total_duration_s")
    chunk_steps = int(raw["chunk_steps_per_inference"])
    command_lead = _finite_float(raw["max_command_lead_rad"], "max_command_lead_rad")
    speed = int(raw["driver_speed_percent"])
    locked_width = _finite_float(raw["locked_gripper_width_m"], "locked_gripper_width_m")
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
    home = _array7(raw["home_joints_rad"], "home_joints_rad", positive=False)
    home_tol = _finite_float(raw["home_tolerance_rad"], "home_tolerance_rad")

    if frequency <= 0 or total <= 0 or deadman <= 0 or feedback_age <= 0 or inference_timeout <= 0:
        raise ValueError("timing limits must be positive")
    if total > 30.0:
        raise ValueError("max_total_duration_s must not exceed 30.0 s for Gate 3")
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
    if not GRIPPER_MIN_WIDTH_M <= locked_width <= GRIPPER_MAX_WIDTH_M:
        raise ValueError("locked_gripper_width_m must be within [0.030, 0.099]")
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
    if np.any(home < soft_lower) or np.any(home > soft_upper):
        raise ValueError("home_joints_rad must lie inside soft-limit envelope")

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
    return Gate3Plan(
        schema_version=1,
        name=raw["name"],
        arm_namespace=namespace,
        execution_enabled=raw["execution_enabled"],
        control_frequency_hz=frequency,
        max_total_duration_s=total,
        chunk_steps_per_inference=chunk_steps,
        max_command_lead_rad=command_lead,
        driver_speed_percent=speed,
        locked_gripper_width_m=locked_width,
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
        home_joints_rad=home,
        home_tolerance_rad=home_tol,
        stages=tuple(stages),
        policy_host=raw["policy_host"],
        policy_port=port,
        notes=tuple(notes),
        canonical_json=canonical,
    )


def authorize_execution(plan: Gate3Plan, token: str | None) -> ExecutionAuthorization:
    if not plan.execution_enabled:
        raise PermissionError("plan has execution_enabled=false")
    if token != plan.approval_token:
        raise PermissionError("approval token does not match this exact plan fingerprint")
    return ExecutionAuthorization(plan_fingerprint=plan.fingerprint)


def locked_gripper_command(plan: Gate3Plan) -> float:
    return float(plan.locked_gripper_width_m)


def joint_names() -> tuple[str, ...]:
    return NERO_JOINT_NAMES
