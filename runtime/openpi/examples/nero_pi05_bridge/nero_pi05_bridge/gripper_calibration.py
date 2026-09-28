"""Pure validation and statistics for NERO Gate 1 gripper calibration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import pairwise
import json
import math
from pathlib import Path
from typing import Any

PLAN_KEYS = {
    "schema_version",
    "name",
    "arm_namespace",
    "execution_enabled",
    "targets",
    "safety",
}
SAFETY_KEYS = {
    "width_min_m",
    "width_max_m",
    "effort_min_n",
    "effort_max_n",
    "max_step_delta_m",
    "required_final_width_min_m",
    "joint_static_tolerance_rad",
    "tcp_static_tolerance_m",
    "gripper_tolerance_m",
    "feedback_timeout_s",
    "baseline_duration_s",
    "step_timeout_s",
    "settle_duration_s",
    "gate_open_duration_s",
}


@dataclass(frozen=True)
class CalibrationTarget:
    width_m: float
    effort_n: float


@dataclass(frozen=True)
class SafetyLimits:
    width_min_m: float
    width_max_m: float
    effort_min_n: float
    effort_max_n: float
    max_step_delta_m: float
    required_final_width_min_m: float
    joint_static_tolerance_rad: float
    tcp_static_tolerance_m: float
    gripper_tolerance_m: float
    feedback_timeout_s: float
    baseline_duration_s: float
    step_timeout_s: float
    settle_duration_s: float
    gate_open_duration_s: float


@dataclass(frozen=True)
class CalibrationPlan:
    schema_version: int
    name: str
    arm_namespace: str
    execution_enabled: bool
    targets: tuple[CalibrationTarget, ...]
    safety: SafetyLimits
    canonical_json: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @property
    def approval_token(self) -> str:
        return f"NERO_GATE1_EXECUTE_{self.fingerprint[:16]}"

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arm_namespace": self.arm_namespace,
            "execution_enabled": self.execution_enabled,
            "target_count": len(self.targets),
            "targets": [{"width_m": target.width_m, "effort_n": target.effort_n} for target in self.targets],
            "fingerprint_sha256": self.fingerprint,
            "required_approval_token": self.approval_token,
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


def load_plan(path: Path) -> CalibrationPlan:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("calibration plan must be a JSON object")
    _require_exact_keys(raw, PLAN_KEYS, "plan")
    if raw["schema_version"] != 1:
        raise ValueError("only calibration schema_version=1 is supported")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ValueError("plan name must be a non-empty string")
    namespace = raw["arm_namespace"]
    if not isinstance(namespace, str) or not namespace.startswith("/"):
        raise ValueError("arm_namespace must be an absolute ROS namespace")
    if namespace.endswith("/") or "//" in namespace:
        raise ValueError("arm_namespace must not end in / or contain //")
    if not isinstance(raw["execution_enabled"], bool):
        raise ValueError("execution_enabled must be boolean")

    safety_raw = raw["safety"]
    if not isinstance(safety_raw, dict):
        raise ValueError("safety must be a JSON object")
    _require_exact_keys(safety_raw, SAFETY_KEYS, "safety")
    safety_values = {key: _finite_float(value, f"safety.{key}") for key, value in safety_raw.items()}
    safety = SafetyLimits(**safety_values)
    if not 0.0 <= safety.width_min_m < safety.width_max_m <= 0.1:
        raise ValueError("width safety range must be within [0.0, 0.1] m")
    if not 0.5 <= safety.effort_min_n <= safety.effort_max_n <= 3.0:
        raise ValueError("effort safety range must be within [0.5, 3.0] N")
    if not safety.width_min_m <= safety.required_final_width_min_m <= safety.width_max_m:
        raise ValueError("required final width must lie inside the safety range")
    positive_fields = (
        "max_step_delta_m",
        "joint_static_tolerance_rad",
        "tcp_static_tolerance_m",
        "gripper_tolerance_m",
        "feedback_timeout_s",
        "baseline_duration_s",
        "step_timeout_s",
        "settle_duration_s",
        "gate_open_duration_s",
    )
    for field in positive_fields:
        if getattr(safety, field) <= 0:
            raise ValueError(f"safety.{field} must be positive")
    if safety.max_step_delta_m > 0.03:
        raise ValueError("max_step_delta_m must not exceed 0.03 m")
    if safety.gate_open_duration_s > 0.5:
        raise ValueError("gate_open_duration_s must not exceed 0.5 s")

    targets_raw = raw["targets"]
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ValueError("targets must be a non-empty list")
    targets = []
    for index, target_raw in enumerate(targets_raw):
        if not isinstance(target_raw, dict):
            raise ValueError(f"targets[{index}] must be an object")
        _require_exact_keys(target_raw, {"width_m", "effort_n"}, f"targets[{index}]")
        target = CalibrationTarget(
            width_m=_finite_float(target_raw["width_m"], f"targets[{index}].width_m"),
            effort_n=_finite_float(target_raw["effort_n"], f"targets[{index}].effort_n"),
        )
        if not safety.width_min_m <= target.width_m <= safety.width_max_m:
            raise ValueError(f"targets[{index}] width is outside the plan safety range")
        if not safety.effort_min_n <= target.effort_n <= safety.effort_max_n:
            raise ValueError(f"targets[{index}] effort is outside the plan safety range")
        targets.append(target)
    for index, (previous, current) in enumerate(pairwise(targets)):
        if abs(current.width_m - previous.width_m) > safety.max_step_delta_m + 1e-12:
            raise ValueError(f"target step {index}->{index + 1} exceeds max_step_delta_m")
    if targets[-1].width_m < safety.required_final_width_min_m:
        raise ValueError("final target does not return the gripper to the required open width")

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return CalibrationPlan(
        schema_version=1,
        name=raw["name"],
        arm_namespace=namespace,
        execution_enabled=raw["execution_enabled"],
        targets=tuple(targets),
        safety=safety,
        canonical_json=canonical,
    )


def authorize_execution(
    plan: CalibrationPlan,
    token: str | None,
) -> ExecutionAuthorization:
    if not plan.execution_enabled:
        raise PermissionError("plan has execution_enabled=false")
    if token != plan.approval_token:
        raise PermissionError("approval token does not match this exact plan fingerprint")
    return ExecutionAuthorization(plan_fingerprint=plan.fingerprint)


def max_abs_delta(current: list[float], baseline: list[float]) -> float:
    if len(current) != len(baseline) or not current:
        raise ValueError("vectors must have the same non-zero length")
    return max(
        abs(current_value - baseline_value) for current_value, baseline_value in zip(current, baseline, strict=True)
    )


def max_width_speed(samples: list[tuple[float, float, float]]) -> float:
    maximum = 0.0
    for previous, current in pairwise(samples):
        delta_time = current[0] - previous[0]
        if delta_time > 0:
            maximum = max(maximum, abs(current[1] - previous[1]) / delta_time)
    return maximum


def measured_step_limit_m(safety: SafetyLimits) -> float:
    """Max |measured_width - next_target| before commanding the next step.

    Plan adjacency is still hard-capped by max_step_delta_m. Measured width may
    sit up to gripper_tolerance_m away from the previous target after settle, so
    the live step check must include that slack or exact 0.03 m plan steps fail
    spuriously (e.g. 0.08022 -> 0.05).
    """
    return safety.max_step_delta_m + safety.gripper_tolerance_m


def check_feedback_freshness(
    feedback_times: dict[str, float],
    *,
    now: float,
    required: set[str],
    maximum_age_s: float,
) -> None:
    """Fail-closed freshness check used by Gate 1 (and unit-tested offline)."""
    if maximum_age_s <= 0:
        raise ValueError("maximum_age_s must be positive")
    stale = {
        name: now - timestamp
        for name, timestamp in feedback_times.items()
        if now - timestamp > maximum_age_s
    }
    missing = sorted(required - feedback_times.keys())
    if missing or stale:
        raise RuntimeError(f"feedback freshness failure: missing={missing}, stale={stale}")


def check_feedback_health(snapshot: dict[str, Any]) -> None:
    """Fail-closed health check on a feedback snapshot (ROS-free)."""
    faults = list(snapshot.get("gripper_faults", []))
    if faults:
        raise RuntimeError(f"gripper faults: {faults}")
    err_status = int(snapshot.get("arm_err_status", 0))
    if err_status != 0:
        raise RuntimeError(f"arm err_status became {err_status}")
    motion_status = int(snapshot.get("arm_motion_status", 0))
    if motion_status != 0:
        raise RuntimeError(f"arm motion_status became {motion_status}")
