"""Fail-closed, ROS-free diagnostics for candidate DROID-to-NERO mappings.

This module is intentionally incapable of publishing robot commands.  A
candidate mapping is useful only for quantifying clipping and joint-limit risk
while the real cross-robot joint mapping remains unverified.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

NERO_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
NERO_POSITION_LOWER = np.array(
    [-2.705261, -1.745330, -2.757621, -1.012291, -2.757621, -0.733039, -1.570797],
    dtype=np.float64,
)
NERO_POSITION_UPPER = np.array(
    [2.705261, 1.745330, 2.757621, 2.146755, 2.757621, 0.959932, 1.570797],
    dtype=np.float64,
)
DROID_ACTION_Q01 = np.array(
    [-0.4580, -0.8076, -0.4472, -0.9268, -0.6456, -0.6460, -0.7616],
    dtype=np.float64,
)
DROID_ACTION_Q99 = np.array(
    [0.4476, 0.7652, 0.4480, 0.7944, 0.6484, 0.6628, 0.7344],
    dtype=np.float64,
)
DROID_STATE_Q01 = np.array(
    [-0.82797, -0.83983, -0.84255, -2.77302, -1.84262, 1.17166, -2.04726],
    dtype=np.float64,
)
DROID_STATE_Q99 = np.array(
    [0.89965, 1.38547, 0.69203, -0.45420, 1.73231, 3.46730, 2.19850],
    dtype=np.float64,
)


def _array7(value: Any, name: str, *, dtype=np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


@dataclass(frozen=True)
class CandidateMapping:
    """A non-commandable mapping hypothesis for offline risk analysis."""

    source_indices: np.ndarray
    direction: np.ndarray
    velocity_scale: np.ndarray
    max_abs_velocity: np.ndarray
    control_period_sec: float
    low_pass_alpha: float
    soft_limit_margin_rad: float
    gripper_max_width_m: float
    mapping_status: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateMapping:
        source_indices = _array7(data["source_indices"], "source_indices", dtype=np.int64)
        if sorted(source_indices.tolist()) != list(range(7)):
            raise ValueError("source_indices must be a permutation of 0..6")
        direction = _array7(data["direction"], "direction")
        if not np.all(np.isin(direction, (-1.0, 1.0))):
            raise ValueError("direction values must be -1 or 1")
        velocity_scale = _array7(data["velocity_scale"], "velocity_scale")
        max_abs_velocity = _array7(data["max_abs_velocity"], "max_abs_velocity")
        if np.any(velocity_scale <= 0) or np.any(max_abs_velocity <= 0):
            raise ValueError("velocity scales and limits must be positive")

        period = float(data["control_period_sec"])
        alpha = float(data["low_pass_alpha"])
        margin = float(data["soft_limit_margin_rad"])
        gripper_width = float(data["gripper_max_width_m"])
        if period <= 0 or not 0 < alpha <= 1:
            raise ValueError("control period must be positive and alpha must be in (0, 1]")
        if margin <= 0 or gripper_width <= 0:
            raise ValueError("soft-limit margin and gripper width must be positive")
        if np.any(NERO_POSITION_LOWER + margin >= NERO_POSITION_UPPER - margin):
            raise ValueError("soft-limit margin leaves an empty range")

        status = str(data.get("mapping_status", "unverified"))
        if status == "verified":
            raise ValueError(
                "This offline-only tool refuses mapping_status='verified'; "
                "verification requires an independent hardware safety process"
            )
        return cls(
            source_indices=source_indices,
            direction=direction,
            velocity_scale=velocity_scale,
            max_abs_velocity=max_abs_velocity,
            control_period_sec=period,
            low_pass_alpha=alpha,
            soft_limit_margin_rad=margin,
            gripper_max_width_m=gripper_width,
            mapping_status=status,
        )

    @classmethod
    def load(cls, path: str | Path) -> CandidateMapping:
        with Path(path).open(encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    @property
    def commandable(self) -> bool:
        return False


@dataclass(frozen=True)
class AdaptedStep:
    candidate_velocity: np.ndarray
    target_position: np.ndarray
    gripper_width_m: float
    source_outside_training_quantiles: np.ndarray
    velocity_clipped: np.ndarray
    position_guarded: np.ndarray
    gripper_clipped: bool
    commandable: bool = False


class OfflineAdapter:
    """Convert one logged DROID action into a guarded NERO hypothesis."""

    def __init__(self, mapping: CandidateMapping):
        self.mapping = mapping

    def adapt(
        self,
        droid_action: Any,
        nero_position: Any,
        previous_velocity: Any | None = None,
    ) -> AdaptedStep:
        action = np.asarray(droid_action, dtype=np.float64)
        if action.shape != (8,) or not np.all(np.isfinite(action)):
            raise ValueError("droid_action must be a finite shape-(8,) vector")
        position = _array7(nero_position, "nero_position")
        if previous_velocity is None:
            previous = np.zeros(7, dtype=np.float64)
        else:
            previous = _array7(previous_velocity, "previous_velocity")

        source_velocity = action[:7]
        raw_velocity = (
            source_velocity[self.mapping.source_indices]
            * self.mapping.direction
            * self.mapping.velocity_scale
        )
        clipped_velocity = np.clip(
            raw_velocity,
            -self.mapping.max_abs_velocity,
            self.mapping.max_abs_velocity,
        )
        velocity_clipped = ~np.isclose(raw_velocity, clipped_velocity)
        filtered_velocity = (
            self.mapping.low_pass_alpha * clipped_velocity
            + (1.0 - self.mapping.low_pass_alpha) * previous
        )

        lower = NERO_POSITION_LOWER + self.mapping.soft_limit_margin_rad
        upper = NERO_POSITION_UPPER - self.mapping.soft_limit_margin_rad
        proposed = position + filtered_velocity * self.mapping.control_period_sec
        target = np.clip(proposed, lower, upper)
        position_guarded = ~np.isclose(proposed, target)

        # If already outside the software envelope, only permit motion inward.
        outward_low = (position <= lower) & (filtered_velocity < 0)
        outward_high = (position >= upper) & (filtered_velocity > 0)
        outward = outward_low | outward_high
        target[outward] = position[outward]
        position_guarded |= outward
        safe_velocity = (target - position) / self.mapping.control_period_sec

        gripper_closedness = float(action[7])
        clipped_closedness = float(np.clip(gripper_closedness, 0.0, 1.0))
        gripper_width = self.mapping.gripper_max_width_m * (1.0 - clipped_closedness)

        return AdaptedStep(
            candidate_velocity=safe_velocity,
            target_position=target,
            gripper_width_m=gripper_width,
            source_outside_training_quantiles=(
                (source_velocity < DROID_ACTION_Q01) | (source_velocity > DROID_ACTION_Q99)
            ),
            velocity_clipped=velocity_clipped,
            position_guarded=position_guarded,
            gripper_clipped=not np.isclose(gripper_closedness, clipped_closedness),
        )
