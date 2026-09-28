"""ROS-free NERO-native action contract and shadow trajectory adapter.

This module does not map another robot's joints to NERO joints. It accepts
only actions produced by a NERO-trained checkpoint with this contract:

    [joint1..joint7 absolute target in rad, absolute gripper closedness]

The adapter is deliberately non-commandable. It retains the requested targets
and creates a separately guarded trajectory for offline checks and RViz ghost
playback. Its limits are conservative diagnostic settings, not certified NERO
hardware limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from nero_pi05_bridge.offline_adapter import NERO_JOINT_NAMES
from nero_pi05_bridge.offline_adapter import NERO_POSITION_LOWER
from nero_pi05_bridge.offline_adapter import NERO_POSITION_UPPER


def _array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _array7(value: Any, name: str) -> np.ndarray:
    return _array(value, name, (7,))


@dataclass(frozen=True)
class NeroNativeContract:
    schema_version: int
    action_semantics: str
    joint_names: tuple[str, ...]
    control_frequency_hz: float
    max_abs_velocity_rad_s: np.ndarray
    max_abs_acceleration_rad_s2: np.ndarray
    soft_limit_margin_rad: float
    max_gripper_speed_m_s: float
    gripper_min_width_m: float
    gripper_max_width_m: float
    commandable: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeroNativeContract:
        if int(data.get("schema_version", 0)) != 2:
            raise ValueError("unsupported NERO native contract schema_version; expected 2")
        semantics = str(data.get("action_semantics", ""))
        expected = "absolute_joint_target_rad+absolute_gripper_closedness"
        if semantics != expected:
            raise ValueError(f"action_semantics must be {expected!r}")
        names = tuple(data.get("joint_names", ()))
        if names != NERO_JOINT_NAMES:
            raise ValueError(f"joint_names must be {NERO_JOINT_NAMES}")
        if bool(data.get("commandable", False)):
            raise ValueError("native shadow contract must not be commandable")

        frequency = float(data["control_frequency_hz"])
        velocity = _array7(data["max_abs_velocity_rad_s"], "max_abs_velocity_rad_s")
        acceleration = _array7(
            data["max_abs_acceleration_rad_s2"],
            "max_abs_acceleration_rad_s2",
        )
        margin = float(data["soft_limit_margin_rad"])
        gripper_speed = float(data["max_gripper_speed_m_s"])
        minimum = float(data["gripper_min_width_m"])
        maximum = float(data["gripper_max_width_m"])
        if frequency <= 0 or np.any(velocity <= 0) or np.any(acceleration <= 0):
            raise ValueError("control frequency, velocity, and acceleration limits must be positive")
        if margin <= 0 or np.any(NERO_POSITION_LOWER + margin >= NERO_POSITION_UPPER - margin):
            raise ValueError("soft_limit_margin_rad is invalid")
        if gripper_speed <= 0:
            raise ValueError("max_gripper_speed_m_s must be positive")
        if minimum < 0 or maximum <= minimum:
            raise ValueError("gripper calibration interval is invalid")
        return cls(
            schema_version=2,
            action_semantics=semantics,
            joint_names=names,
            control_frequency_hz=frequency,
            max_abs_velocity_rad_s=velocity,
            max_abs_acceleration_rad_s2=acceleration,
            soft_limit_margin_rad=margin,
            max_gripper_speed_m_s=gripper_speed,
            gripper_min_width_m=minimum,
            gripper_max_width_m=maximum,
        )

    @classmethod
    def load(cls, path: str | Path) -> NeroNativeContract:
        with Path(path).open(encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    @property
    def control_period_sec(self) -> float:
        return 1.0 / self.control_frequency_hz


@dataclass(frozen=True)
class NativeTrajectory:
    requested_joint_positions: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_accelerations: np.ndarray
    requested_gripper_width_m: np.ndarray
    gripper_width_m: np.ndarray
    target_jump_exceeded: np.ndarray
    velocity_limit_exceeded: np.ndarray
    acceleration_limit_exceeded: np.ndarray
    soft_limit_violated: np.ndarray
    gripper_closedness_clipped: np.ndarray
    gripper_speed_limit_exceeded: np.ndarray
    commandable: bool = False


class NeroNativeShadowAdapter:
    """Check absolute NERO targets and produce non-commandable ghost data."""

    def __init__(self, contract: NeroNativeContract):
        self.contract = contract

    def adapt_chunk(self, actions: Any, initial_state: Any) -> NativeTrajectory:
        chunk = np.asarray(actions, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 8 or not np.all(np.isfinite(chunk)):
            raise ValueError("actions must be a finite shape-(horizon, 8) array")
        state = _array(initial_state, "initial_state", (8,))
        position = state[:7].copy()
        initial_closedness = float(state[7])
        if not 0.0 <= initial_closedness <= 1.0:
            raise ValueError("initial_state gripper closedness must be in [0, 1]")

        lower = NERO_POSITION_LOWER + self.contract.soft_limit_margin_rad
        upper = NERO_POSITION_UPPER - self.contract.soft_limit_margin_rad
        if np.any(position < lower) or np.any(position > upper):
            raise ValueError("initial_state joints are outside the NERO soft-limit envelope")

        period = self.contract.control_period_sec
        previous_velocity = np.zeros(7, dtype=np.float64)
        gripper_width = self._closedness_to_width(initial_closedness)
        requested_positions = []
        positions = []
        velocities = []
        accelerations = []
        requested_widths = []
        widths = []
        jump_flags = []
        velocity_flags = []
        acceleration_flags = []
        soft_limit_flags = []
        gripper_clip_flags = []
        gripper_speed_flags = []

        for action in chunk:
            requested = action[:7].copy()
            soft_limit_violated = (requested < lower) | (requested > upper)
            soft_target = np.clip(requested, lower, upper)

            requested_delta = soft_target - position
            target_jump_exceeded = (
                np.abs(requested_delta) > self.contract.max_abs_velocity_rad_s * period
            )
            requested_velocity = requested_delta / period
            velocity_limit_exceeded = (
                np.abs(requested_velocity) > self.contract.max_abs_velocity_rad_s
            )
            velocity_limited = np.clip(
                requested_velocity,
                -self.contract.max_abs_velocity_rad_s,
                self.contract.max_abs_velocity_rad_s,
            )

            requested_acceleration = (requested_velocity - previous_velocity) / period
            acceleration_limit_exceeded = (
                np.abs(requested_acceleration) > self.contract.max_abs_acceleration_rad_s2
            )
            acceleration_to_limited_velocity = (velocity_limited - previous_velocity) / period
            acceleration = np.clip(
                acceleration_to_limited_velocity,
                -self.contract.max_abs_acceleration_rad_s2,
                self.contract.max_abs_acceleration_rad_s2,
            )
            guarded_velocity = previous_velocity + acceleration * period
            guarded_target = np.clip(position + guarded_velocity * period, lower, upper)
            guarded_velocity = (guarded_target - position) / period
            acceleration = (guarded_velocity - previous_velocity) / period

            raw_closedness = float(action[7])
            closedness = float(np.clip(raw_closedness, 0.0, 1.0))
            requested_width = self._closedness_to_width(closedness)
            requested_width_delta = requested_width - gripper_width
            max_width_delta = self.contract.max_gripper_speed_m_s * period
            gripper_speed_exceeded = abs(requested_width_delta) > max_width_delta
            guarded_width = gripper_width + float(
                np.clip(requested_width_delta, -max_width_delta, max_width_delta)
            )

            position = guarded_target
            previous_velocity = guarded_velocity
            gripper_width = guarded_width
            requested_positions.append(requested)
            positions.append(position.copy())
            velocities.append(guarded_velocity.copy())
            accelerations.append(acceleration.copy())
            requested_widths.append(requested_width)
            widths.append(gripper_width)
            jump_flags.append(target_jump_exceeded)
            velocity_flags.append(velocity_limit_exceeded)
            acceleration_flags.append(acceleration_limit_exceeded)
            soft_limit_flags.append(soft_limit_violated)
            gripper_clip_flags.append(not np.isclose(raw_closedness, closedness))
            gripper_speed_flags.append(gripper_speed_exceeded)

        return NativeTrajectory(
            requested_joint_positions=np.asarray(requested_positions),
            joint_positions=np.asarray(positions),
            joint_velocities=np.asarray(velocities),
            joint_accelerations=np.asarray(accelerations),
            requested_gripper_width_m=np.asarray(requested_widths),
            gripper_width_m=np.asarray(widths),
            target_jump_exceeded=np.asarray(jump_flags),
            velocity_limit_exceeded=np.asarray(velocity_flags),
            acceleration_limit_exceeded=np.asarray(acceleration_flags),
            soft_limit_violated=np.asarray(soft_limit_flags),
            gripper_closedness_clipped=np.asarray(gripper_clip_flags),
            gripper_speed_limit_exceeded=np.asarray(gripper_speed_flags),
        )

    def _closedness_to_width(self, closedness: float) -> float:
        return self.contract.gripper_max_width_m - closedness * (
            self.contract.gripper_max_width_m - self.contract.gripper_min_width_m
        )
