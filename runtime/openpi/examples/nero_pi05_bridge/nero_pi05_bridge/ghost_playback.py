"""Load non-commandable NERO FK projections for isolated RViz playback."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

JOINT_NAMES = (*(f"joint{index}" for index in range(1, 8)), "gripper")
GRIPPER_MIN_WIDTH_M = 0.030
GRIPPER_MAX_WIDTH_M = 0.099


@dataclass(frozen=True)
class GhostTrajectory:
    group: str
    initial_positions: tuple[float, ...]
    frames: tuple[tuple[float, ...], ...]
    initial_tcp_position: tuple[float, float, float]
    tcp_path: tuple[tuple[float, float, float], ...]
    initial_limit_violations: tuple[dict[str, Any], ...]


def _joint_positions(state: list[float]) -> tuple[float, ...]:
    if len(state) != 8:
        raise ValueError(f"expected 8 state values, got {len(state)}")
    closedness = min(max(float(state[7]), 0.0), 1.0)
    width = GRIPPER_MAX_WIDTH_M - closedness * (GRIPPER_MAX_WIDTH_M - GRIPPER_MIN_WIDTH_M)
    return (*(float(value) for value in state[:7]), width)


def load_ghost_trajectory(
    path: Path,
    *,
    group: str,
    trajectory_mode: str,
    initial_mode: str,
) -> GhostTrajectory:
    report = json.loads(path.read_text())
    if report.get("commandable") is not False:
        raise ValueError("projection must explicitly contain commandable=false")
    if group not in report.get("groups", {}):
        raise ValueError(f"projection does not contain group {group!r}")
    if trajectory_mode not in {"raw", "clipped"}:
        raise ValueError("trajectory_mode must be raw or clipped")
    if initial_mode not in {"requested", "clipped"}:
        raise ValueError("initial_mode must be requested or clipped")

    initial_key = "initial_state" if initial_mode == "requested" else "initial_clipped_state"
    initial_state = report.get(initial_key, report.get("initial_state"))
    if initial_state is None:
        raise ValueError("projection is missing its initial state")
    group_report = report["groups"][group]
    actions = group_report.get("raw_action_mean")
    if not isinstance(actions, list) or len(actions) < 1:
        raise ValueError("projection must contain a non-empty raw_action_mean")
    horizon = len(actions)

    lower = [float(value) for value in report["joint_limits_lower"]]
    upper = [float(value) for value in report["joint_limits_upper"]]
    frame_names = report.get("frame_names", [])
    try:
        tcp_index = frame_names.index("gripper_base")
    except ValueError as error:
        raise ValueError("projection frame_names must contain gripper_base") from error
    frame_positions_key = f"{trajectory_mode}_frame_positions"
    projected_positions = group_report.get(frame_positions_key)
    if not isinstance(projected_positions, list) or len(projected_positions) != horizon:
        raise ValueError(
            f"projection must contain {horizon}-step {frame_positions_key}, "
            f"got {0 if not isinstance(projected_positions, list) else len(projected_positions)}"
        )

    initial_frames_key = (
        "initial_frame_positions"
        if initial_mode == "requested"
        else "initial_clipped_frame_positions"
    )
    initial_frame_positions = report.get(initial_frames_key)
    if not isinstance(initial_frame_positions, list):
        raise ValueError(f"projection is missing {initial_frames_key}")

    def tcp_position(frames: list[Any], *, label: str) -> tuple[float, float, float]:
        try:
            position = frames[tcp_index]
        except IndexError as error:
            raise ValueError(f"{label} is missing gripper_base frame") from error
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"{label} gripper_base position must have 3 values")
        return tuple(float(value) for value in position)

    frames = []
    for index, action in enumerate(actions):
        if not isinstance(action, list) or len(action) != 8:
            raise ValueError(f"action step {index} does not contain 8 values")
        state = [float(value) for value in action]
        if trajectory_mode == "clipped":
            state[:7] = [
                min(max(value, minimum), maximum)
                for value, minimum, maximum in zip(state[:7], lower, upper, strict=True)
            ]
            state[7] = min(max(state[7], 0.0), 1.0)
        frames.append(_joint_positions(state))
    return GhostTrajectory(
        group=group,
        initial_positions=_joint_positions([float(value) for value in initial_state]),
        frames=tuple(frames),
        initial_tcp_position=tcp_position(initial_frame_positions, label=initial_frames_key),
        tcp_path=tuple(
            tcp_position(step, label=f"{frame_positions_key}[{index}]")
            for index, step in enumerate(projected_positions)
        ),
        initial_limit_violations=tuple(report.get("initial_limit_violations", [])),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline-only NERO RViz ghost playback")
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--group", default="approach")
    parser.add_argument("--trajectory-mode", choices=("raw", "clipped"), default="clipped")
    parser.add_argument("--initial-mode", choices=("requested", "clipped"), default="requested")
    parser.add_argument("--rate-hz", type=float, default=8.0)
    parser.add_argument("--initial-hold-s", type=float, default=2.0)
    parser.add_argument("--loop", choices=("true", "false"), default="true")
    from rclpy.utilities import remove_ros_args

    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    if args.rate_hz <= 0.0 or args.initial_hold_s < 0.0:
        parser.error("rate-hz must be positive and initial-hold-s must be non-negative")
    trajectory = load_ghost_trajectory(
        args.projection,
        group=args.group,
        trajectory_mode=args.trajectory_mode,
        initial_mode=args.initial_mode,
    )

    from nero_pi05_bridge.ghost_playback_ros import run_playback

    return run_playback(
        trajectory,
        rate_hz=args.rate_hz,
        initial_hold_s=args.initial_hold_s,
        loop=args.loop == "true",
    )


if __name__ == "__main__":
    raise SystemExit(main())
