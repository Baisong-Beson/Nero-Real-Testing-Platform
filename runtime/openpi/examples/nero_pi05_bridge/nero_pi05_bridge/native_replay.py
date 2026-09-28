"""Replay NERO-native absolute action chunks through the schema-v2 shadow adapter.

This CLI is non-commandable. It reads action chunks (from JSONL shadow logs,
NPZ, or JSON), runs ``NeroNativeShadowAdapter``, and writes a ghost projection
JSON compatible with ``ghost_playback`` diagnostics.

Examples:

  PYTHONPATH=examples/nero_pi05_bridge python3 -m nero_pi05_bridge.native_replay \\
    --actions-npz path/to/actions.npy \\
    --initial-state 0,1.57,-1.57,1.57,0,0,0,0 \\
    --output artifacts/nero_native_ghost/projection.json

  PYTHONPATH=examples/nero_pi05_bridge python3 -m nero_pi05_bridge.native_replay \\
    --shadow-jsonl /tmp/nero_pi05_bridge/shadow_left_*.jsonl \\
    --output artifacts/nero_native_ghost/from_shadow.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from nero_pi05_bridge.native_adapter import NeroNativeContract
from nero_pi05_bridge.native_adapter import NeroNativeShadowAdapter
from nero_pi05_bridge.offline_adapter import NERO_POSITION_LOWER
from nero_pi05_bridge.offline_adapter import NERO_POSITION_UPPER

DEFAULT_CONTRACT = (
    Path(__file__).resolve().parents[1] / "config" / "nero_native_shadow_contract.json"
)


def _parse_state(text: str) -> np.ndarray:
    values = [float(part) for part in text.split(",") if part.strip() != ""]
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (8,):
        raise ValueError(f"initial-state must have 8 values, got {arr.shape}")
    return arr


def _load_actions_npz(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        # mean over batch if present
        arr = arr.mean(axis=0)
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError(f"Expected actions shape (H, >=8), got {arr.shape}")
    return arr[:, :8]


def _load_actions_from_shadow_jsonl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    states = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") != "inference" or not row.get("ok"):
                continue
            actions = row.get("actions", {})
            chunk = actions.get("chunk")
            if chunk is None:
                first = actions.get("first_action")
                if first is None:
                    continue
                chunk = [first]
            chunk_arr = np.asarray(chunk, dtype=np.float64)
            if chunk_arr.ndim != 2 or chunk_arr.shape[1] < 8:
                continue
            obs = row.get("observation", {})
            joints = obs.get("joint_position")
            grip = obs.get("gripper_closedness", obs.get("gripper_position"))
            if joints is None or grip is None:
                continue
            grip_val = float(grip[0] if isinstance(grip, list) else grip)
            state = np.concatenate(
                [np.asarray(joints, dtype=np.float64), np.asarray([grip_val], dtype=np.float64)]
            )
            chunks.append(chunk_arr[:, :8])
            states.append(state)
    if not chunks:
        raise ValueError(f"No usable inference chunks in {path}")
    # Use the first successful chunk for ghost projection.
    return chunks[0], states[0]


FRAME_NAMES = (
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "gripper_flange",
    "gripper_base",
    "gripper_link1",
    "gripper_link2",
)


def _configuration(joints: np.ndarray, closedness: float) -> np.ndarray:
    width = 0.099 - float(closedness) * (0.099 - 0.030)
    return np.concatenate((joints, [width, width / 2.0, -width / 2.0]))


def _frame_positions(model: Any, data: Any, frame_ids: list[int], q: np.ndarray) -> np.ndarray:
    import pinocchio as pin

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.asarray([data.oMf[frame_id].translation.copy() for frame_id in frame_ids])


def _add_fk_fields(
    report: dict[str, Any],
    *,
    urdf: Path,
    group: str,
    initial_state: np.ndarray,
    requested_joints: np.ndarray,
    guarded_joints: np.ndarray,
    requested_closedness: np.ndarray,
    guarded_closedness: np.ndarray,
) -> None:
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf))
    if model.nq != 10:
        raise ValueError(f"Expected NERO model nq=10, got {model.nq}")
    frame_ids = [model.getFrameId(name) for name in FRAME_NAMES]
    if any(frame_id >= len(model.frames) for frame_id in frame_ids):
        raise ValueError("NERO URDF is missing a required visualization frame")
    data = model.createData()

    raw_positions = []
    clipped_positions = []
    for req_q, req_g, grd_q, grd_g in zip(
        requested_joints,
        requested_closedness,
        guarded_joints,
        guarded_closedness,
        strict=True,
    ):
        raw_positions.append(
            _frame_positions(model, data, frame_ids, _configuration(req_q, float(req_g))).tolist()
        )
        clipped_positions.append(
            _frame_positions(model, data, frame_ids, _configuration(grd_q, float(grd_g))).tolist()
        )

    report["urdf"] = str(urdf)
    report["frame_names"] = list(FRAME_NAMES)
    report["initial_frame_positions"] = _frame_positions(
        model, data, frame_ids, _configuration(initial_state[:7], float(initial_state[7]))
    ).tolist()
    report["initial_clipped_frame_positions"] = report["initial_frame_positions"]
    report["initial_clipped_state"] = report["initial_state"]
    report["groups"][group]["raw_frame_positions"] = raw_positions
    report["groups"][group]["clipped_frame_positions"] = clipped_positions


def _trajectory_to_projection(
    trajectory: Any,
    *,
    initial_state: np.ndarray,
    group: str,
) -> dict[str, Any]:
    requested = np.asarray(trajectory.requested_joint_positions)
    guarded = np.asarray(trajectory.joint_positions)
    widths = np.asarray(trajectory.gripper_width_m)
    requested_widths = np.asarray(trajectory.requested_gripper_width_m)

    def pack_state(joints: np.ndarray, width_m: float) -> list[float]:
        closedness = float(np.clip((0.099 - width_m) / (0.099 - 0.030), 0.0, 1.0))
        return [*(float(v) for v in joints), closedness]

    horizon = requested.shape[0]
    raw_action_mean = [
        pack_state(requested[i], float(requested_widths[i])) for i in range(horizon)
    ]
    guarded_action_mean = [
        pack_state(guarded[i], float(widths[i])) for i in range(horizon)
    ]

    return {
        "mode": "nero_native_shadow_projection",
        "commandable": False,
        "schema_version": 2,
        "action_semantics": "absolute_joint_target_rad+absolute_gripper_closedness",
        "joint_limits_lower": NERO_POSITION_LOWER.tolist(),
        "joint_limits_upper": NERO_POSITION_UPPER.tolist(),
        "initial_state": pack_state(
            initial_state[:7], float(0.099 - initial_state[7] * (0.099 - 0.030))
        ),
        "groups": {
            group: {
                "raw_action_mean": raw_action_mean,
                "guarded_action_mean": guarded_action_mean,
                "target_jump_exceeded_rate": float(np.mean(trajectory.target_jump_exceeded)),
                "velocity_limit_exceeded_rate": float(np.mean(trajectory.velocity_limit_exceeded)),
                "acceleration_limit_exceeded_rate": float(
                    np.mean(trajectory.acceleration_limit_exceeded)
                ),
                "soft_limit_violated_rate": float(np.mean(trajectory.soft_limit_violated)),
                "gripper_speed_limit_exceeded_rate": float(
                    np.mean(trajectory.gripper_speed_limit_exceeded)
                ),
            }
        },
        "notes": [
            "Produced by NeroNativeShadowAdapter; not robot commands.",
            "Use clipped_frame_positions (guarded) for conservative ghost visualization.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--actions-npz", type=Path, default=None)
    parser.add_argument("--shadow-jsonl", type=Path, default=None)
    parser.add_argument(
        "--initial-state",
        type=str,
        default=None,
        help="Comma-separated 8D state: joint1..7 rad + gripper closedness",
    )
    parser.add_argument("--group", type=str, default="native")
    parser.add_argument("--urdf", type=Path, default=None, help="Optional NERO URDF for FK frames")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.actions_npz is None and args.shadow_jsonl is None:
        parser.error("Provide --actions-npz or --shadow-jsonl")

    contract = NeroNativeContract.load(args.contract)
    adapter = NeroNativeShadowAdapter(contract)

    if args.shadow_jsonl is not None:
        actions, initial_state = _load_actions_from_shadow_jsonl(args.shadow_jsonl)
        if args.initial_state is not None:
            initial_state = _parse_state(args.initial_state)
    else:
        if args.initial_state is None:
            parser.error("--initial-state is required with --actions-npz")
        actions = _load_actions_npz(args.actions_npz)
        initial_state = _parse_state(args.initial_state)

    trajectory = adapter.adapt_chunk(actions, initial_state)
    report = _trajectory_to_projection(trajectory, initial_state=initial_state, group=args.group)
    report["flags"] = {
        "target_jump_exceeded": trajectory.target_jump_exceeded.tolist(),
        "velocity_limit_exceeded": trajectory.velocity_limit_exceeded.tolist(),
        "acceleration_limit_exceeded": trajectory.acceleration_limit_exceeded.tolist(),
        "soft_limit_violated": trajectory.soft_limit_violated.tolist(),
        "gripper_closedness_clipped": trajectory.gripper_closedness_clipped.tolist(),
        "gripper_speed_limit_exceeded": trajectory.gripper_speed_limit_exceeded.tolist(),
    }
    report["requested_joint_positions"] = trajectory.requested_joint_positions.tolist()
    report["guarded_joint_positions"] = trajectory.joint_positions.tolist()
    report["guarded_gripper_width_m"] = trajectory.gripper_width_m.tolist()

    if args.urdf is not None:
        requested_closedness = np.clip(
            (0.099 - np.asarray(trajectory.requested_gripper_width_m)) / (0.099 - 0.030),
            0.0,
            1.0,
        )
        guarded_closedness = np.clip(
            (0.099 - np.asarray(trajectory.gripper_width_m)) / (0.099 - 0.030),
            0.0,
            1.0,
        )
        _add_fk_fields(
            report,
            urdf=args.urdf,
            group=args.group,
            initial_state=initial_state,
            requested_joints=np.asarray(trajectory.requested_joint_positions),
            guarded_joints=np.asarray(trajectory.joint_positions),
            requested_closedness=requested_closedness,
            guarded_closedness=guarded_closedness,
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote non-commandable native ghost projection to {output}")
    print(
        "jump_rate="
        f"{float(np.mean(trajectory.target_jump_exceeded)):.4f} "
        "soft_limit_rate="
        f"{float(np.mean(trajectory.soft_limit_violated)):.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
