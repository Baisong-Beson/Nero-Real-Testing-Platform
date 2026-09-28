"""Replay NERO shadow JSONL logs through the non-commandable adapter."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from nero_pi05_bridge.offline_adapter import DROID_STATE_Q01
from nero_pi05_bridge.offline_adapter import DROID_STATE_Q99
from nero_pi05_bridge.offline_adapter import NERO_JOINT_NAMES
from nero_pi05_bridge.offline_adapter import CandidateMapping
from nero_pi05_bridge.offline_adapter import OfflineAdapter


def replay(paths: list[str], mapping: CandidateMapping) -> dict:
    adapter = OfflineAdapter(mapping)
    counts = {
        "rows": 0,
        "state_quantile_exceed": np.zeros(7, dtype=np.int64),
        "source_quantile_exceed": np.zeros(7, dtype=np.int64),
        "velocity_clipped": np.zeros(7, dtype=np.int64),
        "position_guarded": np.zeros(7, dtype=np.int64),
        "gripper_clipped": 0,
    }
    target_min = np.full(7, np.inf)
    target_max = np.full(7, -np.inf)

    for path in paths:
        previous_velocity = np.zeros(7, dtype=np.float64)
        with Path(path).open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                if row.get("event") != "inference" or row.get("ok") is not True:
                    continue
                action = row["actions"].get("first_action")
                position = row["observation"]["joint_position"]
                step = adapter.adapt(action, position, previous_velocity)
                previous_velocity = step.candidate_velocity
                counts["rows"] += 1
                counts["state_quantile_exceed"] += (np.asarray(position) < DROID_STATE_Q01) | (
                    np.asarray(position) > DROID_STATE_Q99
                )
                counts["source_quantile_exceed"] += step.source_outside_training_quantiles
                counts["velocity_clipped"] += step.velocity_clipped
                counts["position_guarded"] += step.position_guarded
                counts["gripper_clipped"] += int(step.gripper_clipped)
                target_min = np.minimum(target_min, step.target_position)
                target_max = np.maximum(target_max, step.target_position)

    total = counts["rows"]
    if total == 0:
        raise ValueError("No successful inference rows were found")

    def fraction(values) -> list[float]:
        return (np.asarray(values, dtype=np.float64) / total).tolist()

    return {
        "mode": "offline_only",
        "commandable": False,
        "mapping_status": mapping.mapping_status,
        "limitations": [
            "Cross-robot joint permutation, direction, and scale are hypotheses only.",
            "Existing logs contain first_action only; full 15-step chunks cannot be replayed.",
            "DROID state quantiles are compared dimension-wise, before any cross-robot remapping.",
            "No collision model or self-collision check is performed.",
        ],
        "files": paths,
        "successful_rows": total,
        "joint_names": list(NERO_JOINT_NAMES),
        "state_quantile_exceed_fraction": fraction(counts["state_quantile_exceed"]),
        "source_quantile_exceed_fraction": fraction(counts["source_quantile_exceed"]),
        "velocity_clipped_fraction": fraction(counts["velocity_clipped"]),
        "position_guarded_fraction": fraction(counts["position_guarded"]),
        "gripper_clipped_fraction": counts["gripper_clipped"] / total,
        "candidate_target_min": target_min.tolist(),
        "candidate_target_max": target_max.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--glob", action="append", dest="patterns", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    paths = sorted({path for pattern in args.patterns for path in glob.glob(pattern)})
    if not paths:
        raise SystemExit("No files matched")
    report = replay(paths, CandidateMapping.load(args.config))
    rendered = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
