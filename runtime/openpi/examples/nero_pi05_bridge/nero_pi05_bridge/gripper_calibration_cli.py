"""Command-line entry point for NERO Gate 1 gripper calibration."""

import argparse
import json
from pathlib import Path

from nero_pi05_bridge.gripper_calibration import authorize_execution
from nero_pi05_bridge.gripper_calibration import load_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed NERO Gate 1 gripper calibration",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("plan", "audit", "execute"),
        default="plan",
        help="plan is offline-only; audit creates subscriptions only",
    )
    parser.add_argument("--approval-token")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/nero_gate1_gripper_calibration"),
    )
    parser.add_argument(
        "--keep-arm-enabled",
        action="store_true",
        help=(
            "After execute, close the control gate but leave the arm enabled so "
            "it holds pose (needed before a follow-on home/park in the experiment loop)"
        ),
    )
    args = parser.parse_args()

    plan = load_plan(args.plan)
    print(json.dumps(plan.summary(), indent=2))
    if args.mode == "plan":
        return 0
    authorization = None
    if args.mode == "execute":
        authorization = authorize_execution(plan, args.approval_token)
    elif args.approval_token is not None:
        parser.error("--approval-token is accepted only in execute mode")

    from nero_pi05_bridge.gripper_calibration_ros import run_calibration

    return run_calibration(
        plan,
        mode=args.mode,
        output_dir=args.output_dir,
        authorization=authorization,
        keep_arm_enabled=bool(args.keep_arm_enabled),
    )


if __name__ == "__main__":
    raise SystemExit(main())
