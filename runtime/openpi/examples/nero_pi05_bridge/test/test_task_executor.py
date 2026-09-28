"""Unit tests for Gate 3 task executor pure logic (no ROS / no hardware)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from nero_pi05_bridge.arm_executor import validate_joint_target
from nero_pi05_bridge.task_executor import StageMachine
from nero_pi05_bridge.task_executor import assert_near_home
from nero_pi05_bridge.task_executor import authorize_execution
from nero_pi05_bridge.task_executor import gripper_width_from_closedness
from nero_pi05_bridge.task_executor import load_plan
from nero_pi05_bridge.task_executor import validate_command_lead
from nero_pi05_bridge.task_executor import validate_gripper_command
import pytest


def _plan_data():
    return {
        "schema_version": 1,
        "name": "test_gate3",
        "arm_namespace": "/right_arm",
        "execution_enabled": False,
        "control_frequency_hz": 20.0,
        "max_total_duration_s": 20.0,
        "chunk_steps_per_inference": 8,
        "max_command_lead_rad": 0.25,
        "driver_speed_percent": 5,
        "locked_gripper_width_m": 0.099,
        "gripper_control": "locked",
        "max_gripper_speed_m_s": 0.15,
        "gripper_effort_n": 1.0,
        "deadman_timeout_s": 0.2,
        "feedback_max_age_s": 1.5,
        "inference_timeout_s": 30.0,
        "max_abs_velocity_rad_s": [2.0] * 7,
        "max_abs_acceleration_rad_s2": [40.0] * 7,
        "soft_limit_margin_rad": 0.05,
        "tcp_jump_threshold_m": 0.05,
        "workspace_aabb": {
            "x_min": -0.7,
            "x_max": 0.7,
            "y_min": -0.7,
            "y_max": 0.7,
            "z_min": -0.3,
            "z_max": 0.9,
        },
        "home_joints_rad": [0.0, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 0.0, 0.0, 0.0],
        "home_tolerance_rad": 0.05,
        "stages": [
            {
                "name": "approach",
                "prompt": "move the left gripper close to the plastic bottle",
                "max_duration_s": 8.0,
            },
            {
                "name": "pick",
                "prompt": "pick up the plastic bottle",
                "max_duration_s": 8.0,
            },
        ],
        "policy_host": "127.0.0.1",
        "policy_port": 8000,
        "notes": ["unit-test"],
    }


def _write_plan(tmp_path, data):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data))
    return path


def test_checked_in_gate3_plan_loads_disabled():
    plan = load_plan(
        Path(__file__).resolve().parents[1] / "config" / "nero_gate3_pick_plan.json"
    )
    assert plan.arm_namespace == "/right_arm"
    assert plan.execution_enabled is False
    assert plan.gripper_control == "locked"
    assert len(plan.stages) == 2
    assert plan.stages[0].prompt.startswith("move the left gripper")
    assert plan.stages[1].prompt == "pick up the plastic bottle"
    assert plan.approval_token.startswith("NERO_GATE3_EXECUTE_")


def test_execution_disabled_even_with_token(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    with pytest.raises(PermissionError, match="execution_enabled=false"):
        authorize_execution(plan, plan.approval_token)


def test_execution_requires_matching_token(tmp_path):
    data = _plan_data()
    data["execution_enabled"] = True
    plan = load_plan(_write_plan(tmp_path, data))
    with pytest.raises(PermissionError, match="does not match"):
        authorize_execution(plan, "wrong")
    auth = authorize_execution(plan, plan.approval_token)
    assert auth.plan_fingerprint == plan.fingerprint


def test_rejects_oversized_total_duration(tmp_path):
    data = _plan_data()
    data["max_total_duration_s"] = 31.0
    with pytest.raises(ValueError, match="max_total_duration_s"):
        load_plan(_write_plan(tmp_path, data))


def test_rejects_chunk_steps_beyond_action_horizon(tmp_path):
    data = _plan_data()
    data["chunk_steps_per_inference"] = 17
    with pytest.raises(ValueError, match="chunk_steps_per_inference"):
        load_plan(_write_plan(tmp_path, data))


def test_rejects_nonpositive_chunk_steps(tmp_path):
    data = _plan_data()
    data["chunk_steps_per_inference"] = 0
    with pytest.raises(ValueError, match="chunk_steps_per_inference"):
        load_plan(_write_plan(tmp_path, data))


def test_rejects_oversized_command_lead(tmp_path):
    data = _plan_data()
    data["max_command_lead_rad"] = 0.6
    with pytest.raises(ValueError, match="max_command_lead_rad"):
        load_plan(_write_plan(tmp_path, data))


def test_accepts_driver_speed_matching_training_pace(tmp_path):
    data = _plan_data()
    data["driver_speed_percent"] = 35
    assert load_plan(_write_plan(tmp_path, data)).driver_speed_percent == 35


def test_rejects_driver_speed_above_training_pace(tmp_path):
    data = _plan_data()
    data["driver_speed_percent"] = 36
    with pytest.raises(ValueError, match="driver_speed_percent"):
        load_plan(_write_plan(tmp_path, data))


def test_command_lead_within_limit_is_clean(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    measured = np.array([0.0, 1.5, 1.5, 1.4, 0.0, 0.0, 0.0])
    requested = measured + 0.1
    assert validate_command_lead(plan, requested_joints=requested, measured_joints=measured) == []


def test_command_lead_hard_stops_when_servo_stops_following(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    measured = np.array([0.0, 1.5, 1.5, 1.4, 0.0, 0.0, 0.0])
    requested = measured.copy()
    requested[3] += 0.4
    violations = validate_command_lead(
        plan, requested_joints=requested, measured_joints=measured
    )
    assert [v.code for v in violations] == ["command_lead"]
    assert "joint4" in violations[0].message


def test_v3_plan_starts_from_training_start_pose_not_terminal_pose():
    plan = load_plan(
        Path(__file__).resolve().parents[1] / "config" / "nero_gate3_pick_v3_plan.json"
    )
    # The policy places the gripper close at horizon step 8..15, so a replay window that
    # stops short of the full 16 steps discards every grasp command.
    assert plan.chunk_steps_per_inference == 16
    assert plan.max_total_duration_s == pytest.approx(30.0)
    # The median start pose sits only ~0.28 rad from the terminal pose
    # [0, pi/2, pi/2, pi/2, 0, 0, 0], so gross distance is not the useful guard. What the
    # live probe actually responds to is these two joints:
    # joint1 is base yaw, and holding it away from 0 both aims the arm at the bottle and
    # keeps the pose out of the terminal cluster that the policy reads as task-complete.
    assert plan.home_joints_rad[0] > 0.2
    # joint4 aims the wrist camera. The training start mean (1.3474) is dragged down by a
    # single outlier episode and makes the policy read the scene as task-complete; the
    # median (1.4975) is what flips the live probe from RETRACT to REACH.
    assert plan.home_joints_rad[3] == pytest.approx(1.4975, abs=5e-3)


def test_rejects_stage_budget_over_total(tmp_path):
    data = _plan_data()
    data["stages"][0]["max_duration_s"] = 15.0
    data["stages"][1]["max_duration_s"] = 15.0
    with pytest.raises(ValueError, match="sum of stage"):
        load_plan(_write_plan(tmp_path, data))


def test_closedness_to_width_endpoints():
    assert gripper_width_from_closedness(0.0) == pytest.approx(0.099)
    assert gripper_width_from_closedness(1.0) == pytest.approx(0.030)
    assert gripper_width_from_closedness(0.5) == pytest.approx(0.0645)


def test_gripper_closedness_small_overshoot_is_clipped_not_hard_stop(tmp_path):
    data = _plan_data()
    data["gripper_control"] = "policy"
    plan = load_plan(_write_plan(tmp_path, data))
    # Live pi05_nero_lora_v3 emitted -0.0026 on the first tick; must not abort.
    violations = validate_gripper_command(
        plan, previous_width_m=None, requested_closedness=-0.0026
    )
    assert violations == []
    assert gripper_width_from_closedness(-0.0026) == pytest.approx(0.099)


def test_gripper_closedness_large_overshoot_hard_stops(tmp_path):
    data = _plan_data()
    data["gripper_control"] = "policy"
    plan = load_plan(_write_plan(tmp_path, data))
    violations = validate_gripper_command(
        plan, previous_width_m=None, requested_closedness=-0.2
    )
    assert any(v.code == "gripper_closedness_range" for v in violations)


def test_gripper_speed_hard_stop(tmp_path):
    data = _plan_data()
    data["gripper_control"] = "policy"
    data["max_gripper_speed_m_s"] = 0.02  # 0.001 m / 0.05 s
    plan = load_plan(_write_plan(tmp_path, data))
    # 0 -> 1 closedness is 0.069 m in one period >> 0.001 m
    violations = validate_gripper_command(
        plan, previous_width_m=0.099, requested_closedness=1.0
    )
    assert any(v.code == "gripper_speed" for v in violations)


def test_assert_near_home(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    home = plan.home_joints_rad.copy()
    assert_near_home(plan, home)
    far = home.copy()
    far[0] = 0.5
    with pytest.raises(RuntimeError, match="home tolerance"):
        assert_near_home(plan, far)


def test_stage_machine_timeout_and_manual_advance(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    machine = StageMachine(stages=plan.stages).start(0.0)
    assert machine.current.name == "approach"
    machine, changed = machine.maybe_advance(1.0, requested=False)
    assert changed is False
    machine, changed = machine.maybe_advance(8.0, requested=False)
    assert changed is True
    assert machine.current.name == "pick"
    machine, changed = machine.maybe_advance(10.0, requested=True)
    assert changed is True
    assert machine.done is True


def test_gate3_plan_satisfies_validate_joint_target_contract(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    home = plan.home_joints_rad.copy()
    ok = validate_joint_target(
        plan,
        current_joints=home,
        previous_joints=None,
        previous_velocity=None,
        requested_joints=home + 0.001,
        is_first_command=True,
    )
    assert ok == []
