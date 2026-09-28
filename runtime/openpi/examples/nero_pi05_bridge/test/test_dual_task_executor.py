"""Unit tests for the dual-arm Gate 3 plan and safety helpers (no ROS)."""

from __future__ import annotations

import json
from pathlib import Path

from nero_pi05_bridge.dual_task_executor import authorize_execution
from nero_pi05_bridge.dual_task_executor import evaluate_dual_step
from nero_pi05_bridge.dual_task_executor import load_plan
import numpy as np
import pytest


def _plan_data():
    return {
        "schema_version": 1,
        "name": "test_dual_gate3",
        "right_arm_namespace": "/right_arm",
        "left_arm_namespace": "/left_arm",
        "deadman_topic": "/right_arm/gate2_deadman",
        "execution_enabled": False,
        "control_frequency_hz": 20.0,
        "max_total_duration_s": 20.0,
        "chunk_steps_per_inference": 8,
        "max_command_lead_rad": 0.25,
        "driver_speed_percent": 5,
        "right_gripper_min_width_m": 0.0,
        "right_gripper_max_width_m": 0.0996,
        "left_gripper_min_width_m": 0.0,
        "left_gripper_max_width_m": 0.1005,
        "locked_right_gripper_width_m": 0.0996,
        "locked_left_gripper_width_m": 0.1005,
        "gripper_control": "policy",
        "max_gripper_speed_m_s": 1.4,
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
        "right_home_joints_rad": [0.0, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 0.0, 0.0, 0.0],
        "left_home_joints_rad": [0.0, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 0.0, 0.0, 0.0],
        "home_tolerance_rad": 0.05,
        "stages": [
            {
                "name": "marker_drawer",
                "prompt": "pick up the marker with the right arm, open the drawer with the left arm, put the marker in the drawer, and close the drawer",
                "max_duration_s": 16.0,
            }
        ],
        "policy_host": "127.0.0.1",
        "policy_port": 8000,
        "notes": ["unit-test"],
    }


def _write_plan(tmp_path, data):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data))
    return path


def _hold_step_kwargs(plan, *, right_delta=0.0, left_delta=0.0, right_tcp=None, left_tcp=None):
    home_r = plan.right_home_joints_rad
    home_l = plan.left_home_joints_rad
    requested_r = home_r + right_delta
    requested_l = home_l + left_delta
    tcp_r = np.zeros(3) if right_tcp is None else np.asarray(right_tcp, dtype=np.float64)
    tcp_l = np.zeros(3) if left_tcp is None else np.asarray(left_tcp, dtype=np.float64)
    return {
        "right_current_joints": home_r,
        "left_current_joints": home_l,
        "right_requested_joints": requested_r,
        "left_requested_joints": requested_l,
        "right_requested_closedness": 0.0,
        "left_requested_closedness": 0.0,
        "right_current_tcp": tcp_r,
        "left_current_tcp": tcp_l,
        "right_requested_tcp": tcp_r,
        "left_requested_tcp": tcp_l,
        "right_previous_joints": None,
        "left_previous_joints": None,
        "right_previous_velocity": None,
        "left_previous_velocity": None,
        "right_previous_tcp": None,
        "left_previous_tcp": None,
        "right_previous_gripper_width_m": 0.0996,
        "left_previous_gripper_width_m": 0.1005,
        "is_first_command": True,
        "check_command_lead": False,
    }


def test_checked_in_dual_plan_loads():
    plan = load_plan(
        Path(__file__).resolve().parents[1] / "config" / "nero_dual_v1_plan.json"
    )
    assert plan.right_arm_namespace == "/right_arm"
    assert plan.left_arm_namespace == "/left_arm"
    assert plan.deadman_topic == "/right_arm/gate2_deadman"
    assert plan.execution_enabled is True
    assert plan.chunk_steps_per_inference == 16
    assert plan.max_total_duration_s == 35.0
    assert plan.stages[0].prompt.startswith("pick up the marker")
    assert plan.approval_token.startswith("NERO_DUAL_GATE3_EXECUTE_")
    assert "UNCALIBRATED" in plan.notes[1]


def test_checked_in_probe_plan_is_short():
    plan = load_plan(
        Path(__file__).resolve().parents[1] / "config" / "nero_dual_v1_probe_plan.json"
    )
    assert plan.max_total_duration_s == 10.0
    assert plan.stages[0].max_duration_s == 8.0


def test_prompt_override_changes_effective_plan_and_token(tmp_path):
    path = _write_plan(tmp_path, _plan_data())
    original = load_plan(path)
    overridden = load_plan(path, prompt_override="  close the drawer  ")
    assert overridden.stages[0].prompt == "close the drawer"
    assert overridden.approval_token != original.approval_token


def test_prompt_override_rejects_blank_text(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        load_plan(_write_plan(tmp_path, _plan_data()), prompt_override="   ")


def test_rejects_unknown_keys(tmp_path):
    data = _plan_data()
    data["extra"] = True
    with pytest.raises(ValueError, match="keys mismatch"):
        load_plan(_write_plan(tmp_path, data))


def test_rejects_identical_namespaces(tmp_path):
    data = _plan_data()
    data["left_arm_namespace"] = "/right_arm"
    with pytest.raises(ValueError, match="must differ"):
        load_plan(_write_plan(tmp_path, data))


def test_execution_requires_matching_token(tmp_path):
    data = _plan_data()
    data["execution_enabled"] = True
    plan = load_plan(_write_plan(tmp_path, data))
    with pytest.raises(PermissionError, match="does not match"):
        authorize_execution(plan, "wrong")
    auth = authorize_execution(plan, plan.approval_token)
    assert auth.plan_fingerprint == plan.fingerprint


def test_execution_disabled_even_with_token(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    with pytest.raises(PermissionError, match="execution_enabled=false"):
        authorize_execution(plan, plan.approval_token)


def test_hold_step_has_no_violations(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    violations = evaluate_dual_step(plan, **_hold_step_kwargs(plan))
    assert violations == []


def test_left_arm_soft_limit_is_prefixed_and_stops_both(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    kwargs = _hold_step_kwargs(plan)
    kwargs["left_requested_joints"] = plan.left_home_joints_rad.copy()
    kwargs["left_requested_joints"][0] = 10.0
    violations = evaluate_dual_step(plan, **kwargs)
    assert violations
    assert all(item.code.startswith("left_") or item.code.startswith("right_") for item in violations)
    assert any(item.code.startswith("left_") for item in violations)


def test_right_arm_velocity_violation_is_independent(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    kwargs = _hold_step_kwargs(plan, right_delta=1.0)
    violations = evaluate_dual_step(plan, **kwargs)
    assert any(item.code == "right_joint_velocity" for item in violations)
    assert not any(item.code.startswith("left_") for item in violations)


def test_any_arm_violation_is_enough_to_hard_stop(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    kwargs = _hold_step_kwargs(plan, left_tcp=[10.0, 0.0, 0.0])
    kwargs["left_requested_tcp"] = np.asarray([10.0, 0.0, 0.0])
    violations = evaluate_dual_step(plan, **kwargs)
    assert any(item.code == "left_workspace" for item in violations)
    assert violations  # ROS loop hard-stops both arms whenever this list is non-empty
