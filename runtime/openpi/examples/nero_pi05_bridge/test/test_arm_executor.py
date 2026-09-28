"""Unit tests for Gate 2 arm executor pure logic (no ROS / no hardware)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from nero_pi05_bridge.arm_executor import authorize_execution
from nero_pi05_bridge.arm_executor import load_plan
from nero_pi05_bridge.arm_executor import locked_gripper_command
from nero_pi05_bridge.arm_executor import validate_joint_target
import pytest


def _plan_data():
    return {
        "schema_version": 1,
        "name": "test_gate2",
        "arm_namespace": "/right_arm",
        "execution_enabled": False,
        "control_frequency_hz": 20.0,
        "test_window_s": 1.5,
        "driver_speed_percent": 5,
        "locked_gripper_width_m": 0.099,
        "deadman_timeout_s": 0.2,
        "feedback_max_age_s": 0.25,
        "inference_timeout_s": 2.0,
        "max_abs_velocity_rad_s": [0.05] * 7,
        "max_abs_acceleration_rad_s2": [0.1] * 7,
        "soft_limit_margin_rad": 0.1,
        "tcp_jump_threshold_m": 0.05,
        "workspace_aabb": {
            "x_min": -0.5,
            "x_max": 0.5,
            "y_min": -0.5,
            "y_max": 0.5,
            "z_min": 0.0,
            "z_max": 0.7,
        },
        "prompt": "pick up the plastic bottle",
        "policy_host": "127.0.0.1",
        "policy_port": 8000,
        "notes": ["unit-test"],
    }


def _write_plan(tmp_path, data):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data))
    return path


def test_checked_in_gate2_plan_loads_disabled():
    plan = load_plan(
        Path(__file__).resolve().parents[1] / "config" / "nero_gate2_arm_plan.json"
    )
    assert plan.arm_namespace == "/right_arm"
    assert plan.execution_enabled is False
    assert plan.driver_speed_percent == 5
    assert plan.tcp_jump_threshold_m == 0.05
    assert locked_gripper_command(plan) == pytest.approx(0.099)


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


def test_rejects_oversized_test_window(tmp_path):
    data = _plan_data()
    data["test_window_s"] = 2.5
    with pytest.raises(ValueError, match="test_window_s"):
        load_plan(_write_plan(tmp_path, data))


def test_hard_stop_on_soft_limit_and_velocity(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    home = np.array([0.0, 1.57, 1.57, 1.57, 0.0, 0.0, 0.0], dtype=np.float64)
    # Tiny step should pass joint checks (no TCP provided).
    ok = validate_joint_target(
        plan,
        current_joints=home,
        previous_joints=None,
        previous_velocity=None,
        requested_joints=home + 0.001,
        is_first_command=True,
    )
    assert ok == []

    huge = home.copy()
    huge[0] = 3.0  # outside soft limits
    violations = validate_joint_target(
        plan,
        current_joints=home,
        previous_joints=None,
        previous_velocity=None,
        requested_joints=huge,
        is_first_command=True,
    )
    codes = {v.code for v in violations}
    assert "requested_soft_limit" in codes

    fast = home.copy()
    fast[0] = home[0] + 0.05  # 0.05 rad in 0.05 s => 1 rad/s >> 0.05
    violations = validate_joint_target(
        plan,
        current_joints=home,
        previous_joints=None,
        previous_velocity=None,
        requested_joints=fast,
        is_first_command=True,
    )
    assert any(v.code == "joint_velocity" for v in violations)


def test_hard_stop_on_tcp_jump_and_workspace(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    home = np.zeros(7, dtype=np.float64)
    current_tcp = np.array([0.0, 0.0, 0.2])
    far_tcp = np.array([0.2, 0.0, 0.2])  # 0.2 m > 0.05 threshold
    violations = validate_joint_target(
        plan,
        current_joints=home,
        previous_joints=None,
        previous_velocity=None,
        requested_joints=home,
        current_tcp=current_tcp,
        requested_tcp=far_tcp,
        is_first_command=True,
    )
    assert any(v.code == "tcp_first_step_jump" for v in violations)

    outside = np.array([0.9, 0.0, 0.2])
    violations = validate_joint_target(
        plan,
        current_joints=home,
        previous_joints=None,
        previous_velocity=None,
        requested_joints=home,
        requested_tcp=outside,
        is_first_command=False,
    )
    assert any(v.code == "workspace" for v in violations)
