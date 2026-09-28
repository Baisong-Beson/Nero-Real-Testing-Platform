import json
from pathlib import Path

from nero_pi05_bridge.gripper_calibration import authorize_execution
from nero_pi05_bridge.gripper_calibration import check_feedback_freshness
from nero_pi05_bridge.gripper_calibration import check_feedback_health
from nero_pi05_bridge.gripper_calibration import load_plan
from nero_pi05_bridge.gripper_calibration import max_abs_delta
from nero_pi05_bridge.gripper_calibration import max_width_speed
from nero_pi05_bridge.gripper_calibration import measured_step_limit_m
import pytest


def _plan_data():
    return {
        "schema_version": 1,
        "name": "test",
        "arm_namespace": "/left_arm",
        "execution_enabled": False,
        "targets": [
            {"width_m": 0.08, "effort_n": 0.5},
            {"width_m": 0.06, "effort_n": 0.5},
            {"width_m": 0.09, "effort_n": 0.5},
        ],
        "safety": {
            "width_min_m": 0.03,
            "width_max_m": 0.099,
            "effort_min_n": 0.5,
            "effort_max_n": 0.5,
            "max_step_delta_m": 0.03,
            "required_final_width_min_m": 0.09,
            "joint_static_tolerance_rad": 0.01,
            "tcp_static_tolerance_m": 0.0001,
            "gripper_tolerance_m": 0.001,
            "feedback_timeout_s": 5.0,
            "baseline_duration_s": 3.0,
            "step_timeout_s": 8.0,
            "settle_duration_s": 0.75,
            "gate_open_duration_s": 0.25,
        },
    }


def _write_plan(tmp_path, data):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data))
    return path


def test_safe_plan_loads_and_has_stable_fingerprint(tmp_path):
    path = _write_plan(tmp_path, _plan_data())
    first = load_plan(path)
    second = load_plan(path)
    assert first.fingerprint == second.fingerprint
    assert first.approval_token.startswith("NERO_GATE1_EXECUTE_")
    assert first.targets[-1].width_m == 0.09


def test_execution_is_disabled_even_with_matching_token(tmp_path):
    plan = load_plan(_write_plan(tmp_path, _plan_data()))
    with pytest.raises(PermissionError, match="execution_enabled=false"):
        authorize_execution(plan, plan.approval_token)


def test_execution_requires_exact_fingerprint_token(tmp_path):
    data = _plan_data()
    data["execution_enabled"] = True
    plan = load_plan(_write_plan(tmp_path, data))
    with pytest.raises(PermissionError, match="does not match"):
        authorize_execution(plan, "wrong")
    authorization = authorize_execution(plan, plan.approval_token)
    assert authorization.plan_fingerprint == plan.fingerprint


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["targets"].__setitem__(1, {"width_m": 0.04, "effort_n": 0.5}), "exceeds"),
        (lambda data: data["targets"][-1].__setitem__("width_m", 0.08), "final target"),
        (lambda data: data["safety"].__setitem__("gate_open_duration_s", 0.6), "must not exceed"),
        (lambda data: data["targets"][0].__setitem__("effort_n", 0.4), "effort is outside"),
    ],
)
def test_unsafe_plans_are_rejected(tmp_path, mutation, message):
    data = _plan_data()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        load_plan(_write_plan(tmp_path, data))


def test_motion_statistics():
    assert max_abs_delta([1.0, -2.0], [1.1, -1.8]) == pytest.approx(0.2)
    samples = [(0.0, 0.09, 0.5), (0.5, 0.08, 0.5), (1.0, 0.05, 0.5)]
    assert max_width_speed(samples) == pytest.approx(0.06)


def test_right_safe_plan_loads_with_repeatability_and_multi_effort():
    plan = load_plan(
        Path(__file__).resolve().parents[1] / "config" / "gripper_calibration_right_safe.json"
    )
    assert plan.arm_namespace == "/right_arm"
    assert plan.execution_enabled is False
    assert len(plan.targets) == 21  # 7-step sequence × 3 rounds (0.5 / 1.0 / 0.5 N)
    efforts = {target.effort_n for target in plan.targets}
    assert efforts == {0.5, 1.0}
    assert plan.safety.effort_max_n == 1.0
    assert plan.targets[-1].width_m >= plan.safety.required_final_width_min_m


def test_measured_step_limit_includes_gripper_tolerance():
    plan = load_plan(
        Path(__file__).resolve().parents[1] / "config" / "gripper_calibration_right_safe.json"
    )
    limit = measured_step_limit_m(plan.safety)
    assert limit == pytest.approx(0.031)
    # Exact plan adjacency is 0.030; measured overshoot of ~0.22 mm must still pass.
    assert abs(0.05 - 0.080224) <= limit + 1e-12


def test_feedback_freshness_rejects_missing_and_stale():
    required = {"joints", "tcp", "gripper", "arm_status"}
    with pytest.raises(RuntimeError, match="missing"):
        check_feedback_freshness({}, now=10.0, required=required, maximum_age_s=0.5)

    fresh = {name: 9.8 for name in required}
    check_feedback_freshness(fresh, now=10.0, required=required, maximum_age_s=0.5)

    stale = dict(fresh)
    stale["joints"] = 9.0
    with pytest.raises(RuntimeError, match="stale"):
        check_feedback_freshness(stale, now=10.0, required=required, maximum_age_s=0.5)


def test_feedback_health_rejects_faults():
    snapshot = {
        "gripper_faults": [],
        "arm_err_status": 0,
        "arm_motion_status": 0,
    }
    check_feedback_health(snapshot)

    with pytest.raises(RuntimeError, match="gripper faults"):
        check_feedback_health({**snapshot, "gripper_faults": ["voltage_too_low"]})
    with pytest.raises(RuntimeError, match="arm err_status"):
        check_feedback_health({**snapshot, "arm_err_status": 3})
    with pytest.raises(RuntimeError, match="arm motion_status"):
        check_feedback_health({**snapshot, "arm_motion_status": 1})
