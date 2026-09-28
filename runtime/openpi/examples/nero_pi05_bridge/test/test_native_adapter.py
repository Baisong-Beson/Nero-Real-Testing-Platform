from nero_pi05_bridge.native_adapter import NeroNativeContract
from nero_pi05_bridge.native_adapter import NeroNativeShadowAdapter
from nero_pi05_bridge.offline_adapter import NERO_POSITION_UPPER
import numpy as np
import pytest


def contract(**overrides):
    data = {
        "schema_version": 2,
        "action_semantics": "absolute_joint_target_rad+absolute_gripper_closedness",
        "joint_names": [f"joint{i}" for i in range(1, 8)],
        "control_frequency_hz": 20.0,
        "max_abs_velocity_rad_s": [0.1] * 7,
        "max_abs_acceleration_rad_s2": [0.2] * 7,
        "soft_limit_margin_rad": 0.1,
        "max_gripper_speed_m_s": 0.03,
        "gripper_min_width_m": 0.030,
        "gripper_max_width_m": 0.099,
        "commandable": False,
    }
    data.update(overrides)
    return NeroNativeContract.from_dict(data)


def test_native_contract_requires_v2_absolute_semantics_and_refuses_commandable():
    assert contract().commandable is False
    assert contract().control_frequency_hz == 20.0
    with pytest.raises(ValueError, match="expected 2"):
        contract(schema_version=1)
    with pytest.raises(ValueError, match="action_semantics"):
        contract(action_semantics="joint_velocity_rad_s+absolute_gripper_closedness")
    with pytest.raises(ValueError, match="must not be commandable"):
        contract(commandable=True)
    with pytest.raises(ValueError, match="joint_names"):
        contract(joint_names=[f"joint{i}" for i in range(7, 0, -1)])


def test_native_chunk_uses_absolute_targets_and_calibrated_gripper():
    limited = contract(
        max_abs_velocity_rad_s=[10.0] * 7,
        max_abs_acceleration_rad_s2=[1000.0] * 7,
        max_gripper_speed_m_s=10.0,
    )
    actions = np.zeros((2, 8))
    actions[0, 0] = 0.005
    actions[1, 0] = 0.010
    actions[0, 7] = 0.0
    actions[1, 7] = 1.0
    result = NeroNativeShadowAdapter(limited).adapt_chunk(actions, np.zeros(8))
    np.testing.assert_allclose(result.requested_joint_positions[:, 0], [0.005, 0.010])
    np.testing.assert_allclose(result.joint_positions[:, 0], [0.005, 0.010])
    np.testing.assert_allclose(result.requested_gripper_width_m, [0.099, 0.030])
    np.testing.assert_allclose(result.gripper_width_m, [0.099, 0.030])
    assert not np.any(result.velocity_limit_exceeded)
    assert result.commandable is False


def test_native_chunk_reports_and_guards_rate_and_acceleration_limits():
    actions = np.zeros((2, 8))
    actions[:, 3] = 0.5
    result = NeroNativeShadowAdapter(contract()).adapt_chunk(actions, np.zeros(8))
    assert result.target_jump_exceeded[0, 3]
    assert result.velocity_limit_exceeded[0, 3]
    assert result.acceleration_limit_exceeded[0, 3]
    assert result.joint_accelerations[0, 3] == pytest.approx(0.2)
    assert result.joint_velocities[0, 3] == pytest.approx(0.01)
    assert result.joint_positions[0, 3] == pytest.approx(0.0005)


def test_native_chunk_reports_soft_limit_and_gripper_guards():
    actions = np.zeros((1, 8))
    actions[0, 0] = NERO_POSITION_UPPER[0]
    actions[0, 7] = 2.0
    result = NeroNativeShadowAdapter(contract()).adapt_chunk(actions, np.zeros(8))
    assert result.soft_limit_violated[0, 0]
    assert result.gripper_closedness_clipped[0]
    assert result.gripper_speed_limit_exceeded[0]
    assert result.requested_gripper_width_m[0] == pytest.approx(0.030)
    assert result.gripper_width_m[0] == pytest.approx(0.0975)


def test_native_chunk_rejects_bad_action_or_initial_state_shape():
    adapter = NeroNativeShadowAdapter(contract())
    with pytest.raises(ValueError, match="horizon, 8"):
        adapter.adapt_chunk(np.zeros((50, 7)), np.zeros(8))
    with pytest.raises(ValueError, match="initial_state"):
        adapter.adapt_chunk(np.zeros((1, 8)), np.zeros(7))
