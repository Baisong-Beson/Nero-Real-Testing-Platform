from nero_pi05_bridge.offline_adapter import NERO_POSITION_UPPER
from nero_pi05_bridge.offline_adapter import CandidateMapping
from nero_pi05_bridge.offline_adapter import OfflineAdapter
import numpy as np
import pytest


def mapping(**overrides):
    data = {
        "mapping_status": "test_hypothesis",
        "source_indices": list(range(7)),
        "direction": [1] * 7,
        "velocity_scale": [1] * 7,
        "max_abs_velocity": [0.1] * 7,
        "control_period_sec": 0.2,
        "low_pass_alpha": 1.0,
        "soft_limit_margin_rad": 0.1,
        "gripper_max_width_m": 0.1,
    }
    data.update(overrides)
    return CandidateMapping.from_dict(data)


def test_mapping_can_never_be_marked_verified_or_commandable():
    assert mapping().commandable is False
    with pytest.raises(ValueError, match="refuses"):
        mapping(mapping_status="verified")


def test_velocity_is_permuted_signed_and_clipped():
    config = mapping(
        source_indices=[6, 5, 4, 3, 2, 1, 0],
        direction=[-1, 1, 1, 1, 1, 1, 1],
    )
    result = OfflineAdapter(config).adapt(
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.25],
        np.zeros(7),
    )
    assert np.allclose(result.candidate_velocity[[0, 6]], [-0.1, 0.1])
    assert result.velocity_clipped[[0, 6]].tolist() == [True, True]
    assert result.commandable is False


def test_position_guard_blocks_outward_motion():
    position = np.zeros(7)
    position[0] = NERO_POSITION_UPPER[0] - 0.05
    result = OfflineAdapter(mapping()).adapt([0.2, 0, 0, 0, 0, 0, 0, 0], position)
    assert result.position_guarded[0]
    assert result.candidate_velocity[0] == 0.0
    assert result.target_position[0] == position[0]


def test_gripper_closedness_converts_to_opening_width_and_clips():
    adapter = OfflineAdapter(mapping())
    closed = adapter.adapt([0, 0, 0, 0, 0, 0, 0, 1.0], np.zeros(7))
    opened = adapter.adapt([0, 0, 0, 0, 0, 0, 0, -0.2], np.zeros(7))
    assert closed.gripper_width_m == 0.0
    assert opened.gripper_width_m == 0.1
    assert opened.gripper_clipped
