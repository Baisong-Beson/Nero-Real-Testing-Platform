"""Unit tests for MCAP conversion helpers (no real bag required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pytest

from nero_pi05_bridge.mcap_convert import DEFAULT_TOPICS
from nero_pi05_bridge.mcap_convert import _Timeline
from nero_pi05_bridge.mcap_convert import _action_from_command_map
from nero_pi05_bridge.mcap_convert import _closedness_from_width
from nero_pi05_bridge.mcap_convert import _control_command_at
from nero_pi05_bridge.mcap_convert import _state_from_feedback
from nero_pi05_bridge.mcap_convert import extract_convert_options
from nero_pi05_bridge.mcap_convert import normalize_topic_map
from nero_pi05_bridge.observation import ARM_JOINT_NAMES
from nero_pi05_bridge.observation import GRIPPER_JOINT_NAME


def test_closedness_endpoints() -> None:
    assert _closedness_from_width(0.099, gripper_min_width_m=0.030, gripper_max_width_m=0.099) == 0.0
    assert _closedness_from_width(0.030, gripper_min_width_m=0.030, gripper_max_width_m=0.099) == 1.0


def test_control_partial_merge_zoh() -> None:
    timeline = _Timeline()
    timeline.append(100, {"joint1": 0.1, "joint2": 0.2})
    timeline.append(200, {GRIPPER_JOINT_NAME: 0.05})
    timeline.append(300, {"joint1": 0.3})

    merged = _control_command_at(timeline, 250)
    assert merged is not None
    assert merged["joint1"] == 0.1
    assert merged["joint2"] == 0.2
    assert merged[GRIPPER_JOINT_NAME] == 0.05

    merged_later = _control_command_at(timeline, 350)
    assert merged_later is not None
    assert merged_later["joint1"] == 0.3
    assert merged_later["joint2"] == 0.2
    assert merged_later[GRIPPER_JOINT_NAME] == 0.05


def test_action_never_zero_fills_missing_joints() -> None:
    state = np.arange(8, dtype=np.float32) * 0.1
    action, meta = _action_from_command_map(
        {GRIPPER_JOINT_NAME: 0.099},
        gripper_min_width_m=0.030,
        gripper_max_width_m=0.099,
        fallback_state=state,
    )
    assert meta["used_state_fallback_for_joints"] is True
    np.testing.assert_allclose(action[:7], state[:7])
    assert action[7] == 0.0


def test_topic_map_accepts_right_and_left_collector_aliases() -> None:
    topics = normalize_topic_map(
        {
            "right_feedback_joint_states": "/right_arm/feedback/joint_states",
            "right_control_joint_states": "/right_arm/control/joint_states",
            "left_feedback_joint_states": "/left_arm/feedback/joint_states",
            "left_vr_pose": "/nero_vr/left/pose",
        }
    )
    assert topics["feedback_joint_states"] == "/right_arm/feedback/joint_states"
    assert topics["control_joint_states"] == "/right_arm/control/joint_states"
    assert topics["left_feedback_joint_states"] == "/left_arm/feedback/joint_states"
    assert topics["left_teleop_pose"] == "/nero_vr/left/pose"


def test_topic_map_override_and_aliases() -> None:
    topics = normalize_topic_map(
        {
            "external_image": "/vr_rig/zed/rgb/compressed",
            "control_joint_states": "/left_arm/vr_control/joint_states",
            "vr_pose": "/vr/controller_left/pose",
            "tracking_valid": "/vr/controller_left/tracking_valid",
        }
    )
    assert topics["external_image"] == "/vr_rig/zed/rgb/compressed"
    assert topics["control_joint_states"] == "/left_arm/vr_control/joint_states"
    assert topics["teleop_pose"] == "/vr/controller_left/pose"
    assert topics["localization_valid"] == "/vr/controller_left/tracking_valid"
    # Untouched keys keep their defaults.
    assert topics["wrist_image"] == DEFAULT_TOPICS["wrist_image"]


def test_extract_convert_options_reads_crop_flag() -> None:
    assert extract_convert_options(None)["crop_external_padding"] is False
    assert extract_convert_options({"crop_external_padding": True})["crop_external_padding"] is True
    assert extract_convert_options({"notes": {"crop_external_padding": True}})[
        "crop_external_padding"
    ]
    with pytest.raises(ValueError, match="boolean"):
        extract_convert_options({"crop_external_padding": "yes"})


def test_topic_map_ignores_notes_and_comments() -> None:
    topics = normalize_topic_map(
        {
            "_comment": "operator notes",
            "notes": {"arm": "right"},
            "external_image": "/vr_collection/external_image",
            "control_joint_states": "/right_arm/control/joint_states",
        }
    )
    assert topics["external_image"] == "/vr_collection/external_image"
    assert topics["control_joint_states"] == "/right_arm/control/joint_states"


def test_topic_map_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown topic key"):
        normalize_topic_map({"externalimage": "/typo"})


def test_topic_map_defaults_when_no_override() -> None:
    assert normalize_topic_map(None) == DEFAULT_TOPICS


def test_discover_nested_vr_episode_layout(tmp_path: Path) -> None:
    from nero_pi05_bridge.mcap_convert import discover_episode_dirs

    nested = tmp_path / "episode000" / "episode000"
    nested.mkdir(parents=True)
    (nested / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    (nested / "episode000_0.mcap").write_bytes(b"mcap")
    found = discover_episode_dirs(tmp_path)
    assert found == [nested.resolve()]


def test_image_topic_compressed_fallback() -> None:
    from nero_pi05_bridge.mcap_convert import resolve_topics_against_bag

    topics = {
        "external_image": "/vr_collection/external_image",
        "wrist_image": "/vr_collection/wrist_image",
        "feedback_joint_states": "/right_arm/feedback/joint_states",
        "control_joint_states": "/right_arm/control/joint_states",
    }
    available = {
        "/vr_collection/external_image/compressed",
        "/vr_collection/wrist_image/compressed",
        "/right_arm/feedback/joint_states",
        "/right_arm/control/joint_states",
    }
    resolved = resolve_topics_against_bag(
        topics,
        available,
        required_keys={"external_image", "wrist_image", "feedback_joint_states"},
        action_keys=("control_joint_states",),
    )
    assert resolved["external_image"] == "/vr_collection/external_image/compressed"
    assert resolved["wrist_image"] == "/vr_collection/wrist_image/compressed"


def test_state_from_complete_feedback() -> None:
    joint_map = {name: float(i) for i, name in enumerate(ARM_JOINT_NAMES)}
    joint_map[GRIPPER_JOINT_NAME] = 0.099
    state, width = _state_from_feedback(
        joint_map,
        gripper_min_width_m=0.030,
        gripper_max_width_m=0.099,
    )
    assert state.shape == (8,)
    assert width == 0.099
    assert state[7] == 0.0
