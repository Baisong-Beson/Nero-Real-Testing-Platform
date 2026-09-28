from nero_pi05_bridge.observation import LEFT_GRIPPER_MAX_M
from nero_pi05_bridge.observation import LEFT_GRIPPER_MIN_M
from nero_pi05_bridge.observation import RIGHT_GRIPPER_MAX_M
from nero_pi05_bridge.observation import RIGHT_GRIPPER_MIN_M
from nero_pi05_bridge.observation import action_summary
from nero_pi05_bridge.observation import build_dual_state
from nero_pi05_bridge.observation import crop_uniform_border
from nero_pi05_bridge.observation import decode_compressed_image
from nero_pi05_bridge.observation import parse_joint_state
from nero_pi05_bridge.observation import resize_with_pad
from nero_pi05_bridge.observation import split_dual_actions
from nero_pi05_bridge.observation import topic_is_compressed
from nero_pi05_bridge.observation import validate_actions
import cv2
import numpy as np
import pytest


def test_crop_uniform_border_removes_headset_letterbox():
    canvas = np.full((200, 200, 3), 30, dtype=np.uint8)
    canvas[40:140, 20:180] = 180
    cropped = crop_uniform_border(canvas)
    assert cropped.shape == (100, 160, 3)
    assert np.all(cropped == 180)


def test_crop_uniform_border_leaves_filled_zed_frame():
    image = np.arange(720 * 1280 * 3, dtype=np.uint8).reshape(720, 1280, 3)
    cropped = crop_uniform_border(image)
    assert cropped.shape == image.shape
    np.testing.assert_array_equal(cropped, image)


def test_crop_uniform_border_rejects_all_pad():
    with pytest.raises(ValueError, match="entirely uniform"):
        crop_uniform_border(np.full((32, 32, 3), 29, dtype=np.uint8))


def test_resize_with_pad_keeps_rgb_shape_and_dtype():
    image = np.full((480, 848, 3), 127, dtype=np.uint8)
    resized = resize_with_pad(image, 224)
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8
    assert np.all(resized[:48] == 0)
    assert np.all(resized[49:175] == 127)


def test_joint_state_is_reordered_and_gripper_uses_calibrated_interval():
    names = ["gripper", "joint3", "joint1", "joint2", "joint4", "joint5", "joint7", "joint6"]
    positions = [0.04725, 3.0, 1.0, 2.0, 4.0, 5.0, 7.0, 6.0]
    joints, gripper = parse_joint_state(
        names,
        positions,
        gripper_min_width_m=0.030,
        gripper_max_width_m=0.099,
    )
    np.testing.assert_allclose(joints, np.arange(1.0, 8.0))
    np.testing.assert_allclose(gripper, [0.75])

    _, opened = parse_joint_state(
        names,
        [0.099, 3.0, 1.0, 2.0, 4.0, 5.0, 7.0, 6.0],
        gripper_min_width_m=0.030,
        gripper_max_width_m=0.099,
    )
    _, closed = parse_joint_state(
        names,
        [0.030, 3.0, 1.0, 2.0, 4.0, 5.0, 7.0, 6.0],
        gripper_min_width_m=0.030,
        gripper_max_width_m=0.099,
    )
    np.testing.assert_allclose(opened, [0.0])
    np.testing.assert_allclose(closed, [1.0])


def test_joint_state_rejects_missing_and_non_finite_values():
    with pytest.raises(ValueError, match="missing"):
        parse_joint_state(
            ["joint1"],
            [0.0],
            gripper_min_width_m=0.030,
            gripper_max_width_m=0.099,
        )
    names = [f"joint{i}" for i in range(1, 8)] + ["gripper"]
    with pytest.raises(ValueError, match="NaN"):
        parse_joint_state(
            names,
            [0.0] * 7 + [np.nan],
            gripper_min_width_m=0.030,
            gripper_max_width_m=0.099,
        )


def test_joint_state_rejects_invalid_gripper_calibration():
    names = [f"joint{i}" for i in range(1, 8)] + ["gripper"]
    with pytest.raises(ValueError, match="calibration interval"):
        parse_joint_state(
            names,
            [0.0] * 8,
            gripper_min_width_m=0.099,
            gripper_max_width_m=0.030,
        )


def test_policy_actions_are_finite_and_trimmed_to_eight_dimensions():
    actions = validate_actions({"actions": np.ones((15, 32), dtype=np.float32)})
    assert actions.shape == (15, 8)
    assert action_summary(actions)["chunk"] == actions.tolist()
    with pytest.raises(ValueError, match="NaN"):
        validate_actions({"actions": np.full((15, 8), np.nan)})


def _named_arm(joints, gripper_width):
    names = [f"joint{i}" for i in range(1, 8)] + ["gripper"]
    return names, list(joints) + [gripper_width]


def test_build_dual_state_uses_per_arm_gripper_endpoints():
    right_names, right_pos = _named_arm(np.arange(1.0, 8.0), RIGHT_GRIPPER_MAX_M)
    left_names, left_pos = _named_arm(np.arange(10.0, 17.0), LEFT_GRIPPER_MIN_M)
    state = build_dual_state(right_names, right_pos, left_names, left_pos)
    assert state.shape == (16,)
    np.testing.assert_allclose(state[:7], np.arange(1.0, 8.0))
    np.testing.assert_allclose(state[7], 0.0)
    np.testing.assert_allclose(state[8:15], np.arange(10.0, 17.0))
    np.testing.assert_allclose(state[15], 1.0)


def test_validate_actions_sixteen_rejects_eight_and_splits():
    with pytest.raises(ValueError, match=">=16"):
        validate_actions({"actions": np.ones((4, 8))}, action_dim=16)
    actions = validate_actions({"actions": np.arange(32, dtype=np.float32).reshape(2, 16)}, action_dim=16)
    assert actions.shape == (2, 16)
    right, left = split_dual_actions(actions)
    np.testing.assert_array_equal(right, actions[:, :8])
    np.testing.assert_array_equal(left, actions[:, 8:16])
    with pytest.raises(ValueError, match="\\[horizon, 16\\]"):
        split_dual_actions(np.ones((2, 8)))


def test_compressed_image_roundtrip_and_topic_helper():
    rgb = np.zeros((16, 20, 3), dtype=np.uint8)
    rgb[4:12, 6:14] = 200
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    decoded = decode_compressed_image(encoded.tobytes())
    assert decoded.shape == (16, 20, 3)
    assert decoded.dtype == np.uint8
    assert topic_is_compressed("/zed_m/left/image_raw/compressed")
    assert not topic_is_compressed("/zed_m/left/image_raw")
