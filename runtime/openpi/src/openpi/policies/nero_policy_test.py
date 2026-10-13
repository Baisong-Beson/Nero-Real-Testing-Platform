import numpy as np
import pytest

from openpi.models import model
from openpi.models import pi0_config
from openpi.policies import nero_policy
from openpi.training import config as training_config


def make_observation() -> dict:
    return {
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.asarray([0.25], dtype=np.float32),
        "observation/exterior_image_1_left": np.zeros(
            (480, 848, 3),
            dtype=np.uint8,
        ),
        "observation/wrist_image_left": np.zeros(
            (480, 848, 3),
            dtype=np.uint8,
        ),
        "prompt": "move close to the red cup",
    }


def test_nero_inputs_build_native_eight_dimensional_state():
    transformed = nero_policy.NeroInputs(model.ModelType.PI05)(
        make_observation()
    )
    np.testing.assert_array_equal(
        transformed["state"],
        np.asarray([0, 1, 2, 3, 4, 5, 6, 0.25]),
    )
    assert transformed["image_mask"]["base_0_rgb"]
    assert transformed["image_mask"]["left_wrist_0_rgb"]
    assert not transformed["image_mask"]["right_wrist_0_rgb"]


def test_nero_inputs_accept_canonical_combined_state():
    observation = make_observation()
    observation["observation/state"] = np.arange(8, dtype=np.float32)
    del observation["observation/joint_position"]
    del observation["observation/gripper_position"]
    transformed = nero_policy.NeroInputs(model.ModelType.PI05)(observation)
    np.testing.assert_array_equal(transformed["state"], np.arange(8, dtype=np.float32))


def test_nero_inputs_reject_invalid_state():
    observation = make_observation()
    observation["observation/joint_position"] = np.zeros(6)
    with pytest.raises(ValueError, match="7 NERO joints"):
        nero_policy.NeroInputs(model.ModelType.PI05)(observation)


def test_raw_outputs_trim_to_eight_dimensions():
    outputs = nero_policy.NeroRawBaseOutputs()(
        {
            "actions": np.ones((50, 32), dtype=np.float32),
            "state": np.zeros(32, dtype=np.float32),
        }
    )
    assert outputs["actions"].shape == (50, 8)
    assert outputs["state"].shape == (8,)


def test_raw_full_outputs_keep_all_model_dimensions():
    outputs = nero_policy.NeroRawFullOutputs()(
        {
            "actions": np.ones((50, 32), dtype=np.float32),
            "state": np.zeros(32, dtype=np.float32),
        }
    )
    assert outputs["actions"].shape == (50, 32)
    assert outputs["state"].shape == (32,)


def test_native_outputs_expose_eight_semantic_dimensions():
    outputs = nero_policy.NeroNativeOutputs()(
        {"actions": np.ones((50, 32), dtype=np.float32)}
    )
    assert outputs["actions"].shape == (50, 8)


def test_nero_inputs_validate_native_training_actions():
    observation = make_observation()
    observation["actions"] = np.zeros((50, 8), dtype=np.float32)
    transformed = nero_policy.NeroInputs(model.ModelType.PI05)(observation)
    assert transformed["actions"].shape == (50, 8)
    observation["actions"] = np.zeros((50, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="7 absolute joint targets"):
        nero_policy.NeroInputs(model.ModelType.PI05)(observation)


def test_nero_data_config_wires_joint_delta_round_trip(tmp_path):
    data_config = training_config.LeRobotNeroDataConfig().create(
        tmp_path,
        pi0_config.Pi0Config(pi05=True),
    )
    assert [type(transform).__name__ for transform in data_config.data_transforms.inputs] == [
        "NeroInputs",
        "DeltaActions",
    ]
    assert [type(transform).__name__ for transform in data_config.data_transforms.outputs] == [
        "AbsoluteActions",
        "NeroNativeOutputs",
    ]

    mask = (True, True, True, True, True, True, True, False)
    assert data_config.data_transforms.inputs[-1].mask == mask
    assert data_config.data_transforms.outputs[0].mask == mask
    assert data_config.action_sequence_keys == ("action",)

    state = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.25])
    absolute_actions = np.tile(state, (2, 1))
    absolute_actions[:, :7] += 0.05
    canonical_sample = {
        "observation.images.external_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.left_wrist_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.state": state.copy(),
        "action": absolute_actions.copy(),
        "prompt": "move close to the bottle",
    }
    repacked = data_config.repack_transforms.inputs[0](canonical_sample)
    transformed = data_config.data_transforms.inputs[0](repacked)
    transformed = data_config.data_transforms.inputs[-1](transformed)
    np.testing.assert_allclose(transformed["actions"][:, :7], 0.05, atol=1e-7)
    np.testing.assert_allclose(transformed["actions"][:, 7], 0.25)
    restored = data_config.data_transforms.outputs[0](transformed)
    np.testing.assert_allclose(restored["actions"], absolute_actions)


def test_nero_inputs_accept_bimanual_sixteen_dimensional_state():
    observation = make_observation()
    observation["observation/state"] = np.arange(16, dtype=np.float32)
    observation["observation/wrist_image_right"] = np.ones((480, 848, 3), dtype=np.uint8)
    del observation["observation/joint_position"]
    del observation["observation/gripper_position"]
    transformed = nero_policy.NeroInputs(model.ModelType.PI05, action_dim=16)(observation)
    np.testing.assert_array_equal(transformed["state"], np.arange(16, dtype=np.float32))
    assert transformed["image_mask"]["right_wrist_0_rgb"]
    assert transformed["image"]["right_wrist_0_rgb"].shape == (480, 848, 3)


def test_nero_bimanual_data_config_wires_dual_arm_delta_mask(tmp_path):
    data_config = training_config.LeRobotNeroDataConfig(bimanual=True).create(
        tmp_path,
        pi0_config.Pi0Config(pi05=True),
    )
    mask = (True, True, True, True, True, True, True, False) * 2
    assert data_config.data_transforms.inputs[-1].mask == mask
    assert data_config.data_transforms.outputs[0].mask == mask

    state = np.arange(16, dtype=np.float32) * 0.1
    absolute_actions = np.tile(state, (2, 1))
    absolute_actions[:, :7] += 0.05
    absolute_actions[:, 8:15] += 0.02
    canonical_sample = {
        "observation.images.external_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.left_wrist_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.right_wrist_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.state": state.copy(),
        "action": absolute_actions.copy(),
        "prompt": "pick up the marker and open the drawer",
    }
    repacked = data_config.repack_transforms.inputs[0](canonical_sample)
    transformed = data_config.data_transforms.inputs[0](repacked)
    transformed = data_config.data_transforms.inputs[-1](transformed)
    np.testing.assert_allclose(transformed["actions"][:, :7], 0.05, atol=1e-7)
    np.testing.assert_allclose(transformed["actions"][:, 7], state[7])
    np.testing.assert_allclose(transformed["actions"][:, 8:15], 0.02, atol=1e-7)
    np.testing.assert_allclose(transformed["actions"][:, 15], state[15])
    restored = data_config.data_transforms.outputs[0](transformed)
    np.testing.assert_allclose(restored["actions"], absolute_actions)
