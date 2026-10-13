"""Transforms for NERO-native, shadow-only π0.5 base inference."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class NeroInputs(transforms.DataTransformFn):
    """Convert one NERO observation into the common π model input.

    ``action_dim`` is 8 for single-arm (right or left) and 16 for bimanual
    ``[right_8, left_8]``. The physical right wrist stays in ``left_wrist_0_rgb``
    so v3 LoRA visual features transfer; the physical left wrist fills
    ``right_wrist_0_rgb``.
    """

    model_type: _model.ModelType
    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        if self.action_dim not in (8, 16):
            raise ValueError(f"NERO action_dim must be 8 or 16, got {self.action_dim}")
        if "observation/state" in data:
            state = np.asarray(data["observation/state"], dtype=np.float32)
            if state.shape != (self.action_dim,):
                raise ValueError(f"Expected a {self.action_dim}D NERO state, got {state.shape}")
        else:
            if self.action_dim != 8:
                raise ValueError("Bimanual NERO observations must provide observation/state")
            joints = np.asarray(data["observation/joint_position"], dtype=np.float32)
            gripper = np.asarray(data["observation/gripper_position"], dtype=np.float32)
            if joints.shape != (7,):
                raise ValueError(f"Expected 7 NERO joints, got {joints.shape}")
            if gripper.shape != (1,):
                raise ValueError(f"Expected one gripper value, got {gripper.shape}")
            state = np.concatenate((joints, gripper))
        if not np.all(np.isfinite(state)):
            raise ValueError("NERO state contains NaN or Inf")

        external_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])
        if self.action_dim == 16:
            right_wrist = _parse_image(data["observation/wrist_image_right"])
            right_mask = np.True_
        else:
            right_wrist = np.zeros_like(external_image)
            right_mask = (
                np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_
            )
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": external_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim < 2 or actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"NERO training actions must have shape [..., horizon, {self.action_dim}] "
                    "with 7 absolute joint targets and one absolute gripper closedness per arm"
                )
            if not np.all(np.isfinite(actions)):
                raise ValueError("NERO actions contain NaN or Inf")
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class NeroRawBaseOutputs(transforms.DataTransformFn):
    """Expose only NERO's 8 dimensions without claiming command semantics."""

    def __call__(self, data: dict) -> dict:
        return {
            "actions": np.asarray(data["actions"][..., :8]),
            "state": np.asarray(data["state"][..., :8]),
        }


@dataclasses.dataclass(frozen=True)
class NeroRawFullOutputs(transforms.DataTransformFn):
    """Retain all model dimensions for deterministic offline diagnostics."""

    def __call__(self, data: dict) -> dict:
        return {
            "actions": np.asarray(data["actions"]),
            "state": np.asarray(data["state"]),
        }


@dataclasses.dataclass(frozen=True)
class NeroNativeOutputs(transforms.DataTransformFn):
    """Decode a NERO-trained policy into its native action dimensions.

    This transform is valid only with NERO normalization statistics and a
    NERO-fine-tuned checkpoint. Single-arm is 8D; bimanual is 16D
    ``[right_8, left_8]``. The training pipeline may represent joint dims as
    deltas internally; ``AbsoluteActions`` restores them before this runs.
    """

    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 2 or actions.shape[-1] < self.action_dim:
            raise ValueError(
                f"Expected NERO policy actions [..., >={self.action_dim}], got {actions.shape}"
            )
        actions = actions[..., : self.action_dim]
        if not np.all(np.isfinite(actions)):
            raise ValueError("NERO policy actions contain NaN or Inf")
        return {"actions": actions}
