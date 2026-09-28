"""Observation conversion and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
GRIPPER_JOINT_NAME = "gripper"

# Single-arm Gate 1 interval (left/right v1–v3). Dual-arm session_20260814
# episode054+ uses the measured per-arm endpoints below.
DEFAULT_GRIPPER_MIN_M = 0.030
DEFAULT_GRIPPER_MAX_M = 0.099
RIGHT_GRIPPER_MIN_M = 0.000
RIGHT_GRIPPER_MAX_M = 0.0996
LEFT_GRIPPER_MIN_M = 0.000
LEFT_GRIPPER_MAX_M = 0.1005


def crop_uniform_border(
    image: np.ndarray,
    *,
    max_delta: int = 12,
    min_keep_frac: float = 0.15,
    noop_frac: float = 0.95,
) -> np.ndarray:
    """Crop a near-uniform letterbox (headset composite gray pad) from an HWC RGB image.

    ZED frames have no matching pad, so they are returned unchanged. The crop uses
    the top-left pixel as the pad color and keeps the bounding box of every pixel
    farther than ``max_delta`` in any channel.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    if not (0 < min_keep_frac < noop_frac <= 1.0):
        raise ValueError("min_keep_frac/noop_frac must satisfy 0 < min_keep_frac < noop_frac <= 1")
    if max_delta < 0:
        raise ValueError(f"max_delta must be non-negative, got {max_delta}")

    height, width = image.shape[:2]
    pad = image[0, 0].astype(np.int16)
    content = np.abs(image.astype(np.int16) - pad).max(axis=2) > max_delta
    if not np.any(content):
        raise ValueError("image is entirely uniform padding")

    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    crop_h, crop_w = y1 - y0, x1 - x0
    if crop_h >= height * noop_frac and crop_w >= width * noop_frac:
        return np.ascontiguousarray(image)
    if crop_h < height * min_keep_frac or crop_w < width * min_keep_frac:
        raise ValueError(
            f"uniform-border crop {crop_w}x{crop_h} would discard too much of {width}x{height}"
        )
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an HWC uint8 image without changing its aspect ratio."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    if size <= 0:
        raise ValueError(f"Image size must be positive, got {size}")

    src_height, src_width = image.shape[:2]
    if src_height <= 0 or src_width <= 0:
        raise ValueError(f"Image has invalid shape {image.shape}")
    if src_height == size and src_width == size:
        return np.ascontiguousarray(image)

    scale = min(size / src_width, size / src_height)
    dst_width = max(1, round(src_width * scale))
    dst_height = max(1, round(src_height * scale))
    resized = cv2.resize(image, (dst_width, dst_height), interpolation=cv2.INTER_AREA)

    output = np.zeros((size, size, 3), dtype=np.uint8)
    offset_x = (size - dst_width) // 2
    offset_y = (size - dst_height) // 2
    output[offset_y : offset_y + dst_height, offset_x : offset_x + dst_width] = resized
    return output


def parse_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    *,
    gripper_min_width_m: float,
    gripper_max_width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return NERO joint positions and calibrated gripper closedness.

    Closedness is zero at ``gripper_max_width_m`` and one at
    ``gripper_min_width_m``.  Values outside the calibrated interval are
    clipped.  This keeps the observation and native NERO training contract in
    the same coordinate system.
    """
    if len(names) != len(positions):
        raise ValueError(f"Joint name/position length mismatch: {len(names)} != {len(positions)}")
    if gripper_min_width_m < 0 or gripper_max_width_m <= gripper_min_width_m:
        raise ValueError("gripper width calibration interval is invalid")

    position_by_name = dict(zip(names, positions, strict=True))
    missing = [name for name in (*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME) if name not in position_by_name]
    if missing:
        raise ValueError(f"JointState is missing: {missing}")

    joints = np.asarray([position_by_name[name] for name in ARM_JOINT_NAMES], dtype=np.float64)
    width = float(position_by_name[GRIPPER_JOINT_NAME])
    if not np.all(np.isfinite(joints)) or not np.isfinite(width):
        raise ValueError("JointState contains NaN or Inf")

    calibrated_width = np.clip(width, gripper_min_width_m, gripper_max_width_m)
    closedness = (gripper_max_width_m - calibrated_width) / (
        gripper_max_width_m - gripper_min_width_m
    )
    return joints, np.asarray([closedness], dtype=np.float64)


def build_dual_state(
    right_names: Sequence[str],
    right_positions: Sequence[float],
    left_names: Sequence[str],
    left_positions: Sequence[float],
    *,
    right_gripper_min_width_m: float = RIGHT_GRIPPER_MIN_M,
    right_gripper_max_width_m: float = RIGHT_GRIPPER_MAX_M,
    left_gripper_min_width_m: float = LEFT_GRIPPER_MIN_M,
    left_gripper_max_width_m: float = LEFT_GRIPPER_MAX_M,
) -> np.ndarray:
    """Build the 16D ``[right_8, left_8]`` observation state."""
    right_joints, right_gripper = parse_joint_state(
        right_names,
        right_positions,
        gripper_min_width_m=right_gripper_min_width_m,
        gripper_max_width_m=right_gripper_max_width_m,
    )
    left_joints, left_gripper = parse_joint_state(
        left_names,
        left_positions,
        gripper_min_width_m=left_gripper_min_width_m,
        gripper_max_width_m=left_gripper_max_width_m,
    )
    return np.concatenate((right_joints, right_gripper, left_joints, left_gripper)).astype(
        np.float64
    )


def validate_actions(response: dict, *, action_dim: int = 8) -> np.ndarray:
    """Validate a pi0.5 response without interpreting or publishing it.

    Default ``action_dim=8`` keeps the single-arm path unchanged. Pass
    ``action_dim=16`` for the bimanual ``[right_8, left_8]`` layout.
    """
    if action_dim not in (8, 16):
        raise ValueError(f"action_dim must be 8 or 16, got {action_dim}")
    if "actions" not in response:
        raise ValueError("Policy response does not contain 'actions'")
    actions = np.asarray(response["actions"])
    if actions.ndim == 1:
        actions = actions[np.newaxis, :]
    if actions.ndim < 2 or actions.shape[-1] < action_dim:
        raise ValueError(f"Expected action shape [..., >={action_dim}], got {actions.shape}")
    actions = np.asarray(actions[..., :action_dim], dtype=np.float64)
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy response contains NaN or Inf")
    return actions


def split_dual_actions(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 16D chunk into right ``[:, :8]`` and left ``[:, 8:16]``."""
    chunk = np.asarray(actions, dtype=np.float64)
    if chunk.ndim == 1:
        chunk = chunk[np.newaxis, :]
    if chunk.ndim != 2 or chunk.shape[-1] != 16:
        raise ValueError(f"Expected dual actions [horizon, 16], got {chunk.shape}")
    return chunk[:, :8].copy(), chunk[:, 8:16].copy()


def decode_compressed_image(data: bytes | np.ndarray) -> np.ndarray:
    """Decode a JPEG/PNG CompressedImage payload to HWC RGB uint8."""
    buffer = np.frombuffer(bytes(data), dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode CompressedImage")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def decode_raw_rgb_image(
    *,
    encoding: str,
    height: int,
    width: int,
    data: bytes | np.ndarray,
) -> np.ndarray:
    """Decode a sensor_msgs/Image payload to HWC RGB uint8."""
    encoding = str(encoding).lower()
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image shape {height}x{width}")
    array = np.frombuffer(bytes(data) if not isinstance(data, np.ndarray) else data, dtype=np.uint8)
    if encoding in {"rgb8", "bgr8"}:
        expected = height * width * 3
        if array.size < expected:
            raise ValueError(f"image buffer too small for {encoding}: {array.size} < {expected}")
        image = array[:expected].reshape((height, width, 3))
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image)
    if encoding in {"rgba8", "bgra8"}:
        expected = height * width * 4
        if array.size < expected:
            raise ValueError(f"image buffer too small for {encoding}: {array.size} < {expected}")
        image = array[:expected].reshape((height, width, 4))[:, :, :3]
        if encoding == "bgra8":
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image)
    raise ValueError(f"unsupported image encoding: {encoding}")


def decode_camera_message(message: Any) -> np.ndarray:
    """Accept a duck-typed ``Image`` or ``CompressedImage`` ROS message."""
    if hasattr(message, "encoding"):
        return decode_raw_rgb_image(
            encoding=message.encoding,
            height=int(message.height),
            width=int(message.width),
            data=message.data,
        )
    if hasattr(message, "data"):
        return decode_compressed_image(message.data)
    raise ValueError(f"unsupported camera message type: {type(message).__name__}")


def topic_is_compressed(topic: str) -> bool:
    return topic.rstrip("/").endswith("/compressed")


def action_summary(actions: np.ndarray) -> dict:
    flattened = actions.reshape(-1)
    return {
        "shape": list(actions.shape),
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
        "mean": float(np.mean(flattened)),
        "l2": float(np.linalg.norm(flattened)),
        "abs_gt_1_fraction": float(np.mean(np.abs(flattened) > 1.0)),
        "first_action": actions.reshape(-1, actions.shape[-1])[0].tolist(),
        # Retain the full policy horizon for later offline analysis. This is
        # diagnostic data only and is never interpreted as a robot command.
        "chunk": actions.tolist(),
    }
