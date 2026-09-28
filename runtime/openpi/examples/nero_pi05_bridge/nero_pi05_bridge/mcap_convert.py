"""MCAP -> synchronized NERO frames for LeRobot conversion.

This module is ROS-runtime free. It reads rosbag2 MCAP files through
``mcap-ros2-support`` and produces causally aligned 20 Hz samples.

Action sources:

* production: ``/left_arm/control/joint_states`` (partial 7-DoF and gripper
  messages merged by name with zero-order hold)
* smoke / Gate 0: candidate IK + gripper topics under
  ``/nero_input_only/candidate/*`` (must not be used for training)
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages

from nero_pi05_bridge.observation import ARM_JOINT_NAMES
from nero_pi05_bridge.observation import DEFAULT_GRIPPER_MAX_M
from nero_pi05_bridge.observation import DEFAULT_GRIPPER_MIN_M
from nero_pi05_bridge.observation import GRIPPER_JOINT_NAME
from nero_pi05_bridge.observation import LEFT_GRIPPER_MAX_M as DEFAULT_LEFT_GRIPPER_MAX_M
from nero_pi05_bridge.observation import LEFT_GRIPPER_MIN_M as DEFAULT_LEFT_GRIPPER_MIN_M
from nero_pi05_bridge.observation import RIGHT_GRIPPER_MAX_M as DEFAULT_RIGHT_GRIPPER_MAX_M
from nero_pi05_bridge.observation import RIGHT_GRIPPER_MIN_M as DEFAULT_RIGHT_GRIPPER_MIN_M
from nero_pi05_bridge.observation import crop_uniform_border
from nero_pi05_bridge.observation import resize_with_pad

CANONICAL_HZ = 20.0
DEFAULT_IMAGE_SIZE = 224

DEFAULT_TOPICS = {
    "external_image": "/external/zed_node/rgb/color/rect/image/compressed",
    "wrist_image": "/gripper/camera_l/color/image_rect_raw/compressed",
    "feedback_joint_states": "/left_arm/feedback/joint_states",
    "control_joint_states": "/left_arm/control/joint_states",
    "tcp_pose": "/left_arm/feedback/tcp_pose",
    # Teleop device pose: Pika handle, VR controller, or any other input rig.
    "teleop_pose": "/pika_pose_l",
    "localization_valid": "/pika_localization_status_l",
    "candidate_arm_joint_states": "/nero_input_only/candidate/arm_l/joint_states",
    "candidate_gripper_joint_state": "/nero_input_only/candidate/gripper_l/joint_state",
    # Optional second arm. Present => 16D [right 8, left 8].
    "left_wrist_image": "/left_wrist/color/image_raw/compressed",
    "left_feedback_joint_states": "/left_arm/feedback/joint_states",
    "left_control_joint_states": "/left_arm/control/joint_states",
    "left_tcp_pose": "/left_arm/feedback/tcp_pose",
    "left_teleop_pose": "/nero_vr/left/pose",
    "left_localization_valid": "/nero_vr/left/pose_valid",
}

# Accepted spellings in a user-supplied topic map, mapped to the logical key.
TOPIC_KEY_ALIASES = {
    "pika_pose": "teleop_pose",
    "vr_pose": "teleop_pose",
    "controller_pose": "teleop_pose",
    "tracking_valid": "localization_valid",
    "teleop_valid": "localization_valid",
    "right_wrist_image": "wrist_image",
    "right_feedback_joint_states": "feedback_joint_states",
    "right_control_joint_states": "control_joint_states",
    "right_tcp_pose": "tcp_pose",
    "right_vr_pose": "teleop_pose",
    "right_vr_tracking_valid": "localization_valid",
    "left_vr_pose": "left_teleop_pose",
    "left_vr_tracking_valid": "left_localization_valid",
    "left_tracking_valid": "left_localization_valid",
    "left_teleop_aim_pose": "left_teleop_pose",
}

REQUIRED_TOPIC_KEYS = ("external_image", "wrist_image", "feedback_joint_states")
BIMANUAL_REQUIRED_KEYS = (
    "left_wrist_image",
    "left_feedback_joint_states",
    "left_control_joint_states",
)
IMAGE_TOPIC_KEYS = ("external_image", "wrist_image", "left_wrist_image")


def _image_topic_fallbacks(topic: str) -> list[str]:
    """Accept both ``.../image`` and ``.../image/compressed`` spellings."""
    candidates = [topic]
    if topic.endswith("/compressed"):
        candidates.append(topic[: -len("/compressed")])
    else:
        candidates.append(f"{topic}/compressed")
    return candidates


def resolve_topics_against_bag(
    topics: dict[str, str],
    available: set[str],
    *,
    required_keys: set[str],
    action_keys: tuple[str, ...],
) -> dict[str, str]:
    """Bind logical keys to topics that actually exist in the bag.

    Image streams recorded by the VR collector flipped mid-session between
    ``/vr_collection/external_image`` and ``.../compressed``; accept either.
    """
    resolved = dict(topics)
    for key in (*required_keys, *action_keys):
        preferred = topics[key]
        candidates = (
            _image_topic_fallbacks(preferred)
            if key in IMAGE_TOPIC_KEYS
            else [preferred]
        )
        match = next((topic for topic in candidates if topic in available), None)
        if match is None:
            raise ValueError(
                f"Topic {preferred!r} (for {key}) is not present in the bag. "
                f"Tried: {candidates}. Use --inspect to list topics and "
                f"--topics-json to remap."
            )
        resolved[key] = match
    return resolved


def normalize_topic_map(overrides: dict[str, str] | None) -> dict[str, str]:
    """Merge user topic overrides onto the defaults, resolving key aliases.

    Keys starting with ``_`` and non-string values (e.g. a ``notes`` object) are
    ignored so operator-facing topic_map.json files can carry metadata.
    """
    topics = dict(DEFAULT_TOPICS)
    if not overrides:
        return topics
    for raw_key, topic in overrides.items():
        if raw_key.startswith("_") or raw_key == "notes":
            continue
        if not isinstance(topic, str):
            continue
        key = TOPIC_KEY_ALIASES.get(raw_key, raw_key)
        if key not in DEFAULT_TOPICS:
            known = ", ".join(sorted(DEFAULT_TOPICS))
            raise ValueError(f"Unknown topic key {raw_key!r}. Known keys: {known}")
        if not topic.strip():
            raise ValueError(f"Topic for {raw_key!r} must be a non-empty string")
        topics[key] = topic.strip()
    return topics


def extract_convert_options(overrides: dict | None) -> dict[str, bool]:
    """Read non-topic convert flags from a topics JSON payload."""
    options = {"crop_external_padding": False}
    if not overrides:
        return options
    value = overrides.get("crop_external_padding")
    notes = overrides.get("notes")
    if value is None and isinstance(notes, dict):
        value = notes.get("crop_external_padding")
    if value is True:
        options["crop_external_padding"] = True
    elif value not in (None, False):
        raise ValueError("crop_external_padding must be a boolean")
    return options


def list_bag_topics(mcap_path: Path) -> dict[str, str]:
    """Return {topic: schema_name} for an MCAP file, for building a topic map."""
    with Path(mcap_path).open("rb") as handle:
        summary = make_reader(handle).get_summary()
        if summary is None:
            raise ValueError(f"{mcap_path} has no summary section")
        return {
            channel.topic: summary.schemas[channel.schema_id].name
            for channel in summary.channels.values()
        }


@dataclass(frozen=True)
class TimedValue:
    stamp_ns: int
    value: Any


@dataclass
class EpisodeQuality:
    episode_id: str
    source_dir: str
    action_source: str
    train_ready: bool
    num_frames: int
    duration_sec: float
    dropped_missing: int = 0
    dropped_stale: int = 0
    zed_repeat_fraction: float = 0.0
    wrist_repeat_fraction: float = 0.0
    control_arm_coverage: float = 0.0
    control_gripper_coverage: float = 0.0
    left_control_arm_coverage: float = 0.0
    left_control_gripper_coverage: float = 0.0
    left_wrist_repeat_fraction: float = 0.0
    bimanual: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SyncedFrame:
    timestamp_ns: int
    frame_index: int
    external_rgb: np.ndarray
    left_wrist_rgb: np.ndarray
    state: np.ndarray
    action: np.ndarray
    task: str
    raw: dict[str, Any]
    right_wrist_rgb: np.ndarray | None = None


def _ns_to_sec(stamp_ns: int) -> float:
    return stamp_ns * 1e-9


def _header_stamp_ns(ros_msg: Any, fallback_ns: int) -> int:
    header = getattr(ros_msg, "header", None)
    if header is None:
        return fallback_ns
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return fallback_ns
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return fallback_ns
    return sec * 1_000_000_000 + nanosec


def _decode_compressed_image(ros_msg: Any) -> np.ndarray:
    data = np.frombuffer(bytes(ros_msg.data), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode CompressedImage")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def _joint_position_map(ros_msg: Any) -> dict[str, float]:
    names = list(ros_msg.name)
    positions = list(ros_msg.position)
    if len(names) != len(positions):
        raise ValueError(
            f"JointState name/position length mismatch: {len(names)} != {len(positions)}"
        )
    return {str(name): float(pos) for name, pos in zip(names, positions, strict=True)}


def _closedness_from_width(
    width_m: float,
    *,
    gripper_min_width_m: float,
    gripper_max_width_m: float,
) -> float:
    calibrated = float(np.clip(width_m, gripper_min_width_m, gripper_max_width_m))
    return (gripper_max_width_m - calibrated) / (gripper_max_width_m - gripper_min_width_m)


def _pose_to_list(ros_msg: Any) -> list[float]:
    pose = ros_msg.pose
    return [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]


class _Timeline:
    def __init__(self) -> None:
        self._stamps: list[int] = []
        self._values: list[Any] = []

    def append(self, stamp_ns: int, value: Any) -> None:
        if self._stamps and stamp_ns < self._stamps[-1]:
            # Bags are usually ordered; tolerate rare out-of-order by insert.
            index = bisect_right(self._stamps, stamp_ns)
            self._stamps.insert(index, stamp_ns)
            self._values.insert(index, value)
            return
        self._stamps.append(stamp_ns)
        self._values.append(value)

    def __len__(self) -> int:
        return len(self._stamps)

    @property
    def stamps(self) -> list[int]:
        return self._stamps

    @property
    def values(self) -> list[Any]:
        return self._values

    def latest_at_or_before(self, stamp_ns: int) -> TimedValue | None:
        if not self._stamps:
            return None
        index = bisect_right(self._stamps, stamp_ns) - 1
        if index < 0:
            return None
        return TimedValue(self._stamps[index], self._values[index])

    def coverage(self, sample_stamps: Sequence[int]) -> float:
        if not sample_stamps:
            return 0.0
        hits = 0
        for stamp in sample_stamps:
            if self.latest_at_or_before(stamp) is not None:
                hits += 1
        return hits / len(sample_stamps)


def _is_bag_dir(path: Path) -> bool:
    return (path / "metadata.yaml").exists() or any(path.glob("*.mcap"))


def discover_episode_dirs(root: Path) -> list[Path]:
    """Return directories that contain an MCAP bag (episode folders or bag roots).

    Supports both flat bags and the VR collector layout
    ``episodeXXX/episodeXXX/*.mcap``.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if _is_bag_dir(root):
        return [root]

    episodes: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _is_bag_dir(child):
            episodes.append(child)
            continue
        # Nested rosbag2 directory: episode000/episode000/{mcap,metadata.yaml}
        nested = sorted(
            candidate
            for candidate in child.iterdir()
            if candidate.is_dir() and _is_bag_dir(candidate)
        )
        if nested:
            episodes.append(nested[0])
            continue
        # Also accept episode000/*.mcap one level deeper via resolve later.
        deep = list(child.rglob("metadata.yaml")) + list(child.rglob("*.mcap"))
        if deep:
            episodes.append(child)
    if not episodes:
        raise FileNotFoundError(f"No MCAP episodes found under {root}")
    return episodes


def resolve_mcap_path(episode_dir: Path) -> Path:
    episode_dir = episode_dir.expanduser().resolve()
    mcaps = sorted(episode_dir.glob("*.mcap"))
    if not mcaps:
        mcaps = sorted(episode_dir.rglob("*.mcap"))
    if not mcaps:
        raise FileNotFoundError(f"No .mcap file in {episode_dir}")
    # Prefer the first chronological file; multi-file bags are uncommon here.
    return mcaps[0]


def load_bag_streams(
    mcap_path: Path,
    topics: dict[str, str],
    *,
    action_source: str,
    require_bimanual: bool = False,
) -> dict[str, _Timeline]:
    required_keys = set(REQUIRED_TOPIC_KEYS)
    if require_bimanual:
        required_keys.update(BIMANUAL_REQUIRED_KEYS)
    for key in required_keys:
        if key not in topics or not topics[key]:
            raise ValueError(f"Missing required topic mapping for {key}")

    if action_source == "control":
        action_keys = ("control_joint_states",)
        if require_bimanual:
            action_keys = ("control_joint_states", "left_control_joint_states")
    elif action_source == "candidate":
        action_keys = ("candidate_arm_joint_states", "candidate_gripper_joint_state")
    else:
        raise ValueError(f"Unknown action_source={action_source!r}")
    for key in action_keys:
        if key not in topics or not topics[key]:
            raise ValueError(f"Missing action topic mapping for {key}")

    optional_keys = (
        "tcp_pose",
        "teleop_pose",
        "localization_valid",
        "left_wrist_image",
        "left_feedback_joint_states",
        "left_control_joint_states",
        "left_tcp_pose",
        "left_teleop_pose",
        "left_localization_valid",
    )
    available = set(list_bag_topics(mcap_path))
    topics = resolve_topics_against_bag(
        topics,
        available,
        required_keys=required_keys,
        action_keys=action_keys,
    )
    load_keys = (*required_keys, *action_keys, *optional_keys)
    topic_to_key = {
        topics[key]: key for key in load_keys if key in topics and topics[key]
    }
    # Drop optional keys whose topics are absent rather than failing hard.
    topic_to_key = {
        topic: key
        for topic, key in topic_to_key.items()
        if topic in available or key in (*required_keys, *action_keys)
    }
    wanted = set(topic_to_key)
    streams = {key: _Timeline() for key in topic_to_key.values()}

    for msg in read_ros2_messages(str(mcap_path), topics=list(wanted)):
        topic = msg.channel.topic
        key = topic_to_key[topic]
        stamp_ns = _header_stamp_ns(msg.ros_msg, msg.log_time_ns)
        ros_msg = msg.ros_msg

        if key in IMAGE_TOPIC_KEYS:
            streams[key].append(stamp_ns, _decode_compressed_image(ros_msg))
        elif key in {
            "feedback_joint_states",
            "control_joint_states",
            "candidate_arm_joint_states",
            "candidate_gripper_joint_state",
            "left_feedback_joint_states",
            "left_control_joint_states",
        }:
            streams[key].append(stamp_ns, _joint_position_map(ros_msg))
        elif key in {"tcp_pose", "teleop_pose", "left_tcp_pose", "left_teleop_pose"}:
            streams[key].append(stamp_ns, _pose_to_list(ros_msg))
        elif key in {"localization_valid", "left_localization_valid"}:
            streams[key].append(stamp_ns, bool(getattr(ros_msg, "data", False)))
        else:
            streams[key].append(stamp_ns, ros_msg)

    return streams


def _merge_named_partials(
    arm_timeline: _Timeline,
    gripper_timeline: _Timeline | None,
    stamp_ns: int,
) -> dict[str, float] | None:
    arm = arm_timeline.latest_at_or_before(stamp_ns)
    if arm is None:
        return None
    merged = dict(arm.value)
    if gripper_timeline is not None:
        grip = gripper_timeline.latest_at_or_before(stamp_ns)
        if grip is not None:
            merged.update(grip.value)
    return merged


def _control_command_at(control_timeline: _Timeline, stamp_ns: int) -> dict[str, float] | None:
    """Merge all control JointState partials up to stamp with ZOH by name."""
    if not control_timeline.stamps:
        return None
    index = bisect_right(control_timeline.stamps, stamp_ns) - 1
    if index < 0:
        return None
    merged: dict[str, float] = {}
    # Walk chronologically so later messages overwrite earlier ones per name.
    for i in range(index + 1):
        merged.update(control_timeline.values[i])
    return merged


def _state_from_feedback(
    joint_map: dict[str, float],
    *,
    gripper_min_width_m: float,
    gripper_max_width_m: float,
) -> tuple[np.ndarray, float]:
    missing = [name for name in (*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME) if name not in joint_map]
    if missing:
        raise ValueError(f"Feedback JointState missing {missing}")
    joints = np.asarray([joint_map[name] for name in ARM_JOINT_NAMES], dtype=np.float32)
    width = float(joint_map[GRIPPER_JOINT_NAME])
    closedness = _closedness_from_width(
        width,
        gripper_min_width_m=gripper_min_width_m,
        gripper_max_width_m=gripper_max_width_m,
    )
    state = np.concatenate([joints, np.asarray([closedness], dtype=np.float32)])
    return state, width


def _action_from_command_map(
    command_map: dict[str, float],
    *,
    gripper_min_width_m: float,
    gripper_max_width_m: float,
    fallback_state: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build absolute 8D action; never invent missing joints as zero."""
    missing_joints = [name for name in ARM_JOINT_NAMES if name not in command_map]
    meta = {
        "missing_joint_names": missing_joints,
        "has_gripper_command": GRIPPER_JOINT_NAME in command_map,
        "used_state_fallback_for_joints": bool(missing_joints),
    }
    if missing_joints:
        # Incomplete command snapshot: hold previous absolute targets from state.
        joints = fallback_state[:7].astype(np.float32).copy()
        for i, name in enumerate(ARM_JOINT_NAMES):
            if name in command_map:
                joints[i] = np.float32(command_map[name])
    else:
        joints = np.asarray([command_map[name] for name in ARM_JOINT_NAMES], dtype=np.float32)

    if GRIPPER_JOINT_NAME in command_map:
        width = float(command_map[GRIPPER_JOINT_NAME])
        closedness = _closedness_from_width(
            width,
            gripper_min_width_m=gripper_min_width_m,
            gripper_max_width_m=gripper_max_width_m,
        )
        meta["gripper_width_command_m"] = width
    else:
        closedness = float(fallback_state[7])
        meta["gripper_width_command_m"] = None
        meta["used_state_fallback_for_gripper"] = True

    action = np.concatenate([joints, np.asarray([closedness], dtype=np.float32)])
    if not np.all(np.isfinite(action)):
        raise ValueError("Action contains NaN or Inf")
    return action, meta


def _stream_len(streams: dict[str, _Timeline], key: str) -> int:
    timeline = streams.get(key)
    return 0 if timeline is None else len(timeline)


def _is_bimanual(streams: dict[str, _Timeline]) -> bool:
    return (
        _stream_len(streams, "left_feedback_joint_states") > 0
        and _stream_len(streams, "left_control_joint_states") > 0
    )


def sync_episode_frames(
    streams: dict[str, _Timeline],
    *,
    episode_id: str,
    source_dir: str,
    action_source: str,
    task: str,
    fps: float = CANONICAL_HZ,
    image_size: int = DEFAULT_IMAGE_SIZE,
    gripper_min_width_m: float = DEFAULT_GRIPPER_MIN_M,
    gripper_max_width_m: float = DEFAULT_GRIPPER_MAX_M,
    left_gripper_min_width_m: float = DEFAULT_LEFT_GRIPPER_MIN_M,
    left_gripper_max_width_m: float = DEFAULT_LEFT_GRIPPER_MAX_M,
    require_bimanual: bool = False,
    max_obs_age_sec: float = 0.5,
    train_ready: bool,
    crop_external_padding: bool = False,
) -> tuple[list[SyncedFrame], EpisodeQuality]:
    external = streams["external_image"]
    wrist = streams["wrist_image"]
    feedback = streams["feedback_joint_states"]
    left_wrist = streams.get("left_wrist_image")
    left_feedback = streams.get("left_feedback_joint_states")
    bimanual = require_bimanual
    if bimanual and not _is_bimanual(streams):
        raise ValueError(
            f"Episode {episode_id} is missing left-arm control/feedback; "
            "bimanual conversion requires both arms"
        )

    if len(external) == 0 or len(wrist) == 0 or len(feedback) == 0:
        raise ValueError(
            f"Episode {episode_id} missing required streams: "
            f"external={len(external)} wrist={len(wrist)} feedback={len(feedback)}"
        )
    if bimanual and (left_wrist is None or len(left_wrist) == 0 or left_feedback is None or len(left_feedback) == 0):
        raise ValueError(
            f"Episode {episode_id} is bimanual but missing left wrist or left feedback"
        )

    overlap_starts = [external.stamps[0], wrist.stamps[0], feedback.stamps[0]]
    overlap_ends = [external.stamps[-1], wrist.stamps[-1], feedback.stamps[-1]]
    if bimanual:
        overlap_starts.extend([left_wrist.stamps[0], left_feedback.stamps[0]])
        overlap_ends.extend([left_wrist.stamps[-1], left_feedback.stamps[-1]])
    start_ns = max(overlap_starts)
    end_ns = min(overlap_ends)
    if end_ns <= start_ns:
        raise ValueError(f"Episode {episode_id} has empty time overlap")

    period_ns = int(round(1e9 / fps))
    sample_stamps = list(range(start_ns, end_ns + 1, period_ns))

    frames: list[SyncedFrame] = []
    dropped_missing = 0
    dropped_stale = 0
    zed_ids: list[int] = []
    wrist_ids: list[int] = []
    left_wrist_ids: list[int] = []
    arm_hits = 0
    gripper_hits = 0
    left_arm_hits = 0
    left_gripper_hits = 0
    notes: list[str] = []

    if action_source == "candidate":
        notes.append(
            "action_source=candidate uses /nero_input_only/candidate/*; "
            "NOT train-ready (shadow IK, not executed control commands)"
        )
        train_ready = False
    if bimanual:
        notes.append("bimanual 16D layout [right_8, left_8]")
    if crop_external_padding:
        notes.append("cropped uniform padding on external_image before resize_with_pad")

    for frame_index, stamp_ns in enumerate(sample_stamps):
        ext = external.latest_at_or_before(stamp_ns)
        wr = wrist.latest_at_or_before(stamp_ns)
        fb = feedback.latest_at_or_before(stamp_ns)
        left_wr = left_wrist.latest_at_or_before(stamp_ns) if bimanual else None
        left_fb = left_feedback.latest_at_or_before(stamp_ns) if bimanual else None
        if ext is None or wr is None or fb is None or (bimanual and (left_wr is None or left_fb is None)):
            dropped_missing += 1
            continue

        max_age_ns = int(max_obs_age_sec * 1e9)
        stale = (
            stamp_ns - ext.stamp_ns > max_age_ns
            or stamp_ns - wr.stamp_ns > max_age_ns
            or stamp_ns - fb.stamp_ns > max_age_ns
        )
        if bimanual:
            stale = stale or stamp_ns - left_wr.stamp_ns > max_age_ns or stamp_ns - left_fb.stamp_ns > max_age_ns
        if stale:
            dropped_stale += 1
            continue

        try:
            state, feedback_width = _state_from_feedback(
                fb.value,
                gripper_min_width_m=gripper_min_width_m,
                gripper_max_width_m=gripper_max_width_m,
            )
            left_state = None
            left_feedback_width = None
            if bimanual:
                left_state, left_feedback_width = _state_from_feedback(
                    left_fb.value,
                    gripper_min_width_m=left_gripper_min_width_m,
                    gripper_max_width_m=left_gripper_max_width_m,
                )
        except ValueError:
            dropped_missing += 1
            continue

        if action_source == "control":
            command_map = _control_command_at(streams["control_joint_states"], stamp_ns)
            if command_map is None:
                dropped_missing += 1
                continue
            if any(name in command_map for name in ARM_JOINT_NAMES):
                arm_hits += 1
            if GRIPPER_JOINT_NAME in command_map:
                gripper_hits += 1
            left_command_map = None
            if bimanual:
                left_command_map = _control_command_at(streams["left_control_joint_states"], stamp_ns)
                if left_command_map is None:
                    dropped_missing += 1
                    continue
                if any(name in left_command_map for name in ARM_JOINT_NAMES):
                    left_arm_hits += 1
                if GRIPPER_JOINT_NAME in left_command_map:
                    left_gripper_hits += 1
        else:
            command_map = _merge_named_partials(
                streams["candidate_arm_joint_states"],
                streams["candidate_gripper_joint_state"],
                stamp_ns,
            )
            if command_map is None:
                dropped_missing += 1
                continue
            if any(name in command_map for name in ARM_JOINT_NAMES):
                arm_hits += 1
            if GRIPPER_JOINT_NAME in command_map:
                gripper_hits += 1
            left_command_map = None

        action, action_meta = _action_from_command_map(
            command_map,
            gripper_min_width_m=gripper_min_width_m,
            gripper_max_width_m=gripper_max_width_m,
            fallback_state=state,
        )
        left_action_meta: dict[str, Any] | None = None
        if bimanual:
            left_action, left_action_meta = _action_from_command_map(
                left_command_map,
                gripper_min_width_m=left_gripper_min_width_m,
                gripper_max_width_m=left_gripper_max_width_m,
                fallback_state=left_state,
            )
            state = np.concatenate([state, left_state])
            action = np.concatenate([action, left_action])

        external_value = crop_uniform_border(ext.value) if crop_external_padding else ext.value
        external_rgb = resize_with_pad(external_value, image_size)
        left_wrist_rgb = resize_with_pad(wr.value, image_size)
        right_wrist_rgb = resize_with_pad(left_wr.value, image_size) if bimanual else None
        zed_ids.append(ext.stamp_ns)
        wrist_ids.append(wr.stamp_ns)
        if bimanual:
            left_wrist_ids.append(left_wr.stamp_ns)

        tcp = streams.get("tcp_pose")
        teleop = streams.get("teleop_pose")
        loc = streams.get("localization_valid")
        tcp_tv = tcp.latest_at_or_before(stamp_ns) if tcp is not None else None
        teleop_tv = teleop.latest_at_or_before(stamp_ns) if teleop is not None else None
        loc_tv = loc.latest_at_or_before(stamp_ns) if loc is not None else None

        raw = {
            "joint_position_feedback": state[:7].astype(np.float64).tolist(),
            "gripper_width_feedback_m": feedback_width,
            "joint_position_command": action[:7].astype(np.float64).tolist(),
            "gripper_width_command_m": action_meta.get("gripper_width_command_m"),
            "gripper_closedness_command": float(action[7]),
            "tcp_pose": tcp_tv.value if tcp_tv is not None else None,
            "teleop_pose": teleop_tv.value if teleop_tv is not None else None,
            "localization_valid": loc_tv.value if loc_tv is not None else None,
            "external_image_stamp_ns": ext.stamp_ns,
            "wrist_image_stamp_ns": wr.stamp_ns,
            "feedback_stamp_ns": fb.stamp_ns,
            "action_meta": action_meta,
        }
        if bimanual:
            raw.update(
                {
                    "left_joint_position_feedback": state[8:15].astype(np.float64).tolist(),
                    "left_gripper_width_feedback_m": left_feedback_width,
                    "left_joint_position_command": action[8:15].astype(np.float64).tolist(),
                    "left_gripper_width_command_m": left_action_meta.get("gripper_width_command_m"),
                    "left_gripper_closedness_command": float(action[15]),
                    "left_wrist_image_stamp_ns": left_wr.stamp_ns,
                    "left_feedback_stamp_ns": left_fb.stamp_ns,
                    "left_action_meta": left_action_meta,
                }
            )

        frames.append(
            SyncedFrame(
                timestamp_ns=stamp_ns,
                frame_index=len(frames),
                external_rgb=external_rgb,
                left_wrist_rgb=left_wrist_rgb,
                right_wrist_rgb=right_wrist_rgb,
                state=state.astype(np.float32),
                action=action.astype(np.float32),
                task=task,
                raw=raw,
            )
        )

    def _repeat_fraction(ids: list[int]) -> float:
        if len(ids) < 2:
            return 0.0
        repeats = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
        return repeats / (len(ids) - 1)

    right_arm_cov = (arm_hits / len(frames)) if frames else 0.0
    right_grip_cov = (gripper_hits / len(frames)) if frames else 0.0
    left_arm_cov = (left_arm_hits / len(frames)) if frames and bimanual else 0.0
    left_grip_cov = (left_gripper_hits / len(frames)) if frames and bimanual else 0.0
    quality = EpisodeQuality(
        episode_id=episode_id,
        source_dir=source_dir,
        action_source=action_source,
        train_ready=bool(train_ready and frames),
        num_frames=len(frames),
        duration_sec=_ns_to_sec(end_ns - start_ns),
        dropped_missing=dropped_missing,
        dropped_stale=dropped_stale,
        zed_repeat_fraction=_repeat_fraction(zed_ids),
        wrist_repeat_fraction=_repeat_fraction(wrist_ids),
        left_wrist_repeat_fraction=_repeat_fraction(left_wrist_ids),
        control_arm_coverage=min(right_arm_cov, left_arm_cov) if bimanual else right_arm_cov,
        control_gripper_coverage=min(right_grip_cov, left_grip_cov) if bimanual else right_grip_cov,
        left_control_arm_coverage=left_arm_cov,
        left_control_gripper_coverage=left_grip_cov,
        bimanual=bimanual,
        notes=notes,
    )
    if quality.zed_repeat_fraction > 0.2:
        quality.notes.append(
            f"ZED ZOH repeat fraction={quality.zed_repeat_fraction:.3f} "
            f"(ZED ~15 Hz vs canonical {fps} Hz)"
        )
    return frames, quality


def convert_episode(
    episode_dir: Path,
    *,
    task: str,
    action_source: str = "control",
    topics: dict[str, str] | None = None,
    fps: float = CANONICAL_HZ,
    image_size: int = DEFAULT_IMAGE_SIZE,
    gripper_min_width_m: float = DEFAULT_GRIPPER_MIN_M,
    gripper_max_width_m: float = DEFAULT_GRIPPER_MAX_M,
    left_gripper_min_width_m: float = DEFAULT_LEFT_GRIPPER_MIN_M,
    left_gripper_max_width_m: float = DEFAULT_LEFT_GRIPPER_MAX_M,
    require_bimanual: bool = False,
    train_ready: bool | None = None,
    crop_external_padding: bool = False,
) -> tuple[list[SyncedFrame], EpisodeQuality]:
    episode_dir = episode_dir.expanduser().resolve()
    topic_map = normalize_topic_map(topics)
    mcap_path = resolve_mcap_path(episode_dir)
    streams = load_bag_streams(
        mcap_path,
        topic_map,
        action_source=action_source,
        require_bimanual=require_bimanual,
    )
    ready = True if train_ready is None else train_ready
    if action_source != "control":
        ready = False
    return sync_episode_frames(
        streams,
        episode_id=episode_dir.name,
        source_dir=str(episode_dir),
        action_source=action_source,
        task=task,
        fps=fps,
        image_size=image_size,
        gripper_min_width_m=gripper_min_width_m,
        gripper_max_width_m=gripper_max_width_m,
        left_gripper_min_width_m=left_gripper_min_width_m,
        left_gripper_max_width_m=left_gripper_max_width_m,
        require_bimanual=require_bimanual,
        train_ready=ready,
        crop_external_padding=crop_external_padding,
    )


def write_quality_report(path: Path, qualities: Iterable[EpisodeQuality]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "canonical_frequency_hz": CANONICAL_HZ,
        "episodes": [q.to_dict() for q in qualities],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_raw_audit(path: Path, frames: Sequence[SyncedFrame]) -> None:
    """Write per-frame raw.* audit fields as JSONL (not required by LeRobot)."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            record = {
                "timestamp_ns": frame.timestamp_ns,
                "frame_index": frame.frame_index,
                "task": frame.task,
                "state": frame.state.tolist(),
                "action": frame.action.tolist(),
                "raw": frame.raw,
            }
            handle.write(json.dumps(record) + "\n")

