import json

from nero_pi05_bridge.ghost_playback import load_ghost_trajectory
import pytest


def _report():
    actions = [[0.0] * 8 for _ in range(50)]
    actions[0][0] = 9.0
    actions[0][7] = -1.0
    frame_names = ["base_link", "gripper_base"]
    raw_frames = [[[0.0, 0.0, 0.0], [float(index), 1.0, 2.0]] for index in range(50)]
    clipped_frames = [[[0.0, 0.0, 0.0], [float(index), 3.0, 4.0]] for index in range(50)]
    return {
        "commandable": False,
        "initial_state": [0.0, 1.0, 1.0, -1.5, 0.0, 0.0, 0.0, 0.01],
        "initial_clipped_state": [0.0, 1.0, 1.0, -1.01, 0.0, 0.0, 0.0, 0.01],
        "initial_limit_violations": [{"joint": "joint4"}],
        "joint_limits_lower": [-2.0] * 7,
        "joint_limits_upper": [2.0] * 7,
        "frame_names": frame_names,
        "initial_frame_positions": [[0.0, 0.0, 0.0], [10.0, 11.0, 12.0]],
        "initial_clipped_frame_positions": [[0.0, 0.0, 0.0], [20.0, 21.0, 22.0]],
        "groups": {
            "approach": {
                "raw_action_mean": actions,
                "raw_frame_positions": raw_frames,
                "clipped_frame_positions": clipped_frames,
            }
        },
    }


def test_clipped_ghost_is_isolated_and_has_50_frames(tmp_path):
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(_report()))
    trajectory = load_ghost_trajectory(
        path,
        group="approach",
        trajectory_mode="clipped",
        initial_mode="requested",
    )
    assert len(trajectory.frames) == 50
    assert trajectory.initial_positions[3] == -1.5
    assert trajectory.frames[0][0] == 2.0
    assert trajectory.frames[0][7] == 0.099
    assert trajectory.initial_tcp_position == (10.0, 11.0, 12.0)
    assert trajectory.tcp_path[0] == (0.0, 3.0, 4.0)
    assert trajectory.tcp_path[-1] == (49.0, 3.0, 4.0)
    assert trajectory.initial_limit_violations[0]["joint"] == "joint4"


def test_variable_horizon_ghost(tmp_path):
    report = _report()
    report["groups"]["approach"]["raw_action_mean"] = report["groups"]["approach"]["raw_action_mean"][:16]
    report["groups"]["approach"]["raw_frame_positions"] = report["groups"]["approach"]["raw_frame_positions"][:16]
    report["groups"]["approach"]["clipped_frame_positions"] = report["groups"]["approach"]["clipped_frame_positions"][:16]
    path = tmp_path / "projection16.json"
    path.write_text(json.dumps(report))
    trajectory = load_ghost_trajectory(
        path,
        group="approach",
        trajectory_mode="raw",
        initial_mode="requested",
    )
    assert len(trajectory.frames) == 16
    assert len(trajectory.tcp_path) == 16


def test_projection_must_be_non_commandable(tmp_path):
    report = _report()
    report["commandable"] = True
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="commandable=false"):
        load_ghost_trajectory(
            path,
            group="approach",
            trajectory_mode="clipped",
            initial_mode="clipped",
        )
