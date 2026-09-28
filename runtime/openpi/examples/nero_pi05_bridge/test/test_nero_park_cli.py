"""Unit tests for the park CLI pure logic (no ROS / no hardware)."""

from __future__ import annotations

import numpy as np
from nero_pi05_bridge.nero_park_cli import DEFAULT_PARK_WAYPOINTS_DEG
from nero_pi05_bridge.nero_park_cli import DEFAULT_TOLERANCE_RAD
from nero_pi05_bridge.nero_park_cli import _waypoints_rad
from nero_pi05_bridge.nero_park_cli import approval_token
from nero_pi05_bridge.nero_park_cli import validate_park_sequence
import pytest


def test_default_sequence_ends_with_flat_wrist():
    """Leg 2 must leave joint4 at 0 so disabling the arm cannot drop the wrist."""
    leg1, leg2 = DEFAULT_PARK_WAYPOINTS_DEG
    assert leg1[3] == pytest.approx(90.0)
    assert leg2[3] == pytest.approx(0.0)


def test_default_sequence_yaws_clear_before_lowering_wrist():
    """Leg 1 yaws the base away so lowering the wrist cannot sweep the workspace."""
    leg1, leg2 = DEFAULT_PARK_WAYPOINTS_DEG
    assert leg1[0] == pytest.approx(-20.0)
    assert leg2[0] == pytest.approx(0.0)
    assert leg1[0] != leg2[0]


def test_default_sequence_is_inside_soft_limits():
    validated = validate_park_sequence(_waypoints_rad())
    assert len(validated) == 2
    assert all(entry["within_soft_limits"] for entry in validated)


def test_tolerance_absorbs_joint4_gravity_residual():
    """move_j holds ~0.046 rad short on joint4; a tighter default would time out."""
    assert DEFAULT_TOLERANCE_RAD > 0.046


def test_rejects_waypoint_outside_soft_limits():
    # joint2 upper limit is 1.74 rad in the URDF, so 100 deg cannot be reached.
    bad = _waypoints_rad([[0.0, 100.0, 90.0, 90.0, 0.0, 0.0, 0.0], [0.0, 90.0, 90.0, 0.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="soft-limit"):
        validate_park_sequence(bad)


def test_rejects_empty_sequence():
    with pytest.raises(ValueError, match="at least one waypoint"):
        validate_park_sequence([])


def test_token_covers_every_waypoint():
    """A token minted for one sequence must not authorize a different one."""
    base = _waypoints_rad()
    other = _waypoints_rad([list(DEFAULT_PARK_WAYPOINTS_DEG[0]), [0.0, 90.0, 90.0, 45.0, 0.0, 0.0, 0.0]])
    kwargs = {"namespace": "/right_arm", "tolerance_rad": DEFAULT_TOLERANCE_RAD}
    assert approval_token(waypoints_rad=base, **kwargs) != approval_token(
        waypoints_rad=other, **kwargs
    )


def test_token_covers_namespace_and_tolerance():
    base = _waypoints_rad()
    token = approval_token(
        namespace="/right_arm", waypoints_rad=base, tolerance_rad=DEFAULT_TOLERANCE_RAD
    )
    assert token != approval_token(
        namespace="/left_arm", waypoints_rad=base, tolerance_rad=DEFAULT_TOLERANCE_RAD
    )
    assert token != approval_token(
        namespace="/right_arm", waypoints_rad=base, tolerance_rad=0.02
    )


def test_token_is_stable_and_prefixed():
    base = _waypoints_rad()
    kwargs = {
        "namespace": "/right_arm",
        "waypoints_rad": base,
        "tolerance_rad": DEFAULT_TOLERANCE_RAD,
    }
    token = approval_token(**kwargs)
    assert token == approval_token(**kwargs)
    assert token.startswith("NERO_PARK_MOVE_J_")


def test_waypoints_convert_degrees_to_radians():
    leg1, leg2 = _waypoints_rad()
    assert leg1[0] == pytest.approx(np.deg2rad(-20.0))
    assert leg2[1] == pytest.approx(np.pi / 2)
