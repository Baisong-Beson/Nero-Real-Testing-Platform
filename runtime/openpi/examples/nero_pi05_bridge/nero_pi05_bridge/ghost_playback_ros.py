"""ROS publisher isolated to /nero_ghost for offline RViz visualization."""

import math

from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

from nero_pi05_bridge.ghost_playback import JOINT_NAMES
from nero_pi05_bridge.ghost_playback import GhostTrajectory


class GhostPlaybackNode(Node):
    def __init__(
        self,
        trajectory: GhostTrajectory,
        *,
        rate_hz: float,
        initial_hold_s: float,
        loop: bool,
    ):
        super().__init__("nero_ghost_playback")
        self._trajectory = trajectory
        self._loop = loop
        self._initial_frames = max(1, math.ceil(initial_hold_s * rate_hz))
        self._frame = -self._initial_frames
        self._publisher = self.create_publisher(
            JointState,
            "/nero_ghost/joint_states",
            1,
        )
        self._marker_publisher = self.create_publisher(
            Marker,
            "/nero_ghost/tcp_path",
            1,
        )
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_next)
        self.get_logger().warning(
            "OFFLINE GHOST ONLY: publishing isolated /nero_ghost/joint_states; "
            "no arm control topic or service exists in this node"
        )
        if trajectory.initial_limit_violations:
            joints = ", ".join(violation["joint"] for violation in trajectory.initial_limit_violations)
            self.get_logger().warning(f"requested initial pose violates URDF limits: {joints}")

    def _publish_next(self) -> None:
        is_initial = self._frame < 0
        positions = self._trajectory.initial_positions if is_initial else self._trajectory.frames[self._frame]
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = list(positions)
        self._publisher.publish(message)
        self._publish_markers(is_initial=is_initial)

        self._frame += 1
        if self._frame >= len(self._trajectory.frames):
            if self._loop:
                self._frame = -self._initial_frames
            else:
                self._timer.cancel()
                rclpy.shutdown()

    def _publish_markers(self, *, is_initial: bool) -> None:
        stamp = self.get_clock().now().to_msg()
        path = Marker()
        path.header.frame_id = "nero_ghost/world"
        path.header.stamp = stamp
        path.ns = f"pi05_base_{self._trajectory.group}"
        path.id = 0
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.pose.orientation.w = 1.0
        path.scale.x = 0.008
        path.color.r = 0.1
        path.color.g = 0.9
        path.color.b = 1.0
        path.color.a = 0.95
        path.points = [Point(x=x, y=y, z=z) for x, y, z in self._trajectory.tcp_path]
        self._marker_publisher.publish(path)

        current_index = max(0, min(self._frame, len(self._trajectory.tcp_path) - 1))
        position = (
            self._trajectory.initial_tcp_position
            if is_initial
            else self._trajectory.tcp_path[current_index]
        )
        cursor = Marker()
        cursor.header.frame_id = "nero_ghost/world"
        cursor.header.stamp = stamp
        cursor.ns = path.ns
        cursor.id = 1
        cursor.type = Marker.SPHERE
        cursor.action = Marker.ADD
        cursor.pose.position = Point(x=position[0], y=position[1], z=position[2])
        cursor.pose.orientation.w = 1.0
        cursor.scale.x = cursor.scale.y = cursor.scale.z = 0.025
        cursor.color.r = 1.0
        cursor.color.g = 0.8
        cursor.color.b = 0.1
        cursor.color.a = 1.0
        self._marker_publisher.publish(cursor)


def run_playback(
    trajectory: GhostTrajectory,
    *,
    rate_hz: float,
    initial_hold_s: float,
    loop: bool,
) -> int:
    rclpy.init()
    node = GhostPlaybackNode(
        trajectory,
        rate_hz=rate_hz,
        initial_hold_s=initial_hold_s,
        loop=loop,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
