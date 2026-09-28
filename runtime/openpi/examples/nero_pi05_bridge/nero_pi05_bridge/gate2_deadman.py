#!/usr/bin/env python3
"""Hold-to-run software deadman heartbeat for Gate 2.

Publishes Empty messages on ``/{arm}/gate2_deadman`` while Space is held.
This is NOT a hardware e-stop. Keep a physical/service emergency stop ready.

Usage (with ROS sourced)::

  ros2 run nero_pi05_bridge gate2_deadman --ros-args -p arm_namespace:=/right_arm
"""

from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty


class DeadmanNode(Node):
    def __init__(self) -> None:
        super().__init__("nero_gate2_deadman")
        self.declare_parameter("arm_namespace", "/right_arm")
        self.declare_parameter("rate_hz", 50.0)
        namespace = str(self.get_parameter("arm_namespace").value)
        rate_hz = float(self.get_parameter("rate_hz").value)
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        topic = f"{namespace}/gate2_deadman"
        self._publisher = self.create_publisher(Empty, topic, 10)
        self._held = False
        self._stop = False
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().warning(
            "SOFTWARE DEADMAN ONLY: hold SPACE to allow Gate 2 motion; "
            "release SPACE to cut heartbeat. Keep hardware e-stop ready. "
            f"Publishing on {topic}"
        )

    def _tick(self) -> None:
        if self._held:
            self._publisher.publish(Empty())

    def set_held(self, held: bool) -> None:
        self._held = held


def _keyboard_loop(node: DeadmanNode) -> None:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    held_until = 0.0
    try:
        tty.setcbreak(fd)
        print("Hold SPACE to arm deadman heartbeat; Ctrl-C to quit.", flush=True)
        while rclpy.ok() and not node._stop:
            readable, _, _ = select.select([fd], [], [], 0.05)
            if readable:
                ch = sys.stdin.read(1)
                if ch == " ":
                    held_until = time.monotonic() + 0.15
                elif ch in {"\x03", "q"}:
                    node._stop = True
            node.set_held(time.monotonic() < held_until)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.set_held(False)


def main() -> int:
    rclpy.init()
    node = DeadmanNode()
    thread = threading.Thread(target=_keyboard_loop, args=(node,), daemon=True)
    thread.start()
    try:
        while rclpy.ok() and not node._stop:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop = True
        thread.join(timeout=0.5)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
