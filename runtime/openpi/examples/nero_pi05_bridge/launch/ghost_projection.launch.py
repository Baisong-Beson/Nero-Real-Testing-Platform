"""Launch an isolated NERO URDF ghost and optional RViz display."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    bridge_share = get_package_share_directory("nero_pi05_bridge")
    description_share = get_package_share_directory("agx_arm_description")
    default_urdf = os.path.join(
        description_share,
        "agx_arm_urdf",
        "nero",
        "urdf",
        "nero_with_gripper_description.urdf",
    )
    default_rviz = os.path.join(bridge_share, "config", "nero_ghost.rviz")
    projection = LaunchConfiguration("projection")
    urdf = LaunchConfiguration("urdf")

    return LaunchDescription(
        [
            DeclareLaunchArgument("projection", default_value="/tmp/nero_projection.json"),
            DeclareLaunchArgument("urdf", default_value=default_urdf),
            DeclareLaunchArgument("group", default_value="approach"),
            DeclareLaunchArgument(
                "trajectory_mode",
                default_value="clipped",
                choices=["raw", "clipped"],
            ),
            DeclareLaunchArgument(
                "initial_mode",
                default_value="requested",
                choices=["requested", "clipped"],
            ),
            DeclareLaunchArgument("rate_hz", default_value="8.0"),
            DeclareLaunchArgument("initial_hold_s", default_value="2.0"),
            DeclareLaunchArgument("loop", default_value="true", choices=["true", "false"]),
            DeclareLaunchArgument(
                "display_rviz",
                default_value="true",
                choices=["true", "false"],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace="nero_ghost",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": ParameterValue(
                            Command(["cat ", urdf]),
                            value_type=str,
                        ),
                        "frame_prefix": "nero_ghost/",
                    }
                ],
            ),
            Node(
                package="nero_pi05_bridge",
                executable="ghost_playback",
                output="screen",
                arguments=[
                    "--projection",
                    projection,
                    "--group",
                    LaunchConfiguration("group"),
                    "--trajectory-mode",
                    LaunchConfiguration("trajectory_mode"),
                    "--initial-mode",
                    LaunchConfiguration("initial_mode"),
                    "--rate-hz",
                    LaunchConfiguration("rate_hz"),
                    "--initial-hold-s",
                    LaunchConfiguration("initial_hold_s"),
                    "--loop",
                    LaunchConfiguration("loop"),
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="nero_ghost_rviz",
                output="screen",
                condition=IfCondition(LaunchConfiguration("display_rviz")),
                arguments=["-d", default_rviz],
            ),
        ]
    )
