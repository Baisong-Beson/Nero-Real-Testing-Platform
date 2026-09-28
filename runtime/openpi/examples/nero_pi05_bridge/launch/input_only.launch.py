"""Fail-closed Pika input launch for Gate 0 data capture.

This launch intentionally contains no arm driver or publisher under an arm
control namespace. Serial inputs and shadow IK are opt-in. Every candidate
action output is confined to ``/nero_input_only/candidate``.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _serial_input_node(side: str) -> Node:
    enabled = LaunchConfiguration(f"enable_{side}_serial")
    serial_port = LaunchConfiguration(f"{side}_serial_port")
    short = "l" if side == "left" else "r"
    return Node(
        package="sensor_tools",
        executable="serial_gripper_imu",
        name=f"nero_input_only_serial_{side}",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(enabled),
        parameters=[
            {
                "serial_port": serial_port,
                "joint_name": LaunchConfiguration(f"{side}_joint_name"),
            }
        ],
        remappings=[
            ("/imu/data", f"/nero_input_only/sensor/imu_{short}"),
            ("/gripper/data", f"/nero_input_only/sensor/gripper_{short}/data"),
            ("/gripper/ctrl", f"/nero_input_only/disabled/gripper_{short}/ctrl"),
            ("/gripper/joint_state", f"/sensor/gripper_{short}/joint_state"),
            (
                "/gripper/joint_state_ctrl",
                f"/nero_input_only/disabled/joint_states_{short}",
            ),
            (
                "/joint_state_info",
                f"/nero_input_only/disabled/joint_state_info_{short}",
            ),
            (
                "/joint_state_gripper",
                f"/nero_input_only/sensor/joint_states_gripper_{short}",
            ),
            (
                "/teleop_trigger",
                f"/nero_input_only/disabled/teleop_trigger_{short}",
            ),
            (
                "/data_capture_status",
                "/nero_input_only/disabled/data_capture_status",
            ),
            (
                "/teleop_status",
                f"/nero_input_only/disabled/teleop_status_{short}",
            ),
            ("/localization_status", f"/pika_localization_status_{short}"),
            (
                "/arm_control_status",
                f"/nero_input_only/disabled/arm_control_status_{short}",
            ),
            (
                "/data_tools_dataCapture/capture_service",
                f"/nero_input_only/disabled/capture_service_{short}",
            ),
        ],
    )


def _left_shadow_nodes() -> list[Node]:
    enabled = IfCondition(LaunchConfiguration("enable_left_shadow"))
    remote_share = get_package_share_directory("pika_remote_agx_arm")
    pika_python_bin = os.environ.get(
        "PIKA_PYTHON_BIN",
        os.path.expanduser("~/miniconda3/envs/pika/bin"),
    )
    shadow_env = {
        "PATH": f"{pika_python_bin}:{os.environ.get('PATH', '')}",
    }
    ik_config = os.path.join(
        remote_share,
        "config",
        "arm_ik_pose_node.nero.yaml",
    )
    delta_pose = "/nero_input_only/candidate/tcp_pose_l"
    candidate_arm = "/nero_input_only/candidate/arm_l/joint_states"
    candidate_gripper = "/nero_input_only/candidate/gripper_l/joint_state"
    return [
        Node(
            package="pika_remote_agx_arm",
            executable="pub_delta_pose.py",
            name="nero_input_only_delta_left",
            output="screen",
            emulate_tty=True,
            condition=enabled,
            additional_env=shadow_env,
            parameters=[
                {
                    "hand_name": "left",
                    "handle_pose_topic": "/pika_pose_l",
                    "feedback_tcp_pose_topic": "/left_arm/feedback/tcp_pose",
                    "delta_pose_topic": delta_pose,
                    "control_joint_topic": candidate_gripper,
                    "teleop_trigger_service": (
                        "/nero_input_only/shadow/trigger_l"
                    ),
                    "gripper_joint_state_topic": (
                        "/sensor/gripper_l/joint_state"
                    ),
                    "gripper_max_range": 0.07,
                    "handle_pose_roll": -1.57,
                    "handle_pose_pitch": 0.0,
                    "handle_pose_yaw": 0.0,
                }
            ],
        ),
        Node(
            package="pika_remote_agx_arm",
            executable="arm_ik_pose_node.py",
            name="nero_input_only_ik_left",
            output="screen",
            emulate_tty=True,
            condition=enabled,
            additional_env=shadow_env,
            parameters=[
                ik_config,
                {
                    # The upstream NERO config omits the active prismatic
                    # joint named "gripper". Shadow gripper data comes from
                    # the Pika serial input, so IK must solve arm joints only.
                    "locked_joints": [
                        "gripper",
                        "gripper_base_joint",
                        "gripper_joint1",
                        "gripper_joint2",
                    ],
                    "pose_stamped_topic": delta_pose,
                    "feedback_joint_topic": (
                        "/left_arm/feedback/joint_states"
                    ),
                    "pin_joint_status_topic": candidate_arm,
                    "enable_collision_check": True,
                },
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    # Keep the environment names used by the upstream Pika launch files.
    left_code = os.environ.get("pika_L_code", "")  # noqa: SIM112
    right_code = os.environ.get("pika_R_code", "")  # noqa: SIM112
    arguments = [
        DeclareLaunchArgument("left_hand_code", default_value=left_code),
        DeclareLaunchArgument("right_hand_code", default_value=right_code),
        DeclareLaunchArgument("publish_rate", default_value="100.0"),
        DeclareLaunchArgument("dist_limit", default_value="1.2"),
        DeclareLaunchArgument("angle_limit", default_value="2.2"),
        DeclareLaunchArgument("linear_limit", default_value="5.0"),
        DeclareLaunchArgument("angular_limit", default_value="20.0"),
        DeclareLaunchArgument(
            "enable_left_serial",
            default_value="false",
            choices=["true", "false"],
            description="Opt in to the isolated left Pika serial input.",
        ),
        DeclareLaunchArgument(
            "enable_right_serial",
            default_value="false",
            choices=["true", "false"],
            description="Opt in to the isolated right Pika serial input.",
        ),
        DeclareLaunchArgument(
            "enable_left_shadow",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Opt in to left shadow TCP/IK/gripper candidates. Outputs "
                "remain under /nero_input_only/candidate."
            ),
        ),
        DeclareLaunchArgument(
            "left_serial_port",
            default_value="/dev/serial/by-id/SET_LEFT_PIKA_SERIAL",
        ),
        DeclareLaunchArgument(
            "right_serial_port",
            default_value="/dev/serial/by-id/SET_RIGHT_PIKA_SERIAL",
        ),
        DeclareLaunchArgument(
            "left_joint_name",
            default_value="gripper_l_center_joint",
        ),
        DeclareLaunchArgument(
            "right_joint_name",
            default_value="gripper_r_center_joint",
        ),
    ]
    locator = Node(
        package="pika_locator",
        executable="pika_double_locator_node",
        name="nero_input_only_pika_locator",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "left_hand_code": LaunchConfiguration("left_hand_code"),
                "right_hand_code": LaunchConfiguration("right_hand_code"),
                "publish_rate": ParameterValue(
                    LaunchConfiguration("publish_rate"),
                    value_type=float,
                ),
                "dist_limit": ParameterValue(
                    LaunchConfiguration("dist_limit"),
                    value_type=float,
                ),
                "angle_limit": ParameterValue(
                    LaunchConfiguration("angle_limit"),
                    value_type=float,
                ),
                "linear_limit": ParameterValue(
                    LaunchConfiguration("linear_limit"),
                    value_type=float,
                ),
                "angular_limit": ParameterValue(
                    LaunchConfiguration("angular_limit"),
                    value_type=float,
                ),
            }
        ],
    )
    return LaunchDescription(
        [
            *arguments,
            locator,
            _serial_input_node("left"),
            _serial_input_node("right"),
            *_left_shadow_nodes(),
        ]
    )
