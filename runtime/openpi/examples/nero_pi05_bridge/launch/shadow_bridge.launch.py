from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arm = LaunchConfiguration("arm")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "arm",
                default_value="left",
                choices=["left", "right", "both"],
                description="NERO arm whose observation stream is used. both = 16D dual.",
            ),
            DeclareLaunchArgument("policy_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("policy_port", default_value="8000"),
            DeclareLaunchArgument("prompt", default_value="pick up the object"),
            DeclareLaunchArgument("inference_rate_hz", default_value="5.0"),
            DeclareLaunchArgument("log_dir", default_value="/tmp/nero_pi05_bridge"),
            DeclareLaunchArgument(
                "left_wrist_image_topic",
                default_value="/left_wrist/color/image_raw/compressed",
            ),
            DeclareLaunchArgument(
                "left_joint_state_topic",
                default_value="/left_arm/feedback/joint_states",
            ),
            Node(
                package="nero_pi05_bridge",
                executable="shadow_bridge",
                name="nero_pi05_bridge",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "arm": arm,
                        "shadow": True,
                        "policy_host": LaunchConfiguration("policy_host"),
                        "policy_port": ParameterValue(
                            LaunchConfiguration("policy_port"),
                            value_type=int,
                        ),
                        "prompt": LaunchConfiguration("prompt"),
                        "inference_rate_hz": ParameterValue(
                            LaunchConfiguration("inference_rate_hz"),
                            value_type=float,
                        ),
                        "log_dir": LaunchConfiguration("log_dir"),
                        "left_wrist_image_topic": LaunchConfiguration("left_wrist_image_topic"),
                        "left_joint_state_topic": LaunchConfiguration("left_joint_state_topic"),
                    }
                ],
            ),
        ]
    )
