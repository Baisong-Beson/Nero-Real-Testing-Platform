"""Start the complete NERO dual-arm Gate 3 hardware stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _include(package: str, launch_file: str, arguments: dict):
    return GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare(package), "launch", launch_file]
                    )
                ),
                launch_arguments=arguments.items(),
            )
        ]
    )


def generate_launch_description() -> LaunchDescription:
    speed = LaunchConfiguration("speed_percent")
    common_arm = {
        "arm_type": "nero",
        "effector_type": "agx_gripper",
        "auto_enable": "false",
        "control_enabled": "false",
        "fast_mode": "false",
        "speed_percent": speed,
        "gripper_default_effort": "0.5",
    }
    common_camera = {
        "depth_module.color_profile": "640x480x15",
        "enable_depth": "false",
        "enable_infra1": "false",
        "enable_infra2": "false",
    }
    return LaunchDescription(
        [
            DeclareLaunchArgument("right_can", default_value="can0"),
            DeclareLaunchArgument("left_can", default_value="can1"),
            DeclareLaunchArgument("right_camera_serial", default_value="_262622274737"),
            DeclareLaunchArgument("left_camera_serial", default_value="_260322274875"),
            DeclareLaunchArgument("speed_percent", default_value="35"),
            _include(
                "agx_arm_ctrl",
                "start_single_agx_arm.launch.py",
                {**common_arm, "can_port": LaunchConfiguration("right_can"), "namespace": "right_arm"},
            ),
            _include(
                "agx_arm_ctrl",
                "start_single_agx_arm.launch.py",
                {**common_arm, "can_port": LaunchConfiguration("left_can"), "namespace": "left_arm"},
            ),
            _include(
                "realsense2_camera",
                "rs_launch.py",
                {
                    **common_camera,
                    "serial_no": LaunchConfiguration("right_camera_serial"),
                    "camera_name": "right_wrist",
                },
            ),
            _include(
                "realsense2_camera",
                "rs_launch.py",
                {
                    **common_camera,
                    "serial_no": LaunchConfiguration("left_camera_serial"),
                    "camera_name": "left_wrist",
                },
            ),
            _include(
                "zed_wrapper",
                "zed_camera.launch.py",
                {
                    "camera_model": "zedm",
                    "camera_name": "zed_m",
                    "node_name": "zed_node",
                    "publish_tf": "false",
                    "publish_map_tf": "false",
                    "publish_urdf": "false",
                    "ros_params_override_path": PathJoinSubstitution(
                        [
                            FindPackageShare("nero_pi05_bridge"),
                            "config",
                            "zedm_gate2_rgb_only.yaml",
                        ]
                    ),
                },
            ),
        ]
    )
