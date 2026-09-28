from glob import glob
import os

from setuptools import find_packages
from setuptools import setup

package_name = "nero_pi05_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml")
            + glob("config/*.json")
            + glob("config/*.rviz")
            + glob("config/*.txt"),
        ),
        (
            os.path.join("share", package_name),
            [
                "EPISODE_PROTOCOL.md",
                "COLLECTION_AND_TRAINING.md",
                "VR_COLLECTION_GUIDE.md",
                "GATE2_ARM_EXECUTOR.md",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Platform Maintainers",
    maintainer_email="maintainer@example.com",
    description="NERO shadow bridge, Gate 1/2 safety tools, and Gate 3 staged task executor.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "offline_replay = nero_pi05_bridge.offline_replay:main",
            "native_replay = nero_pi05_bridge.native_replay:main",
            "gripper_calibration = nero_pi05_bridge.gripper_calibration_cli:main",
            "ghost_playback = nero_pi05_bridge.ghost_playback:main",
            "shadow_bridge = nero_pi05_bridge.bridge_node:main",
            "gate2_arm_executor = nero_pi05_bridge.arm_executor_ros:main",
            "gate2_deadman = nero_pi05_bridge.gate2_deadman:main",
            "gate3_task_executor = nero_pi05_bridge.task_executor_ros:main",
            "gate3_dual_task_executor = nero_pi05_bridge.dual_task_executor_ros:main",
            "nero_home = nero_pi05_bridge.nero_home_cli:main",
            "nero_park = nero_pi05_bridge.nero_park_cli:main",
            "nero_experiment = nero_pi05_bridge.experiment_cli:main",
        ],
    },
)
