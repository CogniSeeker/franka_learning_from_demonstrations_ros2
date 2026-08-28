#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("object_localization"),
                "launch",
                "camera_launch.py",
            )
        ),
        launch_arguments={
            "camera_name": "camera3",
            "serial_no": LaunchConfiguration("serial_no"),
            "enable_sync": "true",
        }.items(),
    )
    calibration_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("panda_a404_calib"),
                "launch",
                "panda_a404_calib.launch.py",
            )
        )
    )
    capture_node = Node(
        package="skills_manager",
        executable="workspace_capture",
        name="workspace_capture",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "steps": ParameterValue(LaunchConfiguration("steps"), value_type=int),
                "output_dir": LaunchConfiguration("output_dir"),
                "camera_names": LaunchConfiguration("camera_names"),
                "camera_serial": ParameterValue(
                    LaunchConfiguration("serial_no"), value_type=str
                ),
                "capture_button": LaunchConfiguration("capture_button"),
                "sync_slop_ms": ParameterValue(
                    LaunchConfiguration("sync_slop_ms"), value_type=float
                ),
                "capture_timeout_s": ParameterValue(
                    LaunchConfiguration("capture_timeout_s"), value_type=float
                ),
            }
        ],
    )
    shutdown_when_done = RegisterEventHandler(
        OnProcessExit(
            target_action=capture_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="workspace capture process finished")
                )
            ],
        )
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("steps", default_value="300"),
            DeclareLaunchArgument("output_dir", default_value=""),
            DeclareLaunchArgument("camera_names", default_value="camera3"),
            DeclareLaunchArgument("capture_button", default_value="check"),
            DeclareLaunchArgument("sync_slop_ms", default_value="30"),
            DeclareLaunchArgument("capture_timeout_s", default_value="2"),
            DeclareLaunchArgument("serial_no", default_value='"318122300789"'),
            camera_launch,
            calibration_launch,
            capture_node,
            shutdown_when_done,
        ]
    )
