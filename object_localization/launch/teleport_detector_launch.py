#!/usr/bin/env python
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # Detection node
    teleport_detector_node = Node(
        package="object_localization",
        executable="teleport_detector",
        name="teleport_detector",
        output="screen",
        parameters=[
            {
                "model_module": LaunchConfiguration("model_module"),
                "model_loader": LaunchConfiguration("model_loader"),
                "model_inference": LaunchConfiguration("model_inference"),
                "model_weights": LaunchConfiguration("model_weights"),
            }
        ],
    )

    # Include the camera launch file
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("object_localization"),
                "launch",
                "camera_launch.py"
            )
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_module",
            default_value="object_localization.mock_perception_module",
        ),
        DeclareLaunchArgument("model_loader", default_value="load_model"),
        DeclareLaunchArgument("model_inference", default_value="inference"),
        DeclareLaunchArgument("model_weights", default_value="unused"),
        teleport_detector_node,
        camera_launch,
    ])
