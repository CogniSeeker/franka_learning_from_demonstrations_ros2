#!/usr/bin/env python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # Detection node
    teleport_detector_node = Node(
        package="object_localization",
        executable="teleport_detector",
        name="teleport_detector",
        output="screen",
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
        teleport_detector_node,
        camera_launch,
    ])
