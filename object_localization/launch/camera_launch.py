#!/usr/bin/env python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

# CAMERA_SERIAL_NO = '"105322250337"' # far-away table corner camera
# CAMERA_SERIAL_NO = '"105322250885"' # table camera near PC # camera2_link from panda_a404_calib
CAMERA_SERIAL_NO = '"318122300789"' # hand camera
# CAMERA_SERIAL_NO = '"920312072256"' #?
# CAMERA_SERIAL_NO = '"309122301116"' # table camera far PC

def generate_launch_description():
    log_level = DeclareLaunchArgument("log_level", default_value="warn")
    serial_no = DeclareLaunchArgument("serial_no", default_value=CAMERA_SERIAL_NO)
    camera_name = DeclareLaunchArgument("camera_name", default_value="camera3")

    # Include the Realsense camera launch file with resolution parameters
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py"
            )
        ),
        launch_arguments={
            "camera_name": LaunchConfiguration("camera_name"),
            "rgb_camera.profile": "1280,720,30",
            "depth_module.profile": "848,480,30",
            "log_level": "info",
            "initial_reset": "true",
            "serial_no": LaunchConfiguration("serial_no"),
        }.items()
    )

    # # Static transform publisher node
    # camera_tf_publisher_node = Node(
    #     package="object_localization",
    #     executable="static_transform_camera",
    #     name="camera_tf_publisher",
    #     output="screen",
    # )

    return LaunchDescription([
        log_level,
        serial_no,
        camera_name,
        realsense_launch,
        # camera_tf_publisher_node
    ])
