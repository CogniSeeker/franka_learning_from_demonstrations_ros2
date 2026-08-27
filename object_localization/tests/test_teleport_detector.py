"""Timing checks for the teleport detector camera warm-up."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from launch.actions import DeclareLaunchArgument

from object_localization import teleport_detector


def test_camera_launch_default_matches_detector_topic_namespace():
    launch_path = Path(__file__).parents[1] / "launch" / "camera_launch.py"
    spec = importlib.util.spec_from_file_location("camera_launch", launch_path)
    camera_launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera_launch)

    camera_name_argument = next(
        entity
        for entity in camera_launch.generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument) and entity.name == "camera_name"
    )
    camera_name = "".join(part.text for part in camera_name_argument.default_value)

    assert f"{camera_name}_link" == teleport_detector.ROBOT_CAMERA_TF_FRAME
    assert teleport_detector.CAMERA_COLOR_TOPIC.startswith(f"/{camera_name}/")


def test_warmup_starts_when_first_image_arrives(monkeypatch):
    node = object.__new__(teleport_detector.TeleportDetectionService)
    node._camera_warmup_until = None
    node._camera_info = None
    logger = type("Logger", (), {"info": lambda self, message: None})()
    monkeypatch.setattr(node, "get_logger", lambda: logger)

    now = 10.0
    monkeypatch.setattr(teleport_detector.time, "monotonic", lambda: now)

    node._camera_info_callback(object())
    assert node._camera_info is None

    node._image_callback(object())
    assert node._camera_warmup_until == 13.0

    now = 12.0
    node._camera_info_callback(object())
    node._image_callback(object())
    assert node._camera_info is None
    assert node._camera_warmup_until == 13.0


def test_current_pose_is_cached_as_base_referenced_end_effector():
    node = object.__new__(teleport_detector.TeleportDetectionService)
    node._current_end_effector_pose = None
    message = SimpleNamespace(
        header=SimpleNamespace(frame_id=""),
        pose=SimpleNamespace(position=object(), orientation=object()),
    )

    node._current_pose_callback(message)

    assert node._current_end_effector_pose is message.pose


def test_perception_wrapper_forwards_current_pose_to_supporting_model():
    received = []
    wrapper = object.__new__(teleport_detector.PerceptionModuleWrapper)
    wrapper._model = SimpleNamespace(
        set_visualization_end_effector_pose=received.append
    )
    pose = SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
    )

    wrapper.set_visualization_end_effector_pose(pose)

    assert received == [
        {
            "translation_xyz_m": [1.0, 2.0, 3.0],
            "quaternion_xyzw": [0.1, 0.2, 0.3, 0.9],
        }
    ]


def test_perception_wrapper_forwards_link_to_optical_tf_to_supporting_model():
    received = []
    wrapper = object.__new__(teleport_detector.PerceptionModuleWrapper)
    wrapper._model = SimpleNamespace(set_visualization_camera_transform=received.append)
    transform = SimpleNamespace(
        translation=SimpleNamespace(x=0.01, y=0.02, z=0.03),
        rotation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
    )

    wrapper.set_visualization_camera_transform(transform)

    assert received == [
        {
            "translation_xyz_m": [0.01, 0.02, 0.03],
            "quaternion_xyzw": [0.1, 0.2, 0.3, 0.9],
        }
    ]


def test_ros_message_accepts_adapted_detection_contract():
    node = object.__new__(teleport_detector.TeleportDetectionService)
    detection = SimpleNamespace(
        object_id="r1c1_green",
        class_name="green",
        state="off",
        confidence=0.8,
        pose_valid=True,
        pose=SimpleNamespace(
            position=(0.7, -0.2, -0.1),
            orientation=(0.0, 0.70710678, 0.0, 0.70710678),
        ),
    )
    transform = teleport_detector.TransformStamped()
    transform.transform.rotation.w = 1.0

    message = node._to_ros_message(
        detection,
        transform,
        transform.header.stamp,
        "camera2_color_optical_frame",
    )

    assert message.object_id == "r1c1_green"
    assert message.class_name == "green"
    assert message.state == "off"
    assert message.pose.position.x == 0.7
