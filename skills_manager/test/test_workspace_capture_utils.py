from types import SimpleNamespace

import numpy as np

from skills_manager.workspace_capture_utils import (
    bracket_messages,
    camera_info_record,
    completion_reason_after_save,
    matrix_from_translation_quaternion,
    select_first_pair_after,
    transform_record,
    transforms_agree,
    transforms_text,
)


def _message(stamp_ns: int):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            )
        )
    )


def test_selects_first_synchronized_pair_after_button():
    colors = [_message(90), _message(110), _message(140)]
    depths = [_message(92), _message(118), _message(141)]

    color, depth = select_first_pair_after(colors, depths, 100, 10)

    assert color.header.stamp.nanosec == 110
    assert depth.header.stamp.nanosec == 118


def test_pair_requires_both_images_after_button_and_within_slop():
    colors = [_message(110)]
    depths = [_message(99), _message(125)]

    assert select_first_pair_after(colors, depths, 100, 10) is None


def test_robot_messages_bracket_image_time():
    messages = [_message(90), _message(105), _message(120)]

    before, after = bracket_messages(messages, 110, 25)

    assert before.header.stamp.nanosec == 105
    assert after.header.stamp.nanosec == 120
    assert bracket_messages(messages, 200, 25) is None


def test_completion_reason_prioritizes_limit_and_honors_early_stop():
    assert completion_reason_after_save(299, 300, False) is None
    assert completion_reason_after_save(299, 300, True) == "circle_early_stop"
    assert completion_reason_after_save(300, 300, False) == "step_limit"
    assert completion_reason_after_save(300, 300, True) == "step_limit"


def test_transform_chain_and_serialization_use_parent_child_convention():
    t_base_ee = matrix_from_translation_quaternion(
        [0.4, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0]
    )
    t_ee_camera = matrix_from_translation_quaternion(
        [0.1, 0.02, 0.0], [0.0, 0.0, 0.0, 1.0]
    )
    t_base_camera = t_base_ee @ t_ee_camera

    record = transform_record(
        "panda_link0",
        "camera3_link",
        t_base_camera,
        source="composed",
        stamp_ns=123,
        static=False,
    )

    np.testing.assert_allclose(record["translation_xyz_m"], [0.5, 0.02, 0.3])
    np.testing.assert_allclose(record["matrix"], t_base_camera)
    assert record["parent_frame"] == "panda_link0"
    assert record["child_frame"] == "camera3_link"
    assert transforms_agree(t_base_camera, t_base_camera.copy())
    assert not transforms_agree(t_base_camera, np.linalg.inv(t_base_camera))


def test_camera_info_serialization_keeps_every_calibration_field():
    message = _message(123)
    message.header.frame_id = "camera3_color_optical_frame"
    message.width = 1280
    message.height = 720
    message.distortion_model = "plumb_bob"
    message.d = [0.1, 0.2, 0.3, 0.4, 0.5]
    message.k = list(range(9))
    message.r = list(range(9, 18))
    message.p = list(range(18, 30))
    message.binning_x = 2
    message.binning_y = 3
    message.roi = SimpleNamespace(
        x_offset=4,
        y_offset=5,
        height=600,
        width=800,
        do_rectify=True,
    )

    record = camera_info_record(message, "/camera3/color/camera_info")

    assert record["stamp_ns"] == 123
    assert record["frame_id"] == "camera3_color_optical_frame"
    assert record["width"] == 1280
    assert record["height"] == 720
    assert record["D"] == message.d
    assert record["K"] == message.k
    assert record["R"] == message.r
    assert record["P"] == message.p
    assert record["roi"]["do_rectify"] is True


def test_human_readable_capture_contains_timestamps_frames_joints_and_matrix():
    transform = transform_record(
        "panda_link0",
        "camera3_color_optical_frame",
        np.eye(4),
        source="tf2 lookup",
        stamp_ns=110,
        static=False,
    )
    joint = {
        "stamp_ns": 105,
        "offset_from_image_ns": -5,
        "frame_id": "panda_link0",
        "name": ["panda_joint1"],
        "position": [0.1],
        "velocity": [0.2],
        "effort": [0.3],
    }
    capture = {
        "step": 1,
        "button_stamp_ns": 100,
        "cameras": {
            "camera3": {
                "color": {
                    "stamp_ns": 110,
                    "frame_id": "camera3_color_optical_frame",
                    "encoding": "rgb8",
                },
                "depth": {
                    "stamp_ns": 112,
                    "frame_id": "camera3_depth_optical_frame",
                    "encoding": "16UC1",
                },
                "sync_delta_ns": 2,
                "color_after_button_ns": 10,
                "depth_after_button_ns": 12,
                "transforms": {"T_base_color_optical": transform},
                "robot_messages": {
                    "/joint_states": {"before": joint, "after": joint}
                },
            }
        },
    }

    text = transforms_text(capture)

    assert "camera3_color_optical_frame" in text
    assert "panda_joint1" in text
    assert "quaternion_xyzw" in text
    assert "[[1." in text
