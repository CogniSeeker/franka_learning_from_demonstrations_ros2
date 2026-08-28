"""Pure helpers for timestamped workspace captures."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation


TRANSFORM_CONVENTION = (
    "T_A_B maps homogeneous coordinates from frame B into frame A: "
    "p_A = T_A_B @ p_B"
)
T = TypeVar("T")


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def message_stamp_ns(message: Any) -> int:
    return stamp_to_ns(message.header.stamp)


def select_first_pair_after(
    colors: Iterable[T],
    depths: Iterable[T],
    after_ns: int,
    slop_ns: int,
    stamp: Callable[[T], int] = message_stamp_ns,
) -> tuple[T, T] | None:
    """Return the earliest color and nearest depth, both after a trigger."""
    eligible_depths = [item for item in depths if stamp(item) >= after_ns]
    for color in sorted(
        (item for item in colors if stamp(item) >= after_ns), key=stamp
    ):
        if not eligible_depths:
            return None
        depth = min(eligible_depths, key=lambda item: abs(stamp(item) - stamp(color)))
        if abs(stamp(depth) - stamp(color)) <= slop_ns:
            return color, depth
    return None


def bracket_messages(
    messages: Sequence[T],
    at_ns: int,
    tolerance_ns: int,
    stamp: Callable[[T], int] = message_stamp_ns,
) -> tuple[T, T] | None:
    """Return the nearest messages on both sides of a timestamp."""
    before = [item for item in messages if stamp(item) <= at_ns]
    after = [item for item in messages if stamp(item) >= at_ns]
    if not before or not after:
        return None
    left = max(before, key=stamp)
    right = min(after, key=stamp)
    if at_ns - stamp(left) > tolerance_ns or stamp(right) - at_ns > tolerance_ns:
        return None
    return left, right


def completion_reason_after_save(
    completed_steps: int, requested_steps: int, stop_requested: bool
) -> str | None:
    if completed_steps >= requested_steps:
        return "step_limit"
    if stop_requested:
        return "circle_early_stop"
    return None


def transforms_agree(left: Any, right: Any, atol: float = 1e-6) -> bool:
    return bool(
        np.allclose(
            np.asarray(left, dtype=np.float64).reshape(4, 4),
            np.asarray(right, dtype=np.float64).reshape(4, 4),
            rtol=0.0,
            atol=atol,
        )
    )


def matrix_from_translation_quaternion(
    translation_xyz: Iterable[float], quaternion_xyzw: Iterable[float]
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        np.asarray(tuple(quaternion_xyzw), dtype=np.float64)
    ).as_matrix()
    matrix[:3, 3] = np.asarray(tuple(translation_xyz), dtype=np.float64)
    return matrix


def matrix_from_transform(transform: Any) -> np.ndarray:
    translation = transform.translation
    rotation = transform.rotation
    return matrix_from_translation_quaternion(
        (translation.x, translation.y, translation.z),
        (rotation.x, rotation.y, rotation.z, rotation.w),
    )


def matrix_from_pose(pose: Any) -> np.ndarray:
    return matrix_from_translation_quaternion(
        (pose.position.x, pose.position.y, pose.position.z),
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
    )


def transform_record(
    parent_frame: str,
    child_frame: str,
    matrix: Any,
    *,
    source: str,
    stamp_ns: int | None,
    static: bool,
) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    quaternion = Rotation.from_matrix(value[:3, :3]).as_quat()
    return {
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "source": source,
        "static": static,
        "stamp_ns": stamp_ns,
        "translation_xyz_m": value[:3, 3].tolist(),
        "quaternion_xyzw": quaternion.tolist(),
        "matrix": value.tolist(),
    }


def camera_info_record(message: Any, topic: str) -> dict[str, Any]:
    roi = message.roi
    return {
        "topic": topic,
        "stamp_ns": message_stamp_ns(message),
        "frame_id": message.header.frame_id,
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": message.distortion_model,
        "D": list(message.d),
        "K": list(message.k),
        "R": list(message.r),
        "P": list(message.p),
        "binning_x": int(message.binning_x),
        "binning_y": int(message.binning_y),
        "roi": {
            "x_offset": int(roi.x_offset),
            "y_offset": int(roi.y_offset),
            "height": int(roi.height),
            "width": int(roi.width),
            "do_rectify": bool(roi.do_rectify),
        },
    }


def joint_state_record(message: Any, at_ns: int) -> dict[str, Any]:
    stamp_ns = message_stamp_ns(message)
    return {
        "stamp_ns": stamp_ns,
        "offset_from_image_ns": stamp_ns - at_ns,
        "frame_id": message.header.frame_id,
        "name": list(message.name),
        "position": list(message.position),
        "velocity": list(message.velocity),
        "effort": list(message.effort),
    }


def pose_stamped_record(message: Any, at_ns: int) -> dict[str, Any]:
    stamp_ns = message_stamp_ns(message)
    pose = message.pose
    return {
        "stamp_ns": stamp_ns,
        "offset_from_image_ns": stamp_ns - at_ns,
        "frame_id": message.header.frame_id,
        "position_xyz_m": [pose.position.x, pose.position.y, pose.position.z],
        "quaternion_xyzw": [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        "matrix": matrix_from_pose(pose).tolist(),
    }


def transforms_text(capture: dict[str, Any]) -> str:
    """Create the human-readable companion to capture.json."""
    lines = [TRANSFORM_CONVENTION, ""]
    lines.append(f"step: {capture['step']}")
    lines.append(f"button_stamp_ns: {capture['button_stamp_ns']}")
    for camera_name, camera in capture["cameras"].items():
        lines.extend(
            [
                "",
                f"camera: {camera_name}",
                f"color_stamp_ns: {camera['color']['stamp_ns']}",
                f"color_frame_id: {camera['color']['frame_id']}",
                f"color_encoding: {camera['color']['encoding']}",
                f"depth_stamp_ns: {camera['depth']['stamp_ns']}",
                f"depth_frame_id: {camera['depth']['frame_id']}",
                f"depth_encoding: {camera['depth']['encoding']}",
                f"rgb_depth_delta_ns: {camera['sync_delta_ns']}",
                f"color_after_button_ns: {camera['color_after_button_ns']}",
                f"depth_after_button_ns: {camera['depth_after_button_ns']}",
            ]
        )
        for name, transform in camera["transforms"].items():
            lines.extend(
                [
                    "",
                    name,
                    f"  {transform['parent_frame']} <- {transform['child_frame']}",
                    f"  source: {transform['source']}",
                    f"  stamp_ns: {transform['stamp_ns']}",
                    f"  translation_xyz_m: {transform['translation_xyz_m']}",
                    f"  quaternion_xyzw: {transform['quaternion_xyzw']}",
                    np.array2string(
                        np.asarray(transform["matrix"]), precision=8, suppress_small=True
                    ),
                ]
            )
        for topic_name, samples in camera["robot_messages"].items():
            lines.append("")
            lines.append(topic_name)
            for side in ("before", "after"):
                sample = samples[side]
                lines.append(
                    f"  {side}: stamp={sample['stamp_ns']} "
                    f"offset={sample['offset_from_image_ns']} ns"
                )
                lines.append(f"    frame_id={sample['frame_id']}")
                if "position" in sample:
                    lines.append(f"    name={sample['name']}")
                    lines.append(f"    position={sample['position']}")
                    lines.append(f"    velocity={sample['velocity']}")
                    lines.append(f"    effort={sample['effort']}")
                else:
                    lines.append(f"    position_xyz_m={sample['position_xyz_m']}")
                    lines.append(f"    quaternion_xyzw={sample['quaternion_xyzw']}")
    return "\n".join(lines) + "\n"
