#!/usr/bin/env python3
"""Capture timestamped RGB-D workspace observations during kinesthetic teaching."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener

from sam3.deck_scene.frame_visualization import save_frame_visualizations
from skills_manager.lfd import LfD
from skills_manager.signalizer import Signalizator
from skills_manager.workspace_capture_utils import (
    TRANSFORM_CONVENTION,
    bracket_messages,
    camera_info_record,
    completion_reason_after_save,
    joint_state_record,
    matrix_from_transform,
    message_stamp_ns,
    pose_stamped_record,
    select_first_pair_after,
    transform_record,
    transforms_agree,
    transforms_text,
)


BASE_FRAME = "panda_link0"
CALIBRATED_EE_FRAME = "panda_end_effector_frame_calibrated"
ROBOT_BRACKET_TOLERANCE_NS = 250_000_000
REQUIRED_ARM_JOINTS = {f"panda_joint{index}" for index in range(1, 8)}


@dataclass
class CameraRuntime:
    name: str
    link_frame: str
    color_topic: str
    depth_topic: str
    color_info_topic: str
    depth_info_topic: str
    colors: deque[Image] = field(default_factory=lambda: deque(maxlen=60))
    depths: deque[Image] = field(default_factory=lambda: deque(maxlen=60))
    color_info: CameraInfo | None = None
    depth_info: CameraInfo | None = None


@dataclass
class CaptureRequest:
    button_stamp_ns: int
    deadline_monotonic: float
    images: dict[str, tuple[Image, Image]] = field(default_factory=dict)


def _clean_frame(frame: str) -> str:
    return str(frame).strip().lstrip("/")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class WorkspaceCapture(LfD):
    """LfD recorder with a Check-button RGB-D snapshot hook."""

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("steps", 300)
        self.declare_parameter("output_dir", "")
        self.declare_parameter("camera_names", "camera3")
        self.declare_parameter("camera_serial", "318122300789")
        self.declare_parameter("capture_button", "check")
        self.declare_parameter("sync_slop_ms", 30.0)
        self.declare_parameter("capture_timeout_s", 2.0)
        self.declare_parameter("startup_timeout_s", 30.0)

        self.requested_steps = int(self.get_parameter("steps").value)
        self.camera_serial = str(
            self.get_parameter("camera_serial").value
        ).strip().strip('"')
        self.capture_button = str(self.get_parameter("capture_button").value).strip()
        self.sync_slop_ns = int(
            float(self.get_parameter("sync_slop_ms").value) * 1_000_000
        )
        self.capture_timeout_s = float(
            self.get_parameter("capture_timeout_s").value
        )
        self.startup_timeout_s = float(
            self.get_parameter("startup_timeout_s").value
        )
        camera_names = [
            item.strip()
            for item in str(self.get_parameter("camera_names").value).split(",")
            if item.strip()
        ]
        if self.requested_steps <= 0:
            raise ValueError("steps must be positive")
        if not camera_names:
            raise ValueError("camera_names must contain at least one name")
        if self.capture_button == "circle":
            raise ValueError("circle is reserved for early stop")
        if self.sync_slop_ns < 0 or self.capture_timeout_s <= 0:
            raise ValueError("sync_slop_ms must be non-negative and timeout positive")

        started = datetime.now(timezone.utc)
        configured_output = str(self.get_parameter("output_dir").value).strip()
        if configured_output:
            self.session_dir = Path(configured_output).expanduser().resolve()
        else:
            ros_home = Path(
                os.environ.get("ROS_HOME", str(Path.home() / ".ros"))
            ).expanduser()
            self.session_dir = (
                ros_home
                / "workspace_captures"
                / started.strftime("%Y%m%dT%H%M%S_%fZ")
            ).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=False)

        self._started_at = started.isoformat()
        self.completed_steps = 0
        self.completion_reason = "running"
        self._capture_enabled = False
        self._busy = False
        self._stop_requested = False
        self._request: CaptureRequest | None = None
        self._writer: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status = "STARTING"
        self._status_color = Signalizator.COLOR_IDLE
        self._status_step = 0
        self._status_revision = 0
        self._shown_status_revision = -1
        self._data_saved_at: float | None = None

        self._camera: dict[str, CameraRuntime] = {}
        self._subscriptions = []
        for name in camera_names:
            prefix = f"/{name}"
            runtime = CameraRuntime(
                name=name,
                link_frame=f"{name}_link",
                color_topic=f"{prefix}/color/image_raw",
                depth_topic=f"{prefix}/depth/image_rect_raw",
                color_info_topic=f"{prefix}/color/camera_info",
                depth_info_topic=f"{prefix}/depth/camera_info",
            )
            self._camera[name] = runtime
            self._subscriptions.extend(
                [
                    self.create_subscription(
                        Image,
                        runtime.color_topic,
                        lambda message, camera=name: self._image_callback(
                            camera, "color", message
                        ),
                        qos_profile_sensor_data,
                    ),
                    self.create_subscription(
                        Image,
                        runtime.depth_topic,
                        lambda message, camera=name: self._image_callback(
                            camera, "depth", message
                        ),
                        qos_profile_sensor_data,
                    ),
                    self.create_subscription(
                        CameraInfo,
                        runtime.color_info_topic,
                        lambda message, camera=name: self._camera_info_callback(
                            camera, "color", message
                        ),
                        qos_profile_sensor_data,
                    ),
                    self.create_subscription(
                        CameraInfo,
                        runtime.depth_info_topic,
                        lambda message, camera=name: self._camera_info_callback(
                            camera, "depth", message
                        ),
                        qos_profile_sensor_data,
                    ),
                ]
            )

        self._poses: deque[PoseStamped] = deque(maxlen=200)
        self._joint_states: deque[JointState] = deque(maxlen=200)
        self._joint_states_calibrated: deque[JointState] = deque(maxlen=200)
        self._subscriptions.extend(
            [
                self.create_subscription(
                    PoseStamped, "/panda/curr_pose", self._pose_callback, 20
                ),
                self.create_subscription(
                    JointState, "/joint_states", self._joint_state_callback, 20
                ),
                self.create_subscription(
                    JointState,
                    "/joint_states_calibrated",
                    self._joint_state_calibrated_callback,
                    20,
                ),
            ]
        )

        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._static_transforms: dict[tuple[str, str], Any] = {}
        self._subscriptions.append(
            self.create_subscription(
                TFMessage, "/tf_static", self._tf_static_callback, static_qos
            )
        )
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._capture_timer = self.create_timer(0.02, self._capture_tick)

        self._session = {
            "schema_version": 1,
            "started_at": self._started_at,
            "ended_at": None,
            "completion_reason": self.completion_reason,
            "transform_convention": TRANSFORM_CONVENTION,
            "requested_steps": self.requested_steps,
            "completed_steps": 0,
            "camera_names": camera_names,
            "parameters": {
                "output_dir": str(self.session_dir),
                "camera_serial": self.camera_serial,
                "capture_button": self.capture_button,
                "sync_slop_ms": self.sync_slop_ns / 1_000_000,
                "capture_timeout_s": self.capture_timeout_s,
                "startup_timeout_s": self.startup_timeout_s,
                "base_frame": BASE_FRAME,
                "calibrated_ee_frame": CALIBRATED_EE_FRAME,
            },
        }
        self._write_session()

    def _set_status(self, state: str, step: int, color: str) -> None:
        with self._lock:
            changed = (state, step, color) != (
                self._status,
                self._status_step,
                self._status_color,
            )
            self._status = state
            self._status_step = step
            self._status_color = color
            if changed:
                self._status_revision += 1
        if changed:
            ansi = {
                Signalizator.COLOR_TELEOPERATING: "\033[96m",
                Signalizator.COLOR_BUTTON_PRESSED: "\033[93m",
                Signalizator.COLOR_DATA_SAVED: "\033[92m",
                Signalizator.COLOR_ERROR: "\033[91m",
            }.get(color, "\033[90m")
            print(
                f"{ansi}{state}: step {step}/{self.requested_steps}\033[0m",
                flush=True,
            )

    def _render_status(self, *, force: bool = False) -> None:
        with self._lock:
            revision = self._status_revision
            state = self._status
            step = self._status_step
            color = self._status_color
        if not force and revision == self._shown_status_revision:
            return
        self.signalizer.signalize_status(state, step, self.requested_steps, color)
        self._shown_status_revision = revision

    def init_additional_flags(self):
        self._render_status(force=True)

    def update_additional_flags(self):
        if (
            self._data_saved_at is not None
            and time.monotonic() - self._data_saved_at >= 0.8
        ):
            with self._lock:
                idle = not self._busy and not self._stop_requested
            if idle:
                self._data_saved_at = None
                self._set_status(
                    "TELEOPERATING",
                    self.completed_steps,
                    Signalizator.COLOR_TELEOPERATING,
                )
        # LfD uses the same Signalizator for its legacy ready/recording states.
        # Reassert the capture state from the main thread on every 10 Hz sample.
        self._render_status(force=True)

    def _camera_info_callback(
        self, camera_name: str, stream: str, message: CameraInfo
    ) -> None:
        runtime = self._camera[camera_name]
        if stream == "color":
            runtime.color_info = message
        else:
            runtime.depth_info = message

    def _image_callback(
        self, camera_name: str, stream: str, message: Image
    ) -> None:
        runtime = self._camera[camera_name]
        with self._lock:
            (runtime.colors if stream == "color" else runtime.depths).append(message)
            self._collect_image_pairs_locked()

    def _pose_callback(self, message: PoseStamped) -> None:
        if message_stamp_ns(message) > 0:
            self._poses.append(message)

    def _joint_state_callback(self, message: JointState) -> None:
        if REQUIRED_ARM_JOINTS.issubset(message.name) and message_stamp_ns(message) > 0:
            self._joint_states.append(message)

    def _joint_state_calibrated_callback(self, message: JointState) -> None:
        if REQUIRED_ARM_JOINTS.issubset(message.name) and message_stamp_ns(message) > 0:
            self._joint_states_calibrated.append(message)

    def _tf_static_callback(self, message: TFMessage) -> None:
        for transform in message.transforms:
            parent = _clean_frame(transform.header.frame_id)
            child = _clean_frame(transform.child_frame_id)
            self._static_transforms[(parent, child)] = transform

    def _direct_static(self, parent: str, child: str) -> Any | None:
        return self._static_transforms.get((_clean_frame(parent), _clean_frame(child)))

    def _static_parent(self, child: str) -> str | None:
        child = _clean_frame(child)
        parents = [
            parent
            for parent, candidate in list(self._static_transforms)
            if candidate == child
        ]
        return parents[0] if len(parents) == 1 else None

    def _ready(self) -> bool:
        for runtime in self._camera.values():
            if runtime.color_info is None or runtime.depth_info is None:
                return False
            optical = _clean_frame(runtime.color_info.header.frame_id)
            color_frame = self._static_parent(optical)
            if color_frame is None:
                return False
            if self._direct_static(runtime.link_frame, color_frame) is None:
                return False
            if self._direct_static(color_frame, optical) is None:
                return False
            if self._direct_static(CALIBRATED_EE_FRAME, runtime.link_frame) is None:
                return False
            if not self.tf_buffer.can_transform(
                BASE_FRAME,
                optical,
                Time(),
                timeout=Duration(seconds=0),
            ):
                return False
        return bool(self._poses and self._joint_states and self._joint_states_calibrated)

    def wait_until_ready(self) -> None:
        self._set_status("WAITING FOR CAMERA", 0, Signalizator.COLOR_BUTTON_PRESSED)
        deadline = time.monotonic() + self.startup_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            self._render_status()
            if self._ready():
                self._save_intrinsics()
                return
            time.sleep(0.1)
        missing = []
        for runtime in self._camera.values():
            if runtime.color_info is None:
                missing.append(runtime.color_info_topic)
            if runtime.depth_info is None:
                missing.append(runtime.depth_info_topic)
        raise RuntimeError(
            "capture prerequisites did not become ready"
            + (": " + ", ".join(missing) if missing else " (TF or robot state missing)")
        )

    def _save_intrinsics(self) -> None:
        data = {
            "schema_version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "cameras": {},
        }
        text_lines = []
        for name, runtime in self._camera.items():
            color = camera_info_record(runtime.color_info, runtime.color_info_topic)
            depth = camera_info_record(runtime.depth_info, runtime.depth_info_topic)
            data["cameras"][name] = {"color": color, "depth": depth}
            for stream_name, info in (("color", color), ("depth", depth)):
                text_lines.extend(
                    [
                        f"[{name} {stream_name}]",
                        f"topic: {info['topic']}",
                        f"frame_id: {info['frame_id']}",
                        f"resolution: {info['width']} x {info['height']}",
                        f"distortion_model: {info['distortion_model']}",
                        f"D: {info['D']}",
                        f"K: {np.asarray(info['K']).reshape(3, 3)}",
                        f"R: {np.asarray(info['R']).reshape(3, 3)}",
                        f"P: {np.asarray(info['P']).reshape(3, 4)}",
                        f"binning: {info['binning_x']} x {info['binning_y']}",
                        f"roi: {info['roi']}",
                        "",
                    ]
                )
        _atomic_write_text(
            self.session_dir / "camera_intrinsics.json",
            json.dumps(data, indent=2) + "\n",
        )
        _atomic_write_text(
            self.session_dir / "camera_intrinsics.txt", "\n".join(text_lines)
        )

    def enable_capture(self) -> None:
        self._capture_enabled = True
        self._set_status(
            "TELEOPERATING", self.completed_steps, Signalizator.COLOR_TELEOPERATING
        )
        self._render_status()

    def franka_on_press(self, key):
        if key == self.capture_button:
            self.request_capture()
        elif key == "circle":
            self.request_early_stop()
        else:
            super().franka_on_press(key)

    def request_capture(self) -> None:
        with self._lock:
            if not self._capture_enabled:
                self.get_logger().warning("capture button ignored: recorder is not ready")
                return
            if self._stop_requested:
                self.get_logger().warning("capture button ignored: shutdown requested")
                return
            if self._busy:
                self.get_logger().warning("capture button ignored: a capture is in progress")
                return
            if self.completed_steps >= self.requested_steps:
                return
            button_stamp_ns = self.get_clock().now().nanoseconds
            self._request = CaptureRequest(
                button_stamp_ns=button_stamp_ns,
                deadline_monotonic=time.monotonic() + self.capture_timeout_s,
            )
            self._busy = True
            self._collect_image_pairs_locked()
            step = self.completed_steps + 1
        self._set_status("BUTTON PRESSED", step, Signalizator.COLOR_BUTTON_PRESSED)

    def request_early_stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            busy = self._busy
            if not busy:
                self.completion_reason = "circle_early_stop"
                self.end = True
        if busy:
            self.get_logger().info("early stop requested; finishing current capture")

    def _collect_image_pairs_locked(self) -> None:
        if self._request is None:
            return
        for name, runtime in self._camera.items():
            if name in self._request.images:
                continue
            pair = select_first_pair_after(
                runtime.colors,
                runtime.depths,
                self._request.button_stamp_ns,
                self.sync_slop_ns,
            )
            if pair is not None:
                self._request.images[name] = pair

    def _capture_tick(self) -> None:
        with self._lock:
            request = self._request
        if request is None:
            return
        if time.monotonic() > request.deadline_monotonic:
            self._fail_request("capture timed out waiting for images, robot state, or TF")
            return
        if len(request.images) != len(self._camera):
            return
        try:
            prepared = self._prepare_capture(request)
        except (ValueError, TransformException) as error:
            self._fail_request(f"capture rejected: {error}")
            return
        if prepared is None:
            return
        record, visual_frames = prepared
        with self._lock:
            if self._request is not request:
                return
            self._request = None
        self._writer = threading.Thread(
            target=self._write_capture,
            args=(request, record, visual_frames),
            daemon=True,
        )
        self._writer.start()

    def _prepare_capture(
        self, request: CaptureRequest
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
        step = self.completed_steps + 1
        record: dict[str, Any] = {
            "schema_version": 1,
            "step": step,
            "button_stamp_ns": request.button_stamp_ns,
            "transform_convention": TRANSFORM_CONVENTION,
            "cameras": {},
        }
        visual_frames: dict[str, np.ndarray] = {BASE_FRAME: np.eye(4)}

        for name, (color, depth) in request.images.items():
            runtime = self._camera[name]
            if depth.encoding.lower() not in {"16uc1", "mono16"}:
                raise ValueError(
                    f"{name} raw depth must be 16UC1/mono16, "
                    f"got {depth.encoding!r}"
                )
            if color.encoding.lower() not in {
                "rgb8",
                "rgba8",
                "bgr8",
                "bgra8",
                "mono8",
                "mono16",
            }:
                raise ValueError(
                    f"{name} has unsupported color encoding {color.encoding!r}"
                )
            color_ns = message_stamp_ns(color)
            depth_ns = message_stamp_ns(depth)
            pose_bracket = bracket_messages(
                list(self._poses), color_ns, ROBOT_BRACKET_TOLERANCE_NS
            )
            joint_bracket = bracket_messages(
                list(self._joint_states), color_ns, ROBOT_BRACKET_TOLERANCE_NS
            )
            calibrated_bracket = bracket_messages(
                list(self._joint_states_calibrated),
                color_ns,
                ROBOT_BRACKET_TOLERANCE_NS,
            )
            if pose_bracket is None or joint_bracket is None or calibrated_bracket is None:
                return None

            optical_frame = _clean_frame(runtime.color_info.header.frame_id)
            depth_optical_frame = _clean_frame(depth.header.frame_id)
            color_frame = self._static_parent(optical_frame)
            if color_frame is None:
                return None
            static_color_optical = self._direct_static(color_frame, optical_frame)
            static_link_color = self._direct_static(runtime.link_frame, color_frame)
            static_ee_link = self._direct_static(
                CALIBRATED_EE_FRAME, runtime.link_frame
            )
            if (
                static_color_optical is None
                or static_link_color is None
                or static_ee_link is None
            ):
                return None

            color_time = Time.from_msg(color.header.stamp)
            depth_time = Time.from_msg(depth.header.stamp)
            required = (
                (BASE_FRAME, CALIBRATED_EE_FRAME, color_time),
                (BASE_FRAME, optical_frame, color_time),
                (BASE_FRAME, depth_optical_frame, depth_time),
            )
            if not all(
                self.tf_buffer.can_transform(
                    target,
                    source,
                    at_time,
                    timeout=Duration(seconds=0),
                )
                for target, source, at_time in required
            ):
                return None

            base_from_ee_msg = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                CALIBRATED_EE_FRAME,
                color_time,
                timeout=Duration(seconds=0),
            )
            base_from_optical_msg = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                optical_frame,
                color_time,
                timeout=Duration(seconds=0),
            )
            base_from_depth_msg = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                depth_optical_frame,
                depth_time,
                timeout=Duration(seconds=0),
            )

            t_base_ee = matrix_from_transform(base_from_ee_msg.transform)
            t_ee_link = matrix_from_transform(static_ee_link.transform)
            t_link_color = matrix_from_transform(static_link_color.transform)
            t_color_optical = matrix_from_transform(static_color_optical.transform)
            t_base_link = t_base_ee @ t_ee_link
            t_base_color = t_base_link @ t_link_color
            t_base_optical = t_base_color @ t_color_optical
            t_base_optical_direct = matrix_from_transform(
                base_from_optical_msg.transform
            )
            if not transforms_agree(t_base_optical, t_base_optical_direct):
                error = float(np.max(np.abs(t_base_optical - t_base_optical_direct)))
                raise ValueError(
                    f"{name} composed T_base_color_optical disagrees with TF2 "
                    f"(max abs error {error:.3e})"
                )

            names = {
                "color_optical": f"T_{color_frame}_{optical_frame}",
                "link_color": f"T_{runtime.link_frame}_{color_frame}",
                "ee_link": f"T_{CALIBRATED_EE_FRAME}_{runtime.link_frame}",
                "base_ee": f"T_{BASE_FRAME}_{CALIBRATED_EE_FRAME}",
                "base_link": f"T_{BASE_FRAME}_{runtime.link_frame}",
                "base_color": f"T_{BASE_FRAME}_{color_frame}",
                "base_optical": f"T_{BASE_FRAME}_{optical_frame}",
                "base_optical_direct": f"T_{BASE_FRAME}_{optical_frame}_direct_tf2",
                "base_depth": f"T_{BASE_FRAME}_{depth_optical_frame}",
            }
            transforms = {
                names["color_optical"]: transform_record(
                    color_frame,
                    optical_frame,
                    t_color_optical,
                    source="/tf_static",
                    stamp_ns=message_stamp_ns(static_color_optical),
                    static=True,
                ),
                names["link_color"]: transform_record(
                    runtime.link_frame,
                    color_frame,
                    t_link_color,
                    source="/tf_static",
                    stamp_ns=message_stamp_ns(static_link_color),
                    static=True,
                ),
                names["ee_link"]: transform_record(
                    CALIBRATED_EE_FRAME,
                    runtime.link_frame,
                    t_ee_link,
                    source="/tf_static",
                    stamp_ns=message_stamp_ns(static_ee_link),
                    static=True,
                ),
                names["base_ee"]: transform_record(
                    BASE_FRAME,
                    CALIBRATED_EE_FRAME,
                    t_base_ee,
                    source="/tf",
                    stamp_ns=color_ns,
                    static=False,
                ),
                names["base_link"]: transform_record(
                    BASE_FRAME,
                    runtime.link_frame,
                    t_base_link,
                    source="composed",
                    stamp_ns=color_ns,
                    static=False,
                ),
                names["base_color"]: transform_record(
                    BASE_FRAME,
                    color_frame,
                    t_base_color,
                    source="composed",
                    stamp_ns=color_ns,
                    static=False,
                ),
                names["base_optical"]: transform_record(
                    BASE_FRAME,
                    optical_frame,
                    t_base_optical,
                    source="composed",
                    stamp_ns=color_ns,
                    static=False,
                ),
                names["base_optical_direct"]: transform_record(
                    BASE_FRAME,
                    optical_frame,
                    t_base_optical_direct,
                    source="tf2 lookup",
                    stamp_ns=color_ns,
                    static=False,
                ),
                names["base_depth"]: transform_record(
                    BASE_FRAME,
                    depth_optical_frame,
                    matrix_from_transform(base_from_depth_msg.transform),
                    source="tf2 lookup",
                    stamp_ns=depth_ns,
                    static=False,
                ),
            }
            record["cameras"][name] = {
                "color": {
                    "path": f"{name}_color.png",
                    "topic": runtime.color_topic,
                    "stamp_ns": color_ns,
                    "frame_id": color.header.frame_id,
                    "encoding": color.encoding,
                    "width": int(color.width),
                    "height": int(color.height),
                },
                "depth": {
                    "path": f"{name}_depth.png",
                    "topic": runtime.depth_topic,
                    "stamp_ns": depth_ns,
                    "frame_id": depth.header.frame_id,
                    "encoding": depth.encoding,
                    "units": "raw uint16 sensor units (normally millimetres for RealSense)",
                    "width": int(depth.width),
                    "height": int(depth.height),
                },
                "sync_delta_ns": depth_ns - color_ns,
                "color_after_button_ns": color_ns - request.button_stamp_ns,
                "depth_after_button_ns": depth_ns - request.button_stamp_ns,
                "robot_messages": {
                    "/panda/curr_pose": {
                        "before": pose_stamped_record(pose_bracket[0], color_ns),
                        "after": pose_stamped_record(pose_bracket[1], color_ns),
                    },
                    "/joint_states": {
                        "before": joint_state_record(joint_bracket[0], color_ns),
                        "after": joint_state_record(joint_bracket[1], color_ns),
                    },
                    "/joint_states_calibrated": {
                        "before": joint_state_record(calibrated_bracket[0], color_ns),
                        "after": joint_state_record(calibrated_bracket[1], color_ns),
                    },
                },
                "transforms": transforms,
                "composition_max_abs_error": float(
                    np.max(np.abs(t_base_optical - t_base_optical_direct))
                ),
            }
            visual_frames.setdefault(CALIBRATED_EE_FRAME, t_base_ee)
            visual_frames[runtime.link_frame] = t_base_link
            visual_frames[color_frame] = t_base_color
            visual_frames[optical_frame] = t_base_optical

        return record, visual_frames

    def _color_array(self, message: Image) -> np.ndarray:
        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        encoding = message.encoding.lower()
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        if encoding in {"bgr8", "bgra8", "mono8", "mono16"}:
            return image
        raise ValueError(f"unsupported color encoding {message.encoding!r}")

    def _depth_array(self, message: Image) -> np.ndarray:
        if message.encoding.lower() not in {"16uc1", "mono16"}:
            raise ValueError(
                f"raw depth must be 16UC1/mono16, got {message.encoding!r}"
            )
        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        if image.dtype != np.uint16 or image.ndim != 2:
            raise ValueError(
                f"raw depth must decode to a 2-D uint16 array, got {image.dtype} {image.shape}"
            )
        return image

    def _write_capture(
        self,
        request: CaptureRequest,
        record: dict[str, Any],
        visual_frames: dict[str, np.ndarray],
    ) -> None:
        step = int(record["step"])
        destination = self.session_dir / f"step_{step:06d}"
        temporary = self.session_dir / (
            f"step_{step:06d}.tmp_{request.button_stamp_ns}"
        )
        try:
            temporary.mkdir()
            for name, (color, depth) in request.images.items():
                if not cv2.imwrite(
                    str(temporary / f"{name}_color.png"), self._color_array(color)
                ):
                    raise OSError(f"failed to write {name}_color.png")
                if not cv2.imwrite(
                    str(temporary / f"{name}_depth.png"), self._depth_array(depth)
                ):
                    raise OSError(f"failed to write {name}_depth.png")
            (temporary / "capture.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            (temporary / "capture.txt").write_text(
                transforms_text(record), encoding="utf-8"
            )
            save_frame_visualizations(
                visual_frames,
                temporary / "frames.html",
                temporary / "frames.png",
                title=f"Workspace capture step {step}",
            )
            temporary.replace(destination)
        except Exception as error:
            self.get_logger().error(
                f"step {step} write failed; partial data left in {temporary}: "
                f"{type(error).__name__}: {error}"
            )
            with self._lock:
                self._busy = False
                stop = self._stop_requested
            self._set_status("ERROR", self.completed_steps, Signalizator.COLOR_ERROR)
            if stop:
                self.completion_reason = "circle_early_stop"
                self.end = True
            return

        with self._lock:
            self.completed_steps = step
        self._data_saved_at = time.monotonic()
        self._set_status("DATA SAVED", step, Signalizator.COLOR_DATA_SAVED)

        session_error: Exception | None = None
        with self._lock:
            completion = completion_reason_after_save(
                step, self.requested_steps, self._stop_requested
            )
            if completion is not None:
                self.completion_reason = completion
            self._session["completed_steps"] = step
            try:
                # Keep Check/Circle callbacks outside the tiny metadata-write
                # window so they observe one unambiguous completed state.
                self._write_session()
            except Exception as error:
                session_error = error
                self.completion_reason = "error"
            self._busy = False
        if session_error is not None:
            self.get_logger().error(
                f"could not update session.json: {session_error}"
            )
            self._set_status("ERROR", step, Signalizator.COLOR_ERROR)
            self.end = True
            return
        if completion is not None:
            self.end = True

    def _fail_request(self, reason: str) -> None:
        with self._lock:
            if self._request is None:
                return
            self._request = None
            self._busy = False
            stop = self._stop_requested
        self.get_logger().error(reason)
        self._set_status("ERROR", self.completed_steps, Signalizator.COLOR_ERROR)
        if stop:
            self.completion_reason = "circle_early_stop"
            self.end = True

    def wait_for_capture(self) -> None:
        """Wait until the timer rejects or the atomic background write finishes."""
        while True:
            with self._lock:
                busy = self._busy
            writer = self._writer
            if not busy and (writer is None or not writer.is_alive()):
                return
            self.update_additional_flags()
            time.sleep(0.05)

    def save_trajectory(self) -> None:
        if self.recorded_traj is None:
            return
        temporary = self.session_dir / "trajectory.tmp.npz"
        np.savez(
            temporary,
            traj=self.recorded_traj,
            ori=self.recorded_ori_wxyz,
            grip=self.recorded_gripper,
            stamp_ns=self.recorded_stamp_ns,
            img_feedback_flag=self.recorded_img_feedback_flag,
            spiral_flag=self.recorded_spiral_flag,
        )
        temporary.replace(self.session_dir / "trajectory.npz")

    def _write_session(self) -> None:
        self._session["completed_steps"] = self.completed_steps
        self._session["completion_reason"] = self.completion_reason
        _atomic_write_text(
            self.session_dir / "session.json",
            json.dumps(self._session, indent=2) + "\n",
        )

    def finalize_session(self, reason: str | None = None) -> None:
        with self._lock:
            self._capture_enabled = False
        if reason is not None and self.completion_reason == "running":
            self.completion_reason = reason
        self._session["ended_at"] = datetime.now(timezone.utc).isoformat()
        self._write_session()
        if self.completion_reason == "error":
            self._set_status("ERROR", self.completed_steps, Signalizator.COLOR_ERROR)
        else:
            self._set_status(
                "COMPLETE", self.completed_steps, Signalizator.COLOR_DATA_SAVED
            )
        self._render_status()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: WorkspaceCapture | None = None
    control_started = False
    failure: Exception | None = None
    try:
        node = WorkspaceCapture()
        node.start()
        control_started = True
        node.get_logger().info("moving robot to home before capture")
        node.home()
        node.wait_until_ready()
        node.frankabuttons_start()
        node.enable_capture()
        node.traj_rec(record_images=False)
    except KeyboardInterrupt:
        if node is not None:
            node.completion_reason = "keyboard_interrupt"
    except Exception as error:
        failure = error
        if node is not None:
            node.completion_reason = "error"
            node.get_logger().error(f"workspace capture failed: {error}")
    finally:
        if node is not None:
            node.wait_for_capture()
            try:
                node.save_trajectory()
            except Exception as error:
                node.get_logger().error(f"could not save trajectory: {error}")
                failure = failure or error
                node.completion_reason = "error"
            if control_started:
                try:
                    node.set_stiffness(
                        node.K_pos,
                        node.K_pos,
                        node.K_pos,
                        node.K_ori,
                        node.K_ori,
                        node.K_ori,
                        0,
                    )
                except Exception as error:
                    node.get_logger().error(f"could not restore stiffness: {error}")
                    failure = failure or error
                    node.completion_reason = "error"
            try:
                node.finalize_session()
            except Exception as error:
                node.get_logger().error(f"could not finalize session: {error}")
                failure = failure or error
            node.signalizer.close()
        if rclpy.ok():
            rclpy.shutdown()
        if node is not None:
            node._spin_thread.join(timeout=1.0)
            node.destroy_node()
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
