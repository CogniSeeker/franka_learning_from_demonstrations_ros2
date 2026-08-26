#!/usr/bin/env python3
"""Encode Perception Module detections as ROS messages and send via publisher."""

from __future__ import annotations

from importlib import import_module
import time
from typing import Any

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from lfd_msgs.msg import DetectedObject as DetectedObjectMsg
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
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_pose_stamped

from object_localization.tf_utils import CustomTransformListener
from skills_manager.ros_utils import SpinningRosNode

DETECTED_OBJECT_TOPIC = "/perception/detected_object"

ROBOT_BASE_TF_FRAME = "panda_link0"
ROBOT_CAMERA_TF_FRAME = "camera2_link"

CAMERA_COLOR_TOPIC = "/camera2/color/image_raw"
CAMERA_INFO_TOPIC = "/camera2/color/camera_info"
CURRENT_POSE_TOPIC = "/panda/curr_pose"
CAMERA_WARMUP_SECONDS = 3.0


def detected_object_qos() -> QoSProfile:
    """Return the durable QoS shared with the detection consumer."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class PerceptionModuleWrapper:
    """Load the external Perception Module and expose one method.

    The configured external module must provide:

      loader(weights_path) -> model

    The loaded model must provide:

      inference(bgr_image, camera_info) -> list[DetectedObject] | Deck
    """

    def __init__(
        self,
        module_name: str,
        loader_name: str,
        inference_name: str,
        weights_path: str,
        logger: Any,
    ) -> None:
        if not module_name:
            raise ValueError("model_module parameter must not be empty")
        if not loader_name:
            raise ValueError("model_loader parameter must not be empty")
        if not inference_name:
            raise ValueError("model_inference parameter must not be empty")
        if not weights_path:
            raise ValueError("model_weights parameter must not be empty")

        logger.info(f"importing perception module {module_name}")
        module = import_module(module_name)
        logger.info(f"perception module imported; calling {loader_name}()")
        loader = getattr(module, loader_name)
        self._model = loader(weights_path)
        logger.info("perception model loaded")
        self._inference = getattr(self._model, inference_name)
        logger.info(f"perception entrypoint ready: {inference_name}()")

    def infer(
        self,
        bgr_image: np.ndarray,
        camera_info: CameraInfo,
    ) -> list[Any]:
        result = self._inference(bgr_image, camera_info)
        detections = (
            result if isinstance(result, list) else getattr(result, "items", None)
        )
        if not isinstance(detections, list):
            raise TypeError(
                "perception inference must return a Deck or a Python list of detections"
            )
        return detections

    def set_visualization_end_effector_pose(self, pose: Any) -> None:
        """Forward an optional base-referenced EE pose to supporting models."""
        setter = getattr(self._model, "set_visualization_end_effector_pose", None)
        if callable(setter):
            setter(
                {
                    "translation_xyz_m": [pose.position.x, pose.position.y, pose.position.z],
                    "quaternion_xyzw": [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ],
                }
            )

    def set_visualization_camera_transform(self, transform: Any) -> None:
        """Forward T_camera_link_optical from TF2 to supporting models."""
        setter = getattr(self._model, "set_visualization_camera_transform", None)
        if callable(setter):
            translation = transform.translation
            rotation = transform.rotation
            setter(
                {
                    "translation_xyz_m": [
                        translation.x,
                        translation.y,
                        translation.z,
                    ],
                    "quaternion_xyzw": [
                        rotation.x,
                        rotation.y,
                        rotation.z,
                        rotation.w,
                    ],
                }
            )


class TeleportDetectionService(CustomTransformListener, SpinningRosNode):
    """Transform and publish detections returned by the perception package.

    The perception module is expected to return a Deck or a list of detections.

    Camera frames are passed to the model automatically. Adapted interaction
    poses use the model's configured output frame. The batch retains the image
    timestamp and waits until the corresponding camera-to-base TF is buffered.
    """

    def __init__(self) -> None:
        super().__init__()
        self.get_logger().info("initializing teleport detector")

        self.declare_parameter("model_module", "")
        self.declare_parameter("model_loader", "")
        self.declare_parameter("model_inference", "")
        self.declare_parameter("model_weights", "")

        self._model = PerceptionModuleWrapper(
            module_name=str(self.get_parameter("model_module").value),
            loader_name=str(self.get_parameter("model_loader").value),
            inference_name=str(self.get_parameter("model_inference").value),
            weights_path=str(self.get_parameter("model_weights").value),
            logger=self.get_logger(),
        )
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._current_end_effector_pose: Any | None = None
        self._published_batch = False
        self._pending_batch: tuple[list[Any], Any, str] | None = None
        self._seen_camera_info = False
        self._seen_image = False
        self._waiting_for_camera_info_logged = False
        self._waiting_for_optical_tf_logged = False
        self._waiting_for_tf_logged = False
        self._camera_warmup_until: float | None = None

        self._publisher = self.create_publisher(
            DetectedObjectMsg,
            DETECTED_OBJECT_TOPIC,
            detected_object_qos(),
        )
        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self._image_subscription = self.create_subscription(
            Image,
            CAMERA_COLOR_TOPIC,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self._current_pose_subscription = self.create_subscription(
            PoseStamped,
            CURRENT_POSE_TOPIC,
            self._current_pose_callback,
            5,
        )
        self._tf_retry_timer = self.create_timer(
            0.02,
            self._try_publish_pending_batch,
        )
        self.get_logger().info(
            f"ready; camera warm-up will start with the first image and last "
            f"{CAMERA_WARMUP_SECONDS:.0f}s; "
            f"subscribed to {CAMERA_COLOR_TOPIC}, {CAMERA_INFO_TOPIC}, and "
            f"{CURRENT_POSE_TOPIC}"
        )

    def _current_pose_callback(self, message: PoseStamped) -> None:
        """Cache T_panda_link0_end_effector published by the robot controller."""
        frame_id = message.header.frame_id.strip()
        if frame_id and frame_id != ROBOT_BASE_TF_FRAME:
            self.get_logger().warning(
                f"ignoring {CURRENT_POSE_TOPIC} pose in unexpected frame {frame_id!r}; "
                f"expected {ROBOT_BASE_TF_FRAME!r} or an empty frame_id"
            )
            return
        self._current_end_effector_pose = message.pose

    def _camera_info_callback(self, camera_info: CameraInfo) -> None:
        """Store the latest camera calibration."""
        if (
            self._camera_warmup_until is None
            or time.monotonic() < self._camera_warmup_until
        ):
            return
        self._camera_info = camera_info
        if not self._seen_camera_info:
            self._seen_camera_info = True
            self.get_logger().info(
                f"received CameraInfo {camera_info.width}x{camera_info.height} "
                f"frame={camera_info.header.frame_id!r}"
            )

    def _image_callback(self, image: Image) -> None:
        """Run inference once and retain its result until matching TF arrives."""
        now = time.monotonic()
        if self._camera_warmup_until is None:
            self._camera_warmup_until = now + CAMERA_WARMUP_SECONDS
            self.get_logger().info(
                f"first image received; warming up camera for "
                f"{CAMERA_WARMUP_SECONDS:.0f}s"
            )
            return
        if now < self._camera_warmup_until:
            return
        if not self._seen_image:
            self._seen_image = True
            self.get_logger().info(
                f"received image {image.width}x{image.height} "
                f"frame={image.header.frame_id!r}"
            )
        if self._published_batch or self._pending_batch is not None:
            return
        if self._camera_info is None:
            if not self._waiting_for_camera_info_logged:
                self._waiting_for_camera_info_logged = True
                self.get_logger().warning("image received; waiting for CameraInfo")
            return

        camera_frame = self._camera_info.header.frame_id.strip()
        if not camera_frame:
            self.get_logger().error("CameraInfo has an empty frame_id")
            return
        if not self.tf_buffer.can_transform(
            ROBOT_CAMERA_TF_FRAME,
            camera_frame,
            Time(),
            timeout=Duration(seconds=0),
        ):
            if not self._waiting_for_optical_tf_logged:
                self._waiting_for_optical_tf_logged = True
                self.get_logger().warning(
                    f"waiting for RealSense TF {ROBOT_CAMERA_TF_FRAME} <- "
                    f"{camera_frame} on /tf_static"
                )
            return

        link_from_optical = self.tf_buffer.lookup_transform(
            ROBOT_CAMERA_TF_FRAME,
            camera_frame,
            Time(),
            timeout=Duration(seconds=0),
        )
        if self._waiting_for_optical_tf_logged:
            self.get_logger().info("camera link-to-optical TF is now available")
            self._waiting_for_optical_tf_logged = False
        self._model.set_visualization_camera_transform(link_from_optical.transform)

        bgr_image = self._bridge.imgmsg_to_cv2(image, "bgr8")
        if self._current_end_effector_pose is not None:
            self._model.set_visualization_end_effector_pose(
                self._current_end_effector_pose
            )
        self.get_logger().info("starting perception inference")
        started = time.perf_counter()
        try:
            detections = self._model.infer(bgr_image, self._camera_info)
        except Exception as error:
            self.get_logger().error(
                f"perception inference failed: {type(error).__name__}: {error}"
            )
            raise
        self.get_logger().info(
            f"perception inference finished in {time.perf_counter() - started:.2f}s; "
            f"items={len(detections)}"
        )
        if not detections:
            self.get_logger().warning("perception returned no deck items")
            return

        self._pending_batch = (
            detections,
            image.header.stamp,
            camera_frame,
        )

    def _try_publish_pending_batch(self) -> None:
        """Publish the retained batch once its image-time TF is buffered."""
        if self._pending_batch is None:
            return

        detections, stamp, camera_frame = self._pending_batch
        at_time = Time.from_msg(stamp)

        if not self.tf_buffer.can_transform(
            ROBOT_BASE_TF_FRAME,
            camera_frame,
            at_time,
            timeout=Duration(seconds=0),
        ):
            if not self._waiting_for_tf_logged:
                self._waiting_for_tf_logged = True
                self.get_logger().warning(
                    f"detections ready; waiting for TF {ROBOT_BASE_TF_FRAME} <- "
                    f"{camera_frame} at image time"
                )
            return

        if self._waiting_for_tf_logged:
            self.get_logger().info("camera-to-base TF is now available")

        transform = self.tf_buffer.lookup_transform(
            ROBOT_BASE_TF_FRAME,
            camera_frame,
            at_time,
            timeout=Duration(seconds=0),
        )

        self.publish_detections(detections, transform, stamp, camera_frame)
        self._pending_batch = None
        self._published_batch = True

    def _to_ros_message(
        self,
        detection: Any,
        transform: Any,
        stamp: Any,
        camera_frame: str,
    ) -> DetectedObjectMsg:
        """Encode one adapted perception detection as a ROS message."""
        msg = DetectedObjectMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = ROBOT_BASE_TF_FRAME
        object_id = str(getattr(detection, "object_id", "")).strip()
        if not object_id:
            layout_id = getattr(detection, "layout_id", None)
            object_id = "" if layout_id is None else str(layout_id).strip()
        if not object_id:
            object_id = f"detection_{int(detection.detection_id)}"
        msg.object_id = object_id
        class_name = str(getattr(detection, "class_name", "")).strip()
        if not class_name:
            class_name = str(getattr(detection, "classified_class", "")).strip()
        yolo_class = getattr(detection, "yolo_class", None)
        if not class_name and yolo_class is not None:
            class_name = str(yolo_class).strip()
        msg.class_name = class_name or "unknown"
        state = getattr(detection.state, "name", detection.state)
        msg.state = (str(state).strip() if state is not None else "") or "unknown"
        confidence = float(detection.confidence)
        msg.confidence = (
            float(np.clip(confidence, 0.0, 1.0))
            if np.isfinite(confidence)
            else 0.0
        )

        if not detection.pose_valid:
            msg.pose.orientation.w = 1.0
            msg.pose_valid = False
            return msg

        python_pose = detection.pose
        camera_pose = PoseStamped()
        camera_pose.header.stamp = stamp
        camera_pose.header.frame_id = camera_frame
        (
            camera_pose.pose.position.x,
            camera_pose.pose.position.y,
            camera_pose.pose.position.z,
        ) = (float(value) for value in python_pose.position)

        (
            camera_pose.pose.orientation.x,
            camera_pose.pose.orientation.y,
            camera_pose.pose.orientation.z,
            camera_pose.pose.orientation.w,
        ) = (float(value) for value in python_pose.orientation)

        base_pose = do_transform_pose_stamped(camera_pose, transform)
        msg.pose = base_pose.pose
        msg.pose_valid = True

        return msg

    def publish_detections(
        self,
        detections: list[Any],
        transform: Any,
        stamp: Any,
        camera_frame: str,
    ) -> int:
        """Transform and publish one ROS message per Python detection.

        Args:
            detections: Results from the perception function.
            transform: ROS transform from the camera to panda_link0.
            stamp: ROS builtin_interfaces/Time of the source image.
            camera_frame: Optical frame in which perception returned poses.

        Returns:
            Number of messages published.
        """
        if not detections:
            return 0

        for detection in detections:
            msg = self._to_ros_message(
                detection,
                transform,
                stamp,
                camera_frame,
            )
            self._publisher.publish(msg)

        self.get_logger().info(
            f"published {len(detections)} detections in {ROBOT_BASE_TF_FRAME}"
        )
        return len(detections)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TeleportDetectionService()
    try:
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node._spin_thread.join(timeout=1.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
