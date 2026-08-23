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
# TODO bad logic. Not adapted for contemporary TF publishing
# "panda_link0 = base frame"
# "camera3_link = camera frame"
# BASE_CAMERA_TF_TOPIC = "panda_link0_to_camera2_link"
ROBOT_BASE_TF_FRAME = "panda_link0"
ROBOT_CAMERA_TF_FRAME = "camera2_link"

CAMERA_COLOR_TOPIC = "/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"


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

      inference(bgr_image, camera_info) -> list[DetectedObject]
    """

    def __init__(
        self,
        module_name: str,
        loader_name: str,
        inference_name: str,
        weights_path: str,
    ) -> None:
        if not module_name:
            raise ValueError("model_module parameter must not be empty")
        if not loader_name:
            raise ValueError("model_loader parameter must not be empty")
        if not inference_name:
            raise ValueError("model_inference parameter must not be empty")
        if not weights_path:
            raise ValueError("model_weights parameter must not be empty")

        module = import_module(module_name)
        loader = getattr(module, loader_name)
        self._model = loader(weights_path)
        self._inference = getattr(self._model, inference_name)

    def infer(
        self,
        bgr_image: np.ndarray,
        camera_info: CameraInfo,
    ) -> list[Any]:
        detections = self._inference(bgr_image, camera_info)
        if not isinstance(detections, list):
            raise TypeError(
                "perception inference must return a Python list of DetectedObject"
            )
        return detections


class TeleportDetectionService(CustomTransformListener, SpinningRosNode):
    """Transform and publish detections returned by the perception package.

    The perception module is expected to return a Python list whose elements
    expose the same fields as lfd_msgs/msg/DetectedObject. Its pose is a Python
    object with position=(x, y, z) and either orientation=(x, y, z, w) or
    orientation=None.

    Camera frames are passed to the model automatically. Detection poses must
    use ROBOT_CAMERA_TF_FRAME coordinates. The batch retains the image timestamp
    and waits until the corresponding camera-to-base TF is buffered.
    """

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("model_module", "")
        self.declare_parameter("model_loader", "")
        self.declare_parameter("model_inference", "")
        self.declare_parameter("model_weights", "")

        self._model = PerceptionModuleWrapper(
            module_name=str(self.get_parameter("model_module").value),
            loader_name=str(self.get_parameter("model_loader").value),
            inference_name=str(self.get_parameter("model_inference").value),
            weights_path=str(self.get_parameter("model_weights").value),
        )
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._published_batch = False
        self._pending_batch: tuple[list[Any], Any] | None = None

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
        self._tf_retry_timer = self.create_timer(
            0.02,
            self._try_publish_pending_batch,
        )

    def _camera_info_callback(self, camera_info: CameraInfo) -> None:
        """Store the latest camera calibration."""
        self._camera_info = camera_info

    def _image_callback(self, image: Image) -> None:
        """Run inference once and retain its result until matching TF arrives."""
        if self._published_batch or self._pending_batch is not None:
            return
        if self._camera_info is None:
            return

        bgr_image = self._bridge.imgmsg_to_cv2(image, "bgr8")
        detections = self._model.infer(bgr_image, self._camera_info)
        if not detections:
            return

        self._pending_batch = (
            detections,
            image.header.stamp,
        )

    def _try_publish_pending_batch(self) -> None:
        """Publish the retained batch once its image-time TF is buffered."""
        if self._pending_batch is None:
            return

        detections, stamp = self._pending_batch
        at_time = Time.from_msg(stamp)

        if not self.tf_buffer.can_transform(
            ROBOT_BASE_TF_FRAME,
            ROBOT_CAMERA_TF_FRAME,
            at_time,
            timeout=Duration(seconds=0),
        ):
            return

        transform = self.tf_buffer.lookup_transform(
            ROBOT_BASE_TF_FRAME,
            ROBOT_CAMERA_TF_FRAME,
            at_time,
            timeout=Duration(seconds=0),
        )

        self.publish_detections(detections, transform, stamp)
        self._pending_batch = None
        self._published_batch = True

    def _to_ros_message(
        self,
        detection: Any,
        transform: Any,
        stamp: Any,
    ) -> DetectedObjectMsg:
        """Encode one Python detection as an lfd_msgs ROS message."""
        msg = DetectedObjectMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = ROBOT_BASE_TF_FRAME
        msg.object_id = str(detection.object_id)
        msg.class_name = str(detection.class_name)
        msg.state = str(detection.state)
        msg.confidence = float(detection.confidence)

        python_pose = detection.pose
        input_pose_valid = bool(detection.pose_valid)

        camera_pose = PoseStamped()
        camera_pose.header.stamp = stamp
        camera_pose.header.frame_id = ROBOT_CAMERA_TF_FRAME
        (
            camera_pose.pose.position.x,
            camera_pose.pose.position.y,
            camera_pose.pose.position.z,
        ) = (float(value) for value in python_pose.position)

        orientation_missing = python_pose.orientation is None
        if orientation_missing:
            camera_pose.pose.orientation.w = 1.0
        else:
            (
                camera_pose.pose.orientation.x,
                camera_pose.pose.orientation.y,
                camera_pose.pose.orientation.z,
                camera_pose.pose.orientation.w,
            ) = (float(value) for value in python_pose.orientation)

        base_pose = do_transform_pose_stamped(camera_pose, transform)
        msg.pose = base_pose.pose
        if orientation_missing:
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0
        msg.pose_valid = input_pose_valid

        return msg

    def publish_detections(
        self,
        detections: list[Any],
        transform: Any,
        stamp: Any,
    ) -> int:
        """Transform and publish one ROS message per Python detection.

        Args:
            detections: Results from the perception function.
            transform: ROS transform from the camera to panda_link0.
            stamp: ROS builtin_interfaces/Time of the source image.

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
