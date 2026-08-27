"""Publish one configurable mock object detection for integration testing."""

import rclpy
from lfd_msgs.msg import DetectedObject
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


DETECTED_OBJECT_TOPIC = "/perception/detected_object"


def detected_object_qos() -> QoSProfile:
    """Return the durable QoS shared with detection consumers."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class MockObjectDetector(Node):
    """Publish one mock detection, then stay alive to retain the sample."""

    def __init__(self) -> None:
        super().__init__("mock_object_detector")

        self.declare_parameter("object_id", "button_1")
        self.declare_parameter("class_name", "button")
        self.declare_parameter("state", "unpressed")
        self.declare_parameter("confidence", 1.0)
        self.declare_parameter("pose_valid", True)
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("position_x", 0.0)
        self.declare_parameter("position_y", 0.0)
        self.declare_parameter("position_z", 0.4)
        self.declare_parameter("orientation_x", 0.0)
        self.declare_parameter("orientation_y", 0.0)
        self.declare_parameter("orientation_z", 0.0)
        self.declare_parameter("orientation_w", 1.0)
        self.declare_parameter("publish_delay_sec", 1.0)

        self._publisher = self.create_publisher(
            DetectedObject,
            DETECTED_OBJECT_TOPIC,
            detected_object_qos(),
        )
        delay = max(
            0.01,
            self.get_parameter("publish_delay_sec").get_parameter_value().double_value,
        )
        self._published = False
        self._timer = self.create_timer(delay, self._publish_once)

    def _publish_once(self) -> None:
        if self._published:
            return

        msg = DetectedObject()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter("frame_id").value
        msg.object_id = self.get_parameter("object_id").value
        msg.class_name = self.get_parameter("class_name").value
        msg.state = self.get_parameter("state").value
        msg.confidence = float(self.get_parameter("confidence").value)
        msg.pose_valid = bool(self.get_parameter("pose_valid").value)
        msg.pose.position.x = float(self.get_parameter("position_x").value)
        msg.pose.position.y = float(self.get_parameter("position_y").value)
        msg.pose.position.z = float(self.get_parameter("position_z").value)
        msg.pose.orientation.x = float(self.get_parameter("orientation_x").value)
        msg.pose.orientation.y = float(self.get_parameter("orientation_y").value)
        msg.pose.orientation.z = float(self.get_parameter("orientation_z").value)
        msg.pose.orientation.w = float(self.get_parameter("orientation_w").value)

        self._publisher.publish(msg)
        self._published = True
        self._timer.cancel()
        self.get_logger().info(
            f"published mock detection {msg.object_id} "
            f"({msg.class_name}, {msg.state})"
        )


def main(args: list[str] | None = None) -> None:
    """Run the mock detector until ROS shuts down."""
    rclpy.init(args=args)
    node = MockObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
