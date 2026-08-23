import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros

ROBOT_BASE_TF_FRAME = "panda_link0"
CAMERA_TF_FRAME = "camera3_link"
# CAMERA_TF_FRAME = "panda_hand"

class CustomTransformListener():
    """TF lookups backed by tf2_ros.

    The previous version kept its own {(parent, child): TransformStamped} dict
    fed from /tf and /tf_static, and matched only a single *directly broadcast*
    edge. That happened to work here -- panda.py broadcasts panda_link0 ->
    panda_hand directly and static_transform_camera.py broadcasts panda_hand ->
    camera_color_optical_frame -- but it always returned the newest message
    regardless of the stamp asked for, so a lookup silently used stale
    extrinsics while the arm moved, and it broke outright for any frame pair
    reached through more than one hop.

    tf2's buffer walks the chain, inverts edges as needed, and interpolates to
    a requested stamp.
    """

    def __init__(self):
        super(CustomTransformListener, self).__init__()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def lookup_relative_transform(self, source_frame, target_frame, at_time=None):
        """Translation+rotation of target_frame expressed in source_frame.

        That is T_source<-target, the same thing the old dict returned for the
        key (source_frame, target_frame). Returns (None, None) when the
        transform is not available.

        Pass at_time (a rclpy.time.Time, e.g. built from an image
        header.stamp) to get the transform as it was at that instant rather
        than the latest one -- that is what keeps a lookup honest while the arm
        is moving.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                source_frame,
                target_frame,
                Time() if at_time is None else at_time,
                # ponytail: zero timeout on purpose. SpinningRosNode runs a
                # SingleThreadedExecutor, so blocking here from inside a
                # service callback would deadlock the very executor that fills
                # the buffer. Callers already retry; give them a fast miss.
                timeout=Duration(seconds=0),
            )
        except tf2_ros.TransformException as error:
            self.get_logger().warning(
                f"Transform {source_frame} -> {target_frame} not available: {error}"
            )
            return None, None
        return transform.transform.translation, transform.transform.rotation


def main(args=None):
    """Smoke check: resolve one chained transform and print it."""
    import threading

    rclpy.init(args=args)

    class _Probe(CustomTransformListener, Node):
        def __init__(self):
            Node.__init__(self, "tf_utils_probe")
            CustomTransformListener.__init__(self)

    node = _Probe()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    rate = node.create_rate(10)

    source_frame, target_frame = ROBOT_BASE_TF_FRAME, CAMERA_TF_FRAME
    try:
        # the buffer needs a moment of spinning before the chain is complete
        for _ in range(50):
            translation, rotation = node.lookup_relative_transform(source_frame, target_frame)
            if translation is not None:
                node.get_logger().info(
                    f"Transform {source_frame} -> {target_frame}:\n"
                    f"Translation: x={translation.x}, y={translation.y}, z={translation.z}\n"
                    f"Rotation: x={rotation.x}, y={rotation.y}, z={rotation.z}, w={rotation.w}"
                )
                break
            rate.sleep()
        else:
            node.get_logger().error(
                f"Transform {source_frame} -> {target_frame} never became available"
            )
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
