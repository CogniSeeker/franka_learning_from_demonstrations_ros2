from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("rclpy")
pytest.importorskip("panda_py")
pytest.importorskip("quaternion")
pytest.importorskip("roboticstoolbox")

from builtin_interfaces.msg import Time  # noqa: E402
from panda_control.panda import _feedback_messages  # noqa: E402


def test_feedback_pose_and_joints_share_nonzero_stamp_and_robot_sample():
    base_from_ee = np.eye(4)
    base_from_ee[:3, 3] = [0.4, -0.1, 0.3]
    state = SimpleNamespace(
        O_T_EE=base_from_ee.reshape(16, order="F"),
        q=np.arange(7, dtype=float),
        dq=np.arange(7, dtype=float) + 10.0,
        tau_J=np.arange(7, dtype=float) + 20.0,
    )
    stamp = Time(sec=12, nanosec=34)

    pose, joints = _feedback_messages(state, gripper_width=0.06, stamp=stamp)

    assert pose.header.stamp == joints.header.stamp
    assert pose.header.stamp.sec == 12
    assert pose.header.stamp.nanosec == 34
    assert pose.header.frame_id == "panda_link0"
    np.testing.assert_allclose(
        [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
        [0.4, -0.1, 0.3],
    )
    np.testing.assert_allclose(joints.position[:7], state.q)
    np.testing.assert_allclose(joints.position[7:], [0.03, 0.03])
