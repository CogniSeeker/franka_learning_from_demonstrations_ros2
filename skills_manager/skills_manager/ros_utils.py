
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading

from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np

from skills_manager.ros_param_manager import get_remote_parameter, set_remote_parameter
from rclpy.exceptions import ParameterAlreadyDeclaredException

class SpinningRosNode(Node):
    def __init__(self):
        super(SpinningRosNode, self).__init__(f"panda_node_{np.random.randint(100000)}") # node name replaced by launch description
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(self)
        spinning_thread = threading.Thread(target=executor.spin, daemon=True)
        spinning_thread.start()

        self.callback_group = ReentrantCallbackGroup()

        self.get_remote_parameter = get_remote_parameter
        self.set_remote_parameter = set_remote_parameter

    def declare_parameter_and_get(self, name, default_value):
        try:
            self.declare_parameter(name, default_value)
        except ParameterAlreadyDeclaredException:
            self.set_remote_parameter(self, name, default_value)
        return self.get_remote_parameter(self, name)


