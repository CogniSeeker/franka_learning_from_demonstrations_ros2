

import time
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from panda_control.pose_transform_functions import position_2_array, pos_quat_2_pose_st, list_2_quaternion, invert_tf
import rclpy

class Player():
    def __init__(self):
        super(Player, self).__init__()
    def execute(self, retry_insertion_flag=0):
        start = self.player_init()
        while self.time_index <( self.recorded_traj.shape[1]):
            self.player_step(start, retry_insertion_flag)

    def player_init(self):
        self.spiralling_occured = False
        start = PoseStamped()

        quat_start = list_2_quaternion(self.recorded_ori_wxyz[:, 0])
        start = pos_quat_2_pose_st(self.recorded_traj[:, 0], quat_start)

        self.go_to_pose_ik(start)
        self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, self.K_ori, 0)
        
        self.time_index=0

        if self.recorded_gripper[0][0] < self.grip_open_width/2 and self.gripper_width > 0.9 * self.grip_open_width:
            print("closing gripper")
            self.grasp_gripper(self.recorded_gripper[0][self.time_index])
            time.sleep(0.1)
        if self.recorded_gripper[0][0] > self.grip_open_width/2:
            print("opening gripper")
            self.move_gripper(self.recorded_gripper[0][self.time_index])
            time.sleep(0.1)
        
        self.trajectory_len = self.recorded_traj.shape[1]

        return start



    def player_step(self, start, retry_insertion_flag):
        
        quat_goal = list_2_quaternion(self.recorded_ori_wxyz[:, self.time_index])
        goal = pos_quat_2_pose_st(self.recorded_traj[:, self.time_index] + self.camera_correction, quat_goal)
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'panda_link0'
        ori_threshold = 0.3
        pos_threshold = 0.1
        
        self.correct()

        if (self.recorded_gripper[0][self.time_index]-self.recorded_gripper[0][max(0,self.time_index-1)]) < -self.grip_open_width/2:
            self.grasp_gripper(self.recorded_gripper[0][self.time_index])
            time.sleep(0.1)

        if (self.recorded_gripper[0][self.time_index]-self.recorded_gripper[0][max(0,self.time_index-1)]) > self.grip_open_width/2:
            self.move_gripper(self.recorded_gripper[0][self.time_index])
            time.sleep(0.1)
        self.move_to_pose_with_stampedpose(goal)

        if self.recorded_img_feedback_flag[0, self.time_index]:
            self.sift_matching()
        
        if self.recorded_spiral_flag[0, self.time_index]:
            if self.force.z > 5:
                spiral_success, offset_correction = self.spiral_search(goal)
                self.spiralling_occured = True
                if spiral_success:
                    self.recorded_traj[0, self.time_index:] += offset_correction[0]
                    self.recorded_traj[1, self.time_index:] += offset_correction[1]

        goal_pos_array = position_2_array(goal.pose.position)
        pos_2_goal_diff = np.linalg.norm(self.curr_pos-goal_pos_array)

        if pos_2_goal_diff <= self.attractor_distance_threshold:
            self.time_index=self.time_index + 1

        force_xy_plane = np.sqrt(self.force.x ** 2 + self.force.y ** 2)
        if retry_insertion_flag and force_xy_plane > self.insertion_force_threshold:
            # print("Camera correction", self.camera_correction)
            if self.retry_counter >= 3:
                self.move_gripper(self.grip_open_width)

                return 'stop'
                
            self.go_to_pose(start)
            self.time_index = 0
            self.retry_counter = self.retry_counter + 1
        self.r.sleep()
        # Stop playback if at end of trajectory (some indices might be deleted by feedback)
        if self.time_index == self.recorded_traj.shape[1]-1:
            return 'stop'


    