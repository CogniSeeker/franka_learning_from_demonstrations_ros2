
import time
import numpy as np
from risk_estimation.models.safety_layer import SafetyLayer
from video_embedding.utils import visualize_labelled_video_frame
from skills_manager.player import Player
import rclpy
import trajectory_data

from panda_control.pose_transform_functions import orientation_2_quaternion, pose_st_2_transformation, position_2_array, pos_quat_2_pose_st, transformation_2_pose, transform_pose, list_2_quaternion, transform_pos_ori, invert_tf

from pathlib import Path

class RiskAwarePlayer(Player):
    """Demonstration player with risk-awareness capabilities
    """    
    def __init__(self):
        super(RiskAwarePlayer, self).__init__()

    def set_decision_unit(self, obj):
        self.state_decider = obj

    def get_current_branch(self):
        try:
            return int(self.filename.split("_")[-1])
        except ValueError:
            return 0

    def execute(self, retry_insertion_flag=0):
        ''' Has trajectory at self.recorded_traj, self.recorded_ori
        '''
        print(f"Executing: {self.filename}", flush=True)
        start = self.player_init()
        self.traj_rec_init()
        self.end = False
        while self.time_index <( self.recorded_traj.shape[1]) and rclpy.ok() and not self.end:
            print("Execution time index: ", self.time_index, flush=True)
            try:
                while(self.pause):
                    self.r.sleep()
                    
                    if self.take_control: # user take control
                        print("recording new branch")
                        self.traj_rec()
                        self.save(f"{self.filename}_branch_at_{self.time_index}")
                        return    
                    
                    if self.continue_feedback:
                        self.save_these_img_data()
                        break

                    if self.switch_flag:
                        
                        branch: str = self.generate_closest_branch_name()

                        self.load(branch)
                        self.time_index = 0
                        self.execute(retry_insertion_flag)
                        return
                
                if self.player_step(start, retry_insertion_flag) == 'stop':
                    break
                
                self.traj_rec_step()
                
                anomaly, suggested_branch = self.state_decider(self.get_observations())
                visualize_labelled_video_frame(self.curr_image, risk_flag=anomaly, risk_val=0.0)
                
                self.pause |= anomaly

                curr_branch: int = self.get_current_branch()
                # system switches branch
                if suggested_branch != curr_branch:
                    self.load(self.get_branch(suggested_branch))
                    self.time_index = 0
                    self.execute(retry_insertion_flag)

            except rclpy.exceptions.ROSInterruptException:
                break



    def go_to_time_index(self, time_index: int, linear: bool = False):
        """
        Args:
            time_index (int): target time index
            linear (bool, optional): Linear motion to time index, non-blocking. Defaults to False.
        """        
        quat_goal = list_2_quaternion(self.recorded_ori_wxyz[:, time_index])
        goal = pos_quat_2_pose_st(self.recorded_traj[:, time_index] + self.camera_correction, quat_goal)

        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'panda_link0'
        
        self.correct()

        if (self.recorded_gripper[0][time_index]-self.recorded_gripper[0][max(0,time_index-1)]) < -self.grip_open_width/2:
            self.grasp_gripper(self.recorded_gripper[0][time_index])
            time.sleep(0.1)

        if (self.recorded_gripper[0][time_index]-self.recorded_gripper[0][max(0,time_index-1)]) > self.grip_open_width/2:
            self.move_gripper(self.recorded_gripper[0][time_index])
            time.sleep(0.1)


        if linear:
            self.go_to_pose_ik(start)
            # self.go_to_pose(goal)
        else:
            self.move_to_pose_with_stampedpose(goal)
            # self.goal_pub.publish(goal)

 
    def finished_correctly(self):
        
        if self.time_index == self.recorded_traj.shape[1]:
            return True
        else:
            return False
        
    def save(self, file: str):
        if Path(trajectory_data.package_path + '/trajectories/' + str(file) + '.npz').is_file():
            print("WARNING FILE EXISTS!\nWARNING FILE EXISTS!\nWARNING FILE EXISTS!", flush=True)

        if self.final_transform is not None:
            self.recorded_traj, self.recorded_ori_wxyz = self.transform_traj_ori(self.recorded_traj, self.recorded_ori_wxyz, invert_tf(self.final_transform))

        np.savez(trajectory_data.package_path + '/trajectories/' + str(file) + '.npz',
                 traj=self.recorded_traj,
                 ori=self.recorded_ori_wxyz,
                 grip=self.recorded_gripper,
                 img=self.recorded_img, 
                 img_feedback_flag=self.recorded_img_feedback_flag,
                 spiral_flag=self.recorded_spiral_flag)
        
