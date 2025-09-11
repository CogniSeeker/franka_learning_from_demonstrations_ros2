



import time
from risk_estimation.scripts.result_img_save import sample_and_save_on_video
import numpy as np
from risk_estimation.models.safety_layer import SafetyLayer
from video_embedding.utils import visualize_labelled_video_frame
from skills_manager.player import Player
import rclpy

from panda_control.pose_transform_functions import orientation_2_quaternion, pose_st_2_transformation, position_2_array, pos_quat_2_pose_st, transformation_2_pose, transform_pose, list_2_quaternion, transform_pos_ori


class RiskAwarePlayer(Player):
    """Demonstration player with risk-awareness capabilities
    """    
    def __init__(self):
        super(RiskAwarePlayer, self).__init__()

    def execute(self, retry_insertion_flag=0):

        # has trajectory at self.recorded_traj, self.recorded_ori


        print("Executing: ", self.filename)

        self.sl = SafetyLayer(skill_name = self.filename, model_not_found_is_ok=True)  

        replay = False 
        switch = False
        start = self.player_init()
        self.traj_rec_init()

        print("Starting execution", flush=True)
        self.end = False
        while self.time_index <( self.recorded_traj.shape[1]) and rclpy.ok() and not self.end:
            print("Execution time index: ", self.time_index, flush=True)
            try:
                while(self.pause):
                    self.r.sleep()  
                if self.player_step(start, retry_insertion_flag) == 'stop':
                    break
                
                self.traj_rec_step()
                
                # system_risk_pred, risk_val = self.sl.get_estimated_risk(self.get_observations())
                # visualize_labelled_video_frame(self.curr_image, risk_flag=system_risk_pred, risk_val=risk_val)
                
                # action, alpha = self.risk_policy.do(self, system_risk_pred, self.risk_flag) # system, human flag

                if self.risk_flag:
                    print("recording new demonstration")
                    self.traj_rec()
                    self.save(self.filename+"_branch_0")
                    return
                elif self.switch_flag is not None:
                    self.load(self.switch_flag)
                    self.time_index = 0
                    self.execute(retry_insertion_flag)
                    return

            except rclpy.exceptions.ROSInterruptException:
                break

            if hasattr(self, 'input_modality') and self.input_modality is not None:
                self.input_modality.step()


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

 
    def recovery_mission(self, alpha: float, linear: bool = True):
        """

        Args:
            alpha (float): taget frame as normalized time phase (0,1)
            linear (bool, optional): Defaults to True.
                If true: Going back via linear motion
                If false: Going back via same path - going backwards
        """        

        target_time_index = int(alpha * self.recorded_traj.shape[1])
            
        if linear:
            self.go_to_time_index(target_time_index, linear=True)
        else:
            while (self.time_index != target_time_index) and rclpy.ok() and not self.end:
                try:
                    system_risk_pred = self.sl.get_estimated_risk(self.get_observations())

                    next_time_index = self.time_index + np.clip(int(target_time_index) - int(self.time_index), -1, 1, dtype=int)
                    at_target = self.time_index == target_time_index
                    print(f"Now time index: {self.time_index}, {target_time_index}, next_time index: {next_time_index}, action: {at_target}")

                    self.go_to_time_index(next_time_index)

                    self.time_index = next_time_index

                    visualize_labelled_video_frame(self.curr_image, risk_flag=system_risk_pred)
                except rclpy.exceptions.ROSInterruptException:
                    break


    def finished_correctly(self):
        
        if self.time_index == self.recorded_traj.shape[1]:
            return True
        else:
            return False
        
    def save(self):
        sample_and_save_on_video(
            video_name=self.sl.video_embedder.name,
            video_embedder=self.sl.video_embedder,
            risk_estimator=self.sl.risk_estimator,
            risk_estimator2=self.sl.risk_estimator2,
            features=self.sl.feature_extractor,
            train_dataloader = None,
            folder="autogen"
        )

class InteractivePlayer(RiskAwarePlayer):
    def __init__(self):
        super(InteractivePlayer, self).__init__()
        
        self.target_time_index = 0
        self.at_target_previously = True
    
    def set_stiffness_once(self, at_target):
        if at_target == self.at_target_previously:
            return
        else:
            print("setting stiffness!")
            if at_target:
                self.set_stiffness(0, 0, 250, 0, 0, 0, 0)
            else:
                self.set_stiffness(3000, 3000, 3000, 40, 40, 40, 0)

        self.at_target_previously = at_target

    def loop(self):
        
        self.sl = SafetyLayer(skill_name = self.filename, enable_risk_estimator=True)  
        start = self.player_init()
        self.traj_rec_init()

        retry_insertion_flag = 0

        while rclpy.ok() and not self.end:
            try:

                if self.player_step(start, retry_insertion_flag) == 'stop':
                    break
                
                self.traj_rec_step()

                system_risk_pred, risk_val = self.sl.get_estimated_risk(self.get_observations())
                
                next_time_index = int(self.time_index + np.clip(int(self.target_time_index) - int(self.time_index), -1, 1, dtype=int))
                at_target = self.time_index == self.target_time_index
                # print(f"Now time index: {self.time_index}, {self.target_time_index}, next_time index: {next_time_index}, action: {at_target}")

                self.set_stiffness_once(at_target)

                if not at_target:
                    self.go_to_time_index(next_time_index)
                else:
                    time.sleep(0.1)
                
                self.time_index = next_time_index

                # visualize image
                visualize_labelled_video_frame(self.curr_image, risk_flag=system_risk_pred, risk_val=risk_val)

                
            except rclpy.exceptions.ROSInterruptException:
                break