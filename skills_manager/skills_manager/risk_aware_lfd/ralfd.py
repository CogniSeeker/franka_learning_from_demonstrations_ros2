
import pathlib
from typing import Iterable, Tuple
import cv2, os
import numpy as np
import risk_estimation
from risk_estimation.scripts.result_img_save import sample_and_save_on_video

from video_embedding.utils import get_session, number_of_saved

from skills_manager.camera_feedback import image_process
from skills_manager.lfd import LfD
from skills_manager.risk_aware_lfd.risk_policy import *
from skills_manager.feedback import Feedback, RiskAwareFeedback

import rclpy
from std_msgs.msg import Float32, String

import torch
from playsound import playsound
from threading import Thread

from video_embedding.utils import get_trajectory_path

import time
import trajectory_data

from panda_control.pose_transform_functions import orientation_2_quaternion, pose_st_2_transformation, position_2_array, pos_quat_2_pose_st, transformation_2_pose, transform_pose, list_2_quaternion, transform_pos_ori, invert_tf

from pathlib import Path

class RALfD(RiskAwareFeedback, LfD):

    def __init__(self, estimator_risk_policy: str = 'ContinueRiskPolicy', risk_patience: int = 2):
        """
        Args:
            estimator_risk_policy (str): What to do when Risk Estimator detects risk
            human_risk_policy (str): What to do when Human signalizes risk
            risk_patience (int): How many risky samples next to each other to trigger risk. Defaulting to 2.
        """        
        super(RALfD, self).__init__()

        self.create_subscription(String, "/target_state", self.target_state_callback, 5)

        self.risk_policy = eval(estimator_risk_policy)(risk_patience)
        
        self.haptic_buzz_pub = self.create_publisher(Float32, "/haptic_feedback", 5)

        self.target_state = ""
        self.last_target_state = 0.0

    def target_state_callback(self, msg):
        self.last_target_state = time.time()
        self.target_state = msg.data

    def skill_exists(self, name):
        return os.path.isfile(f"{get_trajectory_path()}/trajectories/{get_session()}/{name}.npz")

    def vibrate(self):
        self.haptic_buzz_pub.publish(Float32(0.5))

    def init_additional_flags(self):
        self.recorded_risk_flag = np.array([0])
        self.recorded_safe_flag = np.array([0])
        self.recorded_novelty_flag = np.array([0])

    def update_additional_flags(self):
        self.recorded_risk_flag = np.c_[self.recorded_risk_flag, self.risk_flag]
        self.recorded_safe_flag = np.c_[self.recorded_safe_flag, self.safe_flag]
        self.recorded_novelty_flag = np.c_[self.recorded_novelty_flag, self.novelty_flag]

    def get_time_phase(self, default_traj_len: int = 400):
        """ (Time Frame / trajectory_len)

        Args:
            default_traj_len (int, optional): Is used when trajectory recording. Defaults to 400.
        """        
        try:
            trajectory_len = self.trajectory_len
        except AttributeError:
            trajectory_len = default_traj_len

        try:
            self.time_index
        except AttributeError:
            self.time_index = 0.0

        return self.time_index / trajectory_len


    def get_observations(self) -> Tuple[torch.tensor, None, None, None, torch.tensor]:
        """Collect current observations as Enum list:
        [1. Image, 2. Risk, 3. Safe, 4. Novelty, 5. Time Phase]

        Returns:
            Tuple[torch.tensor, None, None, None, torch.tensor]
        """

        last_image_square = cv2.resize(self.get_rec_image(), (64, 64), interpolation=cv2.INTER_AREA)
        last_image_square = last_image_square[np.newaxis, np.newaxis, :, :] # (x, 1, 64, 64)

        frame_number = np.array([self.get_time_phase()]) # (x, 1)
        frame_number = frame_number[np.newaxis,:]

        return [
            torch.tensor(last_image_square, dtype=torch.float32).cuda(), # 1. Image
            None, # 2. Risk Label flag
            None, # 3. Safe Label flag
            None, # 4. Novelty Label flag
            torch.tensor(frame_number, dtype=torch.float32).cuda() # 5. Frame number normalized (0-1)
        ]

    def communicate_recovery_action(self):
        music_thread = Thread(target=self.play_recovery_action)
        music_thread.start()

    def play_recovery_action(self):
        playsound(f'{risk_estimation.path}/sounds/device-added.oga')
    
    def communicate_risk_detected(self):
        music_thread = Thread(target=self.play_risk_detected)
        music_thread.start()
        # self.vibrate()

    def play_risk_detected(self):
        playsound(f'{risk_estimation.path}/sounds/dialog-warning.oga')
        playsound(f'{risk_estimation.path}/sounds/dialog-warning.oga')
        playsound(f'{risk_estimation.path}/sounds/dialog-warning.oga')

    def get_current_branch(self):
        try:
            return int(self.filename.split("_")[-1])
        except ValueError:
            return 0

    def execute(self):
        ''' Has trajectory at self.loaded_traj, self.loaded_ori
        '''
        start = self.player_init()
        self.recorded_risk_flag = np.array([0])
        self.recorded_safe_flag = np.array([0])
        self.recorded_novelty_flag = np.array([0])

        while self.time_index <( self.loaded_traj.shape[1]) and rclpy.ok() and not self.end:
            try:
                t0 = time.perf_counter()
                while(self.pause):
                    self.r.sleep()
                    
                    if self.take_control: # user take control
                        print("branch request")
                        return ("branch request at", self.time_index, f"{self.filename}_branch_at_{self.time_index}")
                        
                    # if self.continue_feedback:
                    #     self.save_these_img_data()
                    #     break

                    # if self.switch_flag:
                        
                    #     branch: str = self.generate_closest_branch_name()

                    #     self.load(branch)
                    #     self.time_index = 0
                    #     self.execute()
                    #     return


                self.recorded_risk_flag = np.c_[self.recorded_risk_flag, self.risk_flag]
                self.recorded_safe_flag = np.c_[self.recorded_safe_flag, self.safe_flag]
                self.recorded_novelty_flag = np.c_[self.recorded_novelty_flag, self.novelty_flag]
                if self.player_step(start) == 'stop':
                    break

                # anomaly, suggested_branch = self.target_state
                anomaly = False
                suggested_branch = 0
                print(self.target_state)

                self.pause |= anomaly

                curr_branch: int = self.get_current_branch()
                # system switches branch
                print(f"[step {self.time_index}] now: {curr_branch} -> suggested: {suggested_branch}. anomaly: {anomaly}")
                if False: #int(suggested_branch) != int(curr_branch):
                    self.load(self.get_branch(suggested_branch))
                    self.time_index = 0
                    self.execute()

                print(f"{round(1.0 / (time.perf_counter()-t0))} samples per second")
            except rclpy.exceptions.ROSInterruptException:
                break

    def go_to_time_index(self, time_index: int, linear: bool = False):
        """
        Args:
            time_index (int): target time index
            linear (bool, optional): Linear motion to time index, non-blocking. Defaults to False.
        """        
        quat_goal = list_2_quaternion(self.loaded_ori_wxyz[:, time_index])
        goal = pos_quat_2_pose_st(self.loaded_traj[:, time_index] + self.camera_correction, quat_goal)

        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'panda_link0'
        
        self.correct()

        self.handle_gripper(self.loaded_gripper[0][self.time_index])

        if linear:
            self.go_to_pose_ik(start)
            # self.go_to_pose(goal)
        else:
            self.move_to_pose_with_stampedpose(goal)
            # self.goal_pub.publish(goal)

 
    def finished_correctly(self):
        
        if self.time_index == self.loaded_traj.shape[1]:
            return True
        else:
            return False
        
    def save(self, file: str='last', risk_exec_trial: bool = False):
        """Saves Trajectory data to file
        If video_embedder and risk_estimator specified, then it is sampled and saved as video.

        Args:
            file (str | Iterable[str]): video file name to be saved.
            risk_exec_trial (bool, optional): Loads trajectory data from execution. Defaults to False.
                Initial trajectory is safe by default (np.ones)
                If some risk_flags observed, then safe_flag is 0
        """        
        if (isinstance(file, Iterable) and not isinstance(file, str)):
            file = file[0]
        
        if risk_exec_trial:
            n = number_of_saved(file, "trial") # trials 0, ..., n-1 exists

            pathlib.Path(f"{get_trajectory_path()}/trajectories/{get_session()}").mkdir(parents=True, exist_ok=True)
            np.savez(f"{get_trajectory_path()}/trajectories/{get_session()}/{file}_trial_{n}.npz",
                 traj=              self.recorded_traj,
                 ori=               self.recorded_ori,
                 grip=              self.recorded_gripper,
                 img=               self.recorded_img, 
                 img_feedback_flag= self.recorded_img_feedback_flag,
                 spiral_flag=       self.recorded_spiral_flag,
                 risk_flag=         self.recorded_risk_flag,
                 safe_flag=         self.recorded_safe_flag,
                 novelty_flag=      self.recorded_novelty_flag,
                )
            
            # if self.sl.video_embedder is not None and self.sl.risk_estimator is not None and self.sl.feature_extractor is not None:
            #     sample_and_save_on_video(f"{file}_trial_{n}", self.sl.video_embedder, self.sl.risk_estimator, self.sl.feature_extractor)

        else:
            self.recorded_safe_flag = np.ones((self.recorded_safe_flag.shape))
            self.recorded_safe_flag[self.recorded_risk_flag != 0] = 0

            pathlib.Path(f"{get_trajectory_path()}/trajectories/{get_session()}").mkdir(parents=True, exist_ok=True)
            np.savez(f"{get_trajectory_path()}/trajectories/{get_session()}/{file}.npz",
                 traj=self.recorded_traj,
                 ori=self.recorded_ori_wxyz,
                 grip=self.recorded_gripper,
                 img=self.recorded_img, 
                 img_feedback_flag=self.recorded_img_feedback_flag,
                 spiral_flag=self.recorded_spiral_flag,
                 risk_flag=self.recorded_risk_flag,
                 safe_flag=self.recorded_safe_flag,
                 novelty_flag=self.recorded_novelty_flag,)
            
    def load(self, file='last'):
        data = np.load(f"{get_trajectory_path()}/trajectories/{get_session()}/{file}.npz")
        self.loaded_traj = data['traj']
        self.loaded_ori_wxyz = data['ori']
        self.loaded_gripper = data['grip']
        self.loaded_img = data['img']
        self.loaded_img_feedback_flag = data['img_feedback_flag']
        self.loaded_spiral_flag = data['spiral_flag']

        if 'risk_flag' not in data.keys():
            self.loaded_risk_flag = np.zeros(data['grip'].shape)
        else:
            self.loaded_risk_flag = data['risk_flag']

        if 'safe_flag' not in data.keys():
            self.loaded_safe_flag = np.zeros(data['grip'].shape)
        else:
            self.loaded_safe_flag = data['safe_flag']

        if 'novelty_flag' not in data.keys():
            self.loaded_novelty_flag = np.zeros(data['grip'].shape)
        else:
            self.loaded_novelty_flag = data['novelty_flag']

        if self.final_transform is not None:
            self.loaded_traj, self.loaded_ori_wxyz = self.transform_traj_ori(self.loaded_traj, self.loaded_ori_wxyz, self.final_transform)
        
        self.filename=str(file) 