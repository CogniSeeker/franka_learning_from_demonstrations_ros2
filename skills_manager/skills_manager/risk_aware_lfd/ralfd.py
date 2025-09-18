
import pathlib
from typing import Iterable, Tuple
import cv2, os
import numpy as np
import risk_estimation
from risk_estimation.scripts.result_img_save import sample_and_save_on_video

from video_embedding.utils import get_session, number_of_saved

from skills_manager.camera_feedback import image_process
from skills_manager.lfd import LfD
from skills_manager.risk_aware_lfd.raplayer import RiskAwarePlayer
from skills_manager.risk_aware_lfd.risk_policy import *
from skills_manager.feedback import Feedback, RiskAwareFeedback

import rclpy
from std_msgs.msg import Float32

import torch
from playsound import playsound
from threading import Thread

from video_embedding.utils import get_trajectory_path

class RALfD(RiskAwarePlayer, RiskAwareFeedback, LfD):

    def __init__(self, state_decider, estimator_risk_policy: str = 'ContinueRiskPolicy', risk_patience: int = 2, button_press_mode: str = "momentary"):
        """

        Args:
            estimator_risk_policy (str): What to do when Risk Estimator detects risk
            human_risk_policy (str): What to do when Human signalizes risk
            risk_patience (int): How many risky samples next to each other to trigger risk. Defaulting to 2.
            button_press_mode (str):
                "toggle" - "r" button to set risk flag, "q" button do reset risk flag
                "momentary" - Pressing a button (e.g. "r") sets flag and releasing the same button to resets flag
        """        
        super(RALfD, self).__init__()

        self.state_decider = state_decider

        self.risk_policy = eval(estimator_risk_policy)(risk_patience)
        
        self.haptic_buzz_pub = self.create_publisher(Float32, "/haptic_feedback", 5)

        self.button_press_mode = button_press_mode

    def skill_exists(self, name):
        return os.path.isfile(f"{get_trajectory_path()}/trajectories/{get_session()}/{name}.npz")

    def vibrate(self):
        self.haptic_buzz_pub.publish(Float32(0.5))

    def init_additional_flags(self):
        self.recorded_risk_flag = np.array([0])
        self.recorded_safe_flag = np.array([0])
        self.recorded_novelty_flag = np.array([0])
        self.recorded_recovery_phase = np.array([-1])

    def update_additional_flags(self):
        self.recorded_risk_flag = np.c_[self.recorded_risk_flag, self.risk_flag]
        self.recorded_safe_flag = np.c_[self.recorded_safe_flag, self.safe_flag]
        self.recorded_novelty_flag = np.c_[self.recorded_novelty_flag, self.novelty_flag]
        self.recorded_recovery_phase = np.c_[self.recorded_recovery_phase, self.recovery_phase]
    
    
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
                 traj=              self.exec_record['traj'],
                 ori=               self.exec_record['ori'],
                 grip=              self.exec_record['gripper'],
                 img=               self.exec_record['img'], 
                 img_feedback_flag= self.exec_record['img_feedback_flag'],
                 spiral_flag=       self.exec_record['spiral_flag'],
                 risk_flag=         self.exec_record['risk_flag'],
                 safe_flag=         self.exec_record['safe_flag'],
                 novelty_flag=      self.exec_record['novelty_flag'],
                 recovery_phase=    self.exec_record['recovery_phase'])
            
            if self.sl.video_embedder is not None and self.sl.risk_estimator is not None and self.sl.feature_extractor is not None:
                sample_and_save_on_video(f"{file}_trial_{n}", self.sl.video_embedder, self.sl.risk_estimator, self.sl.feature_extractor)

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
                 novelty_flag=self.recorded_novelty_flag,
                 recovery_phase=self.recorded_recovery_phase,)
    
    def load(self, file='last'):
        data = np.load(f"{get_trajectory_path()}/trajectories/{get_session()}/{file}.npz")
        self.recorded_traj = data['traj']
        self.recorded_ori_wxyz = data['ori']
        self.recorded_gripper = data['grip']
        self.recorded_img = data['img']
        self.recorded_img_feedback_flag = data['img_feedback_flag']
        self.recorded_spiral_flag = data['spiral_flag']

        if 'risk_flag' not in data.keys():
            self.recorded_risk_flag = np.zeros(data['grip'].shape)
        else:
            self.recorded_risk_flag = data['risk_flag']

        if 'safe_flag' not in data.keys():
            self.recorded_safe_flag = np.zeros(data['grip'].shape)
        else:
            self.recorded_safe_flag = data['safe_flag']

        if 'novelty_flag' not in data.keys():
            self.recorded_novelty_flag = np.zeros(data['grip'].shape)
        else:
            self.recorded_novelty_flag = data['novelty_flag']
        ## This check is temporary
        if 'recovery_phase' not in data.keys():
            self.recorded_recovery_phase = np.zeros(data['grip'].shape)
        else:    
            self.recorded_recovery_phase = data['recovery_phase']
        if self.final_transform is not None:
            self.recorded_traj, self.recorded_ori_wxyz = self.transform_traj_ori(self.recorded_traj, self.recorded_ori_wxyz, self.final_transform)
        
        self.filename=str(file) 

        

    def traj_rec_init(self):
        self.exec_record = {}
        
        self.exec_record['traj'] = self.curr_pos
        self.exec_record['ori'] = self.curr_ori_xyzw
        if self.gripper_width < self.grip_open_width * 0.9:
            self.grip_value = 0
        else:
            self.grip_value = self.grip_open_width
        self.exec_record['gripper']= self.grip_value
        self.exec_record['img_feedback_flag'] = np.array([0])
        self.exec_record['spiral_flag'] = np.array([0])

        resized_img_gray=image_process(self.curr_image, self.ds_factor,  self.row_crop_pct_top , self.row_crop_pct_bot,
                                        self.col_crop_pct_left, self.col_crop_pct_right)
        resized_img_msg = self.bridge.cv2_to_imgmsg(resized_img_gray)
        self.cropped_img_pub.publish(resized_img_msg)
        self.exec_record['img'] = resized_img_gray.reshape((1, resized_img_gray.shape[0], resized_img_gray.shape[1]))

        self.exec_record['risk_flag'] = np.array([0])
        self.exec_record['safe_flag'] = np.array([0])
        self.exec_record['novelty_flag'] = np.array([0])
        self.exec_record['recovery_phase'] = np.array([-1.0])

    def traj_rec_step(self):
        self.exec_record['traj'] = np.c_[self.exec_record['traj'], self.curr_pos]
        self.exec_record['ori']  = np.c_[self.exec_record['ori'], self.curr_ori_xyzw]
        self.exec_record['gripper'] = np.c_[self.exec_record['gripper'], self.grip_value]

        # print("shape current image", self.curr_image.shape)
        resized_img_gray=image_process(self.curr_image, self.ds_factor, self.row_crop_pct_top , self.row_crop_pct_bot,self.col_crop_pct_left, self.col_crop_pct_right)
        # print("shape current image", resized_img_gray.shape)
        resized_img_msg = self.bridge.cv2_to_imgmsg(resized_img_gray)
        # print("shape current image 2", resized_img_gray.shape)
        self.cropped_img_pub.publish(resized_img_msg)
        self.exec_record['img'] = np.r_[self.exec_record['img'], resized_img_gray.reshape((1, resized_img_gray.shape[0], resized_img_gray.shape[1]))]
        self.exec_record['img_feedback_flag'] = np.c_[self.exec_record['img_feedback_flag'], self.img_feedback_flag]
        self.exec_record['spiral_flag'] = np.c_[self.exec_record['spiral_flag'], self.spiral_flag]

        self.exec_record['risk_flag'] = np.c_[self.exec_record['risk_flag'], self.risk_flag]
        self.exec_record['safe_flag'] = np.c_[self.exec_record['safe_flag'], self.safe_flag]
        self.exec_record['novelty_flag'] = np.c_[self.exec_record['novelty_flag'], self.novelty_flag]
        self.exec_record['recovery_phase'] = np.c_[self.exec_record['recovery_phase'], self.recovery_phase]

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
        resized_img_gray=image_process(self.curr_image, self.ds_factor, self.row_crop_pct_top , self.row_crop_pct_bot,self.col_crop_pct_left, self.col_crop_pct_right)
        resized_img_gray.reshape((1, resized_img_gray.shape[0], resized_img_gray.shape[1]))

        last_image_square = cv2.resize(resized_img_gray, (64, 64), interpolation=cv2.INTER_AREA)
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

