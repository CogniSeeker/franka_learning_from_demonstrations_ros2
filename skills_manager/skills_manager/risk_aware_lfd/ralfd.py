
from dataclasses import dataclass
import pathlib, cv2, os, time, math
import numpy as np
from typing import Iterable, Tuple
from copy import deepcopy

from video_embedding.utils import get_session, number_of_saved

from skills_manager.lfd import LfD
from skills_manager.risk_aware_lfd.risk_policy import *
from skills_manager.feedback import RiskAwareFeedback
import trajectory_data
from panda_control.pose_transform_functions import pos_quat_2_pose_st, list_2_quaternion

import rclpy
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from lfd_msgs.srv import StringService
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

import torch
from playsound import playsound
from threading import Thread

from nocode_robot_programming.state_decision.utils import Filename

EXPECTED_TARGET_STATE_PUB_FREQ = 0.15 # sec

@dataclass
class Request():
    action: str = "play"
    timestep: int = 0
    task_name: str = ""
    valid_actions = {"play", "rec", "done"}

    def __post_init__(self):
        if self.action not in self.valid_actions:
            raise ValueError(f"Invalid action: '{self.action}'. Valid actions are: {self.valid_actions}")


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

        self.target_state = ""
        self.last_target_state = 0.0

        # self.haptic_buzz_pub = self.create_publisher(Float32, "/haptic_feedback", 5)

        self.retrain_client = self.create_client(StringService, 'state_decider_retrain', qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT), callback_group=self.callback_group)

    def retrain(self, task_name: str):
        self.retrain_client.call(StringService.Request(text=str(task_name)))

    def target_state_callback(self, msg):
        self.last_target_state = time.time()
        self.target_state = msg.data

    def skill_exists(self, name):
        return os.path.isfile(f"{trajectory_data.package_path}/trajectories/{get_session()}/{name}.npz")

    def init_additional_flags(self):
        self.recorded_risk_flag = np.array([0])
        self.recorded_safe_flag = np.array([0])
        self.recorded_novelty_flag = np.array([0])

    def update_additional_flags(self):
        self.recorded_risk_flag = np.c_[self.recorded_risk_flag, self.risk_flag]
        self.recorded_safe_flag = np.c_[self.recorded_safe_flag, self.safe_flag]
        self.recorded_novelty_flag = np.c_[self.recorded_novelty_flag, self.novelty_flag]

    # def get_observations(self) -> Tuple[torch.tensor, None, None, None, torch.tensor]:
    #     """Collect current observations as Enum list:
    #     [1. Image, 2. Risk, 3. Safe, 4. Novelty, 5. Time Phase]

    #     Returns:
    #         Tuple[torch.tensor, None, None, None, torch.tensor]
    #     """

    #     last_image_square = cv2.resize(self.pub_rec_image(), (64, 64), interpolation=cv2.INTER_AREA)
    #     last_image_square = last_image_square[np.newaxis, np.newaxis, :, :] # (x, 1, 64, 64)

    #     frame_number = np.array([self.time_phase]) # (x, 1)
    #     frame_number = frame_number[np.newaxis,:]

    #     return [
    #         torch.tensor(last_image_square, dtype=torch.float32).cuda(), # 1. Image
    #         None, # 2. Risk Label flag
    #         None, # 3. Safe Label flag
    #         None, # 4. Novelty Label flag
    #         torch.tensor(frame_number, dtype=torch.float32).cuda() # 5. Frame number normalized (0-1)
    #     ]

    def play_skill(self, name_skill, object_template_name, localize_box=True):
        if localize_box:
            if not self.set_localizer_client.wait_for_service(timeout_sec=5.0):
                raise Exception("Service not available after waiting")
            ret = self.set_localizer_client.call(SetTemplate.Request(template_name=object_template_name))
            if not ret.success:
                print("Returned because localizer not succesful", flush=True)
                return
            self.move_template_start()
            self.active_localizer_client.call(Trigger.Request())
            self.compute_final_transform() 

        try:
            request = Request(task_name = name_skill)
            while request.action != "done":
                print(f"New request: {request}")
                if request.action == "play":
                    self.show(request.task_name)
                    self.load(request.task_name)
                    new_request = self.execute()

                    # Proposal FIX: correct label around DS
                    # --------------------------
                    # anomaly was triggered with new potential DS, 
                    # we need to make sure here that DS window is saved with correct label
                    if new_request.action in ["play", "rec"]:
                        window_size = 10
                        # 1.) Saving DS with an anomaly label
                        self.save(request.task_name, is_exec_trial=True, split=slice(None, -window_size))
                        # 2.) Saving the initial part with the original label
                        self.save(new_request.task_name, is_exec_trial=True, split=slice(-window_size, None))
    
                    else:
                        self.save(request.task_name, is_exec_trial=True)
                
                elif request.action == "rec":
                    self.traj_rec()
                    self.save(request.task_name, is_exec_trial=False)
                    new_request = Request(action="done")

                else: raise Exception("action not valid")

                request = new_request # update request

        except KeyboardInterrupt:
            print("Keyboard interrupted", flush=True)
        return

    def execute(self) -> Request:
        ''' Has trajectory at self.loaded_traj, self.loaded_ori
        '''
        self.signalizer.signalize_execution()
        self.player_init()
        self.recorded_risk_flag = np.array([0])
        self.recorded_safe_flag = np.array([0])
        self.recorded_novelty_flag = np.array([0])

        while (time.time() - self.last_target_state) > EXPECTED_TARGET_STATE_PUB_FREQ:
            print("waiting for target state", flush=True)
            self.pub_rec_image()
            time.sleep(1.0)

        while self.time_index <( self.loaded_traj.shape[1]) and rclpy.ok() and not self.end:
            try:
                t0 = time.perf_counter()
                vel = 0
                init_pos = deepcopy(self.curr_pos)
                while(self.pause):
                    self.r.sleep()
                    
                    if self.end or not rclpy.ok(): # user take control
                        offset = Filename(self.filename).offset
                        task = Filename(self.filename).task
                        save_name = f"{task}_branch_from_{offset}_at_{self.time_index}"
                        return Request(action="rec", timestep=self.time_index, task_name=save_name)
                    
                    vel = math.sqrt((self.curr_pos[0]-init_pos[0])**2 + (self.curr_pos[1]-init_pos[1])**2 + (self.curr_pos[2]-init_pos[2])**2)
                    if vel > 0.02:
                        print("Robot was moved manually! Continuing")
                        self.pause = False

                self.recorded_risk_flag = np.c_[self.recorded_risk_flag, self.risk_flag]
                self.recorded_safe_flag = np.c_[self.recorded_safe_flag, self.safe_flag]
                self.recorded_novelty_flag = np.c_[self.recorded_novelty_flag, self.novelty_flag]
                self.player_step()
                
                suggested_branch: str = self.target_state
                curr_branch: str = self.filename

                print(f"[step {self.time_index:3}]({int(round(1.0 / (time.perf_counter()-t0))):3}S/s) now: {curr_branch} -> suggested: {self.target_state}. anomaly: {suggested_branch == 'anomaly'}")

                if suggested_branch == "anomaly":
                    self.pause = True
                    continue

                if suggested_branch != curr_branch:
                    return Request(action="play", timestep=self.time_index, task_name=suggested_branch)
            except rclpy.exceptions.ROSInterruptException:
                print("manually interrupted", flush=True)
        
        if self.time_index < self.loaded_traj.shape[1]: # not finished ending
            offset = Filename(self.filename).offset
            task = Filename(self.filename).task
            save_name = f"{task}_branch_from_{offset}_at_{self.time_index}"
            return Request(action="rec", timestep=self.time_index, task_name=save_name)

        self.signalizer.signalize_idle()
        return Request(action="done", timestep=self.time_index, task_name=self.filename)

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
        
    def save(self, file: str='last', is_exec_trial: bool = False, 
            # additional
            tag: str = "",
            split: slice | None = None,  
            ):
        """Saves Trajectory data to file
        If video_embedder and risk_estimator specified, then it is sampled and saved as video.

        Args:
            file (str | Iterable[str]): video file name to be saved.
            is_exec_trial (bool, optional): Loads trajectory data from execution. Defaults to False.
                Initial trajectory is safe by default (np.ones)
                If some risk_flags observed, then safe_flag is 0
        """        
        if (isinstance(file, Iterable) and not isinstance(file, str)):
            file = file[0]
        
        if is_exec_trial:
            n = number_of_saved(file, "trial") # trials 0, ..., n-1 exists
            added_file_suffix = f"_trial_{n}"
        else:
            added_file_suffix = ""

        pathlib.Path(f"{trajectory_data.package_path}/trajectories/{get_session()}").mkdir(parents=True, exist_ok=True)
        if split is None:
            np.savez(f"{trajectory_data.package_path}/trajectories/{get_session()}/{file}{added_file_suffix}.npz",
                    traj=              self.recorded_traj,
                    ori=               self.recorded_ori_wxyz,
                    grip=              self.recorded_gripper,
                    img=               self.recorded_img, 
                    img_feedback_flag= self.recorded_img_feedback_flag,
                    spiral_flag=       self.recorded_spiral_flag,
                    risk_flag=         self.recorded_risk_flag,
                    safe_flag=         self.recorded_safe_flag,
                    novelty_flag=      self.recorded_novelty_flag,
                    tag = tag,
                )
        else:
            assert isinstance(split, slice)
            if len(self.recorded_img[split,:,:]) == 0:
                print("short demonstration, saving only DS")
                return
            
            np.savez(f"{trajectory_data.package_path}/trajectories/{get_session()}/{file}{added_file_suffix}.npz",
                traj=              self.recorded_traj[:, split],
                ori=               self.recorded_ori_wxyz[:, split],
                grip=              self.recorded_gripper[:, split],
                img=               self.recorded_img[split,:,:], 
                img_feedback_flag= self.recorded_img_feedback_flag[:, split],
                spiral_flag=       self.recorded_spiral_flag[:, split],
                risk_flag=         self.recorded_risk_flag[:, split],
                safe_flag=         self.recorded_safe_flag[:, split],
                novelty_flag=      self.recorded_novelty_flag[:, split],
                tag = tag,
            )

            
    def load(self, file='last'):
        data = np.load(f"{trajectory_data.package_path}/trajectories/{get_session()}/{file}.npz")
        self.loaded_traj = data['traj']
        self.loaded_ori_wxyz = data['ori']
        self.loaded_gripper = data['grip']
        self.loaded_img = data['img']
        self.loaded_img_feedback_flag = data['img_feedback_flag']
        self.loaded_spiral_flag = data['spiral_flag']
        if 'tag' not in data.keys():
            self.tag = ""
        else:
            self.tag = str(data['tag'])

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

    # def vibrate(self):
    #     self.haptic_buzz_pub.publish(Float32(data=0.5))

    def communicate_risk_detected(self):
        music_thread = Thread(target=self.play_risk_detected)
        music_thread.start()
        # self.vibrate()

    def play_risk_detected(self):
        for _ in range(3):
            playsound('/usr/share/sounds/gnome/default/alerts/glass.ogg')
