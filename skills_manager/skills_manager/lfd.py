#%%
#!/usr/bin/env python
import time, math, time
from quaternion import quaternion
import numpy as np
import tf2_ros
from skills_manager.camera_feedback import CameraFeedback, image_process
from geometry_msgs.msg import PoseStamped
from panda_control import Panda, SpinningRosNode
from skills_manager.feedback import Feedback
from skills_manager.insertion import Insertion
from skills_manager.transfom import Transform 
from panda_control.pose_transform_functions import position_2_array, pos_quat_2_pose_st, list_2_quaternion, invert_tf
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from skills_manager.ros_param_manager import get_remote_parameters
import trajectory_data
from copy import deepcopy
import spatialmath as sm

from lfd_msgs.srv import SetTemplate
from std_srvs.srv import Trigger
from std_msgs.msg import Int32

from trajectory_data.skill_visualizer import show_skill
class SkillVis():
    def show_skill(self, name_skill: str):
        if name_skill[-4:] != ".npz":
            name_skill = name_skill + ".npz"
        show_skill(name_skill)

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

class LfD(Feedback, Panda, Insertion, Transform, CameraFeedback, SpinningRosNode, SkillVis):
    def __init__(self):
        super(LfD, self).__init__()
        self.freq = 10
        self.r = self.create_rate(self.freq)

        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.curr_image = None
        self.recorded_traj = None
        self.recorded_ori_wxyz = None
        self.loaded_traj = None
        self.loaded_ori_qxyz = None

        self.end = False

        self.insertion_force_threshold = 6
        self.retry_counter = 0

        self.set_localizer_client = self.create_client(SetTemplate, 'set_localizer', callback_group=self.callback_group)
        self.active_localizer_client = self.create_client(Trigger, 'active_localizer', qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT), callback_group=self.callback_group)
        self.start_publishing_scene_call = self.create_client(Trigger, 'start_publishing_scene', qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT), callback_group=self.callback_group)
        self.stop_publishing_scene_call = self.create_client(Trigger, 'stop_publishing_scene', qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT), callback_group=self.callback_group)

        self.timestep_pub = self.create_publisher(Int32, "/timestep", 5)

        time.sleep(1)

    @property
    def loaded_traj_n(self):
        return 0 if self.loaded_traj is None else self.loaded_traj.shape[1]

    def traj_rec(self, trigger: float = 0.005, roll_redution_alpha: float = 0.4):
        """ Demonstrate a trajectory with either joystic, gestures, or kinesthetic teaching.
        Fills: 
            self.recorded_traj
            self.recorded_ori_wxyz
            self.recorded_gripper
            self.recorded_img_feedback_flag
            self.recorded_spiral_flag
            ...

        Args:
            roll_reduction_alpha: float. When controlled externally (joystick/gestures), we let roll->0 as the user cannot control it.
        """
        self.end = False
        self.pause = False
        self.set_stiffness(0,0,0,0,0,0,0)

        init_pos = self.curr_pos
        vel = 0 
        init_feedback = deepcopy(self.feedback)
        print("Move robot to start recording.", flush=True)
        while vel < trigger:
            self.r.sleep()
            set_stifness_once = sum(self.feedback - init_feedback) != 0  # feedback changed -> not kinesthetic teaching
            if set_stifness_once:
                time.sleep(0.1)
                print("Externally control, setting stiffness!", flush=True)
                if self.modality_in_control == 'joystick':
                    self.set_stiffness(1000, 1000, 1000, 80, 80, 80, 0)
                elif self.modality_in_control == 'gestures':
                    self.set_stiffness(500, 100, 100, 80, 80, 80, 0)
                else:
                    raise Exception("modality_in_control not defined")
                break
            vel = math.sqrt((self.curr_pos[0]-init_pos[0])**2 + (self.curr_pos[1]-init_pos[1])**2 + (self.curr_pos[2]-init_pos[2])**2)

        self.recorded_traj = self.curr_pos
        self.recorded_ori_wxyz = self.curr_ori_wxyz
        self.recorded_gripper= self.grip_value
        self.recorded_img_feedback_flag = np.array([0])
        self.recorded_spiral_flag = np.array([0])
        self.init_additional_flags()
        self.recorded_img = self.get_rec_image()

        print("Recording started. Press e to stop.")
        while not self.end:
            while(self.pause):
                print("Paused", flush=True)
                time.sleep(0.5)
            t0 = time.perf_counter()
            self.recorded_traj = np.c_[self.recorded_traj, self.curr_pos]
            self.recorded_ori_wxyz  = np.c_[self.recorded_ori_wxyz, self.curr_ori_wxyz]
            self.recorded_gripper = np.c_[self.recorded_gripper, self.grip_value]
            self.recorded_img = np.r_[self.recorded_img, self.get_rec_image()]
            
            self.recorded_img_feedback_flag = np.c_[self.recorded_img_feedback_flag, self.img_feedback_flag]
            self.recorded_spiral_flag = np.c_[self.recorded_spiral_flag, self.spiral_flag]
            
            # Joystick increments (radians): Z (yaw), Y (pitch)
            dqz = sm.UnitQuaternion.Rz(self.feedback[3])
            dqy = sm.UnitQuaternion.Ry(self.feedback[4])

            cx, cy, cz, cw = self.curr_ori_xyzw
            q_curr = sm.UnitQuaternion([cw, cx, cy, cz])  # [w,x,y,z]
            goal = PoseStamped()
            goal.pose.position = Point(
                x=self.curr_pos[0] + self.feedback[0],
                y=self.curr_pos[1] + self.feedback[1],
                z=self.curr_pos[2] + self.feedback[2],
            )
            q_pre = q_curr * dqz * dqy
            trans_speed = np.linalg.norm(self.feedback[0:3])
            alpha_when_moving = 0.02
            alpha = alpha_when_moving + (roll_redution_alpha - alpha_when_moving) * np.exp(-4.0*trans_speed)

            q_goal = Transform.step_toward_roll(q_pre, alpha=alpha)
            qx, qy, qz, qw = q_goal.vec_xyzs  # returns (x,y,z,w)
            goal.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

            self.move_to_pose_with_stampedpose(goal)
            if self.feedback_gripper == "grasp":
                print("closing gripper")

                if not self.gripper_state.is_grasped:
                    self.grasp_gripper(0)
                time.sleep(0.1)
                self.feedback_gripper = ""

            if self.feedback_gripper == "open":
                print("open gripper")
                self.move_gripper(0.08)
                time.sleep(0.1)
                self.feedback_gripper = ""

            self.update_additional_flags()
            if (time.perf_counter() - t0) * 0.8 > (1.0 / self.freq):
                print(f"WARN: trajectory recording at {round(1.0 / (time.perf_counter() - t0))} samples per sec")
            self.r.sleep()

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"

        goal.pose.position.x = self.curr_pos[0]
        goal.pose.position.y = self.curr_pos[1]
        goal.pose.position.z = self.curr_pos[2]
        
        goal.pose.orientation.w = self.curr_ori_wxyz[0]
        goal.pose.orientation.x = self.curr_ori_wxyz[1]
        goal.pose.orientation.y = self.curr_ori_wxyz[2]
        goal.pose.orientation.z = self.curr_ori_wxyz[3]
        
        self.move_to_pose_with_stampedpose(goal)

        self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, self.K_ori, 0)
        self.get_logger().info("Ending trajectory recording")

    def save(self, file='last'):
        if self.recorded_traj is None or self.recorded_ori_wxyz is None:
            print("Cannot save, recording is empty", flush=True)
            return

        if self.final_transform is not None:
            self.recorded_traj, self.recorded_ori_wxyz = self.transform_traj_ori(self.recorded_traj, self.recorded_ori_wxyz, invert_tf(self.final_transform))

        np.savez(trajectory_data.package_path + '/trajectories/' + str(file) + '.npz',
                 traj=self.recorded_traj,
                 ori=self.recorded_ori_wxyz,
                 grip=self.recorded_gripper,
                 img=self.recorded_img, 
                 img_feedback_flag=self.recorded_img_feedback_flag,
                 spiral_flag=self.recorded_spiral_flag)
    
    def load(self, file='last'):
        data = np.load(trajectory_data.package_path + '/trajectories/' + str(file) + '.npz')
        self.loaded_traj = data['traj']
        self.loaded_ori_wxyz = data['ori']
        self.loaded_gripper = data['grip']
        self.loaded_img = data['img']
        self.loaded_img_feedback_flag = data['img_feedback_flag']
        self.loaded_spiral_flag = data['spiral_flag']
        if self.final_transform is not None:
            self.loaded_traj, self.loaded_ori_wxyz = self.transform_traj_ori(self.loaded_traj, self.loaded_ori_wxyz, self.final_transform)
        
        self.filename=str(file)

    def init_additional_flags(self):
        pass
    def update_additional_flags(self):
        pass

    def localize(self, object_template_name: str = ""):
        """ DRAFT """
        if object_template_name == "":
            print("No given object_template_name", flush=True)
            return False

        if not self.set_localizer_client.wait_for_service(timeout_sec=5.0):
            raise Exception("Service not available after waiting")
        ret = self.set_localizer_client.call(SetTemplate.Request(template_name=object_template_name))
        if not ret.success:
            print("Returned because localizer not succesful", flush=True)
            return False
        self.move_template_start()
        self.active_localizer_client.call(Trigger.Request())
        self.compute_final_transform() 

    def play_skill(self, name_skill, object_template_name, localize_box=True):
        if localize_box:
            if not self.set_localizer_client.wait_for_service(timeout_sec=5.0):
                raise Exception("Service not available after waiting")
            ret = self.set_localizer_client.call(SetTemplate.Request(template_name=object_template_name))
            if not ret.success:
                print("Returned because localizer not succesful", flush=True)
                return False
            self.move_template_start()
            self.active_localizer_client.call(Trigger.Request())
            self.compute_final_transform() 
        else:
            pass # move to start?

        try:
            self.load(name_skill)
            print(f"Execution", flush=True)
            return self.execute()
        except KeyboardInterrupt:
            return False
        return True

    def move_template_start(self):
        pose = get_remote_parameters(self, param_names=[
            "position_x", "position_y", "position_z", 
            "orientation_w", "orientation_x", "orientation_y", "orientation_z"],
            server="localizer_node")

        assert pose[2] > 0.02
        pos_array = pose[:3]
        quat_wxyz = quaternion(pose[3], pose[4], pose[5], pose[6])
        
        goal = pos_quat_2_pose_st(pos_array, quat_wxyz)
        goal.header.stamp = self.get_clock().now().to_msg()

        print(f"Move to start: x={goal.pose.position.x} y={goal.pose.position.y} y={goal.pose.position.z}", flush=True)
        
        self.go_to_pose_ik(goal)    

        if not np.allclose(self.curr_pos, pose[:3], atol=2e-3) or not np.allclose(self.curr_ori_wxyz, pose[3:], atol=2e-2):
            self.set_stiffness(2000,2000,2000,150,150,150,0)
            self.go_to_pose_ik(goal)    
            self.set_stiffness(1000,1000,1000,80,80,80,0)

    # player
    def execute(self):
        start = self.player_init()
        while self.time_index <( self.loaded_traj_n):
            self.player_step(start)


    def gripper_step(self, target_gripper: float):
        
        if self.is_open(target_gripper) and not self.is_open():
            self.move_gripper(0.08)
        print(self.is_open(target_gripper), self.is_open(), self.is_grasped())
        if not self.is_open(target_gripper) and self.is_open():
            if not self.is_grasped():
                self.grasp_gripper(0.0)

    def get_rec_image(self):
        resized_img_gray=image_process(self.curr_image, self.ds_factor,  self.row_crop_pct_top , self.row_crop_pct_bot, self.col_crop_pct_left, self.col_crop_pct_right)
        
        resized_img_msg = self.bridge.cv2_to_imgmsg(resized_img_gray)
        self.cropped_img_pub.publish(resized_img_msg)

        return resized_img_gray.reshape((1, resized_img_gray.shape[0], resized_img_gray.shape[1]))

    def player_init(self):
        assert self.loaded_traj is not None, "Trajectory not loaded"

        print(f"Executing: {self.filename}", flush=True)
        # init states
        self.time_index=0
        self.end = False
        self.pause = False
        self.spiralling_occured = False

        # init pose
        start = PoseStamped()
        quat_start = list_2_quaternion(self.loaded_ori_wxyz[:, 0])
        start = pos_quat_2_pose_st(self.loaded_traj[:, 0], quat_start)
        self.go_to_pose_ik(start)

        self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, self.K_ori, 0)
        
        self.gripper_step(self.loaded_gripper[0][0])            
        
        self.trajectory_len = self.loaded_traj.shape[1]

        # init recording of new execution attempt
        self.recorded_traj = self.curr_pos
        self.recorded_ori = self.curr_ori_xyzw
        self.recorded_gripper = self.grip_value
        self.recorded_img_feedback_flag = np.array([0])
        self.recorded_spiral_flag = np.array([0])
        self.recorded_img = self.get_rec_image()

        return start

    def player_step(self, start: PoseStamped):
        assert self.loaded_traj is not None, "Trajectory not loaded"

        self.timestep_pub.publish(Int32(data=int(self.time_index)))
        quat_goal = list_2_quaternion(self.loaded_ori_wxyz[:, self.time_index])
        goal = pos_quat_2_pose_st(self.loaded_traj[:, self.time_index] + self.camera_correction, quat_goal)
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'panda_link0'
        
        self.correct()

        self.gripper_step(self.loaded_gripper[0][self.time_index])
        
        self.move_to_pose_with_stampedpose(goal)

        # if self.loaded_img_feedback_flag[0, self.time_index]:
        #     self.sift_matching()
        
        if self.loaded_spiral_flag[0, self.time_index]:
            if self.force.z > 5:
                spiral_success, offset_correction = self.spiral_search(goal)
                self.spiralling_occured = True
                if spiral_success:
                    self.loaded_traj[0, self.time_index:] += offset_correction[0]
                    self.loaded_traj[1, self.time_index:] += offset_correction[1]

        goal_pos_array = position_2_array(goal.pose.position)
        pos_2_goal_diff = np.linalg.norm(self.curr_pos-goal_pos_array)

        if pos_2_goal_diff <= self.attractor_distance_threshold:
            self.time_index=self.time_index + 1

        force_xy_plane = np.sqrt(self.force.x ** 2 + self.force.y ** 2)
        if False and force_xy_plane > self.insertion_force_threshold:
            # print("Camera correction", self.camera_correction)
            if self.retry_counter >= 3:
                self.move_gripper(self.grip_open_width)

                return 'stop'
                
            self.go_to_pose(start)
            self.time_index = 0
            self.retry_counter = self.retry_counter + 1
        self.r.sleep()

        # save step sample
        self.recorded_traj = np.c_[self.recorded_traj, self.curr_pos]
        self.recorded_ori  = np.c_[self.recorded_ori, self.curr_ori_xyzw]
        self.recorded_gripper = np.c_[self.recorded_gripper, self.grip_value]

        self.recorded_img = np.r_[self.recorded_img, self.get_rec_image()]
        self.recorded_img_feedback_flag = np.c_[self.recorded_img_feedback_flag, self.img_feedback_flag]
        self.recorded_spiral_flag = np.c_[self.recorded_spiral_flag, self.spiral_flag]

        # Stop playback if at end of trajectory (some indices might be deleted by feedback)
        if self.time_index == self.loaded_traj.shape[1]-1:
            return 'stop'


    def start_publishing_scene(self):
        self.start_publishing_scene_call.call(Trigger.Request())

    def stop_publishing_scene(self):
        self.stop_publishing_scene_call.call(Trigger.Request())


