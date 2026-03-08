import numpy as np
import time
import cv2
import os

try:
    from piper_sdk import *
    PiperSDK = C_PiperInterface
except ImportError:
    print("警告：piper_sdk 未安装，将使用模拟模式")
    PiperSDK = None

try:
    from Camera_Module import DepthCameraModule
except ImportError:
    print("警告：Camera_Module 未找到，相机功能不可用")
    DepthCameraModule = None

# AprilTag 支持
try:
    from pupil_apriltags import Detector
    HAS_APRILTAGS = True
except ImportError:
    print("警告：pupil-apriltags 未安装，AprilTag 功能不可用")
    print("运行: pip install pupil-apriltags")
    HAS_APRILTAGS = False


class PiperRobot:
    def __init__(self, use_sim=False, camera_width=256, camera_height=256,
                 obj_pos=None, goal_pos=None,
                 use_apriltag=False, tag_size=0.05,
                 camera_calibration_file='camera_calibration.npz',
                 hand_eye_calibration_file='hand_eye_calibration.npz'):
        self.use_sim = use_sim
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.use_apriltag = use_apriltag
        self.tag_size = tag_size
        
        self.piper = None
        
        if not use_sim and PiperSDK is not None:
            try:
                self.piper = PiperSDK(
                    can_name="can0",
                    judge_flag=True,
                    can_auto_init=False,
                    dh_is_offset=1,
                    start_sdk_joint_limit=False,
                    start_sdk_gripper_limit=False,
                    logger_level=LogLevel.WARNING,
                    log_to_file=False
                )
                
                self.piper.CreateCanBus(can_name="can0")
                self.piper.ConnectPort()
                self.piper.MasterSlaveConfig(0xFC, 0, 0, 0)
                time.sleep(0.5)
                print("机械臂连接成功")
            except Exception as e:
                print(f"警告：无法连接机械臂：{e}，将使用模拟模式")
                self.use_sim = True
                self.piper = None
        
        self.camera = None
        if not use_sim and DepthCameraModule is not None:
            try:
                self.camera = DepthCameraModule(
                    color_width=640,
                    color_height=480,
                    depth_width=640,
                    depth_height=480,
                    fps=30
                )
            except Exception as e:
                print(f"警告：无法初始化相机：{e}")
                self.camera = None
        
        self.current_joint_pos = np.zeros(6)
        self.current_end_effector_pos = np.zeros(3)
        
        if obj_pos is None:
            self.obj_pos = np.array([0.0, 0.6, 0.0])
        else:
            self.obj_pos = np.array(obj_pos)
            
        if goal_pos is None:
            self.goal_pos = np.array([0.0, 0.75, 0.0])
        else:
            self.goal_pos = np.array(goal_pos)
            
        self.move_spd_rate_ctrl = 50
        
        # AprilTag 初始化
        self.apriltag_detector = None
        self.camera_params = None
        self.T_cam2robot = None
        self.last_detected_obj_pos = None
        
        if not use_sim and use_apriltag and HAS_APRILTAGS:
            self._init_apriltag(camera_calibration_file, hand_eye_calibration_file)
        
        if self.use_sim:
            self._update_end_effector_pos_sim()
    
    def _init_apriltag(self, camera_calibration_file, hand_eye_calibration_file):
        """初始化 AprilTag 检测器"""
        try:
            # 加载相机标定
            if os.path.exists(camera_calibration_file):
                data = np.load(camera_calibration_file)
                self.camera_params = (
                    float(data['fx']),
                    float(data['fy']),
                    float(data['cx']),
                    float(data['cy'])
                )
                print(f"✓ 加载相机标定")
            else:
                print(f"⚠️  未找到相机标定文件: {camera_calibration_file}")
                print("   AprilTag 将无法估计 3D 位置")
            
            # 尝试加载手眼标定（优先简单标定）
            self.hand_eye_offset = None
            self.T_cam2robot = None
            
            # 先尝试简单标定
            simple_calib_file = 'simple_hand_eye.json'
            if os.path.exists(simple_calib_file):
                import json
                with open(simple_calib_file, 'r') as f:
                    calib_data = json.load(f)
                self.hand_eye_offset = np.array(calib_data['offset'])
                print(f"✓ 加载简单手眼标定，偏移量: {self.hand_eye_offset}")
            elif os.path.exists(hand_eye_calibration_file):
                # 再尝试完整手眼标定
                data = np.load(hand_eye_calibration_file)
                self.T_cam2robot = data['T_cam2gripper']
                print(f"✓ 加载完整手眼标定")
            else:
                print(f"⚠️  未找到手眼标定文件")
                print(f"   请先运行 simple_hand_eye_calibration.py")
                print("   将使用简单坐标转换（可能不准确）")
            
            # 初始化检测器
            self.apriltag_detector = Detector(
                families='tag36h11',
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0
            )
            
            print("✓ AprilTag 检测器初始化成功")
            
        except Exception as e:
            print(f"✗ AprilTag 初始化失败: {e}")
            self.apriltag_detector = None
    
    def _update_end_effector_pos_sim(self):
        self.current_end_effector_pos = np.array([
            np.sin(self.current_joint_pos[0]) * 0.3,
            np.cos(self.current_joint_pos[1]) * 0.3,
            np.sin(self.current_joint_pos[2]) * 0.1 + 0.2
        ])
        
    def _update_obj_position_sim(self, action):
        dist_to_obj = np.linalg.norm(self.current_end_effector_pos - self.obj_pos)
        if dist_to_obj < 0.05:
            target_pos = self.goal_pos
            push_dir = target_pos[:2] - self.obj_pos[:2]
            if np.linalg.norm(push_dir) > 0.001:
                push_dir = push_dir / np.linalg.norm(push_dir)
                self.obj_pos[:2] += push_dir * 0.005
                
    def get_joint_pos(self):
        if not self.use_sim and self.piper is not None:
            try:
                joint_pose = self.piper.GetArmJointPoseMsgs()
                self.current_joint_pos = np.array([
                    joint_pose.joint_pos.J1,
                    joint_pose.joint_pos.J2,
                    joint_pose.joint_pos.J3,
                    joint_pose.joint_pos.J4,
                    joint_pose.joint_pos.J5,
                    joint_pose.joint_pos.J6
                ])
            except Exception as e:
                print(f"获取关节位置错误：{e}")
        return self.current_joint_pos.copy()
        
    def set_joint_pos(self, joint_pos, speed=None):
        joint_pos = np.array(joint_pos)
        if not self.use_sim and self.piper is not None:
            try:
                spd = speed if speed is not None else self.move_spd_rate_ctrl
                self.piper.MoveJ(
                    joint_pos[0],
                    joint_pos[1],
                    joint_pos[2],
                    joint_pos[3],
                    joint_pos[4],
                    joint_pos[5],
                    spd,
                    1,
                    0
                )
            except Exception as e:
                print(f"设置关节位置错误：{e}")
        self.current_joint_pos = joint_pos.copy()
        if self.use_sim:
            self._update_end_effector_pos_sim()
            
    def get_end_effector_pos(self):
        if not self.use_sim and self.piper is not None:
            try:
                end_pose = self.piper.GetArmEndPoseMsgs()
                self.current_end_effector_pos = np.array([
                    end_pose.end_pose.X_axis / 1000.0,
                    end_pose.end_pose.Y_axis / 1000.0,
                    end_pose.end_pose.Z_axis / 1000.0
                ])
            except Exception as e:
                print(f"获取末端位置错误：{e}")
        return self.current_end_effector_pos.copy()
            
    def get_obj_pos(self):
        if self.use_sim:
            return self.obj_pos.copy()
        
        # 真实世界：尝试用 AprilTag 检测
        if self.use_apriltag and self.apriltag_detector is not None and self.camera is not None:
            try:
                color_array_rgb, _ = self.camera.get_frame()
                
                if color_array_rgb is not None and self.camera_params is not None:
                    gray = cv2.cvtColor(color_array_rgb, cv2.COLOR_RGB2GRAY)
                    detections = self.apriltag_detector.detect(
                        gray,
                        estimate_tag_pose=True,
                        camera_params=self.camera_params,
                        tag_size=self.tag_size
                    )
                    
                    if detections:
                        tag = detections[0]
                        tag_pos_camera = tag.pose_t.flatten()
                        
                        # 转换到机械臂坐标系
                        obj_pos = self._camera_to_robot(tag_pos_camera)
                        
                        self.last_detected_obj_pos = obj_pos
                        return obj_pos
            
            except Exception as e:
                print(f"⚠️  AprilTag 检测错误: {e}")
        
        # 如果检测失败，返回最后一次检测到的位置或默认位置
        if self.last_detected_obj_pos is not None:
            return self.last_detected_obj_pos.copy()
        
        return self.obj_pos.copy()
    
    def _camera_to_robot(self, camera_pos):
        """
        将相机坐标系转换为机械臂坐标系
        
        优先级：
        1. 简单手眼标定偏移量（推荐初学者）
        2. 完整手眼标定矩阵
        3. 简单坐标转换（默认）
        """
        if self.hand_eye_offset is not None:
            # 使用简单手眼标定偏移量
            # 假设相机坐标系和机械臂坐标系对齐
            # 只需要加上偏移量
            robot_pos = camera_pos + self.hand_eye_offset
            return robot_pos
        
        if self.T_cam2robot is not None:
            # 使用完整手眼标定结果
            p_cam = np.ones(4)
            p_cam[:3] = camera_pos
            p_robot = self.T_cam2robot @ p_cam
            return p_robot[:3]
        
        # 简单坐标转换（需要根据实际情况调整）
        robot_pos = np.zeros(3)
        
        # 假设相机坐标系：
        #   x: 相机右方
        #   y: 相机下方
        #   z: 相机前方
        #
        # 转换为机械臂坐标系（示例）
        robot_pos[0] = camera_pos[0]
        robot_pos[1] = camera_pos[2]
        robot_pos[2] = -camera_pos[1]
        
        # 添加相机相对于机械臂底座的偏移（需要根据实际安装位置调整）
        robot_pos[0] += 0.0
        robot_pos[1] += 0.0
        robot_pos[2] += 0.0
        
        return robot_pos
        
    def set_obj_pos(self, obj_pos):
        self.obj_pos = np.array(obj_pos)
        
    def get_goal_pos(self):
        return self.goal_pos.copy()
        
    def set_goal_pos(self, goal_pos):
        self.goal_pos = np.array(goal_pos)
        
    def get_camera_image(self):
        if self.camera is not None:
            try:
                color_array_rgb, _ = self.camera.get_frame()
                if color_array_rgb is not None:
                    resized = cv2.resize(color_array_rgb, (self.camera_width, self.camera_height))
                    return resized
            except Exception as e:
                print(f"获取相机图像错误：{e}")
        
        if self.use_sim:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        else:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            
    def reset(self):
        self.current_joint_pos = np.zeros(6)
        if not hasattr(self, '_initial_obj_pos'):
            self._initial_obj_pos = self.obj_pos.copy()
        if not hasattr(self, '_initial_goal_pos'):
            self._initial_goal_pos = self.goal_pos.copy()
        self.obj_pos = self._initial_obj_pos.copy()
        self.goal_pos = self._initial_goal_pos.copy()
        
        if not self.use_sim and self.piper is not None:
            self.set_joint_pos(np.zeros(6))
            time.sleep(2.0)
            
        if self.use_sim:
            self._update_end_effector_pos_sim()
            
    def step(self, action, dt=0.01):
        action = np.clip(action, -1.0, 1.0)
        joint_delta = action * 0.1
        new_joint_pos = self.current_joint_pos + joint_delta
        self.set_joint_pos(new_joint_pos)
        
        if self.use_sim:
            self._update_obj_position_sim(action)
            
        time.sleep(dt)
        
    def close(self):
        if self.piper is not None:
            try:
                self.piper.DisconnectPort()
            except:
                pass
        if self.camera is not None:
            try:
                self.camera.stop()
            except:
                pass
