import numpy as np
import time
import cv2
import os
from piper_sdk import *

PiperSDK = None

try:
    from Camera_Module import DepthCameraModule
except ImportError:
    print("警告：Camera_Module 未找到，相机功能不可用")
    DepthCameraModule = None

try:
    from pupil_apriltags import Detector
    HAS_APRILTAGS = True
except ImportError:
    print("警告：pupil-apriltags 未安装，AprilTag 功能不可用")
    HAS_APRILTAGS = False


class PiperRobot:
    def __init__(self, use_sim=False, camera_width=256, camera_height=256,
                 obj_pos=None, goal_pos=None,
                 use_apriltag=False, tag_size=0.05,
                 camera_calibration_file='camera_calibration.npz',
                 hand_eye_calibration_file='simple_hand_eye.json'):
        self.use_sim = use_sim
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.use_apriltag = use_apriltag
        self.tag_size = tag_size
        
        self.piper = None
        # 【关键修改】夹爪初始状态：张开（对齐 manual_collect.py）
        self.gripper_pos = 0.08  # 0.08m = 张开，0.0m = 闭合
        
        self.apriltag_visible = False
        
        # 【关键修改】机械臂初始位姿：完全复制 manual_collect.py
        self.factor = 1000
        self.current_end_pos = np.array([300.614 / 1000.0, -12.185 / 1000.0, 282.341 / 1000.0])  # mm → m
        self.current_end_rpy = np.array([-179.351 * np.pi / 180.0, 23.933 * np.pi / 180.0, 177.934 * np.pi / 180.0])  # 度 → 弧度
        
        if not use_sim and PiperSDK is not None:
            try:
                self.piper = C_PiperInterface_V2("can0")
                self.piper.ConnectPort()
                
                while not self.piper.EnablePiper():
                    time.sleep(0.01)
                
                # 【关键修改】初始化夹爪为张开
                self.piper.GripperCtrl(round(0.08 * 1000 * 1000), 1000, 0x01, 0)
                
                # 【关键修改】初始化机械臂到 manual_collect.py 初始位姿
                self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                X = round(300.614 * self.factor)
                Y = round(-12.185 * self.factor)
                Z = round(282.341 * self.factor)
                RX = round(-179.351 * self.factor)
                RY = round(23.933 * self.factor)
                RZ = round(177.934 * self.factor)
                self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
                time.sleep(0.1)
                
                print("✓ 机械臂连接成功（初始位姿已对齐 manual_collect.py）")
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
        
        if obj_pos is None:
            self.obj_pos = np.array([0.0, 0.6, 0.0])
        else:
            self.obj_pos = np.array(obj_pos)
            
        if goal_pos is None:
            self.goal_pos = np.array([0.0, 0.75, 0.0])
        else:
            self.goal_pos = np.array(goal_pos)
        
        self.hand_eye_offset = None
        self.T_cam2robot = None
        self.apriltag_detector = None
        self.camera_params = None
        self.last_detected_obj_pos = None
        
        if not use_sim and use_apriltag and HAS_APRILTAGS:
            self._init_apriltag(camera_calibration_file, hand_eye_calibration_file)
    
    def _init_apriltag(self, camera_calibration_file, hand_eye_calibration_file):
        try:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            cam_calib_paths = [
                camera_calibration_file,
                os.path.join(script_dir, camera_calibration_file),
                'camera_calibration.npz',
                os.path.join(script_dir, 'camera_calibration.npz')
            ]
            
            cam_file_found = None
            for path in cam_calib_paths:
                if os.path.exists(path):
                    cam_file_found = path
                    break
            
            if cam_file_found:
                data = np.load(cam_file_found)
                self.camera_params = (
                    float(data['fx']),
                    float(data['fy']),
                    float(data['cx']),
                    float(data['cy'])
                )
                print(f"✓ 加载相机标定: {cam_file_found}")
            else:
                print(f"⚠️  未找到相机标定文件")
            
            import json
            hand_eye_paths = [
                'simple_hand_eye.json',
                os.path.join(script_dir, 'simple_hand_eye.json'),
                hand_eye_calibration_file,
                os.path.join(script_dir, hand_eye_calibration_file)
            ]
            
            hand_eye_file_found = None
            for path in hand_eye_paths:
                if os.path.exists(path):
                    hand_eye_file_found = path
                    break
            
            if hand_eye_file_found:
                with open(hand_eye_file_found, 'r') as f:
                    calib_data = json.load(f)
                if 'camera_to_robot_offset' in calib_data:
                    self.hand_eye_offset = np.array(calib_data['camera_to_robot_offset'])
                elif 'offset' in calib_data:
                    self.hand_eye_offset = np.array(calib_data['offset'])
                print(f"✓ 加载简单手眼标定: {hand_eye_file_found}")
            
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
    
    def _camera_to_robot(self, camera_pos):
        if self.hand_eye_offset is not None:
            robot_pos = camera_pos + self.hand_eye_offset
            return robot_pos
        return camera_pos
    
    def get_end_effector_pos(self):
        if not self.use_sim and self.piper is not None:
            try:
                end_pose = self.piper.GetArmEndPoseMsgs()
                self.current_end_pos = np.array([
                    end_pose.end_pose.X_axis / 1000.0,
                    end_pose.end_pose.Y_axis / 1000.0,
                    end_pose.end_pose.Z_axis / 1000.0
                ])
            except Exception as e:
                print(f"获取末端位置错误：{e}")
        return self.current_end_pos.copy()
    
    def set_end_pose(self, end_pos, rpy=None, gripper_pos=None):
        end_pos = np.array(end_pos)
        
        if rpy is not None:
            self.current_end_rpy = np.array(rpy)
        
        if gripper_pos is not None:
            self.gripper_pos = gripper_pos
        
        if not self.use_sim and self.piper is not None:
            try:
                factor = 1000
                
                X = round(end_pos[0] * 1000.0)
                Y = round(end_pos[1] * 1000.0)
                Z = round(end_pos[2] * 1000.0)
                RX = round(self.current_end_rpy[0] * factor)
                RY = round(self.current_end_rpy[1] * factor)
                RZ = round(self.current_end_rpy[2] * factor)
                
                # 【关键修改】夹爪控制：对齐 manual_collect.py 的数值范围
                gripper_cmd = round(abs(self.gripper_pos) * 1000 * 1000)
                
                self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
                self.piper.GripperCtrl(gripper_cmd, 1000, 0x01, 0)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"设置末端姿态错误：{e}")
                import traceback
                traceback.print_exc()
        
        self.current_end_pos = end_pos.copy()
    
    def get_obj_pos(self):
        if self.use_sim:
            self.apriltag_visible = True
            return self.obj_pos.copy()
        
        self.apriltag_visible = False
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
                        self.apriltag_visible = True
                        tag = detections[0]
                        tag_pos_camera = tag.pose_t.flatten()
                        obj_pos = self._camera_to_robot(tag_pos_camera)
                        self.last_detected_obj_pos = obj_pos
                        return obj_pos
            
            except Exception as e:
                pass
        
        if self.last_detected_obj_pos is not None:
            return self.last_detected_obj_pos.copy()
        
        return self.obj_pos.copy()
    
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
                pass
        
        if self.use_sim:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        else:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            
    def reset(self):
        """【关键修改】重置函数：完全对齐 manual_collect.py 的初始状态"""
        # 重置夹爪为张开
        self.gripper_pos = 0.08  # 0.08m = 张开
        
        if not hasattr(self, '_initial_obj_pos'):
            self._initial_obj_pos = self.obj_pos.copy()
        if not hasattr(self, '_initial_goal_pos'):
            self._initial_goal_pos = self.goal_pos.copy()
        self.obj_pos = self._initial_obj_pos.copy()
        self.goal_pos = self._initial_goal_pos.copy()
        
        if not self.use_sim and self.piper is not None:
            print("[Reset] 机械臂回到 manual_collect.py 初始位置...")
            # 【关键修改】使用 manual_collect.py 的初始位姿
            initial_pos = np.array([300.614 / 1000.0, -12.185 / 1000.0, 282.341 / 1000.0])
            initial_rpy = np.array([-179.351 * np.pi / 180.0, 23.933 * np.pi / 180.0, 177.934 * np.pi / 180.0])
            self.set_end_pose(initial_pos, initial_rpy, gripper_pos=0.08)  # 夹爪张开
            time.sleep(1.5)
            print("[Reset] ✓ 机械臂已回到 manual_collect.py 初始位置（夹爪张开）")
        else:
            # 模拟模式下也对齐初始位姿
            self.current_end_pos = np.array([300.614 / 1000.0, -12.185 / 1000.0, 282.341 / 1000.0])
            self.current_end_rpy = np.array([-179.351 * np.pi / 180.0, 23.933 * np.pi / 180.0, 177.934 * np.pi / 180.0])
            self.gripper_pos = 0.08
            
    def step(self, action, dt=0.01):
        action = np.clip(action, -1.0, 1.0)
        
        pos_delta = action[:3] * 0.02
        new_end_pos = self.current_end_pos + pos_delta
        
        # 夹爪动作：对齐 manual_collect.py 的归一化逻辑
        gripper_pos = (action[3] + 1.0) / 2.0 * 0.08  # [-1,1] → [0, 0.08]
        
        self.set_end_pose(new_end_pos, gripper_pos=gripper_pos)
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
