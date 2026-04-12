import numpy as np
import time
import cv2
import os
import logging
import threading

# 禁用 Piper SDK 的日志输出，避免 ERROR 日志阻塞
logging.getLogger('piper_sdk').setLevel(logging.CRITICAL)
logging.getLogger('PIPER').setLevel(logging.CRITICAL)


# ============================================
# 📌 Piper 机械臂配置区域 - 在这里修改初始位置和限制
# ============================================
#
# 初始位置（关节角度，单位：弧度）
# 每次环境 reset 时会回到这个位置
#
# 注意：如果需要使用示教模式来设置初始位置：
# 1. 运行 test_piper_camera.py 或使用示教按钮手动移动机械臂
# 2. 记录下满意的关节角度
# 3. 将角度转换为弧度（1度 = π/180 弧度）
# 4. 填入下方的 INITIAL_JOINT_POS
#
# ============================================

# 初始关节位置（弧度）- 每次 reset 时回到这个位置
# 示例：[0.0, 1.57, -1.57, 0.0, 0.0, 0.0] 对应 0°, 90°, -90°, 0°, 0°, 0°
# 修改为更安全的"中性"姿态，避免探索时打结：[0°, 45°, -45°, 0°, 0°, 0°]
INITIAL_JOINT_POS = [0.0, 0.7854, -0.7854, 0.0, 0.0, 0.0]

# ============================================
# 机械臂末端位置限制（毫米）
# 如果需要启用限制，设置 ENABLE_POSITION_LIMITS = True
# ============================================
ENABLE_POSITION_LIMITS = True

# X 轴限制（毫米）- 根据 test_piper_camera 实测数据调整
MIN_X_MM = 162041.0   # X 轴最小值
MAX_X_MM = 306197.0  # X 轴最大值（物体在 268.5mm，需覆盖）

# Y 轴限制（毫米）
MIN_Y_MM = -96658.0  # Y 轴最小值
MAX_Y_MM = 210930.0  # Y 轴最大值（目标在 215.4mm）

# Z 轴限制（毫米）
MIN_Z_MM = 73746.0   # Z 轴最小值
MAX_Z_MM = 287052.0  # Z 轴最大值

# ============================================
# 📌 配置区域结束
# ============================================

try:
    from piper_sdk import *
    PiperSDK = C_PiperInterface_V2
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
                 debug_mode=False,
                 use_apriltag=False, tag_size=0.05,
                 camera_calibration_file='camera_calibration.npz',
                 hand_eye_calibration_file='hand_eye_calibration.npz'):
        self.use_sim = use_sim
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.use_apriltag = use_apriltag
        self.tag_size = tag_size
        self.debug_mode = debug_mode
        
        self.piper = None
        self.factor = 57295.7795
        self.gripper_pos = 0
        self.apriltag_visible = False
        
        if not use_sim and PiperSDK is not None:
            try:
                self.piper = PiperSDK("can0")
                self.piper.ConnectPort()
                
                # 等待机械臂连接稳定（增加等待时间）
                time.sleep(0.5)
                
                # 使能机械臂，带重试机制
                max_retries = 20  # 增加重试次数
                for attempt in range(max_retries):
                    try:
                        if self.piper.EnablePiper():
                            print(f"✓ 机械臂使能成功（尝试 {attempt + 1}/{max_retries}）")
                            break
                    except Exception as e:
                        print(f"使能尝试 {attempt + 1}/{max_retries} 失败: {e}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(0.2)  # 增加等待时间
                else:
                    print("警告：机械臂使能失败，将使用模拟模式")
                    raise RuntimeError("机械臂使能失败，请检查CAN连接和电源")
                
                # 设置夹爪初始状态
                try:
                    self.piper.GripperCtrl(0, 1000, 0x01, 0)
                    time.sleep(0.05)
                except:
                    pass  # 夹爪命令失败不影响主要功能
                
                # 配置控制模式并使能关节（增加重试和错误处理）
                for attempt in range(3):
                    try:
                        self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
                        time.sleep(0.02)
                        self.piper.EnableArm(7, 0x02)
                        time.sleep(0.02)
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(0.1)
                        else:
                            print(f"警告：机械臂配置失败：{e}")
                            # 尝试继续，后续命令可能正常工作
                
                # 清除所有关节错误并配置加速度
                try:
                    self.piper.JointConfig(7, 0x00, 0x00, 500, 0xAE)
                except:
                    pass  # 配置失败不影响主要功能
                time.sleep(0.1)
                
                # 使用 SDK 原生功能设置关节限制（更可靠）
                # 设置 Joint 1: [-150°, 150°] = [-2.6179, 2.6179] rad
                self.piper.SetSDKJointLimitParam("j1", -2.6179, 2.6179)
                # 设置 Joint 2: [-10°, 180°] = [-0.1745, 3.14] rad
                self.piper.SetSDKJointLimitParam("j2", -0.1745, 3.14)
                # 设置 Joint 3: [-180°, 10°] = [-3.1416, 0.1745] rad
                self.piper.SetSDKJointLimitParam("j3", -3.1416, 0.1745)
                # 设置 Joint 4: [-100°, 100°] = [-1.745, 1.745] rad
                self.piper.SetSDKJointLimitParam("j4", -1.745, 1.745)
                # 设置 Joint 5: [-70°, 70°] = [-1.22, 1.22] rad
                self.piper.SetSDKJointLimitParam("j5", -1.22, 1.22)
                # 设置 Joint 6: [-120°, 120°] = [-2.09439, 2.09439] rad
                self.piper.SetSDKJointLimitParam("j6", -2.09439, 2.09439)
                time.sleep(0.05)
                
                print("✓ 机械臂连接成功，关节限制已设置")
            except Exception as e:
                print(f"警告：无法连接机械臂：{e}，将使用模拟模式")
                import traceback
                traceback.print_exc()
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
        
        # 机械臂卡住检测和限制检测
        self.last_joint_pos = None
        self.stuck_counter = 0
        self.safe_joint_pos = None  # 上一个安全的关节位置
        self.last_valid_joint_pos = None  # 最后一个有效的关节位置（用于限制变化量）
        self.position_limit_violated = False  # 位置限制是否被触发
        self.stuck_threshold = 15  # 连续多少步认为卡住（已放宽）
        
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
            # 获取脚本所在目录，支持相对路径
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 加载相机标定 - 尝试多个路径
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
                print(f"   尝试路径: {cam_calib_paths}")
                print("   AprilTag 将无法估计 3D 位置")
            
            # 尝试加载手眼标定（优先简单标定）
            self.hand_eye_offset = None
            self.T_cam2robot = None
            
            # 尝试手眼标定文件 - 多个路径
            hand_eye_paths = [
                'simple_hand_eye.json',
                os.path.join(script_dir, 'simple_hand_eye.json'),
                hand_eye_calibration_file,
                os.path.join(script_dir, hand_eye_calibration_file)
            ]
            
            hand_eye_file_found = None
            hand_eye_type = None
            
            for path in hand_eye_paths:
                if os.path.exists(path):
                    hand_eye_file_found = path
                    if 'simple' in path:
                        hand_eye_type = 'simple'
                    else:
                        hand_eye_type = 'full'
                    break
            
            if hand_eye_file_found and hand_eye_type == 'simple':
                import json
                with open(hand_eye_file_found, 'r') as f:
                    calib_data = json.load(f)
                self.hand_eye_offset = np.array(calib_data['offset'])
                print(f"✓ 加载简单手眼标定: {hand_eye_file_found}")
                print(f"  偏移量: {self.hand_eye_offset}")
            elif hand_eye_file_found and hand_eye_type == 'full':
                data = np.load(hand_eye_file_found)
                self.T_cam2robot = data['T_cam2gripper']
                print(f"✓ 加载完整手眼标定: {hand_eye_file_found}")
            else:
                print(f"⚠️  未找到手眼标定文件")
                print(f"   请先运行 easy_hand_eye_calibration.py")
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
                joint_pose = self.piper.GetArmJointMsgs()
                self.current_joint_pos = np.array([
                    joint_pose.joint_state.joint_1 / self.factor,
                    joint_pose.joint_state.joint_2 / self.factor,
                    joint_pose.joint_state.joint_3 / self.factor,
                    joint_pose.joint_state.joint_4 / self.factor,
                    joint_pose.joint_state.joint_5 / self.factor,
                    joint_pose.joint_state.joint_6 / self.factor
                ])
            except Exception as e:
                print(f"获取关节位置错误：{e}")
        return self.current_joint_pos.copy()
        
    def set_joint_pos(self, joint_pos, gripper_pos=None, speed=None):
        joint_pos = np.array(joint_pos)
        
        if gripper_pos is not None:
            self.gripper_pos = gripper_pos
        
        # 重置位置限制触发标志
        self.position_limit_violated = False
        
        if not self.use_sim and self.piper is not None:
            try:
                spd = speed if speed is not None else 20  # 超低速模式，约 1-2 cm/s 的末端速度
                
                # 关节角度限制（单位：弧度）- 已在 SDK 初始化时设置
                # SDK 会自动限制关节角度，这里只需要检查变化量
                joint_limits = [
                    (-2.6179, 2.6179),  # Joint 1: [-150°, 150°] (SDK 限制)
                    (-0.1745, 3.14),    # Joint 2: [-10° 到 180°] (原 0° 到 180°，向下扩展 10°)
                    (-3.1416, 0.1745),  # Joint 3: [-180° 到 10°] (原 -170° 到 0°，向下扩展 10°)
                    (-1.745, 1.745),    # Joint 4: [-100° 到 100°] (保持 SDK 限制)
                    (-1.22, 1.22),      # Joint 5: [-70° 到 70°] (保持 SDK 限制)
                    (-2.09439, 2.09439) # Joint 6: [-120° 到 120°] (保持 SDK 限制)
                ]
                
                # 限制关节角度在安全范围内（双重保护）
                joint_pos_clipped = np.clip(joint_pos, 
                                            [lim[0] for lim in joint_limits],
                                            [lim[1] for lim in joint_limits])
                
                # 安全检查：如果关节角度被裁剪，说明 RL 发出了危险指令
                if np.any(np.abs(joint_pos_clipped - joint_pos) > 0.01):
                    if self.debug_mode:
                        print(f"[安全警告] 关节角度超出限制，已自动裁剪")
                        print(f"  原始：{joint_pos}")
                        print(f"  裁剪：{joint_pos_clipped}")
                
                # 检查关节变化量，避免突变导致 CAN 通信问题
                if self.last_valid_joint_pos is not None:
                    delta = np.abs(joint_pos_clipped - self.last_valid_joint_pos)
                    max_delta = np.max(delta)
                    if max_delta > 1.0:  # 单次移动不超过 1.0 弧度（约 57.3 度），放宽限制以允许更多探索
                        if self.debug_mode:
                            print(f"[安全警告] 关节变化量过大 ({max_delta:.3f} rad)，限制在 1.0 rad 以内")
                        # 限制最大变化量
                        scale = 1.0 / max_delta
                        joint_pos_clipped = self.last_valid_joint_pos + (joint_pos_clipped - self.last_valid_joint_pos) * scale
                
                joint_0 = round(joint_pos_clipped[0] * self.factor)
                joint_1 = round(joint_pos_clipped[1] * self.factor)
                joint_2 = round(joint_pos_clipped[2] * self.factor)
                joint_3 = round(joint_pos_clipped[3] * self.factor)
                joint_4 = round(joint_pos_clipped[4] * self.factor)
                joint_5 = round(joint_pos_clipped[5] * self.factor)
                
                gripper_cmd = round(abs(self.gripper_pos) * 1000 * 1000)
                
                if self.debug_mode:
                    print(f"[DEBUG] joint_0-5: {joint_0}, {joint_1}, {joint_2}, {joint_3}, {joint_4}, {joint_5}")
                    print(f"[DEBUG] gripper_cmd: {gripper_cmd}")
                
                # 如果启用了位置限制，在发送指令前先检查
                # 注意：由于 GetArmEndPoseMsgs() 返回的是当前位置，不是预测位置
                # 我们暂时只使用关节限制和变化量限制来保证安全
                # 位置限制的检查会在移动后进行，如果超出限制会回退到安全位置
                use_safe_pos = False
                # 暂时禁用基于 FK 的预测检查，因为需要更复杂的正向运动学计算
                # if ENABLE_POSITION_LIMITS and self.safe_joint_pos is not None:
                #     end_pose = self.piper.GetArmEndPoseMsgs()
                #     ...
                
                # 如果需要使用安全位置
                if use_safe_pos and self.safe_joint_pos is not None:
                    joint_0 = round(self.safe_joint_pos[0] * self.factor)
                    joint_1 = round(self.safe_joint_pos[1] * self.factor)
                    joint_2 = round(self.safe_joint_pos[2] * self.factor)
                    joint_3 = round(self.safe_joint_pos[3] * self.factor)
                    joint_4 = round(self.safe_joint_pos[4] * self.factor)
                    joint_5 = round(self.safe_joint_pos[5] * self.factor)
                
                # 确保机械臂处于 CAN 命令控制模式并使能电机
                self.piper.ModeCtrl(0x01, 0x01, spd, 0x00)
                self.piper.EnableArm(7, 0x02)
                # 清除错误并确保电机准备好
                self.piper.JointConfig(7, 0x00, 0x00, 500, 0xAE)
                time.sleep(0.01)  # 等待命令生效
                
                # 发送关节指令（带重试机制）
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                        self.piper.GripperCtrl(gripper_cmd, 1000, 0x01, 0)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            if self.debug_mode:
                                print(f"[CAN 重试] 第{attempt+1}次发送失败：{e}，重试中...")
                            time.sleep(0.02)
                        else:
                            print(f"[CAN 错误] 关节指令发送失败：{e}")
                            raise
                
                # 等待移动完成
                time.sleep(0.1)
                
                # 更新最后有效的关节位置
                if not use_safe_pos:
                    self.last_valid_joint_pos = joint_pos_clipped.copy()
                else:
                    # 没有启用限制，或者是第一次移动，更新安全位置
                    self.safe_joint_pos = joint_pos_clipped.copy()
                
                # 获取机械臂状态用于调试（仅在 debug_mode 下）
                if self.debug_mode:
                    arm_status = self.piper.GetArmStatus()
                    print(f"[DEBUG] Arm Status: {arm_status}")
            except Exception as e:
                print(f"设置关节位置错误：{e}")
                import traceback
                traceback.print_exc()
        
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
    
    def move_end_effector_delta(self, dx=0, dy=0, dz=0, gripper=0, scale=0.02):
        """
        笛卡尔空间末端移动（简化版，使用伪逆雅可比近似）
        
        Args:
            dx, dy, dz: 末端位置的增量（米）
            gripper: 夹爪位置 (-1 到 1)
            scale: 移动缩放因子
            
        Returns:
            执行后的末端实际位置
        """
        # 获取当前末端位置
        current_pos = self.get_end_effector_pos()
        
        # 计算新位置
        new_pos = current_pos + np.array([dx, dy, dz]) * scale
        
        # 简单限幅
        new_pos = np.clip(new_pos, 
                         [0.1, -0.15, 0.03],  # 最小位置（米）
                         [0.35, 0.25, 0.32])  # 最大位置（米）
        
        # 简化的笛卡尔到关节转换
        # 使用当前关节位置 + 伪逆雅可比近似
        delta_pos = new_pos - current_pos
        
        # 简化雅可比矩阵（假设 6DOF 臂，主要影响前 3 个关节）
        # 这是一个近似，实际需要精确的雅可比
        jacobian_inv = np.array([
            [0.5, 0.3, 0.1],   # X 影响
            [0.1, 0.6, 0.2],   # Y 影响
            [0.0, 0.2, 0.5],   # Z 影响
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        # 计算关节增量
        joint_delta = jacobian_inv @ delta_pos
        
        # 添加到当前关节位置
        new_joint_pos = self.current_joint_pos + joint_delta
        
        # 限幅关节位置
        new_joint_pos = np.clip(new_joint_pos, 
                                np.array([-1.5, -0.5, -2.0, -2.0, -2.5, -3.0]),
                                np.array([1.5, 1.5, 0.5, 2.0, 2.5, 3.0]))
        
        # 设置夹爪
        gripper_pos = (gripper + 1.0) / 2.0  # -1~1 转为 0~1
        
        # 执行移动
        self.set_joint_pos(new_joint_pos, gripper_pos)
        
        return self.get_end_effector_pos()
            
    def get_obj_pos(self):
        if self.use_sim:
            self.apriltag_visible = True
            return self.obj_pos.copy()
        
        # 真实世界：尝试用 AprilTag 检测
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
            # offset 是毫米，需要转换为米
            offset_m = self.hand_eye_offset / 1000.0
            
            # 先尝试简单的转换，根据 easy_hand_eye_calibration.py 的 offset 计算方式
            # robot_pos = tag_pos + offset
            robot_pos = camera_pos + offset_m
            
            # 打印用于调试的原始值（仅在debug_mode时打印）
            if self.debug_mode:
                print(f"[DEBUG] camera_pos (tag_pos): {camera_pos}")
                print(f"[DEBUG] offset_m: {offset_m}")
                print(f"[DEBUG] robot_pos: {robot_pos}")
            
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
        self.gripper_pos = 0
        # 重置卡住检测和限制检测状态
        self.last_joint_pos = None
        self.stuck_counter = 0
        self.safe_joint_pos = np.array(INITIAL_JOINT_POS).copy()
        self.last_valid_joint_pos = np.array(INITIAL_JOINT_POS).copy()  # 初始化最后有效位置
        self.position_limit_violated = False
        
        if not hasattr(self, '_initial_obj_pos'):
            self._initial_obj_pos = self.obj_pos.copy()
        if not hasattr(self, '_initial_goal_pos'):
            self._initial_goal_pos = self.goal_pos.copy()
        self.obj_pos = self._initial_obj_pos.copy()
        self.goal_pos = self._initial_goal_pos.copy()
        
        if not self.use_sim and self.piper is not None:
            # 使用配置的初始关节位置
            self.set_joint_pos(INITIAL_JOINT_POS, gripper_pos=0, speed=30)
            time.sleep(1.0)
        else:
            self.current_joint_pos = np.array(INITIAL_JOINT_POS).copy()
            
        if self.use_sim:
            self._update_end_effector_pos_sim()
            
    def update_stuck_detection(self):
        """更新卡住检测状态（仅在step中调用一次）"""
        if self.last_joint_pos is None:
            self.last_joint_pos = self.current_joint_pos.copy()
            return False
        
        # 计算关节位置变化
        joint_change = np.linalg.norm(self.current_joint_pos - self.last_joint_pos)
        if joint_change < 0.0001:  # 关节位置几乎没有变化（阈值已放宽10倍）
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        self.last_joint_pos = self.current_joint_pos.copy()
        
        return self.stuck_counter >= self.stuck_threshold
    
    def is_stuck(self):
        """检测机械臂是否卡住（只读，不修改状态）"""
        return self.stuck_counter >= self.stuck_threshold
            
    def step(self, action, dt=0.01):
        action = np.clip(action, -1.0, 1.0)
        # 动作增量：0.05 对应约 ±2.9°，速度约 2-3 cm/s（柔和控制，避免打结）
        joint_delta = action[:6] * 0.05
        new_joint_pos = self.current_joint_pos + joint_delta
        # 把 action[6] 从 [-1,1] 映射到 [0,1]
        new_gripper_pos = (action[6] + 1.0) / 2.0
                
        # 使用带超时的方式执行机械臂控制命令
        self._step_with_timeout(new_joint_pos, new_gripper_pos, timeout=0.5)


        # 更新卡住检测（每步只调用一次）
        self.update_stuck_detection()

        if self.use_sim:
            self._update_obj_position_sim(action)

        time.sleep(dt * 5)  # dt*5 = 0.05s，配合速度参数实现 2-3 cm/s 的末端速度
            
    def _step_with_timeout(self, joint_pos, gripper_pos, timeout=0.5):
        """使用线程超时机制执行机械臂命令，避免 CAN 阻塞"""
        result = {'error': None, 'done': False}
        
        def _target():
            try:
                self.set_joint_pos(joint_pos, gripper_pos)
            except Exception as e:
                result['error'] = e
            finally:
                result['done'] = True
        
        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            # 线程超时，尝试恢复机械臂
            print(f"[超时] 机械臂命令执行超时 ({timeout}s)，尝试恢复...")
            self._recover_arm()
        elif result['error']:
            # 执行出错
            print(f"[错误] 机械臂命令执行失败: {result['error']}")
            self._recover_arm()
        
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