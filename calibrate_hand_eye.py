import json
import cv2
import numpy as np
import time
import sys
import os

# 导入深度相机模块
sys.path.insert(0, os.path.dirname(__file__))
try:
    from Camera_Module import DepthCameraModule
except ImportError:
    print("错误：DepthCameraModule 未找到！")
    sys.exit(1)

# 导入机械臂 SDK
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'piper'))
try:
    from piper_sdk import C_PiperInterface, LogLevel
except ImportError:
    print("错误：piper_sdk 未找到！")
    sys.exit(1)

# AprilTag 支持
try:
    from pupil_apriltags import Detector
    HAS_APRILTAG = True
except ImportError:
    print("警告：pupil-apriltags 未安装！")
    print("运行: pip install pupil-apriltags")
    HAS_APRILTAG = False


def compute_hand_eye(robot_points, camera_points):
    """计算相机到机械臂的偏移量"""
    if len(robot_points) < 3:
        print("警告：样本点太少，至少需要 3 个点！")
        return np.zeros(3)
    
    robot_array = np.array(robot_points)
    camera_array = np.array(camera_points)
    
    offsets = robot_array - camera_array
    avg_offset = np.mean(offsets, axis=0)
    std_offset = np.std(offsets, axis=0)
    
    print(f"\n平均偏移: X={avg_offset[0]:.4f}, Y={avg_offset[1]:.4f}, Z={avg_offset[2]:.4f}")
    print(f"标准差:   X={std_offset[0]:.4f}, Y={std_offset[1]:.4f}, Z={std_offset[2]:.4f}")
    
    return avg_offset


class AprilTagDetector:
    """基于 RealSense + pupil-apriltags 的检测器（修复版）"""
    
    def __init__(self, tag_size=0.05):
        if not HAS_APRILTAG:
            raise ImportError("pupil-apriltags 未安装！")
        
        self.tag_size = tag_size
        self.depth_camera = None
        self.camera_params = None
        self.intrinsics = None
        
        # 初始化 RealSense 相机
        self._init_realsense_camera()
        
        # 初始化 AprilTag 检测器
        self.detector = Detector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        print("✓ AprilTag 检测器初始化成功")
        print(f"  - Tag 大小: {tag_size}m")
        if self.camera_params:
            print(f"  - 相机内参: fx={self.camera_params[0]:.1f}, fy={self.camera_params[1]:.1f}")
    
    def _init_realsense_camera(self):
        """初始化 RealSense 相机（使用正确的接口）"""
        print("\n正在初始化 RealSense 深度相机...")
        try:
            # 使用和原手眼标定代码一致的参数
            self.depth_camera = DepthCameraModule(
                color_width=640,    # 降低分辨率确保兼容性
                color_height=480,
                depth_width=640,
                depth_height=480,
                fps=30,
                is_decimate=False   # 关闭降采样
            )
            
            # 获取相机内参
            self.intrinsics = self.depth_camera.intrinsics
            self.camera_params = (
                self.intrinsics.fx,
                self.intrinsics.fy,
                self.intrinsics.ppx,
                self.intrinsics.ppy
            )
            
            print("✓ RealSense 相机初始化成功")
            print(f"  相机内参: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
            
            # 等待相机稳定（关键！）
            time.sleep(2.0)
            
        except Exception as e:
            print(f"✗ RealSense 相机初始化失败: {e}")
            self.depth_camera = None
    
    def get_frame(self):
        """获取相机帧（使用正确的 get_image() 接口）"""
        if self.depth_camera is None:
            return None, None
        
        try:
            # 使用 Camera_Module 的 get_image() 方法
            # 返回: (frames_cat, depth_m, color_bgr)
            frames_cat, depth_m, color_bgr = self.depth_camera.get_image()
            
            if color_bgr is None or depth_m is None:
                return None, None
            
            # 将 BGR 转换为 RGB（供 AprilTag 检测使用）
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            
            return color_rgb, depth_m, color_bgr  # 返回 RGB、深度图、原始 BGR
            
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None, None, None
    
    def detect_tag(self, color_rgb):
        """检测 AprilTag"""
        if color_rgb is None or not self.camera_params:
            return None
        
        gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
        
        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size
        )
        
        if not detections:
            return None
        
        tag = detections[0]
        return {
            'id': tag.tag_id,
            'center': tag.center,
            'corners': tag.corners,
            'position': tag.pose_t.flatten()  # 相机坐标系 3D 坐标
        }
    
    def draw_detection(self, color_bgr, detection, robot_pos=None):
        """绘制检测结果（直接在 BGR 图上绘制）"""
        if detection is None:
            return color_bgr
        
        img = color_bgr.copy()
        
        # 绘制 Tag 边框
        corners = detection['corners'].astype(int)
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(img, p1, p2, (0, 255, 0), 2)
        
        # 绘制中心
        center = tuple(detection['center'].astype(int))
        cv2.circle(img, center, 5, (0, 0, 255), -1)
        
        # 绘制 ID 和坐标
        cv2.putText(img, f"Tag {detection['id']}", 
                   (center[0] - 20, center[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        pos = detection['position']
        cv2.putText(img, f"Cam: X:{pos[0]:.3f} Y:{pos[1]:.3f} Z:{pos[2]:.3f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        if robot_pos is not None:
            cv2.putText(img, f"Robot: X:{robot_pos[0]:.3f} Y:{robot_pos[1]:.3f} Z:{robot_pos[2]:.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img
    
    def release(self):
        if self.depth_camera is not None:
            self.depth_camera.stop()
            print("✓ RealSense 相机已释放")


class PiperArm:
    """Piper 机械臂交互类"""
    
    def __init__(self):
        self.robot = None
    
    def connect(self):
        print("\n正在初始化 Piper 机械臂...")
        try:
            self.robot = C_PiperInterface(
                can_name="can0",
                judge_flag=True,
                can_auto_init=False,
                dh_is_offset=1,
                start_sdk_joint_limit=False,
                start_sdk_gripper_limit=False,
                logger_level=LogLevel.WARNING,
                log_to_file=False
            )
            
            self.robot.CreateCanBus(can_name="can0")
            self.robot.ConnectPort()
            self.robot.MasterSlaveConfig(0xFC, 0, 0, 0)
            time.sleep(0.5)
            
            while not self.robot.EnablePiper():
                time.sleep(0.01)
            
            print("✓ Piper 机械臂连接成功")
            return True
        except Exception as e:
            print(f"✗ Piper 机械臂连接失败: {e}")
            return False
    
    def get_end_position(self):
        robot_pos = np.zeros(3)
        if self.robot is not None:
            try:
                end_pose = self.robot.GetArmEndPoseMsgs()
                robot_pos[0] = end_pose.end_pose.X_axis / 1000.0
                robot_pos[1] = end_pose.end_pose.Y_axis / 1000.0
                robot_pos[2] = end_pose.end_pose.Z_axis / 1000.0
            except Exception as e:
                print(f"获取机械臂位置失败: {e}")
        return robot_pos
    
    def disconnect(self):
        if self.robot is not None:
            try:
                self.robot.DisconnectPort()
                print("✓ Piper 机械臂已断开")
            except:
                pass


def save_config(config, filename="simple_hand_eye.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✓ 标定配置已保存到: {filename}")
    except Exception as e:
        print(f"\n✗ 保存配置失败: {e}")


def calibrate_hand_eye():
    print("="*70)
    print("     Piper 机械臂-RealSense 手眼标定工具")
    print("="*70)
    
    output_file = "simple_hand_eye.json"
    config = {
        "camera_to_robot_offset": [0.0, 0.0, 0.0],
        "camera_intrinsics": {"fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0},
        "num_samples": 0
    }
    
    # 初始化硬件
    arm = PiperArm()
    arm_connected = arm.connect()
    
    try:
        detector = AprilTagDetector(tag_size=0.05)
        
        if detector.camera_params:
            config['camera_intrinsics'] = {
                "fx": detector.camera_params[0],
                "fy": detector.camera_params[1],
                "cx": detector.camera_params[2],
                "cy": detector.camera_params[3]
            }
        
        robot_points = []
        camera_points = []
        
        print("\n开始标定...")
        print("按键: c=采集, s=保存计算, q=退出, h=帮助")
        
        while True:
            # 获取帧
            color_rgb, depth_m, color_bgr = detector.get_frame()
            
            if color_bgr is None:
                print("等待相机帧...")
                time.sleep(0.1)
                continue
            
            # 检测 Tag
            tag_info = detector.detect_tag(color_rgb)
            
            # 获取机械臂位置
            robot_pos = arm.get_end_position() if arm_connected else np.zeros(3)
            
            # 绘制结果
            display_img = detector.draw_detection(color_bgr, tag_info, robot_pos)
            
            # 绘制样本数
            cv2.putText(display_img, f"Samples: {len(robot_points)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            cv2.imshow("Hand-Eye Calibration", display_img)
            
            # 按键处理
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                print("\n用户退出")
                break
            
            elif key == ord('h'):
                print("\n=== 帮助 ===")
                print("c: 采集当前位置")
                print("s: 计算并保存结果")
                print("q: 退出")
                print("============")
            
            elif key == ord('c'):
                if tag_info is not None:
                    robot_points.append(robot_pos.copy())
                    camera_points.append(tag_info['position'].copy())
                    print(f"✓ 已采集样本 {len(robot_points)}")
                else:
                    print("⚠️ 未检测到 Tag！")
            
            elif key == ord('s'):
                if len(robot_points) >= 3:
                    offset = compute_hand_eye(robot_points, camera_points)
                    config['camera_to_robot_offset'] = offset.tolist()
                    config['num_samples'] = len(robot_points)
                    save_config(config, output_file)
                else:
                    print(f"⚠️ 至少需要 3 个样本，当前: {len(robot_points)}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cv2.destroyAllWindows()
        detector.release()
        arm.disconnect()
        print("\n程序退出")


if __name__ == "__main__":
    if not HAS_APRILTAG:
        sys.exit(1)
    calibrate_hand_eye()
