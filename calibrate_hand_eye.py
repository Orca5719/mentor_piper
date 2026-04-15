import json
import cv2
import numpy as np
import time
import sys
import os
import pyrealsense2 as rs

# 导入深度相机模块（假设 Camera_Module.py 在同级目录）
sys.path.insert(0, os.path.dirname(__file__))
try:
    from Camera_Module import DepthCameraModule
except ImportError:
    print("错误：DepthCameraModule 未找到！请确保 Camera_Module.py 存在。")
    sys.exit(1)

# 导入机械臂 SDK
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'piper'))
try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    print("错误：piper_sdk 未找到！请确保 Piper 机械臂 SDK 已正确安装。")
    sys.exit(1)


def compute_hand_eye(robot_points, camera_points):
    """
    计算相机到机械臂的变换矩阵
    使用简单的偏移计算（假设只有平移，没有旋转）
    
    Args:
        robot_points: 机械臂坐标系下的点列表 (N, 3)
        camera_points: 相机坐标系下的点列表 (N, 3)
    
    Returns:
        offset: 相机到机械臂的偏移量 (3,)
    """
    if len(robot_points) < 3:
        print("警告：样本点太少，至少需要 3 个点！")
        return np.zeros(3)
    
    robot_array = np.array(robot_points)
    camera_array = np.array(camera_points)
    
    print(f"\n计算变换矩阵...")
    print(f"机械臂点范围: X={robot_array[:,0].min():.3f}~{robot_array[:,0].max():.3f}")
    print(f"               Y={robot_array[:,1].min():.3f}~{robot_array[:,1].max():.3f}")
    print(f"               Z={robot_array[:,2].min():.3f}~{robot_array[:,2].max():.3f}")
    print(f"相机点范围:   X={camera_array[:,0].min():.3f}~{camera_array[:,0].max():.3f}")
    print(f"               Y={camera_array[:,1].min():.3f}~{camera_array[:,1].max():.3f}")
    print(f"               Z={camera_array[:,2].min():.3f}~{camera_array[:,2].max():.3f}")
    
    offsets = robot_array - camera_array
    avg_offset = np.mean(offsets, axis=0)
    
    print(f"\n平均偏移: X={avg_offset[0]:.4f}, Y={avg_offset[1]:.4f}, Z={avg_offset[2]:.4f}")
    
    return avg_offset


class AprilTagDetector:
    """基于 RealSense 深度相机的 AprilTag 检测器"""
    
    def __init__(self, tag_size=0.05, use_decimate=False):
        self.tag_size = tag_size
        self.use_decimate = use_decimate
        
        # 初始化 RealSense 深度相机
        self.depth_camera = None
        self._init_realsense_camera()
        
        # 初始化 AprilTag 检测器
        try:
            import cv2.aruco as aruco
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
            self.parameters = aruco.DetectorParameters()
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
            self.has_aruco = True
            print("✓ OpenCV ArUco/AprilTag 检测器初始化成功")
        except (ImportError, AttributeError):
            # 兼容不同 OpenCV 版本
            try:
                import cv2.aruco as aruco
                self.aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
                self.parameters = aruco.DetectorParameters_create()
                self.has_aruco = True
                print("✓ OpenCV 旧版 ArUco 检测器初始化成功")
            except:
                print("⚠️  OpenCV ArUco 模块未找到/不支持 AprilTag")
                self.has_aruco = False
    
    def _init_realsense_camera(self):
        """初始化 RealSense 深度相机"""
        print("\n正在初始化 RealSense 深度相机...")
        try:
            # 初始化深度相机（可根据需要调整分辨率）
            self.depth_camera = DepthCameraModule(
                color_width=1280, 
                color_height=720,
                depth_width=640,
                depth_height=480,
                fps=30,
                is_decimate=self.use_decimate
            )
            # 获取相机内参（已对齐到彩色流）
            self.intrinsics = self.depth_camera.intrinsics
            # 转换为 OpenCV 格式的内参矩阵 [fx, 0, cx; 0, fy, cy; 0, 0, 1]
            self.camera_matrix = np.array([
                [self.intrinsics.fx, 0, self.intrinsics.ppx],
                [0, self.intrinsics.fy, self.intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float32)
            # 畸变系数
            self.dist_coeffs = np.array(self.intrinsics.coeffs, dtype=np.float32)
            
            print("✓ RealSense 相机初始化成功")
            print(f"  相机内参: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
            print(f"            cx={self.intrinsics.ppx:.1f}, cy={self.intrinsics.ppy:.1f}")
        except Exception as e:
            print(f"✗ RealSense 相机初始化失败: {e}")
            self.depth_camera = None
    
    def get_frame(self):
        """获取彩色帧、深度帧（米）"""
        if self.depth_camera is None:
            return None, None
        
        # 获取对齐后的帧数据
        frames_cat, depth_m, color_bgr = self.depth_camera.get_image()
        if color_bgr is None or depth_m is None:
            return None, None
        
        return color_bgr, depth_m
    
    def detect_tag(self, color_frame, depth_frame):
        """
        检测 AprilTag 并计算其在相机坐标系下的 3D 坐标
        
        Args:
            color_frame: 彩色帧 (BGR格式)
            depth_frame: 深度帧 (米为单位)
        
        Returns:
            dict: 包含检测结果（id、center、position、corners），或 None
        """
        if not self.has_aruco or color_frame is None or depth_frame is None:
            return None
        
        # 转换为灰度图
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        
        # 检测 AprilTag
        try:
            # 兼容不同 OpenCV 版本的检测接口
            if hasattr(self, 'detector'):
                corners, ids, rejected = self.detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.parameters
                )
        except:
            return None
        
        if ids is None or len(ids) == 0:
            return None
        
        # 处理第一个检测到的 Tag
        tag_id = ids[0][0]
        tag_corners = corners[0][0]  # (4,2) 像素坐标
        
        # 计算 Tag 中心像素坐标
        cx = np.mean(tag_corners[:, 0])
        cy = np.mean(tag_corners[:, 1])
        
        # 确保坐标在有效范围内
        h, w = depth_frame.shape
        cx_clamped = np.clip(int(cx), 0, w-1)
        cy_clamped = np.clip(int(cy), 0, h-1)
        
        # 获取中心像素的深度值（米）
        tag_depth_m = depth_frame[cy_clamped, cx_clamped]
        
        # 过滤无效深度
        if tag_depth_m <= 0.01 or tag_depth_m > 3.0:
            print(f"⚠️  Tag 深度无效: {tag_depth_m:.3f} 米")
            return None
        
        # 像素坐标转相机坐标系 3D 点
        # 公式: X = (u - cx) * Z / fx
        #       Y = (v - cy) * Z / fy
        #       Z = 深度值
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.ppx
        cy = self.intrinsics.ppy
        
        x_cam = (cx_clamped - cx) * tag_depth_m / fx
        y_cam = (cy_clamped - cy) * tag_depth_m / fy
        z_cam = tag_depth_m
        
        tag_position = np.array([x_cam, y_cam, z_cam], dtype=np.float32)
        
        return {
            'id': tag_id,
            'center': (cx, cy),
            'position': tag_position,  # 相机坐标系下的 3D 坐标 (米)
            'corners': tag_corners,
            'depth': tag_depth_m
        }
    
    def release(self):
        """释放相机资源"""
        if self.depth_camera is not None:
            self.depth_camera.stop()
            print("✓ RealSense 相机已释放")


def calibrate_hand_eye():
    print("="*70)
    print("     Piper 机械臂-RealSense 深度相机 手眼标定工具")
    print("     【Eye-to-Hand】相机固定，不在机械臂末端")
    print("="*70)
    print()
    
    output_file = "simple_hand_eye.json"
    
    # 加载现有配置
    config = {
        "camera_to_robot_offset": [0.0, 0.0, 0.0],
        "rotation_offset": 0.0,
        "scale_factor": 1.0,
        "camera_intrinsics": {
            "fx": 0.0, "fy": 0.0,
            "cx": 0.0, "cy": 0.0
        }
    }
    
    if os.path.exists(output_file):
        print(f"找到现有的标定文件: {output_file}")
        try:
            with open(output_file, 'r') as f:
                config = json.load(f)
            print("已加载现有配置")
        except:
            print("加载失败，使用默认配置")
    else:
        print("未找到现有标定文件，将创建新文件")
    
    print()
    print("="*70)
    print("标定步骤说明 (Eye-to-Hand):")
    print("1. 将 AprilTag 贴在机械臂末端")
    print("2. 移动机械臂到多个不同位置（至少5-10个）")
    print("   - 覆盖工作空间的不同区域")
    print("   - 确保 AprilTag 始终在相机视野内")
    print("3. 在每个位置按 'c' 采集样本")
    print("4. 按 's' 保存并计算标定结果")
    print("5. 按 'q' 退出")
    print("="*70)
    print()
    
    # 初始化机械臂
    print("正在初始化 Piper 机械臂...")
    piper = None
    try:
        piper = C_PiperInterface_V2(can_name="can0")
        piper.ConnectPort()
        while not piper.EnablePiper():
            time.sleep(0.01)
        print("✓ Piper 机械臂连接成功")
    except Exception as e:
        print(f"✗ Piper 机械臂连接失败: {e}")
        print("将使用模拟模式...")
        piper = None
    
    # 初始化 AprilTag 检测器（基于 RealSense）
    detector = AprilTagDetector(tag_size=0.05, use_decimate=False)
    
    # 保存相机内参到配置
    if detector.depth_camera is not None:
        config['camera_intrinsics'] = {
            "fx": detector.intrinsics.fx,
            "fy": detector.intrinsics.fy,
            "cx": detector.intrinsics.ppx,
            "cy": detector.intrinsics.ppy
        }
    
    # 存储采样点
    robot_points = []
    camera_points = []
    
    print()
    print("开始标定... (按 'h' 查看帮助)")
    print()
    
    try:
        while True:
            # 获取相机帧
            color_frame, depth_frame = detector.get_frame()
            
            if color_frame is None or depth_frame is None:
                print("✗ 无法获取相机帧数据")
                time.sleep(0.1)
                continue
            
            # detector.get_frame() 返回的已经是 BGR 格式
            display_frame = color_frame.copy()
            tag_info = None
            
            # 检测 AprilTag
            tag_info = detector.detect_tag(color_frame, depth_frame)
            
            # 绘制检测结果
            if tag_info is not None:
                # 绘制 Tag 边框
                corners = tag_info['corners'].astype(np.int32)
                cv2.polylines(display_frame, [corners], True, (0, 255, 0), 2)
                
                # 绘制中心圆点
                cx, cy = map(int, tag_info['center'])
                cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # 绘制深度和 3D 坐标文本
                pos = tag_info['position']
                depth = tag_info['depth']
                cv2.putText(display_frame, f"Tag {tag_info['id']} | Depth: {depth:.3f}m", 
                           (cx - 50, cy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"X:{pos[0]:.3f} Y:{pos[1]:.3f} Z:{pos[2]:.3f}", 
                           (cx - 50, cy + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 获取机械臂末端位置
            robot_pos = np.zeros(3)
            if piper is not None:
                try:
                    end_pose = piper.GetArmEndPoseMsgs()
                    robot_pos[0] = end_pose.end_pose.X_axis / 1000.0  # 转换为米
                    robot_pos[1] = end_pose.end_pose.Y_axis / 1000.0
                    robot_pos[2] = end_pose.end_pose.Z_axis / 1000.0
                except Exception as e:
                    print(f"获取机械臂位置失败: {e}")
            
            # 绘制状态信息
            y_pos = 30
            line_spacing = 25
            
            cv2.putText(display_frame, f"Samples: {len(robot_points)}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_pos += line_spacing
            
            cv2.putText(display_frame, f"Robot: X={robot_pos[0]:.3f} Y={robot_pos[1]:.3f} Z={robot_pos[2]:.3f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_pos += line_spacing
            
            if tag_info is not None:
                pos = tag_info['position']
                cv2.putText(display_frame, f"Camera: X={pos[0]:.3f} Y={pos[1]:.3f} Z={pos[2]:.3f}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                y_pos += line_spacing
            
            cv2.putText(display_frame, "c=Collect, s=Save, q=Quit, h=Help", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            
            # 调整窗口大小（适配高分辨率）
            display_frame = cv2.resize(display_frame, (800, 450))
            cv2.imshow("RealSense Hand-Eye Calibration", display_frame)
            
            # 按键处理
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                print("\n用户退出")
                break
            
            if key == ord('h'):
                print("\n=== 帮助信息 ===")
                print("c: 采集当前 Tag 位置和机械臂位置")
                print("s: 计算偏移并保存标定结果")
                print("q: 退出程序")
                print("h: 显示此帮助信息")
                print("================")
            
            if key == ord('s'):
                if len(robot_points) >= 3:
                    # 计算手眼偏移
                    offset = compute_hand_eye(robot_points, camera_points)
                    config['camera_to_robot_offset'] = offset.tolist()
                    # 保存配置
                    save_config(config, output_file)
                else:
                    print(f"\n⚠️  样本点太少！至少需要 3 个，当前: {len(robot_points)}")
            
            if key == ord('c'):
                if tag_info is not None:
                    # 采集样本
                    robot_points.append(robot_pos.copy())
                    camera_points.append(tag_info['position'].copy())
                    print(f"\n✓ 已采集样本 {len(robot_points)}")
                    print(f"  机械臂位置: X={robot_pos[0]:.3f} Y={robot_pos[1]:.3f} Z={robot_pos[2]:.3f}")
                    print(f"  相机位置:   X={tag_info['position'][0]:.3f} Y={tag_info['position'][1]:.3f} Z={tag_info['position'][2]:.3f}")
                else:
                    print("\n⚠️  未检测到 AprilTag，无法采集样本！")
    
    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        # 清理资源
        cv2.destroyAllWindows()
        detector.release()
        if piper is not None:
            piper.DisconnectPort()
            print("✓ Piper 机械臂已断开")
        print("\n标定程序已清理资源并退出")


def save_config(config, filename):
    """保存标定配置到 JSON 文件"""
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✓ 标定配置已保存到: {filename}")
        print("\n当前标定配置:")
        print(json.dumps(config, indent=2))
    except Exception as e:
        print(f"\n✗ 保存配置失败: {e}")


if __name__ == "__main__":
    calibrate_hand_eye()