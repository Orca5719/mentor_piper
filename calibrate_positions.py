import time
import sys
import os
import numpy as np

try:
    from piper_sdk import *
except ImportError:
    print("错误：piper_sdk 未安装！")
    print("请确保 piper_sdk 已正确安装")
    sys.exit(1)

# AprilTag 支持
try:
    from pupil_apriltags import Detector
    from Camera_Module import DepthCameraModule
    HAS_APRILTAG = True
except ImportError:
    HAS_APRILTAG = False


class PositionCalibrator:
    """位置标定工具：支持示教模式和 AprilTag 两种方式"""
    
    def __init__(self, use_apriltag=False, tag_size=0.05):
        self.use_apriltag = use_apriltag
        self.tag_size = tag_size
        
        self.robot = None
        self.camera = None
        self.apriltag_detector = None
        self.camera_params = None
        
        # 尝试加载相机标定
        self._load_camera_calibration()
    
    def _load_camera_calibration(self):
        """加载相机标定"""
        if os.path.exists('camera_calibration.npz'):
            try:
                data = np.load('camera_calibration.npz')
                self.camera_params = (
                    float(data['fx']),
                    float(data['fy']),
                    float(data['cx']),
                    float(data['cy'])
                )
                print(f"✓ 加载相机标定")
            except Exception as e:
                print(f"⚠️  加载相机标定失败: {e}")
    
    def connect(self):
        """连接机械臂和摄像头"""
        print("\n[1/4] 连接硬件...")
        
        # 连接机械臂
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
            print("✓ 机械臂连接成功")
        except Exception as e:
            print(f"✗ 机械臂连接失败: {e}")
            return False
        
        # 连接摄像头（如果需要 AprilTag）
        if self.use_apriltag and HAS_APRILTAG:
            try:
                self.camera = DepthCameraModule(
                    color_width=640,
                    color_height=480,
                    depth_width=640,
                    depth_height=480,
                    fps=30
                )
                
                self.apriltag_detector = Detector(
                    families='tag36h11',
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0
                )
                print("✓ 摄像头和 AprilTag 初始化成功")
            except Exception as e:
                print(f"⚠️  摄像头初始化失败: {e}")
                print("   将只使用示教模式")
        
        return True
    
    def get_position_teach_mode(self, position_name):
        """用示教模式获取位置"""
        print(f"\n请用示教模式移动机械臂到 {position_name}")
        print("\n操作步骤：")
        print("  1. 按一下机械臂上的示教按钮（指示灯变亮）")
        print(f"  2. 用手轻轻移动机械臂末端到 {position_name}")
        print("  3. 到达位置后，再按一下示教按钮退出示教模式")
        
        input("\n准备好了吗？按回车继续...")
        
        print(f"\n读取 {position_name}...")
        time.sleep(1.0)
        
        end_pose = self.robot.GetArmEndPoseMsgs()
        
        x = end_pose.end_pose.X_axis / 1000.0
        y = end_pose.end_pose.Y_axis / 1000.0
        z = end_pose.end_pose.Z_axis / 1000.0
        
        pos = np.array([x, y, z])
        
        print(f"\n✓ {position_name} 读取成功！")
        print(f"  坐标: [{x:.4f}, {y:.4f}, {z:.4f}]")
        
        return pos
    
    def get_position_apriltag(self, position_name):
        """用 AprilTag 获取位置"""
        if not HAS_APRILTAG or self.camera is None or self.apriltag_detector is None:
            print("⚠️  AprilTag 不可用，将使用示教模式")
            return self.get_position_teach_mode(position_name)
        
        print(f"\n请将贴有 AprilTag 的物体放在 {position_name}")
        input("准备好了吗？按回车继续...")
        
        print(f"\n检测 AprilTag...")
        
        for _ in range(10):
            color_array_rgb, _ = self.camera.get_frame()
            
            if color_array_rgb is not None and self.camera_params is not None:
                import cv2
                gray = cv2.cvtColor(color_array_rgb, cv2.COLOR_RGB2GRAY)
                detections = self.apriltag_detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=self.camera_params,
                    tag_size=self.tag_size
                )
                
                if detections:
                    tag = detections[0]
                    tag_pos = tag.pose_t.flatten()
                    
                    print(f"✓ 检测到 AprilTag!")
                    print(f"  相机坐标系: [{tag_pos[0]:.4f}, {tag_pos[1]:.4f}, {tag_pos[2]:.4f}]")
                    print("\n注意：这是相机坐标系下的位置")
                    print("      需要手眼标定后才能转换为机械臂坐标系")
                    print("      当前直接作为机械臂坐标系使用（需要标定）")
                    
                    # 暂时直接使用，后续需要手眼标定转换
                    pos = tag_pos.copy()
                    
                    print(f"\n✓ {position_name} 读取成功！")
                    print(f"  坐标: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                    
                    return pos
            
            time.sleep(0.1)
        
        print("✗ 未检测到 AprilTag")
        return None
    
    def calibrate(self):
        """执行标定"""
        print("="*70)
        print(" Piper 位置标定工具")
        print("="*70)
        
        print("\n这个工具可以帮助你标定：")
        print("  1. 物体初始位置 (obj_pos)")
        print("  2. 目标位置 (goal_pos)")
        
        print("\n选择标定方式：")
        print("  1. 示教模式（用手移动机械臂）")
        print("  2. AprilTag 模式（推荐，更准确）")
        
        choice = input("\n请输入 (1/2): ").strip()
        
        use_apriltag = (choice == '2')
        self.use_apriltag = use_apriltag and HAS_APRILTAG
        
        if use_apriltag and not HAS_APRILTAG:
            print("\n⚠️  AprilTag 不可用，将使用示教模式")
            print("   请确保安装: pip install pupil-apriltags")
        
        # 连接硬件
        if not self.connect():
            return False
        
        try:
            # 标定物体位置
            print("\n" + "="*70)
            print(" 第 1/2 步：标定物体初始位置 (obj_pos)")
            print("="*70)
            
            if self.use_apriltag:
                obj_pos = self.get_position_apriltag("物体初始位置")
            else:
                obj_pos = self.get_position_teach_mode("物体初始位置")
            
            if obj_pos is None:
                print("\n✗ 物体位置标定失败")
                return False
            
            # 标定目标位置
            print("\n" + "="*70)
            print(" 第 2/2 步：标定目标位置 (goal_pos)")
            print("="*70)
            
            if self.use_apriltag:
                goal_pos = self.get_position_apriltag("目标位置")
            else:
                goal_pos = self.get_position_teach_mode("目标位置")
            
            if goal_pos is None:
                print("\n✗ 目标位置标定失败")
                return False
            
            # 生成配置
            print("\n" + "="*70)
            print(" 生成配置文件")
            print("="*70)
            
            config_content = f"""# 目标位置设置 (单位: 米)
# X: 左右方向 (右为正)
# Y: 前后方向 (前为正)
# Z: 上下方向 (上为正)
goal_pos: [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]
obj_pos: [{obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f}]
"""
            
            print("\n" + "="*70)
            print("请将以下内容复制到 piper/cfgs/config.yaml 中：")
            print("="*70)
            print(config_content)
            print("="*70)
            
            # 询问是否保存到文件
            save = input("\n是否保存到 config_positions.yaml？(y/n): ").strip().lower()
            if save == 'y':
                with open('config_positions.yaml', 'w', encoding='utf-8') as f:
                    f.write(config_content)
                print("✓ 已保存到 config_positions.yaml")
            
            print("\n" + "="*70)
            print("标定完成！")
            print("="*70)
            
            return True
            
        except Exception as e:
            print(f"\n✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.disconnect()
    
    def disconnect(self):
        """断开连接"""
        if self.robot is not None:
            try:
                self.robot.DisconnectPort()
                print("\n已断开机械臂连接")
            except:
                pass
        
        if self.camera is not None:
            try:
                self.camera.stop()
                print("已断开摄像头连接")
            except:
                pass


def main():
    calibrator = PositionCalibrator()
    success = calibrator.calibrate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
