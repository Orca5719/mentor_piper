import time
import sys
import os
import json
import numpy as np

try:
    from piper_sdk import *
except ImportError:
    print("错误：piper_sdk 未安装！")
    sys.exit(1)

# AprilTag 支持
try:
    from pupil_apriltags import Detector
    from Camera_Module import DepthCameraModule
    HAS_APRILTAG = True
except ImportError:
    HAS_APRILTAG = False


class EasyHandEyeCalibrator:
    """简易手眼标定工具（使用示教模式）"""
    
    def __init__(self, tag_size=0.05):
        self.tag_size = tag_size
        self.samples = []
        
        self.robot = None
        self.camera = None
        self.apriltag_detector = None
        self.camera_params = None
        
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
        """连接硬件"""
        print("\n[1/5] 连接硬件...")
        
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
        
        # 连接摄像头
        if HAS_APRILTAG:
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
                print(f"✗ 摄像头初始化失败: {e}")
                return False
        else:
            print("✗ pupil-apriltags 未安装！")
            print("   运行: pip install pupil-apriltags")
            return False
        
        return True
    
    def get_robot_position(self):
        """用示教模式获取机械臂坐标"""
        print("\n请用示教模式移动机械臂：")
        print("  1. 按一下机械臂上的示教按钮（指示灯变亮）")
        print("  2. 用手轻轻移动机械臂末端，对准 AprilTag 中心")
        print("  3. 到达位置后，再按一下示教按钮退出示教模式")
        
        input("\n准备好了吗？按回车继续...")
        
        print("\n读取机械臂位置...")
        time.sleep(1.0)
        
        end_pose = self.robot.GetArmEndPoseMsgs()
        
        x = end_pose.end_pose.X_axis / 1000.0
        y = end_pose.end_pose.Y_axis / 1000.0
        z = end_pose.end_pose.Z_axis / 1000.0
        
        pos = np.array([x, y, z])
        
        print(f"✓ 机械臂位置读取成功！")
        print(f"  坐标: [{x:.4f}, {y:.4f}, {z:.4f}]")
        
        return pos
    
    def detect_apriltag(self):
        """检测 AprilTag 获取相机坐标"""
        print("\n检测 AprilTag...")
        
        for _ in range(20):
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
                    print(f"  相机坐标: [{tag_pos[0]:.4f}, {tag_pos[1]:.4f}, {tag_pos[2]:.4f}]")
                    
                    return tag_pos, color_array_rgb
            
            time.sleep(0.1)
        
        print("✗ 未检测到 AprilTag")
        return None, None
    
    def add_sample(self, sample_num):
        """添加一个样本"""
        print(f"\n{'='*70}")
        print(f" 采集样本 {sample_num}/5")
        print(f"{'='*70}")
        
        print("\n步骤：")
        print("  1. 确保贴有 AprilTag 的咖啡杯在摄像头视野内")
        print("  2. 用示教模式把机械臂末端移到 AprilTag 中心")
        print("  3. 按提示操作")
        
        input("\n准备好了吗？按回车开始...")
        
        # 先检测 AprilTag
        tag_pos, img = self.detect_apriltag()
        if tag_pos is None:
            return False
        
        # 再获取机械臂位置
        robot_pos = self.get_robot_position()
        if robot_pos is None:
            return False
        
        # 保存样本
        self.samples.append({
            'robot_pos': robot_pos.tolist(),
            'tag_pos': tag_pos.tolist()
        })
        
        print(f"\n✓ 样本 {sample_num} 采集完成！")
        return True
    
    def calibrate(self):
        """执行标定"""
        print("="*70)
        print(" 简易手眼标定工具（示教模式）")
        print("="*70)
        
        print("\n原理：")
        print("  对于每个采集点，我们获取：")
        print("    1. 机械臂坐标系下的位置（用示教模式）")
        print("    2. 相机坐标系下的位置（用 AprilTag）")
        print("  然后计算两者之间的偏移量！")
        
        print("\n需要采集 3-5 个点")
        print("建议的点分布：")
        print("  - 点 1: 正前方")
        print("  - 点 2: 右侧")
        print("  - 点 3: 左侧")
        print("  - 点 4: 稍远处")
        print("  - 点 5: 稍近处")
        
        # 连接硬件
        if not self.connect():
            return False
        
        try:
            # 采集样本
            num_samples = 5
            for i in range(1, num_samples + 1):
                success = self.add_sample(i)
                if not success:
                    print(f"\n⚠️  样本 {i} 采集失败，跳过")
                    continue
                
                if i < num_samples:
                    choice = input(f"\n继续采集下一个点？(y/n): ").strip().lower()
                    if choice != 'y':
                        break
            
            if len(self.samples) < 3:
                print(f"\n✗ 至少需要 3 个样本，当前只有 {len(self.samples)} 个")
                return False
            
            # 计算偏移量
            print(f"\n{'='*70}")
            print(f" 计算偏移量")
            print(f"{'='*70}")
            
            robot_poses = np.array([s['robot_pos'] for s in self.samples])
            tag_poses = np.array([s['tag_pos'] for s in self.samples])
            
            # 计算每个样本的偏移
            offsets = robot_poses - tag_poses
            
            # 平均偏移
            mean_offset = np.mean(offsets, axis=0)
            std_offset = np.std(offsets, axis=0)
            
            print(f"\n采集的 {len(self.samples)} 个点：")
            for i, (r, t, o) in enumerate(zip(robot_poses, tag_poses, offsets)):
                print(f"\n点 {i+1}:")
                print(f"  机械臂: [{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}]")
                print(f"  相机  : [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
                print(f"  偏移  : [{o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f}]")
            
            print(f"\n{'='*70}")
            print(f" 标定结果")
            print(f"{'='*70}")
            print(f"\n平均偏移量 (robot_pos = tag_pos + offset):")
            print(f"  offset_x = {mean_offset[0]:.4f}  (标准差: {std_offset[0]:.4f})")
            print(f"  offset_y = {mean_offset[1]:.4f}  (标准差: {std_offset[1]:.4f})")
            print(f"  offset_z = {mean_offset[2]:.4f}  (标准差: {std_offset[2]:.4f})")
            
            # 保存结果
            result = {
                'offset': mean_offset.tolist(),
                'offset_std': std_offset.tolist(),
                'num_samples': len(self.samples),
                'samples': self.samples
            }
            
            with open('simple_hand_eye.json', 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ 结果已保存到: simple_hand_eye.json")
            
            print(f"\n{'='*70}")
            print(f"标定完成！")
            print(f"{'='*70}")
            print(f"\n现在你可以运行训练了！")
            print(f"piper/robot.py 会自动加载 simple_hand_eye.json")
            
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
    calibrator = EasyHandEyeCalibrator()
    success = calibrator.calibrate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
