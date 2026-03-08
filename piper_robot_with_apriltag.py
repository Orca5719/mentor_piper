import numpy as np
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from piper.robot import PiperRobot
except ImportError:
    print("错误：piper.robot 导入失败！")
    sys.exit(1)

try:
    from april_tag_tracker import AprilTagTracker, HAS_APRILTAGS
except ImportError:
    print("错误：april_tag_tracker 导入失败！")
    sys.exit(1)


class PiperRobotWithAprilTag(PiperRobot):
    """集成 AprilTag 追踪的 Piper 机械臂"""
    
    def __init__(self, 
                 use_sim: bool = False,
                 camera_width: int = 256,
                 camera_height: int = 256,
                 obj_pos: np.ndarray = None,
                 goal_pos: np.ndarray = None,
                 tag_size: float = 0.05,
                 camera_calibration_file: str = 'camera_calibration.npz'):
        """
        初始化 Piper 机械臂（带 AprilTag 追踪）
        
        参数:
            use_sim: 是否使用模拟模式
            camera_width: 摄像头宽度
            camera_height: 摄像头高度
            obj_pos: 物体初始位置
            goal_pos: 目标位置
            tag_size: AprilTag 大小（米）
            camera_calibration_file: 相机标定文件路径
        """
        # 调用父类初始化
        super().__init__(
            use_sim=use_sim,
            camera_width=camera_width,
            camera_height=camera_height,
            obj_pos=obj_pos,
            goal_pos=goal_pos
        )
        
        self.tag_tracker = None
        self.camera_params = None
        self.last_detected_pos = None
        
        # 如果是真实模式且有 AprilTag，初始化追踪器
        if not use_sim and HAS_APRILTAGS:
            self._init_apriltag_tracker(tag_size, camera_calibration_file)
    
    def _init_apriltag_tracker(self, tag_size: float, calibration_file: str):
        """初始化 AprilTag 追踪器"""
        try:
            # 尝试加载相机标定
            if os.path.exists(calibration_file):
                data = np.load(calibration_file)
                self.camera_params = (
                    float(data['fx']),
                    float(data['fy']),
                    float(data['cx']),
                    float(data['cy'])
                )
                print(f"✓ 加载相机标定: {self.camera_params}")
            else:
                print(f"⚠️  未找到相机标定文件: {calibration_file}")
                print("   将不会估计 3D 位置")
                self.camera_params = None
            
            # 初始化追踪器
            self.tag_tracker = AprilTagTracker(
                tag_family='tag36h11',
                tag_size=tag_size,
                camera_params=self.camera_params
            )
            
            print("✓ AprilTag 追踪器初始化成功")
            
        except Exception as e:
            print(f"✗ AprilTag 初始化失败: {e}")
            self.tag_tracker = None
    
    def get_obj_pos(self) -> np.ndarray:
        """
        获取物体位置
        
        - 模拟模式：返回模拟位置
        - 真实模式：尝试用 AprilTag 检测，失败则返回默认位置
        """
        if self.use_sim:
            return self.obj_pos.copy()
        
        # 真实模式：尝试用 AprilTag 检测
        if self.tag_tracker is not None and self.camera is not None:
            try:
                color_array_rgb, _ = self.camera.get_frame()
                
                if color_array_rgb is not None:
                    detection = self.tag_tracker.detect(color_array_rgb)
                    
                    if detection is not None and 'position' in detection:
                        # 获取 AprilTag 在相机坐标系下的位置
                        tag_pos_camera = detection['position']
                        
                        # 将相机坐标系转换为机械臂坐标系
                        # 这里需要根据你的相机安装位置进行标定！
                        # 下面是一个示例转换，你需要根据实际情况修改
                        obj_pos = self._camera_to_robot(tag_pos_camera)
                        
                        self.last_detected_pos = obj_pos
                        return obj_pos
            
            except Exception as e:
                print(f"⚠️  AprilTag 检测错误: {e}")
        
        # 如果检测失败，返回最后一次检测到的位置或默认位置
        if self.last_detected_pos is not None:
            return self.last_detected_pos.copy()
        
        return self.obj_pos.copy()
    
    def _camera_to_robot(self, camera_pos: np.ndarray) -> np.ndarray:
        """
        将相机坐标系转换为机械臂坐标系
        
        ⚠️ 注意：这里需要根据你的相机安装位置进行实际标定！
        
        参数:
            camera_pos: 相机坐标系下的位置 (x, y, z)
            
        返回:
            机械臂坐标系下的位置
        """
        # 这是一个示例转换，你需要根据实际情况修改！
        # 需要考虑：
        #   1. 相机相对于机械臂底座的位置
        #   2. 相机的朝向
        #   3. 坐标系的约定
        
        # 示例：假设相机安装在机械臂前方，稍微向上
        # 这里只是示意，必须根据实际情况标定！
        
        robot_pos = np.zeros(3)
        
        # 简单的坐标转换示例（需要实际标定）
        # camera_pos 的坐标系：
        #   x: 相机右方
        #   y: 相机下方
        #   z: 相机前方
        #
        # 需要转换为机械臂坐标系
        
        # 这只是一个占位符，实际使用时需要进行手眼标定！
        robot_pos[0] = camera_pos[0]  # X 轴示例
        robot_pos[1] = camera_pos[2]  # Y 轴示例（相机 Z -> 机械臂 Y）
        robot_pos[2] = -camera_pos[1]  # Z 轴示例（相机 -Y -> 机械臂 Z）
        
        # 添加相机相对于机械臂底座的偏移（需要标定）
        # 例如：相机安装在 (0.1, 0, 0.5) 相对于底座
        robot_pos[0] += 0.0
        robot_pos[1] += 0.0
        robot_pos[2] += 0.0
        
        return robot_pos


def test_apriltag_integration():
    """测试 AprilTag 集成"""
    print("="*70)
    print(" Piper + AprilTag 集成测试")
    print("="*70)
    
    print("\n提示：")
    print("  - 这个测试展示如何集成 AprilTag")
    print("  - 在实际使用前，你需要：")
    print("    1. 打印并贴 AprilTag")
    print("    2. 标定相机")
    print("    3. 进行手眼标定（相机->机械臂坐标转换）")
    
    # 先测试模拟模式
    print("\n" + "="*70)
    print(" 测试模拟模式")
    print("="*70)
    
    robot_sim = PiperRobotWithAprilTag(use_sim=True)
    print(f"模拟物体位置: {robot_sim.get_obj_pos()}")
    
    # 测试真实模式（如果硬件可用）
    print("\n" + "="*70)
    print(" 提示：真实模式需要连接硬件")
    print("="*70)
    print("\n使用步骤：")
    print("  1. 运行 calibrate_camera.py 标定相机")
    print("  2. 运行 april_tag_tracker.py 测试 AprilTag 检测")
    print("  3. 修改 _camera_to_robot() 进行手眼标定")
    print("  4. 在 piper/env.py 中使用 PiperRobotWithAprilTag")
    
    print("\n✓ 测试完成")


if __name__ == "__main__":
    test_apriltag_integration()
