import cv2
import numpy as np
from typing import Optional, Tuple

try:
    from pupil_apriltags import Detector
    HAS_APRILTAGS = True
except ImportError:
    print("警告：pupil-apriltags 未安装！")
    print("运行: pip install pupil-apriltags")
    HAS_APRILTAGS = False


class AprilTagTracker:
    def __init__(self, 
                 tag_family: str = 'tag36h11',
                 tag_size: float = 0.05,  # 标签大小（米）
                 camera_params: Optional[Tuple[float, float, float, float]] = None):
        """
        AprilTag 追踪器
        
        参数:
            tag_family: AprilTag 家族，默认 'tag36h11'
            tag_size: 标签的物理大小（米），默认 0.05m (5cm)
            camera_params: 相机内参 (fx, fy, cx, cy)，如果为 None 会尝试自动估计
        """
        if not HAS_APRILTAGS:
            raise ImportError("pupil-apriltags 未安装！")
        
        self.tag_size = tag_size
        self.camera_params = camera_params
        
        # 初始化检测器
        self.detector = Detector(
            families=tag_family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        print(f"✓ AprilTag 检测器初始化成功")
        print(f"  - Tag 家族: {tag_family}")
        print(f"  - Tag 大小: {tag_size}m")
    
    def detect(self, color_image: np.ndarray) -> Optional[dict]:
        """
        检测图像中的 AprilTag
        
        参数:
            color_image: RGB 图像 (H, W, 3)
            
        返回:
            detection dict，包含:
                - 'id': tag ID
                - 'position': 3D 位置 (x, y, z)，单位米
                - 'pose_R': 旋转矩阵
                - 'pose_t': 平移向量
                - 'center': 像素坐标中心
            如果未检测到，返回 None
        """
        if color_image is None:
            return None
        
        # 转换为灰度图
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        
        # 检测 tags
        if self.camera_params is not None:
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self.camera_params,
                tag_size=self.tag_size
            )
        else:
            detections = self.detector.detect(gray)
        
        if not detections:
            return None
        
        # 返回第一个检测到的 tag
        tag = detections[0]
        
        result = {
            'id': tag.tag_id,
            'center': tag.center,
            'corners': tag.corners
        }
        
        # 如果估计了姿态
        if hasattr(tag, 'pose_t') and tag.pose_t is not None:
            # pose_t 是 (3, 1) 的数组，转换为 (3,)
            result['position'] = tag.pose_t.flatten()
            result['pose_R'] = tag.pose_R
            result['pose_t'] = tag.pose_t
        
        return result
    
    def draw_detection(self, color_image: np.ndarray, detection: dict) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        参数:
            color_image: RGB 图像
            detection: detect() 返回的结果
            
        返回:
            绘制后的图像
        """
        if detection is None:
            return color_image
        
        img = color_image.copy()
        
        # 绘制角点
        corners = detection['corners'].astype(int)
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(img, p1, p2, (0, 255, 0), 2)
        
        # 绘制中心
        center = tuple(detection['center'].astype(int))
        cv2.circle(img, center, 5, (0, 0, 255), -1)
        
        # 绘制 ID
        cv2.putText(img, f"ID: {detection['id']}", 
                   (center[0] - 20, center[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制位置（如果有）
        if 'position' in detection:
            pos = detection['position']
            pos_text = f"X: {pos[0]:.3f} Y: {pos[1]:.3f} Z: {pos[2]:.3f}"
            cv2.putText(img, pos_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img


def test_april_tag():
    """测试 AprilTag 检测"""
    print("="*60)
    print(" AprilTag 检测测试")
    print("="*60)
    
    try:
        from Camera_Module import DepthCameraModule
    except ImportError:
        print("错误：Camera_Module.py 未找到！")
        return
    
    # 初始化相机
    print("\n初始化相机...")
    camera = DepthCameraModule(
        color_width=640,
        color_height=480,
        depth_width=640,
        depth_height=480,
        fps=30
    )
    
    # 初始化 AprilTag 检测器
    print("\n初始化 AprilTag 检测器...")
    print("提示：需要先进行相机标定才能获取准确的 3D 位置")
    print("      当前仅演示检测功能，不估计 3D 位置")
    
    tracker = AprilTagTracker(
        tag_family='tag36h11',
        tag_size=0.05,
        camera_params=None  # 先不估计 3D 位置
    )
    
    print("\n开始检测...")
    print("提示：")
    print("  - 将 AprilTag 放在摄像头前")
    print("  - 按 'q' 键退出")
    print("  - 按 's' 键保存当前帧")
    
    save_count = 0
    
    try:
        while True:
            # 获取图像
            color_array_rgb, _ = camera.get_frame()
            
            if color_array_rgb is not None:
                # 检测 AprilTag
                detection = tracker.detect(color_array_rgb)
                
                # 绘制检测结果
                display_img = tracker.draw_detection(color_array_rgb, detection)
                
                # 转换为 BGR 用于 OpenCV 显示
                display_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                
                # 显示
                cv2.imshow('AprilTag Detection', display_bgr)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户退出")
                    break
                elif key == ord('s'):
                    save_count += 1
                    cv2.imwrite(f'apriltag_frame_{save_count:03d}.png', display_bgr)
                    print(f"✓ 已保存帧 {save_count}")
    
    finally:
        cv2.destroyAllWindows()
        camera.stop()
        print("\n测试完成")


if __name__ == "__main__":
    test_april_tag()
