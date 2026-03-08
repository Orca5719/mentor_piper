import cv2
import numpy as np
import glob
import os
from typing import List, Tuple


def generate_chessboard_images():
    """生成棋盘格标定图像（使用真实相机）"""
    print("="*70)
    print(" 相机标定 - 采集棋盘格图像")
    print("="*70)
    
    print("\n说明：")
    print("  1. 打印一张棋盘格（推荐 9x6 内角点）")
    print("  2. 将棋盘格放在摄像头前不同位置、不同角度")
    print("  3. 按 'c' 键采集图像")
    print("  4. 按 'q' 键结束采集（至少采集 10 张）")
    print("  5. 采集完成后会自动进行标定")
    
    try:
        from Camera_Module import DepthCameraModule
    except ImportError:
        print("错误：Camera_Module.py 未找到！")
        return None, None
    
    # 初始化相机
    print("\n初始化相机...")
    camera = DepthCameraModule(
        color_width=640,
        color_height=480,
        fps=30
    )
    
    # 创建保存目录
    save_dir = "calibration_images"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n图像将保存到: {save_dir}/")
    
    images = []
    count = 0
    
    try:
        while True:
            color_array_rgb, _ = camera.get_frame()
            
            if color_array_rgb is not None:
                # 转换为 BGR
                display_img = cv2.cvtColor(color_array_rgb, cv2.COLOR_RGB2BGR)
                
                # 显示提示
                cv2.putText(display_img, f"Collected: {count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_img, "Press 'c' to capture, 'q' to quit", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Camera Calibration', display_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    if count >= 10:
                        print(f"\n✓ 采集完成，共 {count} 张图像")
                        break
                    else:
                        print(f"\n⚠️  至少需要 10 张图像，当前只有 {count} 张")
                        print("   继续采集或按 'q' 强制退出")
                        if cv2.waitKey(0) & 0xFF == ord('q'):
                            break
                elif key == ord('c'):
                    count += 1
                    filename = os.path.join(save_dir, f"calib_{count:03d}.png")
                    cv2.imwrite(filename, display_img)
                    images.append(color_array_rgb)
                    print(f"✓ 已保存: {filename}")
    
    finally:
        cv2.destroyAllWindows()
        camera.stop()
    
    if count < 10:
        print("\n⚠️  图像数量不足，跳过标定")
        return None, None
    
    return save_dir, (640, 480)


def calibrate_from_images(image_dir: str, 
                          chessboard_size: Tuple[int, int] = (9, 6),
                          square_size: float = 0.025) -> dict:
    """
    从图像进行相机标定
    
    参数:
        image_dir: 图像目录
        chessboard_size: 棋盘格内角点数 (columns, rows)
        square_size: 棋盘格每个格子的大小（米）
        
    返回:
        标定结果字典，包含:
            - 'camera_matrix': 相机内参矩阵
            - 'dist_coeffs': 畸变系数
            - 'fx', 'fy', 'cx', 'cy': 相机内参
    """
    print("="*70)
    print(" 相机标定 - 计算参数")
    print("="*70)
    
    # 棋盘格设置
    cb_rows, cb_cols = chessboard_size
    
    # 准备对象点
    objp = np.zeros((cb_rows * cb_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)
    objp *= square_size
    
    objpoints = []  # 3D 点
    imgpoints = []  # 2D 点
    
    # 读取图像
    images = glob.glob(os.path.join(image_dir, '*.png')) + \
             glob.glob(os.path.join(image_dir, '*.jpg'))
    
    if not images:
        print(f"错误：在 {image_dir} 中未找到图像！")
        return None
    
    print(f"\n找到 {len(images)} 张图像")
    
    img_shape = None
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = gray.shape[::-1]
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (cb_cols, cb_rows), None)
        
        if ret:
            objpoints.append(objp)
            
            # 亚像素精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            print(f"✓ {fname}: 检测到角点")
        else:
            print(f"✗ {fname}: 未检测到角点")
    
    if len(objpoints) < 3:
        print("\n错误：有效图像太少！")
        return None
    
    print(f"\n使用 {len(objpoints)} 张图像进行标定...")
    
    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if not ret:
        print("错误：标定失败！")
        return None
    
    # 提取参数
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    
    print("\n" + "="*70)
    print(" 标定结果")
    print("="*70)
    print(f"\n相机内参矩阵:")
    print(mtx)
    print(f"\n畸变系数:")
    print(dist)
    print(f"\n相机参数 (fx, fy, cx, cy):")
    print(f"  fx = {fx:.2f}")
    print(f"  fy = {fy:.2f}")
    print(f"  cx = {cx:.2f}")
    print(f"  cy = {cy:.2f}")
    
    # 保存结果
    result = {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'img_shape': img_shape
    }
    
    # 保存到文件
    np.savez('camera_calibration.npz', **result)
    print(f"\n✓ 结果已保存到: camera_calibration.npz")
    
    # 生成配置文件片段
    config_text = f"""# 相机内参 (由标定生成)
camera_params: [{fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f}]
"""
    with open('camera_config.yaml', 'w') as f:
        f.write(config_text)
    print(f"✓ 配置片段已保存到: camera_config.yaml")
    
    return result


def load_calibration(filename: str = 'camera_calibration.npz') -> dict:
    """加载相机标定结果"""
    if not os.path.exists(filename):
        print(f"错误：标定文件 {filename} 不存在！")
        return None
    
    data = np.load(filename)
    return {
        'camera_matrix': data['camera_matrix'],
        'dist_coeffs': data['dist_coeffs'],
        'fx': float(data['fx']),
        'fy': float(data['fy']),
        'cx': float(data['cx']),
        'cy': float(data['cy'])
    }


def main():
    print("="*70)
    print(" 相机标定工具")
    print("="*70)
    
    # 询问是否已有图像
    choice = input("\n选择操作:\n  1. 采集新图像并标定\n  2. 从已有图像标定\n\n请输入 (1/2): ").strip()
    
    if choice == '1':
        # 采集新图像
        image_dir, img_shape = generate_chessboard_images()
        if image_dir is not None:
            calibrate_from_images(image_dir)
    elif choice == '2':
        # 从已有图像
        image_dir = input("请输入图像目录路径: ").strip()
        if os.path.exists(image_dir):
            calibrate_from_images(image_dir)
        else:
            print(f"错误：目录 {image_dir} 不存在！")
    else:
        print("无效选择！")


if __name__ == "__main__":
    main()
