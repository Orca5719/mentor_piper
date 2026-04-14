import json
import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'piper'))

from piper.robot import PiperRobot


def calibrate_hand_eye():
    print("="*60)
    print("          Piper 机械臂手眼标定（简化版）")
    print("="*60)
    print()
    
    config = {
        "camera_to_robot_offset": [0.0, 0.0, 0.0],
        "rotation_offset": 0.0,
        "scale_factor": 1.0
    }
    
    output_file = "simple_hand_eye.json"
    
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
    print("当前配置:")
    print(f"  camera_to_robot_offset: {config['camera_to_robot_offset']}")
    print(f"  rotation_offset: {config['rotation_offset']}")
    print(f"  scale_factor: {config['scale_factor']}")
    print()
    
    print("请选择操作:")
    print("  1. 直接保存当前配置")
    print("  2. 启动相机和机械臂进行标定")
    print("  3. 退出")
    
    choice = input("\n请输入选项 (1/2/3): ").strip()
    
    if choice == '1':
        save_config(config, output_file)
        return
    
    if choice == '3':
        print("退出")
        return
    
    if choice != '2':
        print("无效选项")
        return
    
    print()
    print("="*60)
    print("步骤 1: 初始化相机和机械臂")
    print("="*60)
    print()
    
    try:
        robot = PiperRobot(
            use_sim=False,
            camera_width=640,
            camera_height=480,
            use_apriltag=True,
            tag_size=0.05
        )
        print("✓ 机械臂和相机初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("="*60)
    print("步骤 2: 显示相机画面")
    print("="*60)
    print()
    print("操作说明:")
    print("  - 按 's' 保存当前配置")
    print("  - 按 'q' 退出")
    print()
    
    try:
        while True:
            frame = robot.get_camera_image()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            display_frame = frame_bgr.copy()
            
            obj_pos = robot.get_obj_pos()
            goal_pos = robot.get_goal_pos()
            
            y_pos = 30
            line_spacing = 30
            
            cv2.putText(display_frame, f"Object Position: {obj_pos}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_pos += line_spacing
            
            cv2.putText(display_frame, f"Goal Position: {goal_pos}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_pos += line_spacing
            
            cv2.putText(display_frame, "Press 's' to save, 'q' to quit", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            
            cv2.imshow("Hand-Eye Calibration", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n用户退出")
                break
            
            if key == ord('s'):
                save_config(config, output_file)
                print("配置已保存，继续显示画面...")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cv2.destroyAllWindows()
        robot.close()
        print("\n清理完成")


def save_config(config, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✓ 标定配置已保存到: {filename}")
        print("\n配置内容:")
        print(json.dumps(config, indent=2))
    except Exception as e:
        print(f"\n✗ 保存失败: {e}")


if __name__ == "__main__":
    calibrate_hand_eye()
