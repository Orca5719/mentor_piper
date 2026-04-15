import numpy as np
import time
import cv2
import json
import os
from collections import deque
import sys


class SimpleSpacemouseCollect:
    def __init__(self):
        print("="*60)
        print("     Piper 3D鼠标数据收集工具 (适配256x256×9输入)")
        print("="*60)
        print()
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'piper'))
        
        # 分离：PiperRobot仅用于相机，piper_sdk用于机械臂控制
        self.robot_cam = None  # 仅用于获取相机图像
        self.piper_arm = None  # 原生SDK控制机械臂
        self._running = True
        self._reward = 0.0
        # 新增：训练输入尺寸配置
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
    
    def init_hardware(self):
        from piper.robot import PiperRobot
        from piper_sdk import C_PiperInterface_V2
        
        print("正在初始化相机...")
        # 修改：相机初始分辨率设为256x256（双重保障，后续仍会resize）
        self.robot_cam = PiperRobot(
            use_sim=False,
            camera_width=self.IMG_WIDTH,
            camera_height=self.IMG_HEIGHT,
            use_apriltag=True,
            tag_size=0.05
        )
        print("✓ 相机初始化成功")
        
        print("正在初始化机械臂...")
        # 初始化原生SDK机械臂控制
        self.piper_arm = C_PiperInterface_V2("can0")
        self.piper_arm.ConnectPort()
        while not self.piper_arm.EnablePiper():
            time.sleep(0.01)
        print("✓ 机械臂使能成功")
        
        # 初始化夹爪
        self.piper_arm.GripperCtrl(0, 1000, 0x01, 0)
        print("✓ 硬件初始化完成")
        print()
    
    # 核心修改：适配256x256×9通道堆叠
    def get_stacked_obs(self, frames):
        """
        生成256x256×9通道的堆叠观测（3帧×3通道）
        :param frames: 3帧RGB图像列表
        :return: (9, 256, 256) 格式的堆叠张量
        """
        if len(frames) < 3:
            while len(frames) < 3:
                frames.append(frames[0])
        
        # 确保每帧都是256x256分辨率
        resized_frames = []
        for frame in frames:
            # 强制resize到256x256（即使相机输出不同分辨率也能适配）
            resized = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            resized_frames.append(resized)
        
        # 堆叠逻辑：3帧拼接成9通道 (256,256,9)
        stacked = np.concatenate(resized_frames, axis=-1)
        # 转置为 (9, 256, 256) 适配训练输入格式
        stacked = np.transpose(stacked, (2, 0, 1))
        return stacked
    
    def collect(self, num_episodes=5, max_steps=200):
        print("="*60)
        print("控制说明:")
        print("  3D鼠标移动: 控制机械臂 X/Y/Z 方向")
        print("  3D鼠标按钮0: 夹爪打开")
        print("  3D鼠标按钮1: 夹爪关闭")
        print("  键盘空格: +10 奖励")
        print("  键盘s: 保存数据")
        print("  键盘q: 退出")
        print("="*60)
        print()
        
        # 初始化硬件
        if self.robot_cam is None or self.piper_arm is None:
            self.init_hardware()
        
        # 机械臂初始位姿（参考你的正确代码）
        factor = 1000
        X = 300.614
        Y = -12.185
        Z = 282.341
        RX = -179.351
        RY = 23.933
        RZ = 177.934
        
        # 单位转换
        X = round(X * factor)
        Y = round(Y * factor)
        Z = round(Z * factor)
        RX = round(RX * factor)
        RY = round(RY * factor)
        RZ = round(RZ * factor)
        joint_6 = round(0.08 * 1000 * 1000)  # 初始夹爪打开
        
        # 机械臂移动到初始位姿
        self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper_arm.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        self.piper_arm.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        time.sleep(1.0)  # 等待机械臂到位
        
        # 数据收集初始化
        data_buffer = []
        episode = 0
        episode_step = 0
        episode_reward = 0
        frames = deque(maxlen=3)
        
        # 预填充帧队列（已适配256x256）
        for _ in range(3):
            frames.append(self.robot_cam.get_camera_image())
        
        # 初始化动作
        action = np.zeros(4, dtype=np.float32)
        
        print(f"开始收集数据 (目标 {num_episodes} 个 episode)...")
        print(f"输出格式: 观测={self.IMG_HEIGHT}x{self.IMG_WIDTH}×9通道 | 动作=4维 | 奖励=标量")
        print()
        
        try:
            import pyspacemouse
            with pyspacemouse.open() as device:
                while episode < num_episodes and self._running:
                    # 读取3D鼠标状态
                    state = device.read()
                    
                    # 3D鼠标控制机械臂
                    state_X = round(state.x * factor)
                    state_Y = round(state.y * factor)
                    state_Z = round(state.z * factor)
                    
                    # 更新位姿（RX/RY/RZ保持不变）
                    X = round(X + state_X)
                    Y = round(Y + state_Y)
                    Z = round(Z + state_Z)
                    RX = round(RX)
                    RY = round(RY)
                    RZ = round(RZ)
                    
                    # 发送控制指令
                    self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                    self.piper_arm.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
                    
                    # 3D鼠标按钮控制夹爪
                    if state.buttons[0]:
                        joint_6 = round(0.08 * 1000 * 1000)  # 打开
                    elif state.buttons[1]:
                        joint_6 = round(0.00 * 1000 * 1000)  # 关闭
                    self.piper_arm.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                    
                    # 构建动作向量（用于数据保存）
                    action = np.array([
                        state.x * 2.0,  # 缩放动作值到[-1, 1]
                        state.y * 2.0,
                        state.z * 2.0,
                        (joint_6 / (1000 * 1000) - 0.04) * 50.0  # 夹爪动作归一化
                    ], dtype=np.float32)
                    action = np.clip(action, -1.0, 1.0)
                    
                    # 处理键盘辅助按键（奖励、保存、退出）
                    cv2_key = cv2.waitKey(1) & 0xFF
                    if cv2_key == ord(' '):
                        self._reward += 10.0
                        print(f"[奖励] +10.0")
                    elif cv2_key == ord('s'):
                        self.save_data(data_buffer)
                    elif cv2_key == ord('q') or cv2_key == ord('Q'):
                        print("\n用户退出")
                        self._running = False
                        break
                    
                    # 获取相机图像（后续堆叠时会resize到256x256）
                    frames.append(self.robot_cam.get_camera_image())
                    
                    # 处理奖励
                    reward = self._reward
                    if self._reward != 0:
                        self._reward = 0
                    
                    # 保存数据（观测已适配256x256×9）
                    if episode_step > 0:
                        obs_prev = self.get_stacked_obs(list(frames)[:-1])
                        obs_curr = self.get_stacked_obs(list(frames))
                        data_buffer.append({
                            'observation': obs_prev,
                            'action': action,
                            'reward': reward,
                            'next_observation': obs_curr,
                            'discount': 1.0
                        })
                    
                    # 更新计数
                    episode_reward += reward
                    episode_step += 1
                    
                    # 显示画面（适配256x256分辨率）
                    frame = self.robot_cam.get_camera_image()
                    frame = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(frame_bgr, f"Episode: {episode + 1}/{num_episodes}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f"Step: {episode_step}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f"Reward: {episode_reward:.1f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f"Data: {len(data_buffer)} transitions", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(frame_bgr, f"Input: {self.IMG_HEIGHT}x{self.IMG_WIDTH}×9", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    cv2.imshow("3D Mouse Collect (256x256×9)", frame_bgr)
                    
                    # Episode结束判断
                    if episode_step >= max_steps:
                        print(f"\n[Episode {episode + 1}/{num_episodes}] 完成，奖励: {episode_reward:.1f}")
                        episode += 1
                        episode_step = 0
                        episode_reward = 0
                        
                        # 机械臂回到初始位姿
                        X = round(300.614 * factor)
                        Y = round(-12.185 * factor)
                        Z = round(282.341 * factor)
                        self.piper_arm.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
                        time.sleep(1.0)
                        
                        # 预填充帧队列
                        for _ in range(3):
                            frames.append(self.robot_cam.get_camera_image())
                        
                        # 定期保存数据
                        if episode % 2 == 0:
                            self.save_data(data_buffer)
                    
                    time.sleep(0.01)
        
        except ImportError:
            print("✗ pyspacemouse 未安装，请先安装: pip install pyspacemouse")
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            cv2.destroyAllWindows()
            # 安全关闭硬件
            if self.piper_arm is not None:
                self.piper_arm.EmergencyStop()
                self.piper_arm.DisconnectPort()
            if self.robot_cam is not None:
                self.robot_cam.close()
        
        print(f"\n数据收集完成！共收集 {len(data_buffer)} 条数据")
        print(f"输出格式验证: 观测形状 = {data_buffer[0]['observation'].shape if data_buffer else '空'}")
        self.save_data(data_buffer)
    
    def save_data(self, data_buffer, filename="spacemouse_data_256x256x9.npz"):
        if len(data_buffer) == 0:
            print("⚠️  没有数据可保存")
            return
        
        try:
            observations = np.array([d['observation'] for d in data_buffer])
            actions = np.array([d['action'] for d in data_buffer])
            rewards = np.array([d['reward'] for d in data_buffer])
            next_observations = np.array([d['next_observation'] for d in data_buffer])
            discounts = np.array([d['discount'] for d in data_buffer])
            
            print(f"\n数据形状验证:")
            print(f"  observations: {observations.shape} (预期: [N, 9, 256, 256])")
            print(f"  actions: {actions.shape} (预期: [N, 4])")
            print(f"  rewards: {rewards.shape} (预期: [N])")
            
            np.savez_compressed(filename,
                               observations=observations,
                               actions=actions,
                               rewards=rewards,
                               next_observations=next_observations,
                               discounts=discounts)
            print(f"\n✓ 数据已保存到: {filename}")
            print(f"  数据量: {len(data_buffer)} transitions")
        except Exception as e:
            print(f"\n✗ 保存失败: {e}")


def main():
    collector = SimpleSpacemouseCollect()
    
    print("\n请选择:")
    print("1. 开始收集数据")
    print("2. 退出")
    
    choice = input("\n请输入选项 (1-2): ").strip()
    
    if choice == '1':
        num_episodes = int(input("收集多少个 episode? (默认 5): ") or "5")
        max_steps = int(input("每个 episode 多少步? (默认 200): ") or "200")
        collector.collect(num_episodes=num_episodes, max_steps=max_steps)
    else:
        print("退出")


if __name__ == '__main__':
    main()