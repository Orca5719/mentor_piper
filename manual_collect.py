import numpy as np
import time
import cv2
import json
import os
from collections import deque
import sys
import threading
from queue import Queue

class SimpleSpacemouseCollect:
    def __init__(self):
        print("="*60)
        print("     Piper 3D鼠标数据收集工具 (适配256x256×9输入)")
        print("="*60)
        print()
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'piper'))
        
        # 硬件对象
        self.robot_cam = None
        self.piper_arm = None
        
        # 控制状态
        self._running = True
        self._reward = 0.0  # 每步奖励缓存
        self._lock = threading.Lock()  # 线程安全锁
        
        # 配置参数
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.factor = 1000  # 和参考代码对齐
        self.control_sleep = 0.01  # 控制循环sleep（参考原代码）
        
        # 机械臂初始位姿（完全复用参考代码）
        self.X = round(300.614 * self.factor)
        self.Y = round(-12.185 * self.factor)
        self.Z = round(282.341 * self.factor)
        self.RX = round(-179.351 * self.factor)
        self.RY = round(23.933 * self.factor)
        self.RZ = round(177.934 * self.factor)
        self.joint_6 = round(0.08 * 1000 * 1000)
        
        # 数据收集相关
        self.data_queue = Queue(maxsize=1000)  # 非阻塞数据队列
        self.frames_queue = deque(maxlen=3)    # 帧缓存队列
        self.episode = 0
        self.episode_step = 0
        self.episode_reward = 0  # 累计episode奖励
        self.data_buffer = []
        
        # 新增：用于显示的最新帧
        self._latest_frame = None

    def init_hardware(self):
        """硬件初始化（和参考代码对齐，简化逻辑）"""
        from piper.robot import PiperRobot
        from piper_sdk import C_PiperInterface_V2
        
        print("正在初始化相机...")
        self.robot_cam = PiperRobot(
            use_sim=False,
            camera_width=self.IMG_WIDTH,
            camera_height=self.IMG_HEIGHT,
            use_apriltag=True,
            tag_size=0.05
        )
        print("✓ 相机初始化成功")
        
        print("正在初始化机械臂...")
        self.piper_arm = C_PiperInterface_V2("can0")
        self.piper_arm.ConnectPort()
        while not self.piper_arm.EnablePiper():
            time.sleep(0.01)
        # 初始位姿设置（参考原代码）
        self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)
        print("✓ 机械臂初始化成功")
        print("✓ 硬件初始化完成")
        print()

    def _image_capture_thread(self):
        """独立线程获取相机图像，避免阻塞控制循环"""
        while self._running:
            try:
                # 获取图像并resize（非阻塞方式）
                frame = self.robot_cam.get_camera_image()
                frame = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self.frames_queue.append(frame)
                    self._latest_frame = frame.copy()  # 保存用于显示
                time.sleep(0.005)  # 图像采集频率略高于控制频率
            except Exception as e:
                print(f"图像采集线程异常: {e}")
                continue

    def get_stacked_obs(self):
        """优化版：快速生成9通道堆叠观测"""
        with self._lock:
            frames = list(self.frames_queue)
        
        # 补全3帧
        while len(frames) < 3:
            frames.append(frames[0] if frames else np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8))
        
        # 堆叠并转置（简化计算）
        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))
        return stacked

    def _process_data_thread(self):
        """独立线程处理数据缓存，不影响控制循环"""
        while self._running:
            if not self.data_queue.empty():
                try:
                    # 从队列取出数据并缓存
                    data = self.data_queue.get()
                    self.data_buffer.append(data)
                except Exception as e:
                    print(f"数据处理线程异常: {e}")
            time.sleep(0.01)

    def collect(self, num_episodes=5, max_steps=200):
        print("="*60)
        print("控制说明:")
        print("  3D鼠标移动: 控制机械臂 X/Y/Z 方向")
        print("  3D鼠标按钮0: 夹爪打开")
        print("  3D鼠标按钮1: 夹爪关闭")
        print("  键盘空格: 立即获得 +10 奖励（每按一次生效一次）")
        print("  键盘q: 退出（会保存已收集数据）")
        print("="*60)
        print(f"📌 本次将收集 {num_episodes} 轮数据，收集完成后统一保存！")
        print()
        
        # 初始化硬件
        self.init_hardware()
        
        # 启动辅助线程（图像采集+数据处理）
        img_thread = threading.Thread(target=self._image_capture_thread, daemon=True)
        data_thread = threading.Thread(target=self._process_data_thread, daemon=True)
        img_thread.start()
        data_thread.start()
        
        # 预填充帧队列
        time.sleep(0.5)
        while len(self.frames_queue) < 3:
            time.sleep(0.01)
        
        # 【关键修改】创建OpenCV窗口（必须有窗口才能捕获按键！）
        cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Data Collection", 640, 480)
        
        print(f"开始收集数据 (目标 {num_episodes} 个 episode)...")
        print(f"输出格式: 观测={self.IMG_HEIGHT}x{self.IMG_WIDTH}×9通道 | 动作=4维 | 奖励=标量")
        print("⚠️  请确保点击了 'Data Collection' 窗口，否则按键可能无效！")
        print()

        try:
            import pyspacemouse
            with pyspacemouse.open() as device:
                # 核心控制循环（参考piper_spacemouse.py的简洁逻辑）
                while self.episode < num_episodes and self._running:
                    # 1. 快速读取3D鼠标状态（优先操作）
                    state = device.read()
                    
                    # 2. 机械臂位姿更新（和参考代码完全对齐）
                    state_X = round(state.x * self.factor)
                    state_Y = round(state.y * self.factor)
                    state_Z = round(state.z * self.factor)
                    
                    self.X = round(self.X + state_X)
                    self.Y = round(self.Y + state_Y)
                    self.Z = round(self.Z + state_Z)
                    self.RX = round(self.RX)
                    self.RY = round(self.RY)
                    self.RZ = round(self.RZ)
                    
                    # 3. 发送机械臂控制指令（无延迟）
                    self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                    self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
                    
                    # 4. 夹爪控制（参考代码逻辑）
                    if state.buttons[0]:
                        self.joint_6 = round(0.08 * 1000 * 1000)
                    elif state.buttons[1]:
                        self.joint_6 = round(0.00 * 1000 * 1000)
                    self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)
                    
                    # 5. 【关键修改】显示图像+键盘处理（必须显示窗口才能捕获按键）
                    key_pressed = None
                    if self._latest_frame is not None:
                        # 转换RGB到BGR用于OpenCV显示
                        display_frame = cv2.cvtColor(self._latest_frame, cv2.COLOR_RGB2BGR)
                        # 在图像上绘制状态信息
                        cv2.putText(display_frame, f"Episode: {self.episode + 1}/{num_episodes}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Step: {self.episode_step}/{max_steps}", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Reward: {self.episode_reward + self._reward:.1f}", (10, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Data: {len(self.data_buffer)}", (10, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        cv2.putText(display_frame, "SPACE=+10 | Q=Quit", (10, 190),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                        # 显示图像
                        cv2.imshow("Data Collection", display_frame)
                        # 【关键】等待按键（1ms，必须配合imshow使用）
                        key_pressed = cv2.waitKey(1) & 0xFF
                    
                    # 6. 处理按键
                    if key_pressed is not None:
                        if key_pressed == ord(' '):
                            self._reward += 10.0  # 每按一次空格 +10奖励
                            current_step = self.episode_step + 1
                            current_ep = self.episode + 1
                            print(f"[奖励] Episode {current_ep} Step {current_step} | +10.0 | 累计奖励: {self.episode_reward + self._reward:.1f}")
                        elif key_pressed == ord('q') or key_pressed == ord('Q'):
                            print("\n用户主动退出，开始保存已收集数据...")
                            self._running = False
                            break
                    
                    # 7. 数据收集（非阻塞方式放入队列）
                    if self.episode_step >= 0 and len(self.frames_queue) >= 3:  # 从step0就开始收集
                        try:
                            # 快速生成观测（不阻塞控制循环）
                            obs_prev = self.get_stacked_obs()
                            # 构建动作向量
                            action = np.array([
                                np.clip(state.x * 2.0, -1.0, 1.0),
                                np.clip(state.y * 2.0, -1.0, 1.0),
                                np.clip(state.z * 2.0, -1.0, 1.0),
                                np.clip((self.joint_6 / (1000 * 1000) - 0.04) * 50.0, -1.0, 1.0)
                            ], dtype=np.float32)
                            # 奖励处理：取出当前奖励，然后重置
                            reward = self._reward
                            self.episode_reward += reward  # 累计到episode奖励
                            self._reward = 0.0  # 重置奖励缓存，避免重复计算
                            # 放入队列（非阻塞）
                            if not self.data_queue.full():
                                self.data_queue.put({
                                    'observation': obs_prev,
                                    'action': action,
                                    'reward': reward,
                                    'next_observation': self.get_stacked_obs(),
                                    'discount': 1.0
                                })
                        except Exception as e:
                            print(f"数据收集临时异常: {e}")
                    
                    # 8. Episode管理（简化逻辑）
                    self.episode_step += 1
                    
                    if self.episode_step >= max_steps:
                        print(f"\n[Episode {self.episode + 1}/{num_episodes}] 完成，总奖励: {self.episode_reward:.1f}")
                        # 机械臂回到初始位姿（快速重置）
                        self.X = round(300.614 * self.factor)
                        self.Y = round(-12.185 * self.factor)
                        self.Z = round(282.341 * self.factor)
                        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
                        self.piper_arm.GripperCtrl(abs(0.08*1000*1000), 1000, 0x01, 0)
                        time.sleep(0.5)  # 缩短等待时间
                        
                        # 重置计数
                        self.episode += 1
                        self.episode_step = 0
                        self.episode_reward = 0
                    
                    # 核心：控制循环sleep（和参考代码一致）
                    time.sleep(self.control_sleep)

        except ImportError:
            print("✗ pyspacemouse 未安装，请先安装: pip install pyspacemouse")
        except KeyboardInterrupt:
            print("\n用户中断，开始保存已收集数据...")
        finally:
            # 停止线程
            self._running = False
            time.sleep(0.5)
            # 清理资源
            cv2.destroyAllWindows()
            if self.piper_arm is not None:
                self.piper_arm.EmergencyStop()
                self.piper_arm.DisconnectPort()
            if self.robot_cam is not None:
                self.robot_cam.close()
        
        # 收集完成/退出后 统一保存所有数据
        print(f"\n📊 数据收集结束！共收集 {len(self.data_buffer)} 条有效数据")
        if len(self.data_buffer) > 0:
            self.save_data()
        else:
            print("⚠️  无有效数据可保存")

    def save_data(self, filename="spacemouse_data_256x256x9.npz"):
        """批量保存数据（优化版）"""
        try:
            # 批量转换为numpy数组（减少循环次数）
            observations = np.stack([d['observation'] for d in self.data_buffer])
            actions = np.stack([d['action'] for d in self.data_buffer])
            rewards = np.array([d['reward'] for d in self.data_buffer])
            next_observations = np.stack([d['next_observation'] for d in self.data_buffer])
            discounts = np.array([d['discount'] for d in self.data_buffer])
            
            print(f"\n📋 数据形状验证:")
            print(f"  observations: {observations.shape} (预期: [N, 9, 256, 256])")
            print(f"  actions: {actions.shape} (预期: [N, 4])")
            print(f"  rewards: {rewards.shape} (预期: [N])")
            
            # 打印奖励统计（验证空格奖励是否生效）
            total_reward = np.sum(rewards)
            reward_count = np.sum(rewards > 0)
            print(f"\n💰 奖励统计:")
            print(f"  总奖励: {total_reward:.1f}")
            print(f"  获得奖励的步数: {reward_count}")
            
            # 压缩保存
            np.savez_compressed(filename,
                               observations=observations,
                               actions=actions,
                               rewards=rewards,
                               next_observations=next_observations,
                               discounts=discounts)
            print(f"\n✅ 数据已统一保存到: {os.path.abspath(filename)}")
            print(f"  总数据量: {len(self.data_buffer)} transitions")
            
            # 保存后清空缓冲区
            self.data_buffer = []
        except Exception as e:
            print(f"\n❌ 数据保存失败: {e}")
            import traceback
            traceback.print_exc()


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
