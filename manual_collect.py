import numpy as np
import time
import cv2
import json
import os
from collections import deque, namedtuple
import sys
import threading
from queue import Queue
from pathlib import Path
from dm_env import specs

# 确保能找到核心模块
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from replay_buffer import ReplayBufferStorage
    import piper.env as piper_env
except ImportError as e:
    print(f"错误：无法导入核心模块: {e}")
    print("请确保 replay_buffer.py 和 piper/ 目录在当前路径下")
    sys.exit(1)

# 定义一个简单的 TimeStep 结构，用于兼容 ReplayBufferStorage
class _TimeStep(namedtuple('_TimeStep', ['observation', 'reward', 'discount', 'first', 'last'])):
    pass

class SimpleSpacemouseCollect:
    def __init__(self):
        print("="*60)
        print("     Piper 3D鼠标数据收集工具 (直连训练Buffer版)")
        print("="*60)
        print()
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'piper'))
        
        # 硬件对象
        self.robot_cam = None
        self.piper_arm = None
        
        # 控制状态
        self._running = True
        self._reward = 0.0
        self._lock = threading.Lock()
        
        # 配置参数
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.factor = 1000
        self.control_sleep = 0.01
        
        # 机械臂初始位姿
        self.X = round(300.614 * self.factor)
        self.Y = round(-12.185 * self.factor)
        self.Z = round(282.341 * self.factor)
        self.RX = round(-179.351 * self.factor)
        self.RY = round(23.933 * self.factor)
        self.RZ = round(177.934 * self.factor)
        self.joint_6 = round(0.08 * 1000 * 1000)
        
        # 数据收集相关
        self.data_queue = Queue(maxsize=1000)
        self.frames_queue = deque(maxlen=3)
        self.episode = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.data_buffer = []
        
        # 新增：ReplayBuffer 相关
        self._buffer_dir = Path.cwd() / 'buffer'
        self.replay_storage = None
        
        # 显示用
        self._latest_frame = None

    def init_hardware_and_buffer(self):
        """初始化硬件和 ReplayBuffer"""
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
        
        self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)
        print("✓ 机械臂初始化成功")
        
        # 初始化 ReplayBuffer
        self._init_replay_storage()
        print("✓ 硬件与 Buffer 初始化完成")
        print()

    def _init_replay_storage(self):
        """初始化与 train_piper.py 兼容的 ReplayBufferStorage"""
        print("\n正在初始化 ReplayBufferStorage...")
        
        # 创建临时环境获取 specs
        temp_env = piper_env.make(
            task_name='piper_push',
            seed=0,
            action_repeat=2,
            size=(256, 256),
            use_sim=True,
            frame_stack=3
        )
        
        data_specs = (
            temp_env.observation_spec(),
            temp_env.action_spec(),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount')
        )
        
        self.replay_storage = ReplayBufferStorage(data_specs, self._buffer_dir)
        temp_env.close()
        print(f"✓ Buffer 路径: {self._buffer_dir.resolve()}")

    def _image_capture_thread(self):
        while self._running:
            try:
                frame = self.robot_cam.get_camera_image()
                frame = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                with self._lock:
                    self.frames_queue.append(frame)
                    self._latest_frame = frame.copy()
                time.sleep(0.005)
            except Exception as e:
                continue

    def get_stacked_obs(self):
        with self._lock:
            frames = list(self.frames_queue)
        
        while len(frames) < 3:
            frames.append(frames[0] if frames else np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8))
        
        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))
        return stacked

    def _process_data_thread(self):
        while self._running:
            if not self.data_queue.empty():
                try:
                    data = self.data_queue.get()
                    self.data_buffer.append(data)
                except Exception as e:
                    pass
            time.sleep(0.01)

    def collect(self, num_episodes=5, max_steps=200):
        print("="*60)
        print("控制说明:")
        print("  3D鼠标移动: 控制机械臂 X/Y/Z 方向")
        print("  3D鼠标按钮0: 夹爪打开 | 按钮1: 夹爪关闭")
        print("  键盘空格: +10 奖励 | 键盘q: 退出并保存")
        print("="*60)
        print(f"📌 目标: {num_episodes} 轮。数据将自动写入训练 Buffer。")
        print()
        
        self.init_hardware_and_buffer()
        
        img_thread = threading.Thread(target=self._image_capture_thread, daemon=True)
        data_thread = threading.Thread(target=self._process_data_thread, daemon=True)
        img_thread.start()
        data_thread.start()
        
        time.sleep(0.5)
        while len(self.frames_queue) < 3:
            time.sleep(0.01)
        
        cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Data Collection", 640, 480)
        
        print(f"开始收集...请点击窗口以捕获按键！")
        print()
        
        try:
            import pyspacemouse
            with pyspacemouse.open() as device:
                while self.episode < num_episodes and self._running:
                    state = device.read()
                    
                    state_X = round(state.x * self.factor)
                    state_Y = round(state.y * self.factor)
                    state_Z = round(state.z * self.factor)
                    
                    self.X = round(self.X + state_X)
                    self.Y = round(self.Y + state_Y)
                    self.Z = round(self.Z + state_Z)
                    
                    self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                    self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
                    
                    if state.buttons[0]:
                        self.joint_6 = round(0.08 * 1000 * 1000)
                    elif state.buttons[1]:
                        self.joint_6 = round(0.00 * 1000 * 1000)
                    self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)
                    
                    key_pressed = None
                    if self._latest_frame is not None:
                        display_frame = cv2.cvtColor(self._latest_frame, cv2.COLOR_RGB2BGR)
                        cv2.putText(display_frame, f"Ep: {self.episode + 1}/{num_episodes} | Step: {self.episode_step}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"BufferSize: {len(self.replay_storage)}", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.imshow("Data Collection", display_frame)
                        key_pressed = cv2.waitKey(1) & 0xFF
                    
                    if key_pressed is not None:
                        if key_pressed == ord(' '):
                            self._reward += 10.0
                            print(f"[奖励] Ep{self.episode+1} Step{self.episode_step} | +10.0")
                        elif key_pressed == ord('q'):
                            print("\n用户退出，正在保存...")
                            self._running = False
                            break
                    
                    # 核心：写入 ReplayBuffer
                    if self.episode_step >= 0 and len(self.frames_queue) >= 3:
                        try:
                            obs_t = self.get_stacked_obs()
                            action_t = np.array([
                                np.clip(state.x * 2.0, -1.0, 1.0),
                                np.clip(state.y * 2.0, -1.0, 1.0),
                                np.clip(state.z * 2.0, -1.0, 1.0),
                                np.clip((self.joint_6 / (1000 * 1000) - 0.04) * 50.0, -1.0, 1.0)
                            ], dtype=np.float32)
                            
                            reward = self._reward
                            self._reward = 0.0
                            
                            # Buffer 写入逻辑
                            if self.episode_step == 0:
                                # 第一步：添加 FIRST 帧
                                ts = _TimeStep(observation=obs_t, reward=np.array(0.0, dtype=np.float32), 
                                              discount=np.array(1.0, dtype=np.float32), first=True, last=False)
                                self.replay_storage.add(ts)
                                self._last_obs = obs_t
                            else:
                                # 中间步：添加 MID 帧（包含上一步的 reward）
                                ts = _TimeStep(observation=obs_t, reward=np.array(reward, dtype=np.float32),
                                              discount=np.array(1.0, dtype=np.float32), first=False, last=False)
                                self.replay_storage.add(ts)
                                self._last_obs = obs_t
                            
                            # 同时保留 npz 备份逻辑
                            if not self.data_queue.full():
                                self.data_queue.put({
                                    'observation': obs_t,
                                    'action': action_t,
                                    'reward': reward,
                                    'next_observation': self.get_stacked_obs(),
                                    'discount': 1.0
                                })
                            
                        except Exception as e:
                            pass
                    
                    self.episode_reward += reward
                    self.episode_step += 1
                    
                    if self.episode_step >= max_steps:
                        print(f"\n[Ep {self.episode + 1}] 完成，奖励: {self.episode_reward:.1f}")
                        
                        # 标记 Episode 结束
                        if hasattr(self, '_last_obs'):
                            ts_last = _TimeStep(observation=self._last_obs, reward=np.array(0.0, dtype=np.float32),
                                                discount=np.array(0.0, dtype=np.float32), first=False, last=True)
                            self.replay_storage.add(ts_last)
                        
                        # 复位
                        self.X = round(300.614 * self.factor)
                        self.Y = round(-12.185 * self.factor)
                        self.Z = round(282.341 * self.factor)
                        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
                        self.piper_arm.GripperCtrl(abs(0.08*1000*1000), 1000, 0x01, 0)
                        time.sleep(0.5)
                        
                        self.episode += 1
                        self.episode_step = 0
                        self.episode_reward = 0
                    
                    time.sleep(self.control_sleep)
        except ImportError:
            print("✗ pyspacemouse 未安装: pip install pyspacemouse")
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self._running = False
            time.sleep(0.5)
            cv2.destroyAllWindows()
            if self.piper_arm is not None:
                self.piper_arm.EmergencyStop()
                self.piper_arm.DisconnectPort()
            if self.robot_cam is not None:
                self.robot_cam.close()
            
            if len(self.data_buffer) > 0:
                self.save_data()
            print(f"\n📊 收集结束！Buffer 中共有 {len(self.replay_storage)} 步有效数据")

    def save_data(self, filename="spacemouse_data_256x256x9.npz"):
        try:
            observations = np.stack([d['observation'] for d in self.data_buffer])
            actions = np.stack([d['action'] for d in self.data_buffer])
            rewards = np.array([d['reward'] for d in self.data_buffer])
            
            np.savez_compressed(filename,
                               observations=observations,
                               actions=actions,
                               rewards=rewards)
            print(f"\n✅ 备份数据已保存到: {os.path.abspath(filename)}")
        except Exception as e:
            pass

def main():
    collector = SimpleSpacemouseCollect()
    print("\n请选择:")
    print("1. 开始收集数据 (并写入训练 Buffer)")
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