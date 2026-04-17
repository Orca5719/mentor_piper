import numpy as np
import time
import cv2
import os
from collections import deque, namedtuple
import sys
import threading
from pathlib import Path
from dm_env import specs

# 确保核心模块路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from replay_buffer import ReplayBufferStorage
    import piper.env as piper_env
except ImportError as e:
    print(f"错误：无法导入核心模块: {e}")
    sys.exit(1)

# ====================== 核心修复1：正确的TimeStep定义（支持字符串索引） ======================
# 字段名必须与下方data_specs的name完全一一对应
_TimeStepBase = namedtuple('_TimeStepBase', [
    'observation', 'action', 'reward', 'discount', 'first', 'is_last'
])

class TimeStep(_TimeStepBase):
    def __getitem__(self, key):
        # 解决replay_buffer中time_step[spec.name]的字符串索引问题
        if isinstance(key, str):
            return getattr(self, key)
        return super().__getitem__(key)
    
    def last(self):
        return self.is_last

class SimpleSpacemouseCollect:
    def __init__(self):
        print("="*60)
        print("     Piper 3D鼠标数据收集工具 (Buffer写入最终修复版)")
        print("="*60)
        
        self.end = False
        self._running = True
        self._lock = threading.Lock()
        
        # 硬件参数
        self.robot_cam = None
        self.piper_arm = None
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.factor = 1000
        self.control_sleep = 0.01
        
        # 机械臂初始位姿
        self.X, self.Y, self.Z = 300614, -12185, 282341
        self.RX, self.RY, self.RZ = -179351, 23933, 177934
        self.joint_6 = 80000 
        
        # 数据缓存
        self.frames_queue = deque(maxlen=3)
        self.data_buffer = []  # NPZ备份
        self._buffer_dir = Path.cwd() / 'buffer'
        self.replay_storage = None
        
        # 状态变量
        self.episode = 0
        self.episode_step = 0
        self._last_action = None
        self._obs_spec = None
        self._act_spec = None
        self._latest_frame = None

    def _init_replay_storage(self):
        """初始化ReplayBuffer，严格匹配spec字段名和类型"""
        print(f"\n[Buffer] 数据将保存至: {self._buffer_dir.resolve()}")
        
        # 创建临时环境获取真实spec
        temp_env = piper_env.make(
            task_name='piper_push', seed=0, action_repeat=2,
            size=(self.IMG_HEIGHT, self.IMG_WIDTH), use_sim=True, frame_stack=3
        )
        self._obs_spec = temp_env.observation_spec()
        self._act_spec = temp_env.action_spec()
        
        # ====================== 核心修复2：spec.name与TimeStep字段完全对应 ======================
        data_specs = (
            specs.Array(self._obs_spec.shape, self._obs_spec.dtype, 'observation'),
            specs.Array(self._act_spec.shape, self._act_spec.dtype, 'action'),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount')
        )
        
        self.replay_storage = ReplayBufferStorage(data_specs, self._buffer_dir)
        temp_env.close()
        
        # 打印spec信息方便调试
        print(f"[Buffer] Observation: {self._obs_spec.shape}, {self._obs_spec.dtype}")
        print(f"[Buffer] Action: {self._act_spec.shape}, {self._act_spec.dtype}")
        print("✅ ReplayBuffer初始化完成")

    def init_hardware(self):
        """初始化硬件（保留你原有的相机/机械臂逻辑）"""
        from piper.robot import PiperRobot
        from piper_sdk import C_PiperInterface_V2
        
        print("\n正在初始化相机...")
        self.robot_cam = PiperRobot(
            use_sim=False,
            camera_width=self.IMG_WIDTH,
            camera_height=self.IMG_HEIGHT,
            use_apriltag=True,
            tag_size=0.05
        )
        print("✅ 相机初始化成功")
        
        print("正在初始化机械臂...")
        self.piper_arm = C_PiperInterface_V2("can0")
        self.piper_arm.ConnectPort()
        while not self.piper_arm.EnablePiper():
            time.sleep(0.01)
        
        self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)
        print("✅ 机械臂初始化成功")
        
        # 初始化Buffer
        self._init_replay_storage()
        
        # 启动图像采集线程
        threading.Thread(target=self._image_thread, daemon=True).start()
        time.sleep(0.5)  # 等待相机预热

    def set_gripper_open(self):
        """夹爪打开"""
        self.joint_6 = 80000
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)
        time.sleep(0.5)

    def get_stacked_obs(self):
        """获取堆叠帧，严格匹配obs_spec的dtype"""
        with self._lock:
            frames = list(self.frames_queue)
        
        while len(frames) < 3:
            frames.append(frames[0] if frames else np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8))
        
        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))
        
        # 自动匹配obs_spec的dtype
        if self._obs_spec.dtype == np.uint8:
            return stacked.astype(np.uint8)
        return (stacked.astype(np.float32) / 255.0)

    # ====================== 核心修复3：Action对齐逻辑 ======================
    def align_action(self, x, y, z, gripper_raw):
        """将3D鼠标输入对齐到环境要求的action形状和范围"""
        # 归一化到[-1, 1]
        dx, dy, dz = [np.clip(v * 2.0, -1.0, 1.0) for v in [x, y, z]]
        dg = np.clip((gripper_raw / 1000000.0 - 0.04) * 50.0, -1.0, 1.0)
        
        # 自动适配不同长度的action维度
        action = np.zeros(self._act_spec.shape, dtype=self._act_spec.dtype)
        if self._act_spec.shape[0] >= 3:
            action[0:3] = [dx, dy, dz]
        if self._act_spec.shape[0] >= 7:
            action[6] = dg  # 7维action：夹爪在第7位
        elif self._act_spec.shape[0] == 4:
            action[3] = dg  # 4维action：夹爪在第4位
        
        return action

    def _image_thread(self):
        """图像采集线程"""
        while self._running:
            try:
                frame = self.robot_cam.get_camera_image()
                if frame is not None:
                    frame = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                    with self._lock:
                        self.frames_queue.append(frame)
                        self._latest_frame = frame.copy()
            except Exception as e:
                print(f"⚠️ 图像采集异常: {e}")
            time.sleep(0.002)

    def collect(self, num_episodes=5, max_steps=200, episode_sleep=2.0):
        """主收集循环"""
        self.init_hardware()
        cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Data Collection", 640, 480)
        
        print("\n" + "="*60)
        print("控制说明:")
        print("  3D鼠标移动: 控制机械臂X/Y/Z")
        print("  按钮0: 夹爪打开 | 按钮1: 夹爪关闭")
        print("  空格: +10奖励 | Q: 退出并保存")
        print("="*60)
        
        try:
            import pyspacemouse
            with pyspacemouse.open() as device:
                while self.episode < num_episodes and self._running:
                    state = device.read()
                    
                    # 1. 机械臂物理控制
                    self.X += round(state.x * self.factor)
                    self.Y += round(state.y * self.factor)
                    self.Z += round(state.z * self.factor)
                    
                    if state.buttons[0]:
                        self.joint_6 = 80000
                    elif state.buttons[1]:
                        self.joint_6 = 0
                    
                    self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
                    self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)
                    
                    # 2. 可视化与键盘输入
                    current_reward = 0.0
                    if self._latest_frame is not None:
                        display = cv2.cvtColor(self._latest_frame, cv2.COLOR_RGB2BGR)
                        cv2.putText(display, f"Ep:{self.episode+1}/{num_episodes} Step:{self.episode_step}/{max_steps}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display, f"Buffer: {len(self.replay_storage)}", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.imshow("Data Collection", display)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        self.end=True
                        current_reward = 10.0
                        print(f"[奖励] Ep{self.episode+1} Step{self.episode_step} | +10.0")
                    elif key == ord('q'):
                        print("\n用户终止收集，正在保存数据...")
                        self._running = False
                        break
                    
                    # ====================== 核心修复4：每一步正确写入TimeStep ======================
                    if len(self.frames_queue) >= 3:
                        obs_t = self.get_stacked_obs()
                        action_t = self.align_action(state.x, state.y, state.z, self.joint_6)
                        self._last_action = action_t
                        
                        # 构造完全符合spec要求的TimeStep
                        ts = TimeStep(
                            observation=obs_t,
                            action=action_t,
                            reward=np.array([current_reward], dtype=np.float32),  # 必须是(1,)数组
                            discount=np.array([1.0], dtype=np.float32),          # 必须是(1,)数组
                            first=(self.episode_step == 0),
                            is_last=False
                        )
                        
                        # 写入ReplayBuffer（不会立即落盘，会缓存到episode结束）
                        self.replay_storage.add(ts)
                        
                        # 同时写入NPZ备份
                        self.data_buffer.append({
                            'observation': obs_t,
                            'action': action_t,
                            'reward': current_reward
                        })
                    
                    self.episode_step += 1
                    
                    # ====================== 核心修复5：Episode结束写入LAST帧（触发落盘） ======================
                    if self.episode_step >= max_steps or self.end:
                        # 必须写入is_last=True的帧，ReplayBuffer才会将整个episode写入磁盘
                        ts_last = TimeStep(
                            observation=self.get_stacked_obs(),
                            action=self._last_action,
                            reward=np.array([0.0], dtype=np.float32),
                            discount=np.array([0.0], dtype=np.float32),  # 结束帧discount必须为0
                            first=False,
                            is_last=True
                        )
                        self.replay_storage.add(ts_last)
                        
                        print(f"\n✅ Episode {self.episode+1} 完成，已写入Buffer")
                        print(f"   当前Buffer总步数: {len(self.replay_storage)}")
                        
                        # 机械臂复位
                        self.X, self.Y, self.Z = 300614, -12185, 282341
                        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
                        self.set_gripper_open()
                        
                        print(f"请重新摆放物体... 休眠 {episode_sleep}s")
                        time.sleep(episode_sleep)
                        
                        # 重置episode状态
                        self.episode += 1
                        self.episode_step = 0
                        self.end = False
                    
                    time.sleep(self.control_sleep)
        
        except ImportError:
            print("❌ 错误：pyspacemouse未安装，请执行 pip install pyspacemouse")
        except KeyboardInterrupt:
            print("\n用户中断收集")
        finally:
            # 资源释放
            self._running = False
            time.sleep(0.3)  # 等待线程退出
            cv2.destroyAllWindows()
            
            if self.piper_arm is not None:
                self.piper_arm.EmergencyStop()
                self.piper_arm.DisconnectPort()
                print("✅ 机械臂已断开")
            
            if self.robot_cam is not None:
                self.robot_cam.close()
                print("✅ 相机已释放")
            
            # 保存NPZ备份
            self.save_npz()
            
            print(f"\n📊 收集完成！Buffer总有效步数: {len(self.replay_storage)}")
            print(f"📂 Buffer文件位置: {self._buffer_dir.resolve()}")

    def save_npz(self, filename="spacemouse_backup.npz"):
        """保存NPZ格式备份"""
        if not self.data_buffer:
            print("⚠️ 没有数据需要备份")
            return
        
        try:
            observations = np.stack([d['observation'] for d in self.data_buffer])
            actions = np.stack([d['action'] for d in self.data_buffer])
            rewards = np.array([d['reward'] for d in self.data_buffer])
            
            np.savez_compressed(
                filename,
                observations=observations,
                actions=actions,
                rewards=rewards
            )
            print(f"✅ NPZ备份已保存: {os.path.abspath(filename)}")
        except Exception as e:
            print(f"❌ NPZ备份失败: {e}")

def main():
    collector = SimpleSpacemouseCollect()
    while True:
        print("\n" + "="*45)
        print("      Piper 数据收集系统")
        print("="*45)
        print(" 1. 开始数据收集")
        print(" 2. 退出")
        
        choice = input("\n请选择 (1/2): ").strip()
        if choice == '1':
            try:
                num_ep = int(input("收集轮数 [默认5]: ") or "5")
                max_st = int(input("每轮步数 [默认200]: ") or "200")
                ep_slp = float(input("轮间休眠(s) [默认2.0]: ") or "2.0")
                collector.collect(num_episodes=num_ep, max_steps=max_st, episode_sleep=ep_slp)
            except ValueError:
                print("❌ 输入错误，请输入数字")
        elif choice == '2':
            print("退出程序")
            break

if __name__ == '__main__':
    main()
