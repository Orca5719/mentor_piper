import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys
import time
import cv2
import numpy as np
import torch
import threading
from collections import deque, namedtuple
from pathlib import Path
from dm_env import StepType, specs
from tqdm import tqdm  # 新增：导入tqdm

# ========== 新增：导入3D鼠标读取模块 ==========
from spacemouse_reader import SpacemouseReader
# =============================================

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from replay_buffer import ReplayBufferStorage, make_replay_loader
    import utils
    from logger import Logger
    import piper.env as piper_env
    import hydra
except ImportError as e:
    print(f"错误：无法导入核心模块: {e}")
    sys.exit(1)

_TimeStepBase = namedtuple('_TimeStepBase', [
    'observation', 'action', 'reward', 'discount', 'first', 'is_last'
])

class TimeStep(_TimeStepBase):
    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return super().__getitem__(key)
    
    def last(self):
        return self.is_last


class PiperRobotTrainer:
    def __init__(self):
        print("="*60)
        print("     Piper 机械臂实时训练 (支持3D鼠标干预)")
        print("="*60)

        self.work_dir = Path.cwd()

        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.frame_stack = 3
        self.batch_size = 256
        self.update_every_steps = 2
        self.save_interval = 1000
        self.seed_steps = 1000

        self.robot_cam = None
        self.piper_arm = None
        self.agent = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._global_step = 0
        self._global_episode = 0
        self.last_save_step = -9999

        self.frames_queue = deque(maxlen=3)
        self.replay_storage = None
        self.replay_loader = None
        self.replay_iter = None

        self._reward = 0.0
        self.episode_reward = 0.0
        self.episode_step = 0
        self._last_action = None
        self._obs_spec = None
        self._act_spec = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = True

        self.factor = 1000
        self.X, self.Y, self.Z = int(300614), int(-12185), int(282341)
        self.RX, self.RY, self.RZ = int(-179351), int(23933), int(177934)
        self.joint_6 = int(80000)

        self._buffer_dir = Path.cwd() / 'buffer_robot'
        
        # 动作缩放参数
        self.action_scale = 20000

        # 初始化输出目录
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path.cwd() / "piper_outputs" / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 训练输出目录: {self.output_dir.resolve()}")

        # 3D鼠标配置
        self.DEAD_ZONE = 0.1
        self.SPACE_MOUSE_ACTION_SCALE = 2.0
        self.spacemouse_reader = None
        self.is_intervening = False
        
        # 随机策略的夹爪状态（保持稳定）
        self.random_gripper_state = 1.0
        self.last_gripper_change_step = 0
        self.gripper_change_interval = 50  # 夹爪切换间隔

    def _init_replay_storage(self):
        temp_env = piper_env.make(
            task_name='piper_push', seed=0, action_repeat=2,
            size=(self.IMG_HEIGHT, self.IMG_WIDTH), use_sim=True, frame_stack=3
        )
        self._obs_spec = temp_env.observation_spec()
        self._act_spec = temp_env.action_spec()

        data_specs = (
            specs.Array(self._obs_spec.shape, self._obs_spec.dtype, 'observation'),
            specs.Array(self._act_spec.shape, self._act_spec.dtype, 'action'),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount')
        )

        self._buffer_dir.mkdir(exist_ok=True)
        self.replay_storage = ReplayBufferStorage(data_specs, self._buffer_dir)

        self.replay_loader = make_replay_loader(
            self._buffer_dir,max_size=100000, batch_size=self.batch_size, num_workers=4,
            save_snapshot=False
        )
        self.replay_iter = iter(self.replay_loader)

        temp_env.close()
        print("✅ 回放缓冲区初始化完成")

    def init_hardware(self):
        print("初始化硬件...")

        from piper.robot import PiperRobot
        from piper_sdk import C_PiperInterface_V2

        # 初始化相机
        self.robot_cam = PiperRobot(
            use_sim=False,
            camera_width=self.IMG_WIDTH,
            camera_height=self.IMG_HEIGHT,
            use_apriltag=True,
            tag_size=0.05
        )

        # 初始化机械臂
        self.piper_arm = C_PiperInterface_V2("can0")
        self.piper_arm.ConnectPort()
        while not self.piper_arm.EnablePiper():
            time.sleep(0.01)
        self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)

        self._init_replay_storage()

        # 初始化3D鼠标
        self.spacemouse_reader = SpacemouseReader(
            dead_zone=self.DEAD_ZONE, 
            action_scale=self.SPACE_MOUSE_ACTION_SCALE
        )
        self.spacemouse_reader.start()
        time.sleep(1.0)

        threading.Thread(target=self._image_thread, daemon=True).start()
        time.sleep(0.5)

        print("✅ 硬件初始化完成")
        print()

    def _image_thread(self):
        while self._running:
            try:
                frame = self.robot_cam.get_camera_image()
                if frame is not None:
                    frame = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                    with self._lock:
                        self.frames_queue.append(frame)
                        self._latest_frame = frame.copy()
            except Exception as e:
                pass
            time.sleep(0.002)

    def get_stacked_obs(self):
        with self._lock:
            frames = list(self.frames_queue)

        while len(frames) < 3:
            frames.append(frames[0] if frames else np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8))

        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))

        if self._obs_spec.dtype == np.uint8:
            return stacked.astype(np.uint8)
        return (stacked.astype(np.float32) / 255.0)

    def apply_action(self, action):
        # 位置更新
        dx = int(round(action[0] * self.action_scale))
        dy = int(round(action[1] * self.action_scale))
        dz = int(round(action[2] * self.action_scale))

        self.X += dx
        self.Y += dy
        self.Z += dz

        self.X = int(np.clip(self.X, 100000, 500000))
        self.Y = int(np.clip(self.Y, -200000, 200000))
        self.Z = int(np.clip(self.Z, 100000, 400000))

        # 夹爪控制
        if self.is_intervening:
            self.joint_6 = int(80000) if action[3] > 0 else int(0)
        else:
            self.joint_6 = int(80000) if action[3] > 0 else int(0)

        # 执行动作
        self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)

        time.sleep(0.01)

    def get_action(self, obs):
        # 优先使用3D鼠标动作
        sm_action, is_intervening = self.spacemouse_reader.get_action()
        self.is_intervening = is_intervening

        if is_intervening and sm_action is not None:
            dx = sm_action[0] if abs(sm_action[0]) > self.DEAD_ZONE else 0.0
            dy = sm_action[1] if abs(sm_action[1]) > self.DEAD_ZONE else 0.0
            dz = sm_action[2] if abs(sm_action[2]) > self.DEAD_ZONE else 0.0

            # 夹爪控制
            sm_gripper = sm_action[6]
            if abs(sm_gripper) > self.DEAD_ZONE:
                gripper_ctrl = 1.0 if sm_gripper > 0 else -1.0
            else:
                gripper_ctrl = 1.0 if self.joint_6 > 0 else -1.0

            # 构造动作
            override_action = np.array([
                dx * self.SPACE_MOUSE_ACTION_SCALE,
                dy * self.SPACE_MOUSE_ACTION_SCALE,
                dz * self.SPACE_MOUSE_ACTION_SCALE,
                gripper_ctrl
            ], dtype=self._act_spec.dtype)
            override_action = np.clip(override_action, self._act_spec.minimum, self._act_spec.maximum)
            
            return override_action

        # 无干预时使用随机/模型动作
        if self.agent is None or self._global_step < self.seed_steps:
            # 随机策略：位置随机，夹爪保持稳定
            action = np.random.uniform(
                low=self._act_spec.minimum,
                high=self._act_spec.maximum,
                size=self._act_spec.shape
            ).astype(self._act_spec.dtype)
            
            # 夹爪只在间隔步数后才随机切换
            if self._global_step - self.last_gripper_change_step >= self.gripper_change_interval:
                self.random_gripper_state = np.random.choice([1.0, -1.0])
                self.last_gripper_change_step = self._global_step
            
            action[3] = self.random_gripper_state
            return action

        with torch.no_grad(), utils.eval_mode(self.agent):
            action = self.agent.act(obs, self._global_step, eval_mode=False)

        return action

    def update_policy(self):
        if self.agent is None or self.replay_loader is None:
            return

        if self._global_step < self.seed_steps:
            return

        if self._global_step % self.update_every_steps != 0:
            return

        try:
            metrics = self.agent.update(self.replay_iter, self._global_step)
            return metrics
        except Exception as e:
            return None

    def save_snapshot(self):
        if self.agent is None:
            return

        snapshot_path = self.output_dir / f'snapshot_robot_{self._global_step}.pt'
        payload = {
            'agent': self.agent,
            '_global_step': self._global_step,
            '_global_episode': self._global_episode
        }

        torch.save(payload, snapshot_path)
        print(f"\n✅ 模型保存: {snapshot_path.name}")

    def load_snapshot(self, snapshot_path):
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.exists():
            print(f"⚠️  快照文件不存在: {snapshot_path}")
            return False

        try:
            payload = torch.load(snapshot_path, map_location=self.device, weights_only=False)

            if 'actor_state_dict' in payload and 'agent' not in payload:
                print("加载预训练Actor权重...")
                if hasattr(self.agent, 'actor'):
                    self.agent.actor.load_state_dict(payload['actor_state_dict'])
                    self.agent.actor.to(self.device)
                    print("✅ Actor权重加载成功")
            else:
                for k, v in payload.items():
                    if k in self.__dict__:
                        self.__dict__[k] = v
                        if k == 'agent':
                            self.agent.to(self.device)
                print("✅ 快照加载成功")

            return True
        except Exception as e:
            print(f"❌ 加载快照失败: {e}")
            return False

    def visualize(self, obs, reward, episode_step):
        if self._latest_frame is None:
            return True

        frame = self._latest_frame.copy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        y_pos = 30
        line_spacing = 25

        # 基础状态显示
        if self._global_step < self.seed_steps:
            cv2.putText(frame_bgr, "SEEDING...", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            y_pos += line_spacing
            cv2.putText(frame_bgr, f"Seed: {self._global_step}/{self.seed_steps}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
        else:
            cv2.putText(frame_bgr, "TRAINING", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += line_spacing

        # 干预状态
        if self.is_intervening:
            cv2.putText(frame_bgr, "INTERVENING", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame_bgr, "Model Control", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += line_spacing

        # 核心进度信息
        cv2.putText(frame_bgr, f"Step: {self._global_step}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing
        cv2.putText(frame_bgr, f"Episode: {self._global_episode + 1}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing
        cv2.putText(frame_bgr, f"Reward: {self.episode_reward:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing

        # 操作提示
        cv2.putText(frame_bgr, "SPACE=+10, s=Save, q=Quit", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow("Piper Robot Training", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            self._reward += 10.0
        elif key == ord('s'):
            self.save_snapshot()
        elif key == ord('q'):
            return False

        return True

    def train(self, num_episodes=100, max_steps_per_episode=200):
        print("\n开始训练...")
        print("="*60)

        # 总Episode进度条
        episodes_bar = tqdm(range(num_episodes), desc="Episodes", unit="episode")
        
        try:
            for episode in episodes_bar:
                self._global_episode = episode
                self.episode_step = 0
                self.episode_reward = 0.0

                # 复位机械臂
                self.X, self.Y, self.Z = int(300614), int(-12185), int(282341)
                self.RX, self.RY, self.RZ = int(-179351), int(23933), int(177934)
                self.joint_6 = int(80000)
                self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
                self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)
                time.sleep(1.0)

                # 重置帧队列
                self.frames_queue.clear()
                for _ in range(self.frame_stack):
                    frame = self.robot_cam.get_camera_image()
                    if frame is not None:
                        self.frames_queue.append(frame)

                obs_prev = self.get_stacked_obs()

                # 单Episode内Step进度条
                step_bar = tqdm(range(max_steps_per_episode), desc=f"Episode {episode+1}", unit="step", leave=False)
                for step in step_bar:
                    self.episode_step = step
                    self._global_step += 1

                    # 获取并执行动作
                    action = self.get_action(obs_prev)
                    self._last_action = action
                    self.apply_action(action)

                    # 更新观测
                    new_frame = self.robot_cam.get_camera_image()
                    if new_frame is not None:
                        self.frames_queue.append(new_frame)
                    obs_curr = self.get_stacked_obs()

                    # 奖励处理
                    reward = self._reward
                    if self._reward != 0:
                        self._reward = 0
                    self.episode_reward += reward

                    # 存入缓冲区
                    ts = TimeStep(
                        observation=obs_prev,
                        action=action,
                        reward=np.array([reward], dtype=np.float32),
                        discount=np.array([1.0], dtype=np.float32),
                        first=(step == 0),
                        is_last=False
                    )
                    self.replay_storage.add(ts)

                    # 策略更新
                    if self._global_step >= self.seed_steps:
                        if self._global_step % self.update_every_steps == 0:
                            self.update_policy()

                    # 模型保存
                    if self._global_step - self.last_save_step >= self.save_interval:
                        self.last_save_step = self._global_step
                        self.save_snapshot()

                    obs_prev = obs_curr

                    # 更新进度条描述
                    step_bar.set_postfix({
                        'Global Step': self._global_step,
                        'Reward': f"{self.episode_reward:.1f}",
                        'Intervene': self.is_intervening
                    })

                    # 可视化与退出判断
                    continue_training = self.visualize(obs_curr, reward, step)
                    if not continue_training:
                        step_bar.close()
                        episodes_bar.close()
                        print("\n用户退出训练")
                        return

                step_bar.close()

                # Episode结束处理
                ts_last = TimeStep(
                    observation=self.get_stacked_obs(),
                    action=self._last_action,
                    reward=np.array([0.0], dtype=np.float32),
                    discount=np.array([0.0], dtype=np.float32),
                    first=False,
                    is_last=True
                )
                self.replay_storage.add(ts_last)

                # 更新总进度条描述
                episodes_bar.set_postfix({
                    'Last Reward': f"{self.episode_reward:.1f}",
                    'Buffer Size': len(self.replay_storage),
                    'Global Step': self._global_step
                })

        except KeyboardInterrupt:
            print("\n用户中断训练")
        finally:
            episodes_bar.close()
            self.cleanup()

    def cleanup(self):
        print("\n清理资源...")
        self._running = False
        time.sleep(0.3)
        cv2.destroyAllWindows()

        # 停止3D鼠标
        if self.spacemouse_reader is not None:
            self.spacemouse_reader.stop()

        # 关闭机械臂
        if self.piper_arm is not None:
            self.piper_arm.EmergencyStop()
            self.piper_arm.DisconnectPort()

        # 关闭相机
        if self.robot_cam is not None:
            self.robot_cam.close()

        print("✅ 训练结束，资源已清理")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Piper 机械臂实时训练 (支持3D鼠标干预)')
    parser.add_argument('--snapshot', type=str, default=r"/home/isee604/mentor_mentor/mentor_piper/piper_outputs/2026-04-17_20-01-46/snapshot_robot_14253.pt", help='预训练权重路径')
    parser.add_argument('--episodes', type=int, default=1000, help='训练 episode 数量')
    parser.add_argument('--steps', type=int, default=1500, help='每个 episode 最大步数')

    args = parser.parse_args()

    try:
        with hydra.initialize(config_path='piper/cfgs', version_base=None):
            cfg = hydra.compose(config_name='config')
    except:
        print("⚠️  无法加载Hydra配置，使用随机策略")
        cfg = None

    trainer = PiperRobotTrainer()
    trainer.init_hardware()

    # 初始化Agent
    if cfg is not None:
        try:
            import agents.mentor_mw as mentor_mw
            obs_spec = trainer._obs_spec
            act_spec = trainer._act_spec
            cfg.obs_shape = obs_spec.shape
            cfg.action_shape = act_spec.shape
            trainer.agent = hydra.utils.instantiate(cfg.agent)
            trainer.agent = trainer.agent.to(trainer.device)
            print("✅ Agent初始化成功")
        except Exception as e:
            print(f"⚠️  Agent初始化失败: {e}")
            trainer.agent = None

    # 加载快照
    if args.snapshot:
        trainer.load_snapshot(args.snapshot)

    # 开始训练
    trainer.train(num_episodes=args.episodes, max_steps_per_episode=args.steps)


if __name__ == '__main__':
    main()
