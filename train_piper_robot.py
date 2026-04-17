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
        print("     Piper 机械臂实时训练 (参考正常代码重写版)")
        print("="*60)
        print()

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
        self.X, self.Y, self.Z = 300614, -12185, 282341
        self.RX, self.RY, self.RZ = -179351, 23933, 177934
        self.joint_6 = 80000

        self._buffer_dir = Path.cwd() / 'buffer_robot'

    def _init_replay_storage(self):
        print("正在初始化回放缓冲区...")

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
            self._buffer_dir, self.batch_size, num_workers=1,
            save_snapshot=False
        )
        self.replay_iter = iter(self.replay_loader)

        temp_env.close()

        print(f"  ✓ Observation: {self._obs_spec.shape}, {self._obs_spec.dtype}")
        print(f"  ✓ Action: {self._act_spec.shape}, {self._act_spec.dtype}")
        print(f"  ✓ Buffer: {self._buffer_dir.resolve()}")
        print()

    def init_hardware(self):
        print("正在初始化硬件...")

        from piper.robot import PiperRobot
        from piper_sdk import C_PiperInterface_V2

        print("  初始化相机...")
        self.robot_cam = PiperRobot(
            use_sim=False,
            camera_width=self.IMG_WIDTH,
            camera_height=self.IMG_HEIGHT,
            use_apriltag=True,
            tag_size=0.05
        )
        print("    ✓ 相机初始化成功")

        print("  初始化机械臂...")
        self.piper_arm = C_PiperInterface_V2("can0")
        self.piper_arm.ConnectPort()
        while not self.piper_arm.EnablePiper():
            time.sleep(0.01)

        self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)
        print("    ✓ 机械臂初始化成功")

        self._init_replay_storage()

        threading.Thread(target=self._image_thread, daemon=True).start()
        time.sleep(0.5)

        print("  ✓ 硬件初始化完成")
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
                print(f"⚠️ 图像采集异常: {e}")
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
        dx, dy, dz = action[0] * 1000.0, action[1] * 1000.0, action[2] * 1000.0

        self.X += round(dx)
        self.Y += round(dy)
        self.Z += round(dz)

        self.X = np.clip(self.X, 100000, 500000)
        self.Y = np.clip(self.Y, -200000, 200000)
        self.Z = np.clip(self.Z, 100000, 400000)

        if action[3] > 0:
            self.joint_6 = 80000
        else:
            self.joint_6 = 0

        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)

        time.sleep(0.01)

    def get_action(self, obs):
        if self.agent is None or self._global_step < self.seed_steps:
            return np.random.uniform(
                low=self._act_spec.minimum,
                high=self._act_spec.maximum,
                size=self._act_spec.shape
            ).astype(self._act_spec.dtype)

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

        snapshot_path = self.work_dir / f'snapshot_robot_{self._global_step}.pt'

        payload = {
            'agent': self.agent,
            '_global_step': self._global_step,
            '_global_episode': self._global_episode
        }

        torch.save(payload, snapshot_path)
        print(f"\n✓ 模型已保存: {snapshot_path}")

    def load_snapshot(self, snapshot_path):
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.exists():
            print(f"⚠️  快照文件不存在: {snapshot_path}")
            return False

        try:
            payload = torch.load(snapshot_path, map_location=self.device)

            if 'actor_state_dict' in payload and 'agent' not in payload:
                print("检测到预训练模型，加载 Actor 权重...")
                if hasattr(self.agent, 'actor'):
                    self.agent.actor.load_state_dict(payload['actor_state_dict'])
                    self.agent.actor.to(self.device)
                    print("✓ Actor 权重加载成功")
            else:
                for k, v in payload.items():
                    if k in self.__dict__:
                        self.__dict__[k] = v
                        if k == 'agent':
                            self.agent.to(self.device)
                print(f"✓ 快照加载成功: {snapshot_path}")

            return True
        except Exception as e:
            print(f"✗ 加载快照失败: {e}")
            return False

    def visualize(self, obs, reward, episode_step):
        if self._latest_frame is None:
            return True

        frame = self._latest_frame.copy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        y_pos = 30
        line_spacing = 25

        if self._global_step < self.seed_steps:
            cv2.putText(frame_bgr, "SEEDING...", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            y_pos += line_spacing
            cv2.putText(frame_bgr, f"Collecting data: {self._global_step}/{self.seed_steps}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
        else:
            cv2.putText(frame_bgr, "TRAINING", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += line_spacing

        cv2.putText(frame_bgr, f"Step: {self._global_step}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing

        cv2.putText(frame_bgr, f"Episode: {self._global_episode + 1}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing

        cv2.putText(frame_bgr, f"Episode Step: {episode_step + 1}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing

        cv2.putText(frame_bgr, f"Reward: {self.episode_reward:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing

        cv2.putText(frame_bgr, "SPACE=+10, s=Save, q=Quit", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow("Piper Robot Training", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            self._reward += 10.0
            print(f"[奖励] +10.0")
        elif key == ord('s'):
            self.save_snapshot()
        elif key == ord('q'):
            return False

        return True

    def train(self, num_episodes=100, max_steps_per_episode=200):
        print("="*60)
        print("开始训练...")
        print("="*60)
        print()

        try:
            for episode in range(num_episodes):
                self._global_episode = episode
                self.episode_step = 0
                self.episode_reward = 0.0

                print(f"\n[Episode {episode + 1}/{num_episodes}]")

                self.X, self.Y, self.Z = 300614, -12185, 282341
                self.joint_6 = 80000
                self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
                self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)
                time.sleep(1.0)

                for _ in range(self.frame_stack):
                    self.frames_queue.append(self.robot_cam.get_camera_image())

                obs_prev = self.get_stacked_obs()

                for step in range(max_steps_per_episode):
                    self.episode_step = step
                    self._global_step += 1

                    action = self.get_action(obs_prev)
                    self._last_action = action
                    self.apply_action(action)

                    self.frames_queue.append(self.robot_cam.get_camera_image())
                    obs_curr = self.get_stacked_obs()

                    reward = self._reward
                    if self._reward != 0:
                        self._reward = 0

                    self.episode_reward += reward

                    ts = TimeStep(
                        observation=obs_prev,
                        action=action,
                        reward=np.array([reward], dtype=np.float32),
                        discount=np.array([1.0], dtype=np.float32),
                        first=(step == 0),
                        is_last=False
                    )
                    self.replay_storage.add(ts)

                    if self._global_step >= self.seed_steps:
                        if self._global_step % self.update_every_steps == 0:
                            self.update_policy()

                    if self._global_step - self.last_save_step >= self.save_interval:
                        self.last_save_step = self._global_step
                        self.save_snapshot()

                    obs_prev = obs_curr

                    continue_training = self.visualize(obs_curr, reward, step)
                    if not continue_training:
                        print("\n用户退出")
                        return

                ts_last = TimeStep(
                    observation=self.get_stacked_obs(),
                    action=self._last_action,
                    reward=np.array([0.0], dtype=np.float32),
                    discount=np.array([0.0], dtype=np.float32),
                    first=False,
                    is_last=True
                )
                self.replay_storage.add(ts_last)

                print(f"  完成，奖励: {self.episode_reward:.1f}")
                print(f"  Buffer: {len(self.replay_storage)}")

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n清理资源...")
        self._running = False
        time.sleep(0.3)
        cv2.destroyAllWindows()
        if self.piper_arm is not None:
            self.piper_arm.EmergencyStop()
            self.piper_arm.DisconnectPort()
            print("  ✓ 机械臂已断开")
        if self.robot_cam is not None:
            self.robot_cam.close()
            print("  ✓ 相机已关闭")
        print("完成！")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Piper 机械臂实时训练')
    parser.add_argument('--snapshot', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--episodes', type=int, default=100, help='训练 episode 数量')
    parser.add_argument('--steps', type=int, default=200, help='每个 episode 最大步数')

    args = parser.parse_args()

    try:
        with hydra.initialize(config_path='piper/cfgs', version_base=None):
            cfg = hydra.compose(config_name='config')
    except:
        print("⚠️  无法加载 Hydra 配置，使用简化模式")
        cfg = None

    trainer = PiperRobotTrainer()

    trainer.init_hardware()

    if cfg is not None:
        try:
            import agents.mentor_mw as mentor_mw

            obs_spec = trainer._obs_spec
            act_spec = trainer._act_spec

            cfg.obs_shape = obs_spec.shape
            cfg.action_shape = act_spec.shape

            trainer.agent = hydra.utils.instantiate(cfg.agent)
            trainer.agent = trainer.agent.to(trainer.device)

            print("✓ Agent 初始化成功")
        except Exception as e:
            print(f"⚠️  Agent 初始化失败: {e}")
            print("  将使用随机策略探索")
            trainer.agent = None

    if args.snapshot:
        trainer.load_snapshot(args.snapshot)

    trainer.train(num_episodes=args.episodes, max_steps_per_episode=args.steps)


if __name__ == '__main__':
    main()
