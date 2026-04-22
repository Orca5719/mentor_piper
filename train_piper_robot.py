import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys
import time
import gc
import cv2
import numpy as np
import torch
import threading
import math
import hydra
from collections import deque
from pathlib import Path
from copy import deepcopy
from dm_env import StepType, specs
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from replay_buffer import ReplayBufferStorage, make_replay_loader
    import utils
    from logger import Logger
    import piper.env as piper_env
except ImportError as e:
    print(f"错误：无法导入核心模块: {e}")
    sys.exit(1)


class TimeStepWithAction:
    def __init__(self, observation, action, reward, discount, step_type, success=False):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.discount = discount
        self.step_type = step_type
        self.success = success
        self.first = (step_type == StepType.FIRST)
        self.is_last = (step_type == StepType.LAST)

    def last(self):
        return self.is_last

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        raise KeyError(f"Invalid key: {key}")


class PiperCollectEnv:
    def __init__(self, factor=1000, img_height=256, img_width=256, action_sleep=0.05):
        self.factor = factor
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.action_sleep = action_sleep

        self.X, self.Y, self.Z = 300614, -12185, 282341
        self.RX, self.RY, self.RZ = -179351, 23933, 177934
        self.joint_6 = 80000

        self.robot_cam = None
        self.piper_arm = None
        self.frames_queue = deque(maxlen=3)
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = True
        self._obs_spec = None
        self._act_spec = None

        self._reward = 0.0
        self._step_count = 0

    def _init_specs(self):
        temp_env = piper_env.make(
            task_name='piper_push', seed=0, action_repeat=2,
            size=(self.IMG_HEIGHT, self.IMG_WIDTH), use_sim=True, frame_stack=3
        )
        self._obs_spec = temp_env.observation_spec()
        self._act_spec = temp_env.action_spec()
        temp_env.close()

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._act_spec

    def init_hardware(self):
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

        self._init_specs()

        threading.Thread(target=self._image_thread, daemon=True).start()
        time.sleep(0.5)

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

    def _get_stacked_obs(self):
        with self._lock:
            frames = list(self.frames_queue)

        while len(frames) < 3:
            frames.append(frames[0] if frames else np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8))

        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))

        if self._obs_spec.dtype == np.uint8:
            return stacked.astype(np.uint8)
        return (stacked.astype(np.float32) / 255.0)

    def _apply_action(self, action):
        dx = int(round(action[0] * 20 * self.factor))
        dy = int(round(action[1] * 20 * self.factor))
        dz = int(round(action[2] * 20 * self.factor))

        self.X += dx
        self.Y += dy
        self.Z += dz

        self.X = int(np.clip(self.X, 100000, 500000))
        self.Y = int(np.clip(self.Y, -200000, 200000))
        self.Z = int(np.clip(self.Z, 100000, 400000))

        action_shape = len(action)
        gripper_idx = 6 if action_shape >= 7 else (3 if action_shape >= 4 else -1)
        
        if gripper_idx >= 0 and gripper_idx < action_shape:
            self.joint_6 = int(80000) if action[gripper_idx] > 0 else int(0)

        self.piper_arm.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)

        time.sleep(self.action_sleep)

    def reset(self):
        self.X, self.Y, self.Z = 300614, -12185, 282341
        self.joint_6 = 80000
        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)

        self.frames_queue.clear()
        for _ in range(3):
            frame = self.robot_cam.get_camera_image()
            if frame is not None:
                self.frames_queue.append(frame)

        self._reward = 0.0
        self._step_count = 0

        obs = self._get_stacked_obs()
        return TimeStepWithAction(
            observation=obs,
            action=np.zeros(self._act_spec.shape, dtype=self._act_spec.dtype),
            reward=np.float32(0.0),
            discount=np.float32(1.0),
            step_type=StepType.FIRST
        )

    def step(self, action, reward=0.0):
        self._apply_action(action)

        new_frame = self.robot_cam.get_camera_image()
        if new_frame is not None:
            self.frames_queue.append(new_frame)

        obs = self._get_stacked_obs()
        self._reward = reward
        self._step_count += 1

        return TimeStepWithAction(
            observation=obs,
            action=action,
            reward=np.float32(self._reward),
            discount=np.float32(1.0),
            step_type=StepType.MID
        )

    def step_last(self, action):
        self._apply_action(action)

        new_frame = self.robot_cam.get_camera_image()
        if new_frame is not None:
            self.frames_queue.append(new_frame)

        obs = self._get_stacked_obs()

        return TimeStepWithAction(
            observation=obs,
            action=action,
            reward=np.float32(0.0),
            discount=np.float32(0.0),
            step_type=StepType.LAST
        )

    def get_latest_frame(self):
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def close(self):
        self._running = False
        time.sleep(0.3)
        if self.piper_arm is not None:
            self.piper_arm.EmergencyStop()
            self.piper_arm.DisconnectPort()
        if self.robot_cam is not None:
            self.robot_cam.close()


class Workspace:
    def __init__(self, cfg=None):
        self.work_dir = Path.cwd()
        self.cfg = cfg

        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.frame_stack = 3
        self.batch_size = 256
        self.update_every_episodes = 2
        self.save_interval = 1000
        self.seed_steps = 1000
        self.action_sleep = 0.05

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._discount = 0.99
        self._discount_alpha = 0.0
        self._discount_alpha_temp = 1.0
        self._discount_beta = 0.0
        self._discount_beta_temp = 1.0
        self._nstep = 3
        self._nstep_alpha = 0.0
        self._nstep_alpha_temp = 1.0

        self._global_step = 0
        self._global_episode = 0
        self.last_save_step = -9999

        self.replay_storage = None
        self.replay_loader = None
        self.replay_iter = None
        self.buffer = None
        self.agent = None

        self._buffer_dir = Path.cwd() / 'buffer'

        self.env = PiperCollectEnv(
            factor=1000,
            img_height=self.IMG_HEIGHT,
            img_width=self.IMG_WIDTH,
            action_sleep=self.action_sleep
        )

        self.human_intervened_this_episode = False
        self.is_intervening = False

        self.DEAD_ZONE = 0.1
        self.SPACE_MOUSE_ACTION_SCALE = 1.0

        self.random_amplitude = 0.8
        self.random_drift_prob = 0.3
        self.last_random_direction = np.zeros(3)
        self.random_gripper_state = 1.0
        self.last_gripper_change_step = 0
        self.gripper_change_interval = 50

        self.logger = None
        self.timer = utils.Timer()

        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path.cwd() / "piper_outputs" / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def discount(self):
        return self._discount - self._discount_alpha * math.exp(
            -self.global_step / self._discount_alpha_temp) - self._discount_beta * math.exp(
                -self.global_step / self._discount_beta_temp)

    @property
    def nstep(self):
        return math.floor(self._nstep + self._nstep_alpha *
                          math.exp(-self.global_step / self._nstep_alpha_temp))

    def setup(self):
        self.logger = Logger(self.work_dir, use_tb=False, use_wandb=False)

        self.env.init_hardware()

        data_specs = (
            self.env.observation_spec(),
            self.env.action_spec(),
            specs.Array((), np.float32, 'reward'),
            specs.Array((), np.float32, 'discount')
        )

        self._buffer_dir.mkdir(exist_ok=True)
        self.replay_storage = ReplayBufferStorage(data_specs, self._buffer_dir)

        self.replay_loader, self.buffer = make_replay_loader(
            self._buffer_dir, max_size=100000, batch_size=self.batch_size,
            num_workers=4, save_snapshot=False,
            nstep=math.floor(self._nstep + self._nstep_alpha),
            discount=self._discount - self._discount_alpha - self._discount_beta
        )
        self._replay_iter = None

        print(f"✅ Buffer路径: {self._buffer_dir.resolve()}")
        print(f"✅ Buffer大小: {len(self.replay_storage)}")

    def update_buffer(self):
        current_nstep = self.nstep
        self.buffer.update_nstep(current_nstep)

    def make_agent(self, cfg):
        cfg.obs_shape = self.env.observation_spec().shape
        cfg.action_shape = self.env.action_spec().shape
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.agent = self.agent.to(self.device)
        print("✅ Agent初始化成功")

    def get_action(self, obs, eval_mode=False):
        sm_action, is_intervening = self._read_spacemouse()
        self.is_intervening = is_intervening

        if self.human_intervened_this_episode:
            if sm_action is None:
                sm_action = np.zeros(self.env.action_spec().shape, dtype=np.float32)
                is_intervening = False

            dx = sm_action[0] if abs(sm_action[0]) > self.DEAD_ZONE else 0.0
            dy = sm_action[1] if abs(sm_action[1]) > self.DEAD_ZONE else 0.0
            dz = sm_action[2] if abs(sm_action[2]) > self.DEAD_ZONE else 0.0

            action_shape = self.env.action_spec().shape[0]
            gripper_idx = 6 if action_shape >= 7 else (3 if action_shape >= 4 else -1)

            if gripper_idx >= 0 and gripper_idx < len(sm_action):
                sm_gripper = sm_action[gripper_idx]
                if abs(sm_gripper) > self.DEAD_ZONE:
                    gripper_ctrl = 1.0 if sm_gripper > 0 else -1.0
                else:
                    gripper_ctrl = 1.0 if self.env.joint_6 > 0 else -1.0
            else:
                gripper_ctrl = 1.0 if self.env.joint_6 > 0 else -1.0

            override_action = np.zeros(action_shape, dtype=self.env.action_spec().dtype)
            override_action[0] = dx * self.SPACE_MOUSE_ACTION_SCALE
            override_action[1] = dy * self.SPACE_MOUSE_ACTION_SCALE
            override_action[2] = dz * self.SPACE_MOUSE_ACTION_SCALE
            if gripper_idx >= 0 and gripper_idx < action_shape:
                override_action[gripper_idx] = gripper_ctrl

            return override_action

        if is_intervening and sm_action is not None:
            self.human_intervened_this_episode = True

            dx = sm_action[0] if abs(sm_action[0]) > self.DEAD_ZONE else 0.0
            dy = sm_action[1] if abs(sm_action[1]) > self.DEAD_ZONE else 0.0
            dz = sm_action[2] if abs(sm_action[2]) > self.DEAD_ZONE else 0.0

            action_shape = self.env.action_spec().shape[0]
            gripper_idx = 6 if action_shape >= 7 else (3 if action_shape >= 4 else -1)

            if gripper_idx >= 0 and gripper_idx < len(sm_action):
                sm_gripper = sm_action[gripper_idx]
                if abs(sm_gripper) > self.DEAD_ZONE:
                    gripper_ctrl = 1.0 if sm_gripper > 0 else -1.0
                else:
                    gripper_ctrl = 1.0 if self.env.joint_6 > 0 else -1.0
            else:
                gripper_ctrl = 1.0 if self.env.joint_6 > 0 else -1.0

            override_action = np.zeros(action_shape, dtype=self.env.action_spec().dtype)
            override_action[0] = dx * self.SPACE_MOUSE_ACTION_SCALE
            override_action[1] = dy * self.SPACE_MOUSE_ACTION_SCALE
            override_action[2] = dz * self.SPACE_MOUSE_ACTION_SCALE
            if gripper_idx >= 0 and gripper_idx < action_shape:
                override_action[gripper_idx] = gripper_ctrl

            return override_action

        if self.agent is None or self._global_step < self.seed_steps:
            action = np.zeros(self.env.action_spec().shape, dtype=self.env.action_spec().dtype)

            if np.random.random() < self.random_drift_prob or np.linalg.norm(self.last_random_direction) == 0:
                direction = np.random.uniform(-1, 1, 3)
                direction = direction / np.linalg.norm(direction)
                self.last_random_direction = direction
            else:
                direction = self.last_random_direction
                direction += np.random.normal(0, 0.2, 3)
                direction = direction / np.linalg.norm(direction)
                self.last_random_direction = direction

            action[:3] = direction * self.random_amplitude

            action_shape = self.env.action_spec().shape[0]
            gripper_idx = 6 if action_shape >= 7 else (3 if action_shape >= 4 else -1)
            if gripper_idx >= 0 and self._global_step - self.last_gripper_change_step >= self.gripper_change_interval:
                self.random_gripper_state = np.random.choice([1.0, -1.0])
                self.last_gripper_change_step = self._global_step

            if gripper_idx >= 0:
                action[gripper_idx] = self.random_gripper_state

            return action

        with torch.no_grad(), utils.eval_mode(self.agent):
            action = self.agent.act(obs, self._global_step, eval_mode=eval_mode)

        return action

    def _read_spacemouse(self):
        try:
            import pyspacemouse
            state = pyspacemouse.read()
            if state is None:
                return None, False

            magnitude = np.sqrt(state.x**2 + state.y**2 + state.z**2)
            is_intervening = magnitude > self.DEAD_ZONE

            action_shape = self.env.action_spec().shape[0]
            action = np.zeros(action_shape, dtype=np.float32)
            action[0] = state.x
            action[1] = state.y
            action[2] = state.z
            
            gripper_idx = 6 if action_shape >= 7 else (3 if action_shape >= 4 else -1)
            if gripper_idx >= 0:
                if hasattr(state, 'buttons') and state.buttons is not None:
                    if len(state.buttons) > 0 and state.buttons[0]:
                        action[gripper_idx] = 1.0
                    elif len(state.buttons) > 1 and state.buttons[1]:
                        action[gripper_idx] = -1.0
                    else:
                        action[gripper_idx] = 0.0
                else:
                    action[gripper_idx] = 0.0
            
            return action, is_intervening
        except:
            return None, False

    def visualize(self, episode, step, max_steps, episode_reward, is_training=True):
        frame = self.env.get_latest_frame()
        if frame is None:
            return True, 0.0

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        y_pos = 30
        line_spacing = 25

        mode_str = "TRAINING" if is_training else "EVAL"
        mode_color = (0, 255, 0) if is_training else (255, 255, 0)
        cv2.putText(frame_bgr, mode_str, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        y_pos += line_spacing

        if self.human_intervened_this_episode:
            cv2.putText(frame_bgr, "HUMAN LOCKED", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.is_intervening:
            cv2.putText(frame_bgr, "INTERVENING", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame_bgr, "Model Control", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += line_spacing

        gripper_status = "GRIPPER: OPEN" if self.env.joint_6 > 40000 else "GRIPPER: CLOSED"
        gripper_color = (0, 255, 0) if self.env.joint_6 > 40000 else (0, 0, 255)
        cv2.putText(frame_bgr, gripper_status, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, gripper_color, 2)
        y_pos += line_spacing

        cv2.putText(frame_bgr, f"Ep: {episode+1} Step: {step+1}/{max_steps}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing
        cv2.putText(frame_bgr, f"Reward: {episode_reward:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing
        cv2.putText(frame_bgr, f"Buffer: {len(self.replay_storage)}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += line_spacing

        cv2.putText(frame_bgr, "SPACE=+10&End, s=Save, q=Quit", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow("Piper Robot", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        manual_reward = 0.0
        if key == ord(' '):
            manual_reward = 10.0
            print(f"[奖励] Ep{episode+1} Step{step+1} | +10.0")
        elif key == ord('s'):
            self.save_snapshot()
        elif key == ord('q'):
            return False, manual_reward

        return True, manual_reward

    def update_policy(self, num_updates=100):
        if self.agent is None or self.replay_loader is None:
            return None

        if self._global_step < self.seed_steps:
            return None

        try:
            self.update_buffer()
            print(f"\n开始更新策略，执行 {num_updates} 次梯度更新...")
            metrics = None
            for i in range(num_updates):
                metrics = self.agent.update(self.replay_iter, self._global_step)
                if i % 20 == 0:
                    print(f"  进度: {i+1}/{num_updates}", end='\r')
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"\n✅ 策略更新完成")
            return metrics
        except Exception as e:
            print(f"❌ 策略更新失败: {e}")
            return None

    def save_snapshot(self, step_id=None):
        if self.agent is None:
            return

        if step_id is None:
            snapshot_path = self.output_dir / 'snapshot.pt'
        else:
            snapshot_path = self.output_dir / f'snapshot_{step_id}.pt'

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

    def train(self, num_episodes=100, max_steps_per_episode=200, episode_sleep=2.0):
        print("\n开始训练...")
        print("="*60)

        episodes_bar = tqdm(range(num_episodes), desc="Episodes", unit="episode")

        try:
            for episode in episodes_bar:
                self._global_episode = episode
                self.human_intervened_this_episode = False
                episode_reward = 0.0

                self.env.X, self.env.Y, self.env.Z = 300614, -12185, 282341
                self.env.joint_6 = 80000
                self.env.piper_arm.EndPoseCtrl(self.env.X, self.env.Y, self.env.Z, self.env.RX, self.env.RY, self.env.RZ)
                self.env.piper_arm.GripperCtrl(abs(self.env.joint_6), 1000, 0x01, 0)
                time.sleep(1.0)

                time_step = self.env.reset()
                self.replay_storage.add(time_step)

                step_bar = tqdm(range(max_steps_per_episode), desc=f"Episode {episode+1}", unit="step", leave=False)
                episode_ended_early = False

                for step in step_bar:
                    obs_prev = time_step.observation
                    action = self.get_action(obs_prev, eval_mode=False)

                    continue_vis, manual_reward = self.visualize(episode, step, max_steps_per_episode, episode_reward, is_training=True)
                    if not continue_vis:
                        step_bar.close()
                        episodes_bar.close()
                        print("\n用户退出训练")
                        return

                    if manual_reward > 0:
                        time_step = self.env.step_last(action)
                        ts = TimeStepWithAction(
                            observation=time_step.observation,
                            action=action,
                            reward=np.float32(manual_reward),
                            discount=np.float32(0.0),
                            step_type=StepType.LAST
                        )
                        self.replay_storage.add(ts)
                        episode_reward += manual_reward
                        self._global_step += 1
                        episode_ended_early = True
                        step_bar.close()
                        break

                    time_step = self.env.step(action, reward=0.0)
                    episode_reward += time_step.reward

                    self.replay_storage.add(time_step)

                    self._global_step += 1

                    if self._global_step % 1000 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                    if self._global_step - self.last_save_step >= self.save_interval:
                        self.last_save_step = self._global_step
                        self.save_snapshot(self._global_step)

                    step_bar.set_postfix({
                        'Global Step': self._global_step,
                        'Reward': f"{episode_reward:.1f}",
                        'Human': self.human_intervened_this_episode
                    })

                if not episode_ended_early:
                    step_bar.close()
                    state = self._read_spacemouse()[0]
                    action_shape = self.env.action_spec().shape[0]
                    gripper_idx = 6 if action_shape >= 7 else (3 if action_shape >= 4 else -1)
                    if state is not None:
                        action = np.zeros(action_shape, dtype=np.float32)
                        action[0] = state[0]
                        action[1] = state[1]
                        action[2] = state[2]
                        if gripper_idx >= 0:
                            action[gripper_idx] = 1.0 if self.env.joint_6 > 0 else -1.0
                    else:
                        action = np.zeros(self.env.action_spec().shape, dtype=self.env.action_spec().dtype)
                        if gripper_idx >= 0:
                            action[gripper_idx] = 1.0

                    ts_last = self.env.step_last(action)
                    self.replay_storage.add(ts_last)

                if (episode + 1) % self.update_every_episodes == 0 and self._global_step >= self.seed_steps:
                    metrics = self.update_policy(num_updates=100)
                    if metrics and self.logger:
                        self.logger.log_metrics(metrics, self.global_step, ty='train')

                episodes_bar.set_postfix({
                    'Last Reward': f"{episode_reward:.1f}",
                    'Buffer Size': len(self.replay_storage),
                    'Global Step': self._global_step
                })

                print(f"请重新摆放物体... 休眠 {episode_sleep}s")
                time.sleep(episode_sleep)

        except KeyboardInterrupt:
            print("\n用户中断训练")
        finally:
            episodes_bar.close()
            self.cleanup()

    def eval(self, num_episodes=10, max_steps_per_episode=200):
        print("\n开始评估...")
        print("="*60)

        if self.agent is None:
            print("❌ 没有可用的agent进行评估")
            return

        total_reward = 0.0
        total_success = 0

        episodes_bar = tqdm(range(num_episodes), desc="Eval Episodes", unit="episode")

        try:
            for episode in episodes_bar:
                self.human_intervened_this_episode = False
                episode_reward = 0.0

                self.env.X, self.env.Y, self.env.Z = 300614, -12185, 282341
                self.env.joint_6 = 80000
                self.env.piper_arm.EndPoseCtrl(self.env.X, self.env.Y, self.env.Z, self.env.RX, self.env.RY, self.env.RZ)
                self.env.piper_arm.GripperCtrl(abs(self.env.joint_6), 1000, 0x01, 0)
                time.sleep(1.0)

                time_step = self.env.reset()

                step_bar = tqdm(range(max_steps_per_episode), desc=f"Eval Ep {episode+1}", unit="step", leave=False)

                for step in step_bar:
                    obs = time_step.observation
                    action = self.get_action(obs, eval_mode=True)

                    continue_vis, _ = self.visualize(episode, step, max_steps_per_episode, episode_reward, is_training=False)
                    if not continue_vis:
                        step_bar.close()
                        episodes_bar.close()
                        return

                    time_step = self.env.step(action, reward=0.0)
                    episode_reward += time_step.reward

                    step_bar.set_postfix({
                        'Reward': f"{episode_reward:.1f}"
                    })

                step_bar.close()

                state = self._read_spacemouse()[0]
                action_shape = self.env.action_spec().shape[0]
                gripper_idx = 6 if action_shape >= 7 else (3 if action_shape >= 4 else -1)
                if state is not None:
                    action = np.zeros(action_shape, dtype=np.float32)
                    action[0] = state[0]
                    action[1] = state[1]
                    action[2] = state[2]
                    if gripper_idx >= 0:
                        action[gripper_idx] = 1.0 if self.env.joint_6 > 0 else -1.0
                else:
                    action = np.zeros(self.env.action_spec().shape, dtype=self.env.action_spec().dtype)
                    if gripper_idx >= 0:
                        action[gripper_idx] = 1.0

                ts_last = self.env.step_last(action)

                total_reward += episode_reward

                episodes_bar.set_postfix({
                    'Avg Reward': f"{total_reward / (episode + 1):.1f}"
                })

                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\n用户中断评估")
        finally:
            episodes_bar.close()

        avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
        print(f"\n✅ 评估完成 | 平均奖励: {avg_reward:.2f}")

    def cleanup(self):
        print("\n清理资源...")
        self._running = False
        time.sleep(0.3)
        cv2.destroyAllWindows()

        if self.env is not None:
            self.env.close()

        print("✅ 训练结束，资源已清理")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Piper 机械臂训练 (支持3D鼠标干预)')
    parser.add_argument('--snapshot', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--episodes', type=int, default=1000, help='训练 episode 数量')
    parser.add_argument('--steps', type=int, default=1500, help='每个 episode 最大步数')
    parser.add_argument('--eval', action='store_true', help='仅进行评估')
    parser.add_argument('--eval_episodes', type=int, default=10, help='评估 episode 数量')
    parser.add_argument('--action_sleep', type=float, default=0.05, help='机械臂动作间隔时间(秒)')
    parser.add_argument('--buffer_dir', type=str, default='buffer', help='Buffer目录路径')

    args = parser.parse_args()

    cfg = None
    try:
        with hydra.initialize(config_path='piper/cfgs', version_base=None):
            cfg = hydra.compose(config_name='config')
    except:
        print("⚠️  无法加载Hydra配置，将使用随机策略")

    ws = Workspace(cfg=cfg)
    ws.action_sleep = args.action_sleep
    ws._buffer_dir = Path.cwd() / args.buffer_dir

    ws.setup()

    if cfg is not None:
        ws.make_agent(cfg)

    if args.snapshot:
        ws.load_snapshot(args.snapshot)

    if args.eval:
        ws.eval(num_episodes=args.eval_episodes, max_steps_per_episode=args.steps)
    else:
        ws.train(num_episodes=args.episodes, max_steps_per_episode=args.steps)


if __name__ == '__main__':
    main()
