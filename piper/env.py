import numpy as np
import time
import cv2
import threading
import gym
from gymnasium import spaces
from dm_env import StepType, specs
from typing import Any, NamedTuple
from collections import deque
from .robot import PiperRobot


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    success: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class NormalizeAction:
    def __init__(self, env, key="action"):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()


class PiperEnv:
    def __init__(self, task_name, seed, action_repeat=2, size=(256, 256),
                 use_sim=False, visualize=False, use_apriltag=False,
                 tag_size=0.05, goal_pos=None, obj_pos=None,
                 camera_calibration_file='camera_calibration.npz',
                 hand_eye_calibration_file='simple_hand_eye.json'):
        self.task_name = task_name
        self.action_repeat = action_repeat
        self.visualize = visualize
        self._window_name = "Piper Training Camera"
        
        self.robot = PiperRobot(
            use_sim=use_sim,
            camera_width=size[0],
            camera_height=size[1],
            obj_pos=obj_pos,
            goal_pos=goal_pos,
            use_apriltag=use_apriltag,
            tag_size=tag_size,
            camera_calibration_file=camera_calibration_file,
            hand_eye_calibration_file=hand_eye_calibration_file
        )
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=size + (3,), dtype=np.uint8
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        self.step_count = 0
        self.obj_init_pos = np.array([0.0, 0.6, 0.0])
        self._episode_reward = 0.0
        
        self._manual_reward = 0.0
        self._keyboard_thread = None
        self._keyboard_running = False
        self._start_keyboard_listener()
    
    def _keyboard_listener(self):
        try:
            import sys
            if sys.platform == 'win32':
                import msvcrt
                while self._keyboard_running:
                    if msvcrt.kbhit():
                        key = msvcrt.getwch()
                        if key == ' ':
                            self._manual_reward += 10.0
                            print(f"\n[手动奖励] +10.0 (按空格键)")
                    time.sleep(0.05)
            else:
                import termios
                import tty
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while self._keyboard_running:
                        try:
                            char = sys.stdin.read(1)
                            if char == ' ':
                                self._manual_reward += 10.0
                                print(f"\n[手动奖励] +10.0 (按空格键)")
                        except Exception:
                            pass
                        time.sleep(0.05)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception as e:
            print(f"键盘监听错误：{e}")
    
    def _start_keyboard_listener(self):
        pass
        # self._keyboard_running = True
        # self._keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        # self._keyboard_thread.start()
        # print("✓ 键盘监听已启动（空格=奖励）")
    
    def _stop_keyboard_listener(self):
        self._keyboard_running = False
        if self._keyboard_thread:
            self._keyboard_thread.join(timeout=1.0)
    
    def _visualize_frame(self, obs, reward, success, step_count):
        if not self.visualize:
            return
        
        display_img = obs.copy()
        
        y_pos = 30
        line_spacing = 25
        
        cv2.putText(display_img, f"Step: {step_count}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += line_spacing
        
        cv2.putText(display_img, f"Reward: {reward:.2f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += line_spacing
        
        cv2.putText(display_img, f"Success: {'Yes' if success else 'No'}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += line_spacing
        
        cv2.putText(display_img, "Press 'q' to close (training continues)", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        y_pos += line_spacing
        
        cv2.putText(display_img, "SPACE=+reward", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        cv2.imshow(self._window_name, display_img_bgr)
        
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     print("\n用户关闭可视化窗口，继续训练...")
        #     self.visualize = False
        #     try:
        #         cv2.destroyWindow(self._window_name)
        #     except:
        #         pass
    
    @property
    def obs_space(self):
        return {
            "image": self.observation_space,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
    
    @property
    def act_space(self):
        return {"action": self.action_space}
    
    def _compute_reward(self, action, obs=None):
        reward = self._manual_reward
        
        if self._manual_reward != 0:
            self._manual_reward = 0
        
        obj_pos = self.robot.get_obj_pos()
        target_pos = self.robot.get_goal_pos()
        obj_to_target = np.linalg.norm(obj_pos - target_pos)
        success = float(obj_to_target <= 0.07)
        
        return reward, success, obj_to_target
    
    def reset(self):
        """【关键修改】环境重置：调用 robot.reset()，确保与 manual_collect.py 一致"""
        self.robot.reset()  # 机器人重置（夹爪张开+位姿初始）
        self.step_count = 0
        self._episode_reward = 0.0
        self._manual_reward = 0.0
        
        if self.robot.use_sim:
            if np.random.random() < 0.3:
                obj_x = np.random.uniform(-0.1, 0.1)
                obj_y = np.random.uniform(0.55, 0.65)
                self.robot.set_obj_pos([obj_x, obj_y, 0.0])
                
                goal_x = np.random.uniform(-0.05, 0.05)
                goal_y = np.random.uniform(0.7, 0.75)
                while np.linalg.norm([obj_x - goal_x, obj_y - goal_y]) < 0.15:
                    goal_x = np.random.uniform(-0.05, 0.05)
                    goal_y = np.random.uniform(0.7, 0.75)
                self.robot.set_goal_pos([goal_x, goal_y, 0.0])
        
        self.obj_init_pos = self.robot.get_obj_pos().copy()
        
        obs = self.robot.get_camera_image()
        
        self._visualize_frame(obs, 0.0, False, 0)
        
        return {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": obs,
            "success": False
        }
    
    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        total_reward = 0.0
        success = 0.0
        obj_to_target = 0.0
        
        for _ in range(self.action_repeat):
            self.robot.step(action["action"])
            reward, suc, ott = self._compute_reward(action["action"])
            success += float(suc)
            total_reward += reward or 0.0
            obj_to_target = ott
        
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        
        obs = self.robot.get_camera_image()
        self.step_count += 1
        self._episode_reward += total_reward
        
        self._visualize_frame(obs, total_reward, bool(success), self.step_count)
        
        return {
            "reward": total_reward,
            "is_first": False,
            "is_last": False,
            "is_terminal": False,
            "image": obs,
            "success": success
        }
    
    def render(self):
        return self.robot.get_camera_image()
    
    def close(self):
        self._stop_keyboard_listener()
        
        try:
            if self.visualize:
                cv2.destroyAllWindows()
        except:
            pass
        
        self.robot.close()


class PiperWrapper:
    def __init__(self, env, frame_stack=3):
        self._env = env
        self.frame_stack = frame_stack
        wos = env.obs_space['image']
        low = np.repeat(wos.low, frame_stack, axis=-1)
        high = np.repeat(wos.high, frame_stack, axis=-1)
        self.stackedobs = np.zeros(low.shape, low.dtype)
        
        self.observation_space = spaces.Box(low=np.transpose(low, (2, 0, 1)), 
                                             high=np.transpose(high, (2, 0, 1)), 
                                             dtype=np.uint8)

    def observation_spec(self):
        return specs.BoundedArray(self.observation_space.shape,
                                  np.uint8,
                                  0,
                                  255,
                                  name='observation')

    def action_spec(self):
        return specs.BoundedArray(self._env.act_space['action'].shape,
                                  np.float32,
                                  self._env.act_space['action'].low,
                                  self._env.act_space['action'].high,
                                  'action')

    def reset(self):
        time_step = self._env.reset()
        obs = time_step['image']
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return ExtendedTimeStep(observation=np.transpose(self.stackedobs, (2, 0, 1)),
                                 step_type=StepType.FIRST,
                                 action=np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype),
                                 reward=0.0,
                                 discount=1.0,
                                success=time_step['success'])

    def step(self, action):
        action = {'action': action}
        time_step = self._env.step(action)
        obs = time_step['image']
        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[-1], axis=-1)
        self.stackedobs[..., -obs.shape[-1]:] = obs
        if time_step['is_first']:
            step_type = StepType.FIRST
        elif time_step['is_last']:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        return ExtendedTimeStep(observation=np.transpose(self.stackedobs, (2, 0, 1)),
                                 step_type=step_type,
                                 action=action['action'],
                                 reward=time_step['reward'],
                                 discount=1.0,
                                success=time_step['success'])

    @property
    def act_space(self):
        return self._env.act_space

    @property
    def obs_space(self):
        os = dict(self._env.obs_space)
        os["image"] = self.observation_space
        return os

    def close(self):
        self._env.close()


def make(task_name, seed, action_repeat=2, size=(256, 256),
         use_sim=False, visualize=False, use_apriltag=False,
         tag_size=0.05, goal_pos=None, obj_pos=None,
         camera_calibration_file='camera_calibration.npz',
         hand_eye_calibration_file='simple_hand_eye.json',
         frame_stack=3):
    env = PiperEnv(
        task_name, seed, action_repeat, size,
        use_sim, visualize, use_apriltag,
        tag_size, goal_pos, obj_pos,
        camera_calibration_file, hand_eye_calibration_file
    )
    
    env = NormalizeAction(env)
    env = TimeLimit(env, 250)
    env = PiperWrapper(env, frame_stack)
    
    return env
