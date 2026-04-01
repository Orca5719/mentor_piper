import numpy as np
import cv2
from gym import spaces
from dm_env import StepType, specs
from typing import Any, NamedTuple
from collections import deque
from piper.robot import PiperRobot


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
            raise AttributeError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})
    
    def reset(self):
        return self._env.reset()


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
            raise AttributeError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            obs["TimeLimit.truncated"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()


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


class PiperEnv:
    def __init__(self, task_name, seed=None, action_repeat=1, 
                 size=(256, 256), use_sim=True, visualize=False,
                 obj_pos=None, goal_pos=None, print_reward=True,
                 debug_mode=False, use_apriltag=False, tag_size=0.05,
                 enable_spacemouse=False, spacemouse_scale=0.05):
        self.task_name = task_name
        self._size = size
        self._action_repeat = action_repeat
        self._visualize = visualize
        self._print_reward = print_reward
        self._debug_mode = debug_mode
        
        np.random.seed(seed)
        
        self.robot = PiperRobot(use_sim=use_sim, 
                                camera_width=size[0], 
                                camera_height=size[1],
                                obj_pos=obj_pos,
                                goal_pos=goal_pos,
                                debug_mode=debug_mode,
                                use_apriltag=use_apriltag,
                                tag_size=tag_size)
        
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=size + (3,), 
            dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(7,), 
            dtype=np.float32
        )
        
        self.step_count = 0
        self.obj_init_pos = np.array([0.0, 0.6, 0.0])
        self._episode_reward = 0.0
        self._window_name = "Piper Training Camera"
        self._manual_reward = 0.0  # 手动奖励
        
        # 初始化 SpaceMouse 控制器（可选）
        self._spacemouse_enabled = enable_spacemouse
        self._spacemouse_controller = None
        if self._spacemouse_enabled:
            try:
                from piper.spacemouse_controller import PiperSpaceMouseController
                self._spacemouse_controller = PiperSpaceMouseController(
                    scale_factor=spacemouse_scale
                )
                print("✓ SpaceMouse 已启用")
            except Exception as e:
                print(f"⚠️ SpaceMouse 初始化失败: {e}")
                self._spacemouse_enabled = False
    
    def _visualize_frame(self, obs, reward, success, step_count, obj_to_target=None, episode_reward=None):
        if not self._visualize:
            return
        
        try:
            display_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            
            y_pos = 30
            line_spacing = 30
            
            cv2.putText(display_img, f"Step: {step_count}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += line_spacing
            
            cv2.putText(display_img, f"Reward: {reward:.2f}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += line_spacing
            
            if episode_reward is not None:
                cv2.putText(display_img, f"Episode Reward: {episode_reward:.2f}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += line_spacing
            
            if obj_to_target is not None:
                color = (0, 255, 0) if obj_to_target <= 0.07 else (0, 165, 255)
                cv2.putText(display_img, f"Obj->Target: {obj_to_target:.4f}m", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += line_spacing
            
            success_color = (0, 255, 0) if success else (0, 0, 255)
            cv2.putText(display_img, f"Success: {'Yes' if success else 'No'}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, success_color, 2)
            y_pos += line_spacing
            
            cv2.putText(display_img, "Press 'q' to close (training continues)", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            y_pos += line_spacing
            
            cv2.putText(display_img, "Press SPACE for +reward", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(self._window_name, display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户关闭可视化窗口，继续训练...")
                self._visualize = False
                try:
                    cv2.destroyWindow(self._window_name)
                except:
                    pass
            elif key == ord(' '):  # 空格键
                self._manual_reward += 10.0  # 按空格键增加 5 点奖励
                print(f"\n[手动奖励] +10.0 (当前累积：{self._manual_reward:.1f})")
                    
        except Exception as e:
            print(f"可视化错误: {e}")
            self._visualize = False
    
    @property
    def obs_space(self):
        return {
            "image": self.observation_space,
            "reward": spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": spaces.Box(0, 1, (), dtype=bool),
            "is_last": spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": spaces.Box(0, 1, (), dtype=bool),
            "success": spaces.Box(0, 1, (), dtype=bool),
        }
    
    @property
    def act_space(self):
        return {"action": self.action_space}
    
    def _compute_reward(self, action, obs=None):
        tcp_pos = self.robot.get_end_effector_pos()
        obj_pos = self.robot.get_obj_pos()
        target_pos = self.robot.get_goal_pos()
        
        # 调试打印（仅在debug_mode时打印）
        if self._debug_mode:
            print(f"[DEBUG] obj_pos: {obj_pos}, target_pos: {target_pos}")
            print(f"[DEBUG] aprilag_visible: {self.robot.apriltag_visible}")
        
        # AprilTag 丢失惩罚（分级处理）
        apriltag_penalty = 0.0
        if not self.robot.apriltag_visible:
            # 分级：短暂丢失 = 轻微惩罚，持续丢失 = 重惩罚
            apriltag_penalty = -0.1  # 轻微警告

        # 位置限制惩罚（分级：超出不同范围有不同惩罚）
        position_limit_penalty = 0.0
        if hasattr(self.robot, 'position_limit_violation_factor'):
            # 使用 violation_factor（超出程度：0-1）
            factor = getattr(self.robot, 'position_limit_violation_factor', 0.0)
            if factor > 0:
                # 基础惩罚 -0.3，最大可达 -1.5
                position_limit_penalty = -0.3 - factor * 1.2
        elif hasattr(self.robot, 'position_limit_violated') and self.robot.position_limit_violated:
            position_limit_penalty = -0.5  # 降级兼容

        # 机械臂卡住惩罚（使用卡住计数器）
        stuck_penalty = 0.0
        if hasattr(self.robot, 'stuck_counter'):
            # 卡住时间越长，惩罚越大
            stuck_count = getattr(self.robot, 'stuck_counter', 0)
            if stuck_count > 0:
                # 每帧 -0.05，最高 -1.5（30帧）
                stuck_penalty = -min(stuck_count * 0.05, 1.5)
        elif hasattr(self.robot, 'is_stuck') and self.robot.is_stuck():
            stuck_penalty = -0.5  # 降级兼容
        
        scale = np.array([2., 2., 1.])
        target_to_obj = (obj_pos - target_pos) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        
        obj_init_pos = self.obj_init_pos
        target_to_obj_init = (obj_init_pos - target_pos) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)
        
        in_place = self._tolerance(
            target_to_obj,
            bounds=(0, 0.05),
            margin=target_to_obj_init,
            sigmoid='long_tail'
        )
        
        tcp_to_obj = np.linalg.norm(obj_pos - tcp_pos)
        
        # 只取前 6 个关节给 gripper reward 函数
        gripper_action = action[:6] if len(action) > 6 else action
        object_grasped = self._gripper_caging_reward(
            gripper_action,
            obj_pos,
            tcp_pos,
            object_reach_radius=0.04,
            obj_radius=0.02,
            pad_success_thresh=0.05,
            xz_thresh=0.05
        )
        
        reward = self._hamacher_product(object_grasped, in_place)
        
        if tcp_to_obj < 0.04:
            reward += 1.0 + 5.0 * in_place
        
        obj_to_target = np.linalg.norm(obj_pos - target_pos)
        if obj_to_target < 0.05:
            reward = 10.0
        
        # 添加所有惩罚（分层机制，避免冲突同时保持约束力）
        reward += apriltag_penalty + position_limit_penalty + stuck_penalty
        
        # 添加手动奖励
        reward += self._manual_reward
        if self._manual_reward > 0:
            self._manual_reward = 0  # 重置手动奖励
        
        success = float(obj_to_target <= 0.07)
        
        return reward, success, obj_to_target
    
    def _tolerance(self, x, bounds=(0, 0), margin=0, sigmoid='gaussian'):
        lower, upper = bounds
        in_bounds = (lower <= x) & (x <= upper)
        
        if sigmoid == 'gaussian':
            d = np.maximum(x - upper, lower - x)
            out_of_bounds = np.exp(-(d ** 2) / (2 * margin ** 2))
        elif sigmoid == 'long_tail':
            d = np.maximum(x - upper, lower - x)
            out_of_bounds = 1 / (1 + (d / margin) ** 2)
        elif sigmoid == 'tanh':
            d = np.maximum(x - upper, lower - x)
            out_of_bounds = (1 - np.tanh(d / margin)) / 2
        else:
            raise ValueError(f"Unknown sigmoid type: {sigmoid}")
        
        return np.where(in_bounds, 1.0, out_of_bounds)
    
    def _hamacher_product(self, a, b):
        return a * b / (a + b - a * b + 1e-8)
    
    def _gripper_caging_reward(self, action, obj_pos, tcp_pos,
                                object_reach_radius=0.04, obj_radius=0.02,
                                pad_success_thresh=0.05, xz_thresh=0.05):
        tcp_to_obj = np.linalg.norm(obj_pos - tcp_pos)
        
        in_object_reach = float(tcp_to_obj < object_reach_radius)
        object_reach_reward = self._tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=0.1
        )
        
        return object_reach_reward
    
    def reset(self):
        self.robot.reset()
        self.step_count = 0
        self._episode_reward = 0.0
        self._manual_reward = 0.0  # 重置手动奖励
        
        # 只在模拟模式下才随机化目标位置
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
        
        # 检查是否有 SpaceMouse 人工干预
        spacemouse_action = None
        if self._spacemouse_enabled and self._spacemouse_controller is not None:
            spacemouse_action, is_active = self._spacemouse_controller.get_action()
            if is_active:
                # 有 SpaceMouse 输入时，使用人工控制
                action["action"] = spacemouse_action
        
        total_reward = 0.0
        success = 0.0
        obj_to_target = 0.0
        
        for _ in range(self._action_repeat):
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
        
        # 打印 reward 信息
        if self._print_reward:
            tcp_pos = self.robot.get_end_effector_pos()
            obj_pos = self.robot.get_obj_pos()
            apriltag_info = ""
            if self.robot.apriltag_visible:
                apriltag_info = f" | Apriltag: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]m"
            else:
                apriltag_info = " | Apriltag: ❌"
            
            # 添加限制和卡住状态信息
            limit_info = ""
            if hasattr(self.robot, 'position_limit_violated') and self.robot.position_limit_violated:
                limit_info += " | Limit: ⚠️"
            if hasattr(self.robot, 'stuck_counter') and self.robot.stuck_counter > 0:
                limit_info += f" | Stuck: {self.robot.stuck_counter}"
            
            # 添加 SpaceMouse 干预信息
            spacemouse_info = ""
            if spacemouse_action is not None and np.any(np.abs(spacemouse_action) > 0.001):
                spacemouse_info = " | 🖱️ SpaceMouse"
            
            print(f"[Step {self.step_count:3d}] "
                  f"Reward: {total_reward:6.2f} | "
                  f"Episode Reward: {self._episode_reward:6.2f} | "
                  f"Obj->Target: {obj_to_target:.4f}m | "
                  f"Success: {'✅' if success else '❌'}"
                  f"{apriltag_info}{limit_info}{spacemouse_info}")
        
        self._visualize_frame(obs, total_reward, bool(success), self.step_count, 
                             obj_to_target, self._episode_reward)
        
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
        try:
            if self._visualize:
                cv2.destroyAllWindows()
        except:
            pass
        # 关闭 SpaceMouse
        if self._spacemouse_controller is not None:
            self._spacemouse_controller.close()
        self.robot.close()


class PiperWrapper:
    def __init__(self, env, frame_stack=3):
        self._env = env
        self.frame_stack = frame_stack
        wos = env.obs_space['image']
        low = np.repeat(wos.low, self.frame_stack, axis=-1)
        high = np.repeat(wos.high, self.frame_stack, axis=-1)
        self.stackedobs = np.zeros(low.shape, low.dtype)

        self.observation_space = spaces.Box(
            low=np.transpose(low, (2, 0, 1)), 
            high=np.transpose(high, (2, 0, 1)), 
            dtype=np.uint8
        )

    def observation_spec(self):
        return specs.BoundedArray(
            self.observation_space.shape,
            np.uint8,
            0,
            255,
            name='observation'
        )

    def action_spec(self):
        return specs.BoundedArray(
            self._env.act_space['action'].shape,
            np.float32,
            self._env.act_space['action'].low,
            self._env.act_space['action'].high,
            'action'
        )

    def reset(self):
        time_step = self._env.reset()
        obs = time_step['image']
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return ExtendedTimeStep(
            observation=np.transpose(self.stackedobs, (2, 0, 1)),
            step_type=StepType.FIRST,
            action=np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype),
            reward=0.0,
            discount=1.0,
            success=time_step['success']
        )
    
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
        
        return ExtendedTimeStep(
            observation=np.transpose(self.stackedobs, (2, 0, 1)),
            step_type=step_type,
            action=action['action'],
            reward=time_step['reward'],
            discount=1.0,
            success=time_step['success']
        )
    
    def render(self):
        latest_frame = self.stackedobs[..., -3:]
        return latest_frame
    
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise AttributeError(name)


def make(name, frame_stack, action_repeat, seed, use_sim=True, visualize=False,
         obj_pos=None, goal_pos=None, print_reward=True,
         debug_mode=False, use_apriltag=False, tag_size=0.05):
    env = PiperEnv(name, seed, action_repeat, (84, 84), 
                   use_sim=use_sim, visualize=visualize,
                   obj_pos=obj_pos, goal_pos=goal_pos,
                   print_reward=print_reward,
                   debug_mode=debug_mode,
                   use_apriltag=use_apriltag, tag_size=tag_size)
    env = NormalizeAction(env)
    env = TimeLimit(env, 250)
    env = PiperWrapper(env, frame_stack)
    return env