import numpy as np
import time
import cv2
import os
from collections import deque
import sys
import threading
from pathlib import Path
from dm_env import specs, TimeStep as DMTimeStep, StepType

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from replay_buffer import ReplayBufferStorage
    import piper.env as piper_env
except ImportError as e:
    print(f"错误：无法导入核心模块: {e}")
    sys.exit(1)


class PiperCollectEnv:
    def __init__(self, factor=1000, img_height=256, img_width=256):
        self.factor = factor
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        
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

    def _init_replay_storage(self):
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
        
        self._init_replay_storage()
        
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

    def _apply_action(self, state_x, state_y, state_z, gripper_btn_0, gripper_btn_1):
        state_x_clip = np.clip(state_x * 40.0, -20.0, 20.0)
        state_y_clip = np.clip(state_y * 40.0, -20.0, 20.0)
        state_z_clip = np.clip(state_z * 40.0, -20.0, 20.0)
        
        self.X += round(state_x_clip * self.factor)
        self.Y += round(state_y_clip * self.factor)
        self.Z += round(state_z_clip * self.factor)
        
        if gripper_btn_0:
            self.joint_6 = 80000
        elif gripper_btn_1:
            self.joint_6 = 0
        
        self.piper_arm.EndPoseCtrl(self.X, self.Y, self.Z, self.RX, self.RY, self.RZ)
        self.piper_arm.GripperCtrl(abs(self.joint_6), 1000, 0x01, 0)

    def _get_action_for_buffer(self, state_x, state_y, state_z, gripper_raw):
        dx = np.clip(state_x * 40.0, -20.0, 20.0) / 20.0
        dy = np.clip(state_y * 40.0, -20.0, 20.0) / 20.0
        dz = np.clip(state_z * 40.0, -20.0, 20.0) / 20.0
        dg = 1.0 if gripper_raw > 40000 else -1.0
        
        action = np.zeros(self._act_spec.shape, dtype=self._act_spec.dtype)
        if self._act_spec.shape[0] >= 3:
            action[0:3] = [dx, dy, dz]
        if self._act_spec.shape[0] >= 7:
            action[6] = dg
        elif self._act_spec.shape[0] == 4:
            action[3] = dg
        
        return action

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
        return DMTimeStep(
            step_type=StepType.FIRST,
            reward=None,
            discount=None,
            observation=obs
        )

    def step(self, state, reward=0.0):
        self._apply_action(state.x, state.y, state.z, state.buttons[0], state.buttons[1])
        
        new_frame = self.robot_cam.get_camera_image()
        if new_frame is not None:
            self.frames_queue.append(new_frame)
        
        obs = self._get_stacked_obs()
        self._reward = reward
        self._step_count += 1
        
        return DMTimeStep(
            step_type=StepType.MID,
            reward=np.float32(self._reward),
            discount=np.float32(1.0),
            observation=obs
        )

    def step_last(self, state):
        self._apply_action(state.x, state.y, state.z, state.buttons[0], state.buttons[1])
        
        new_frame = self.robot_cam.get_camera_image()
        if new_frame is not None:
            self.frames_queue.append(new_frame)
        
        obs = self._get_stacked_obs()
        
        return DMTimeStep(
            step_type=StepType.LAST,
            reward=np.float32(0.0),
            discount=np.float32(0.0),
            observation=obs
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


class TimeStepWithAction:
    def __init__(self, observation, action, reward, discount, step_type):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.discount = discount
        self.step_type = step_type
        self.first = (step_type == StepType.FIRST)
        self.is_last = (step_type == StepType.LAST)

    def last(self):
        return self.is_last

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        raise KeyError(f"Invalid key: {key}")


class SimpleSpacemouseCollect:
    def __init__(self):
        print("="*60)
        print("     Piper 3D鼠标数据收集工具 (重构版)")
        print("="*60)
        
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        
        self._buffer_dir = Path.cwd() / 'buffer'
        self.replay_storage = None
        self.data_buffer = []
        
        self.env = PiperCollectEnv(
            factor=1000,
            img_height=self.IMG_HEIGHT,
            img_width=self.IMG_WIDTH
        )
        
        self.episode = 0
        self.episode_step = 0
        self._global_step = 0

    def _init_replay_storage(self):
        print(f"\n[Buffer] 数据将保存至: {self._buffer_dir.resolve()}")
        
        data_specs = (
            self.env.observation_spec(),
            self.env.action_spec(),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount')
        )
        
        self.replay_storage = ReplayBufferStorage(data_specs, self._buffer_dir)
        
        print(f"[Buffer] Observation: {self.env.observation_spec().shape}, {self.env.observation_spec().dtype}")
        print(f"[Buffer] Action: {self.env.action_spec().shape}, {self.env.action_spec().dtype}")
        print("✅ ReplayBuffer初始化完成")

    def collect(self, num_episodes=5, max_steps=200, episode_sleep=2.0):
        self.env.init_hardware()
        self._init_replay_storage()
        
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
                while self.episode < num_episodes:
                    time_step = self.env.reset()
                    
                    zero_action = np.zeros(self.env.action_spec().shape, dtype=self.env.action_spec().dtype)
                    ts_init = TimeStepWithAction(
                        observation=time_step.observation,
                        action=zero_action,
                        reward=np.array([0.0], dtype=np.float32),
                        discount=np.array([1.0], dtype=np.float32),
                        step_type=StepType.FIRST
                    )
                    self.replay_storage.add(ts_init)
                    
                    self.episode_step = 0
                    episode_reward = 0.0
                    
                    print(f"\n🎬 Episode {self.episode+1}/{num_episodes} 开始")
                    
                    for step in range(max_steps):
                        state = device.read()
                        
                        current_reward = 0.0
                        episode_ended_early = False
                        frame = self.env.get_latest_frame()
                        if frame is not None:
                            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            cv2.putText(display, f"Ep:{self.episode+1}/{num_episodes} Step:{step+1}/{max_steps}", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(display, f"Buffer: {len(self.replay_storage)}", 
                                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            cv2.imshow("Data Collection", display)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(' '):
                            current_reward = 10.0
                            print(f"[奖励] Ep{self.episode+1} Step{step+1} | +10.0 | 结束episode")
                            episode_ended_early = True
                        elif key == ord('q'):
                            print("\n用户终止收集，正在保存数据...")
                            self.env.close()
                            cv2.destroyAllWindows()
                            self.save_npz()
                            return
                        
                        action = self.env._get_action_for_buffer(state.x, state.y, state.z, self.env.joint_6)
                        
                        if episode_ended_early:
                            ts_last = self.env.step_last(state)
                            ts_final = TimeStepWithAction(
                                observation=ts_last.observation,
                                action=action,
                                reward=np.array([current_reward], dtype=np.float32),
                                discount=np.array([0.0], dtype=np.float32),
                                step_type=StepType.LAST
                            )
                            self.replay_storage.add(ts_final)
                            episode_reward += current_reward
                            self.episode_step += 1
                            self._global_step += 1
                            break
                        
                        time_step = self.env.step(state, reward=current_reward)
                        episode_reward += current_reward
                        
                        ts = TimeStepWithAction(
                            observation=time_step.observation,
                            action=action,
                            reward=np.array([time_step.reward], dtype=np.float32),
                            discount=np.array([time_step.discount], dtype=np.float32),
                            step_type=time_step.step_type
                        )
                        
                        self.replay_storage.add(ts)
                        
                        self.data_buffer.append({
                            'observation': ts.observation,
                            'action': action,
                            'reward': current_reward
                        })
                        
                        self.episode_step += 1
                        self._global_step += 1
                        
                        time.sleep(0.01)
                    
                    state = device.read()
                    ts_last = self.env.step_last(state)
                    action = self.env._get_action_for_buffer(state.x, state.y, state.z, self.env.joint_6)
                    
                    ts_final = TimeStepWithAction(
                        observation=ts_last.observation,
                        action=action,
                        reward=np.array([0.0], dtype=np.float32),
                        discount=np.array([0.0], dtype=np.float32),
                        step_type=StepType.LAST
                    )
                    
                    if not episode_ended_early:
                        self.replay_storage.add(ts_final)
                    
                    print(f"\n✅ Episode {self.episode+1} 完成 | Reward: {episode_reward:.1f} | Buffer: {len(self.replay_storage)}")
                    
                    self.env.X, self.env.Y, self.env.Z = 300614, -12185, 282341
                    self.env.piper_arm.EndPoseCtrl(self.env.X, self.env.Y, self.env.Z, self.env.RX, self.env.RY, self.env.RZ)
                    self.env.joint_6 = 80000
                    self.env.piper_arm.GripperCtrl(abs(self.env.joint_6), 1000, 0x01, 0)
                    
                    print(f"请重新摆放物体... 休眠 {episode_sleep}s")
                    time.sleep(episode_sleep)
                    
                    self.episode += 1
        
        except ImportError:
            print("❌ 错误：pyspacemouse未安装，请执行 pip install pyspacemouse")
        except KeyboardInterrupt:
            print("\n用户中断收集")
        finally:
            self.env.close()
            cv2.destroyAllWindows()
            self.save_npz()
            
            print(f"\n📊 收集完成！Buffer总有效步数: {len(self.replay_storage)}")
            print(f"📂 Buffer文件位置: {self._buffer_dir.resolve()}")

    def save_npz(self, filename="spacemouse_backup.npz"):
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
