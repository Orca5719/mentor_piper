import numpy as np
import time
import cv2
import json
import os
import threading
from collections import deque
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import hydra
from pathlib import Path
import utils
import torch
from dm_env import specs
import copy

from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import VideoRecorder
import math
import re
from copy import deepcopy

from piper.robot import PiperRobot

torch.backends.cudnn.benchmark = True


class ManualTrainer:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        print("#"*60)
        print("         Piper 手动训练与数据收集")
        print("#"*60)
        print(f"\n工作目录: {self.work_dir}")
        
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._discount_alpha = cfg.discount_alpha
        self._discount_alpha_temp = cfg.discount_alpha_temp
        self._discount_beta = cfg.discount_beta
        self._discount_beta_temp = cfg.discount_beta_temp
        self._nstep = cfg.nstep
        self._nstep_alpha = cfg.nstep_alpha
        self._nstep_alpha_temp = cfg.nstep_alpha_temp
        
        self.setup()
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        
        self._manual_reward = 0.0
        self._keyboard_running = True
        self._keyboard_thread = None
        self._start_keyboard_listener()
    
    def _start_keyboard_listener(self):
        self._keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._keyboard_thread.start()
        print("\n✓ 键盘监听已启动")
        print("  按键说明:")
        print("    空格  = +10 奖励")
        print("    s     = 保存模型")
        print("    q     = 退出")
        print()
    
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
                            print(f"\n[手动奖励] +10.0")
                        elif key == 's':
                            print(f"\n正在保存模型...")
                            self.save_snapshot()
                        elif key == 'q':
                            print(f"\n用户退出...")
                            self._keyboard_running = False
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
                                print(f"\n[手动奖励] +10.0")
                            elif char == 's':
                                print(f"\n正在保存模型...")
                                self.save_snapshot()
                            elif char == 'q':
                                print(f"\n用户退出...")
                                self._keyboard_running = False
                        except Exception:
                            pass
                        time.sleep(0.05)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception as e:
            print(f"键盘监听错误：{e}")
    
    def setup(self):
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=False)
        
        use_sim = getattr(self.cfg, 'use_sim', True)
        visualize = getattr(self.cfg, 'visualize', True)
        obj_pos = getattr(self.cfg, 'obj_pos', None)
        goal_pos = getattr(self.cfg, 'goal_pos', None)
        use_apriltag = getattr(self.cfg, 'use_apriltag', False)
        tag_size = getattr(self.cfg, 'tag_size', 0.05)
        
        self.robot = PiperRobot(
            use_sim=use_sim,
            camera_width=256,
            camera_height=256,
            obj_pos=obj_pos,
            goal_pos=goal_pos,
            use_apriltag=use_apriltag,
            tag_size=tag_size
        )
        
        self.observation_space = specs.BoundedArray(
            (256, 256, 9),
            np.uint8,
            0,
            255,
            name='observation'
        )
        
        self.action_space = specs.BoundedArray(
            (4,),
            np.float32,
            -1.0,
            1.0,
            'action'
        )
        
        data_specs = (
            self.observation_space,
            self.action_space,
            specs.Array((1, ), np.float32, 'reward'),
            specs.Array((1, ), np.float32, 'discount')
        )
        
        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_snapshot,
            math.floor(self._nstep + self._nstep_alpha),
            self._discount - self._discount_alpha - self._discount_beta)
        self._replay_iter = None
        
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
    
    @property
    def global_step(self):
        return self._global_step
    
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def save_snapshot(self, step_id=None):
        if step_id is None:
            snapshot = self.work_dir / 'manual_snapshot.pt'
        else:
            if not os.path.exists(str(self.work_dir) + '/snapshots'):
                os.makedirs(str(self.work_dir) + '/snapshots')
            snapshot = self.work_dir / 'snapshots' / f'manual_snapshot_{step_id}.pt'
        
        try:
            if hasattr(self, 'agent'):
                keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
                payload = {k: self.__dict__[k] for k in keys_to_save if hasattr(self, k)}
                with snapshot.open('wb') as f:
                    torch.save(payload, f)
                print(f"✓ 模型已保存: {snapshot}")
            else:
                print("⚠️  未初始化 agent，只保存了数据到 replay buffer")
        except Exception as e:
            print(f"✗ 保存失败: {e}")
    
    def get_observation(self, frames):
        if len(frames) < 3:
            while len(frames) < 3:
                frames.append(frames[0])
        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))
        return stacked
    
    def collect_data_with_spacemouse(self, num_episodes=10, train_while_collect=False):
        print("\n" + "="*60)
        print("使用 3D 鼠标收集数据")
        print("="*60)
        
        try:
            import pyspacemouse
        except ImportError:
            print("✗ pyspacemouse 未安装，请先安装: pip install pyspacemouse")
            return
        
        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError:
            print("✗ piper_sdk 未找到")
            return
        
        if train_while_collect:
            self.init_agent()
        
        piper = C_PiperInterface_V2("can0")
        piper.ConnectPort()
        while not piper.EnablePiper():
            time.sleep(0.01)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        
        factor = 1000
        
        X = 300
        Y = 0
        Z = 200
        RX = -179
        RY = 24
        RZ = 178
        
        X = round(X)
        Y = round(Y)
        Z = round(Z)
        RX = round(RX * factor)
        RY = round(RY * factor)
        RZ = round(RZ * factor)
        joint_6 = round(0.0 * 1000 * 1000)
        
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        time.sleep(1.0)
        
        print("\n✓ 机械臂已准备好")
        print("开始使用 3D 鼠标控制...")
        
        episode = 0
        episode_step = 0
        episode_reward = 0
        frames = deque(maxlen=3)
        prev_action = np.zeros(4, dtype=np.float32)
        
        for _ in range(3):
            frames.append(self.robot.get_camera_image())
        
        with pyspacemouse.open() as device:
            while episode < num_episodes and self._keyboard_running:
                state = device.read()
                
                state_X = round(state.x * 5)
                state_Y = round(state.y * 5)
                state_Z = round(state.z * 5)
                
                X = round(X + state_X)
                Y = round(Y + state_Y)
                Z = round(Z + state_Z)
                
                piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
                
                if state.buttons[0]:
                    joint_6 = round(0.08 * 1000 * 1000)
                elif state.buttons[1]:
                    joint_6 = round(0.00 * 1000 * 1000)
                piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                
                action = np.array([
                    state.x * 2.0,
                    state.y * 2.0,
                    state.z * 2.0,
                    (joint_6 / (1000 * 1000) - 0.04) * 50.0
                ], dtype=np.float32)
                action = np.clip(action, -1.0, 1.0)
                
                frames.append(self.robot.get_camera_image())
                obs = self.get_observation(list(frames))
                
                reward = self._manual_reward
                if self._manual_reward != 0:
                    self._manual_reward = 0
                
                discount = 1.0
                
                if episode_step > 0:
                    ts_prev = type('', (), {})()
                    ts_prev.observation = self.get_observation(list(frames)[:-1])
                    ts_prev.reward = reward
                    ts_prev.discount = discount
                    self.replay_storage.add(ts_prev)
                
                if train_while_collect and self._global_step > self.cfg.num_seed_frames:
                    if self._global_step % self.cfg.update_every_steps == 0:
                        metrics = self.agent.update(self.replay_iter, self._global_step)
                        self.logger.log_metrics(metrics, self._global_step * self.cfg.action_repeat, ty='train')
                
                episode_reward += reward
                episode_step += 1
                self._global_step += 1
                
                if episode_step >= 250:
                    print(f"\n[Episode {episode + 1}/{num_episodes}] 完成，奖励: {episode_reward:.1f}")
                    episode += 1
                    episode_step = 0
                    episode_reward = 0
                    
                    if self.cfg.save_snapshot and episode % 5 == 0:
                        self.save_snapshot(episode)
                
                time.sleep(0.01)
        
        print("\n数据收集完成！")
        self.save_snapshot()
    
    def collect_data_with_keyboard(self, num_episodes=10, train_while_collect=False):
        print("\n" + "="*60)
        print("使用键盘收集数据")
        print("="*60)
        print("\n控制说明:")
        print("  W/S: 前后移动")
        print("  A/D: 左右移动")
        print("  Q/E: 上下移动")
        print("  1/2: 夹爪开合")
        print("  空格: +10奖励")
        print("  s: 保存")
        print("  q: 退出")
        print()
        
        if train_while_collect:
            self.init_agent()
        
        self.robot.reset()
        
        episode = 0
        episode_step = 0
        episode_reward = 0
        frames = deque(maxlen=3)
        
        for _ in range(3):
            frames.append(self.robot.get_camera_image())
        
        action = np.zeros(4, dtype=np.float32)
        
        print("开始收集数据...")
        
        try:
            while episode < num_episodes and self._keyboard_running:
                import sys
                key = -1
                if sys.platform == 'win32':
                    import msvcrt
                    if msvcrt.kbhit():
                        key = ord(msvcrt.getwch())
                
                action = np.zeros(4, dtype=np.float32)
                
                if key == ord('w') or key == ord('W'):
                    action[1] = 0.5
                elif key == ord('s') or key == ord('S'):
                    action[1] = -0.5
                elif key == ord('a') or key == ord('A'):
                    action[0] = -0.5
                elif key == ord('d') or key == ord('D'):
                    action[0] = 0.5
                elif key == ord('q') or key == ord('Q'):
                    action[2] = 0.5
                elif key == ord('e') or key == ord('E'):
                    action[2] = -0.5
                elif key == ord('1'):
                    action[3] = -1.0
                elif key == ord('2'):
                    action[3] = 1.0
                
                self.robot.step(action)
                
                frames.append(self.robot.get_camera_image())
                obs = self.get_observation(list(frames))
                
                reward = self._manual_reward
                if self._manual_reward != 0:
                    self._manual_reward = 0
                
                discount = 1.0
                
                if episode_step > 0:
                    ts_prev = type('', (), {})()
                    ts_prev.observation = self.get_observation(list(frames)[:-1])
                    ts_prev.reward = reward
                    ts_prev.discount = discount
                    self.replay_storage.add(ts_prev)
                
                if train_while_collect and self._global_step > self.cfg.num_seed_frames:
                    if self._global_step % self.cfg.update_every_steps == 0:
                        metrics = self.agent.update(self.replay_iter, self._global_step)
                        self.logger.log_metrics(metrics, self._global_step * self.cfg.action_repeat, ty='train')
                
                episode_reward += reward
                episode_step += 1
                self._global_step += 1
                
                frame = self.robot.get_camera_image()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"Episode: {episode + 1}/{num_episodes}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Step: {episode_step}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Reward: {episode_reward:.1f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Manual Control", frame_bgr)
                
                cv2_key = cv2.waitKey(1) & 0xFF
                if cv2_key == ord('q'):
                    print("\n用户退出")
                    self._keyboard_running = False
                    break
                
                if episode_step >= 250:
                    print(f"\n[Episode {episode + 1}/{num_episodes}] 完成，奖励: {episode_reward:.1f}")
                    episode += 1
                    episode_step = 0
                    episode_reward = 0
                    self.robot.reset()
                    
                    for _ in range(3):
                        frames.append(self.robot.get_camera_image())
                    
                    if self.cfg.save_snapshot and episode % 5 == 0:
                        self.save_snapshot(episode)
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            cv2.destroyAllWindows()
            self.robot.close()
        
        print("\n数据收集完成！")
        self.save_snapshot()
    
    def init_agent(self):
        from train_piper import make_agent
        self.agent = make_agent(self.observation_space, self.action_space, self.cfg.agent)
        print("\n✓ Agent 已初始化，将在收集数据的同时进行训练")


@hydra.main(config_path='piper/cfgs', config_name='config')
def main(cfg):
    print("\n请选择模式:")
    print("1. 使用 3D 鼠标收集数据（同时训练）")
    print("2. 使用 3D 鼠标收集数据（仅收集）")
    print("3. 使用键盘收集数据（同时训练）")
    print("4. 使用键盘收集数据（仅收集）")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    trainer = ManualTrainer(cfg)
    
    if choice == '1':
        num_episodes = int(input("收集多少个 episode? (默认 10): ") or "10")
        trainer.collect_data_with_spacemouse(num_episodes=num_episodes, train_while_collect=True)
    elif choice == '2':
        num_episodes = int(input("收集多少个 episode? (默认 10): ") or "10")
        trainer.collect_data_with_spacemouse(num_episodes=num_episodes, train_while_collect=False)
    elif choice == '3':
        num_episodes = int(input("收集多少个 episode? (默认 10): ") or "10")
        trainer.collect_data_with_keyboard(num_episodes=num_episodes, train_while_collect=True)
    elif choice == '4':
        num_episodes = int(input("收集多少个 episode? (默认 10): ") or "10")
        trainer.collect_data_with_keyboard(num_episodes=num_episodes, train_while_collect=False)
    else:
        print("无效选项")


if __name__ == '__main__':
    main()
