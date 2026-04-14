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


class SimpleManualCollector:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        print("#"*60)
        print("      Piper 手动数据收集与训练工具")
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
        self._should_exit = False
        self._should_save = False
        
        self._key_state = {
            'w': False, 's': False, 'a': False, 'd': False,
            'q': False, 'e': False, '1': False, '2': False
        }
        
        self._start_keyboard_listener()
    
    def _start_keyboard_listener(self):
        self._keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._keyboard_thread.start()
        print("\n✓ 键盘监听已启动")
        print("  按键说明:")
        print("    W/S: 前后移动")
        print("    A/D: 左右移动")
        print("    Q/E: 上下移动")
        print("    1/2: 夹爪开合")
        print("    空格: +10 奖励")
        print("    S: 保存模型 & 数据")
        print("    Q: 退出 (长按)")
        print()
    
    def _keyboard_listener(self):
        try:
            import sys
            if sys.platform == 'win32':
                import msvcrt
                while not self._should_exit:
                    if msvcrt.kbhit():
                        key = msvcrt.getwch().lower()
                        if key == ' ':
                            self._manual_reward += 10.0
                            print(f"\n[手动奖励] +10.0")
                        elif key == 's':
                            self._should_save = True
                        elif key in self._key_state:
                            self._key_state[key] = True
                    time.sleep(0.01)
            else:
                import termios
                import tty
                import select
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while not self._should_exit:
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
                        if rlist:
                            char = sys.stdin.read(1).lower()
                            if char == ' ':
                                self._manual_reward += 10.0
                                print(f"\n[手动奖励] +10.0")
                            elif char == 's':
                                self._should_save = True
                            elif char in self._key_state:
                                self._key_state[char] = True
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception as e:
            print(f"键盘监听错误：{e}")
    
    def setup(self):
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=False)
        
        use_sim = getattr(self.cfg, 'use_sim', False)
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
                print(f"\n✓ 模型已保存: {snapshot}")
            else:
                print(f"\n✓ 数据已保存到 replay buffer (共 {len(self.replay_storage)} 步)")
        except Exception as e:
            print(f"\n✗ 保存失败: {e}")
    
    def get_observation(self, frames):
        if len(frames) < 3:
            while len(frames) < 3:
                frames.append(frames[0] if frames else np.zeros((256, 256, 3), dtype=np.uint8))
        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))
        return stacked
    
    def get_action_from_keys(self):
        action = np.zeros(4, dtype=np.float32)
        
        if self._key_state['w']:
            action[1] = 0.5
            self._key_state['w'] = False
        if self._key_state['s']:
            action[1] = -0.5
            self._key_state['s'] = False
        if self._key_state['a']:
            action[0] = -0.5
            self._key_state['a'] = False
        if self._key_state['d']:
            action[0] = 0.5
            self._key_state['d'] = False
        if self._key_state['q']:
            action[2] = 0.5
            self._key_state['q'] = False
        if self._key_state['e']:
            action[2] = -0.5
            self._key_state['e'] = False
        if self._key_state['1']:
            action[3] = -1.0
            self._key_state['1'] = False
        if self._key_state['2']:
            action[3] = 1.0
            self._key_state['2'] = False
        
        return action
    
    def collect_and_train(self, num_episodes=10, train_while_collect=True):
        print("\n" + "="*60)
        if train_while_collect:
            print("模式: 收集数据 + 同时训练")
        else:
            print("模式: 仅收集数据")
        print("="*60)
        
        if train_while_collect:
            self.init_agent()
        
        print("\n正在初始化机械臂...")
        self.robot.reset()
        
        episode = 0
        episode_step = 0
        episode_reward = 0
        frames = deque(maxlen=3)
        
        for _ in range(3):
            frames.append(self.robot.get_camera_image())
        
        last_obs = self.get_observation(list(frames))
        last_action = np.zeros(4, dtype=np.float32)
        
        print("\n✓ 准备就绪！开始操作...")
        
        pbar = tqdm(total=num_episodes, desc='收集进度', unit='episode')
        
        try:
            while episode < num_episodes and not self._should_exit:
                if self._should_save:
                    self.save_snapshot(self._global_step)
                    self._should_save = False
                
                action = self.get_action_from_keys()
                
                self.robot.step(action)
                
                frames.append(self.robot.get_camera_image())
                current_obs = self.get_observation(list(frames))
                
                reward = self._manual_reward
                if self._manual_reward != 0:
                    self._manual_reward = 0
                
                discount = 1.0
                
                if episode_step > 0:
                    ts = type('', (), {})()
                    ts.observation = last_obs
                    ts.reward = reward
                    ts.discount = discount
                    ts.action = last_action
                    
                    ts_storage = type('', (), {})()
                    ts_storage.observation = last_obs
                    ts_storage.reward = np.array([reward], dtype=np.float32)
                    ts_storage.discount = np.array([discount], dtype=np.float32)
                    
                    self.replay_storage.add(ts_storage)
                
                if train_while_collect and self._global_step > self.cfg.num_seed_frames:
                    if self._global_step % self.cfg.update_every_steps == 0:
                        metrics = self.agent.update(self.replay_iter, self._global_step)
                        self.logger.log_metrics(metrics, self._global_step * self.cfg.action_repeat, ty='train')
                
                last_obs = current_obs
                last_action = action
                episode_reward += reward
                episode_step += 1
                self._global_step += 1
                
                frame = self.robot.get_camera_image()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                y_pos = 30
                line_spacing = 25
                
                cv2.putText(frame_bgr, f"Episode: {episode + 1}/{num_episodes}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += line_spacing
                
                cv2.putText(frame_bgr, f"Step: {episode_step}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += line_spacing
                
                cv2.putText(frame_bgr, f"Reward: {episode_reward:.1f}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += line_spacing
                
                cv2.putText(frame_bgr, f"Total Steps: {self._global_step}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                y_pos += line_spacing
                
                cv2.putText(frame_bgr, "WASD/QE=Move, 1/2=Gripper, SPACE=Reward, S=Save", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                
                cv2.imshow("Manual Control", frame_bgr)
                
                key = cv2.waitKey(50) & 0xFF
                if key == 27 or key == ord('q'):
                    print("\n用户退出")
                    self._should_exit = True
                    break
                
                if episode_step >= 250:
                    print(f"\n[Episode {episode + 1}/{num_episodes}] 完成，总奖励: {episode_reward:.1f}")
                    pbar.update(1)
                    episode += 1
                    episode_step = 0
                    episode_reward = 0
                    
                    print("机械臂复位中...")
                    self.robot.reset()
                    
                    for _ in range(3):
                        frames.append(self.robot.get_camera_image())
                    last_obs = self.get_observation(list(frames))
                    last_action = np.zeros(4, dtype=np.float32)
                    
                    if self.cfg.save_snapshot and episode % 5 == 0:
                        self.save_snapshot(self._global_step)
        
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            pbar.close()
            cv2.destroyAllWindows()
            self.robot.close()
        
        print("\n" + "="*60)
        print("数据收集完成！")
        print(f"总步数: {self._global_step}")
        print(f"总 Episode: {episode}")
        print(f"Buffer 大小: {len(self.replay_storage)}")
        print("="*60)
        
        self.save_snapshot()
    
    def init_agent(self):
        from train_piper import make_agent
        self.agent = make_agent(self.observation_space, self.action_space, self.cfg.agent)
        print("\n✓ Agent 已初始化，将在收集数据的同时进行训练")


@hydra.main(config_path='piper/cfgs', config_name='config')
def main(cfg):
    print("\n" + "="*60)
    print("请选择操作模式:")
    print("="*60)
    print("1. 收集数据 + 同时训练 (推荐)")
    print("2. 仅收集数据 (之后再训练)")
    print("3. 退出")
    
    choice = input("\n请输入选项 (1-3): ").strip()
    
    if choice == '3':
        print("退出")
        return
    
    num_episodes_input = input("\n收集多少个 Episode? (默认 10): ").strip()
    num_episodes = int(num_episodes_input) if num_episodes_input.isdigit() else 10
    
    trainer = SimpleManualCollector(cfg)
    
    if choice == '1':
        trainer.collect_and_train(num_episodes=num_episodes, train_while_collect=True)
    elif choice == '2':
        trainer.collect_and_train(num_episodes=num_episodes, train_while_collect=False)
    else:
        print("无效选项")


if __name__ == '__main__':
    main()
