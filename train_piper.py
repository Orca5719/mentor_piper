import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
from pathlib import Path

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import hydra
import numpy as np
import utils
import torch
from dm_env import specs

import piper.env as piper_env

from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import math
import re

from utils import models_tuple
from copy import deepcopy
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    agent = hydra.utils.instantiate(cfg)
    # 确保Agent初始化后移到指定设备
    agent = agent.to(cfg.device)
    return agent


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        print("#"*50)
        print(f'\n工作目录: {self.work_dir}')
        print("#"*50)
        self.last_save_step = -9999
        
        # 初始化WandB（提前初始化，确保加载快照后能同步步数）
        self.wandb_init = False
        if self.cfg.use_wandb:
            exp_name = '_'.join([cfg.task_name, str(cfg.seed)])
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            name_1 = cfg.task_name
            name_2 = group_name
            try:
                name_2 += '_' + cfg.title
            except:
                pass
            name_3 = exp_name
            wandb.init(project=name_1,
                       group=name_2,
                       name=name_3,
                       config=cfg)
            self.wandb_init = True
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._discount_alpha = cfg.discount_alpha
        self._discount_alpha_temp = cfg.discount_alpha_temp
        self._discount_beta = cfg.discount_beta
        self._discount_beta_temp = cfg.discount_beta_temp
        self._nstep = cfg.nstep
        self._nstep_alpha = cfg.nstep_alpha
        self._nstep_alpha_temp = cfg.nstep_alpha_temp
        
        # 先初始化环境，再初始化Agent
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent)
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        
        # 读取环境配置（增加容错）
        use_sim = getattr(self.cfg, 'use_sim', True)
        visualize = getattr(self.cfg, 'visualize', False)
        obj_pos = getattr(self.cfg, 'obj_pos', None)
        goal_pos = getattr(self.cfg, 'goal_pos', None)
        use_apriltag = getattr(self.cfg, 'use_apriltag', False)
        tag_size = getattr(self.cfg, 'tag_size', 0.05)
        camera_calibration_file = getattr(self.cfg, 'camera_calibration_file', 'camera_calibration.npz')
        hand_eye_calibration_file = getattr(self.cfg, 'hand_eye_calibration_file', 'simple_hand_eye.json')
        
        # 初始化训练环境
        self.train_env = piper_env.make(
            self.cfg.task_name, 
            self.cfg.seed,
            self.cfg.action_repeat,
            (256, 256),
            use_sim=use_sim,
            visualize=visualize,
            obj_pos=obj_pos,
            goal_pos=goal_pos,
            use_apriltag=use_apriltag,
            tag_size=tag_size,
            camera_calibration_file=camera_calibration_file,
            hand_eye_calibration_file=hand_eye_calibration_file,
            frame_stack=self.cfg.frame_stack
        )
        
        # 初始化评估环境
        self.eval_env = piper_env.make(
            self.cfg.task_name, 
            self.cfg.seed,
            self.cfg.action_repeat,
            (256, 256),
            use_sim=True,
            visualize=False,
            obj_pos=obj_pos,
            goal_pos=goal_pos,
            use_apriltag=False,
            tag_size=tag_size,
            camera_calibration_file=camera_calibration_file,
            hand_eye_calibration_file=hand_eye_calibration_file,
            frame_stack=self.cfg.frame_stack
        )
        
        # 初始化回放缓冲区
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'))

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
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def discount(self):
        return self._discount - self._discount_alpha * math.exp(
            -self.global_step /
            self._discount_alpha_temp) - self._discount_beta * math.exp(
                -self.global_step / self._discount_beta_temp)

    @property
    def nstep(self):
        return math.floor(self._nstep + self._nstep_alpha *
                          math.exp(-self.global_step / self._nstep_alpha_temp))

    def update_buffer(self):
        """更新回放缓冲区的nstep参数（动态调整）"""
        current_nstep = self.nstep
        self.buffer.update_nstep(current_nstep)
        # 可选：打印nstep更新日志（便于调试）
        if self.global_step % 1000 == 0:
            self.logger.log('nstep', current_nstep, self.global_frame)
        return
    
    def eval(self):
        """评估函数（优化进度条和容错）"""
        step, episode, total_reward, total_sr = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        pbar = tqdm(total=self.cfg.num_eval_episodes, desc='评估中', leave=False)
        while eval_until_episode(episode):
            episode_sr = False
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                # 容错处理：避免success属性不存在
                episode_sr = episode_sr or getattr(time_step, 'success', False)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            total_sr += episode_sr
            episode += 1
            pbar.update(1)
            self.video_recorder.save(f'{self.global_frame}.mp4')
        pbar.close()
        
        # 记录评估日志
        avg_sr = total_sr / episode if episode > 0 else 0.0
        avg_reward = total_reward / episode if episode > 0 else 0.0
        avg_length = step * self.cfg.action_repeat / episode if episode > 0 else 0.0
        
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_success_rate', avg_sr)
            log('episode_reward', avg_reward)
            log('episode_length', avg_length)
            log('episode', self.global_episode)
            log('step', self.global_step)
        
        print(f"\n[评估完成] 成功率: {avg_sr:.2%}, 平均奖励: {avg_reward:.2f}, 平均长度: {avg_length:.1f}")

    def train(self):
        """训练主循环（修复核心逻辑）"""
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward, episode_sr = 0, 0, False
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        metrics = None
        print("\n开始训练...")
        
        total_steps = self.cfg.num_train_frames // self.cfg.action_repeat
        pbar = tqdm(total=total_steps, desc='训练进度', unit='step')
        
        while train_until_step(self.global_step):
            # 回合结束处理
            if time_step.last():
                self._global_episode += 1
                
                # 记录训练日志
                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_success_rate', episode_sr)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                        log('current_nstep', self.nstep)
                        log('current_discount', self.discount)
                
                # 保存TP集合（如果有）
                if hasattr(self.agent, 'tp_set'):
                    self.agent.tp_set.add(
                        episode_reward,
                        deepcopy(self.agent.actor),
                        deepcopy(self.agent.critic),
                        deepcopy(self.agent.critic_target),
                        deepcopy(self.agent.value_predictor),
                        moe=deepcopy(self.agent.actor.moe.experts) if hasattr(self.agent.actor, 'moe') else None,
                        gate=deepcopy(self.agent.actor.moe.gate) if hasattr(self.agent.actor, 'moe') else None
                    )
                
                # 重置环境和回合状态
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                
                # 保存快照
                if self.cfg.save_snapshot and self.global_step - self.last_save_step >= self.cfg.save_interval:
                    self.last_save_step = self.global_step
                    self.save_snapshot(self.global_step)
                
                episode_sr = False
                episode_step = 0
                episode_reward = 0

            # 定期评估
            if eval_every_step(self.global_step):
                pbar.set_description('评估中...')
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
                pbar.set_description('训练进度')

            # 生成动作：种子阶段用随机动作，非种子阶段用Agent预测
            if seed_until_step(self.global_step):
                # 种子帧阶段：随机探索
                action_spec = self.train_env.action_spec()
                action = np.random.uniform(
                    low=action_spec.minimum,
                    high=action_spec.maximum,
                    size=action_spec.shape
                ).astype(action_spec.dtype)
            else:
                # 非种子阶段：Agent决策
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=False)

            # 模型更新（非种子阶段，且达到更新步长）
            if not seed_until_step(self.global_step) and self.global_step % self.cfg.update_every_steps == 0:   
                # 先更新缓冲区nstep
                self.update_buffer()
                # 更新Agent
                metrics = self.agent.update(
                    self.replay_iter, self.global_step
                )
                # 记录TP集合日志（如果有）
                if hasattr(self.agent, 'tp_set'):
                    metrics = self.agent.tp_set.log(metrics)
                # 记录训练指标
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # 执行环境步骤
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            # 更新回合成功率（容错处理）
            episode_sr = episode_sr or getattr(time_step, 'success', False)
            # 保存到回放缓冲区
            self.replay_storage.add(time_step)
            
            # 更新步数
            episode_step += 1
            self._global_step += 1
            pbar.update(1)
            
            # 更新进度条后缀
            pbar.set_postfix({
                'episode': self.global_episode,
                'reward': f'{episode_reward:.1f}',
                'success': 'Yes' if episode_sr else 'No',
                'nstep': self.nstep,
                'step': self.global_step
            })
        
        pbar.close()
        # 训练结束后保存最终快照
        self.save_snapshot(self.global_step)
        print("\n训练完成！")

    def save_snapshot(self, step_id=None):
        """优化快照保存逻辑（Path拼接，跨平台兼容）"""
        if step_id is None:
            snapshot_path = self.work_dir / 'snapshot.pt'
        else:
            snapshots_dir = self.work_dir / 'snapshots'
            snapshots_dir.mkdir(exist_ok=True)
            snapshot_path = snapshots_dir / f'snapshot_{step_id}.pt'
        
        # 要保存的参数
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        
        # 保存
        with snapshot_path.open('wb') as f:
            torch.save(payload, f)
        print(f"模型已保存: {snapshot_path}")

    def load_snapshot(self, step_id=None, snapshot_path=None):
        """优化快照加载逻辑（设备迁移+WandB同步）"""
        # 确定快照路径
        if snapshot_path:
            snapshot_path = Path(snapshot_path)
        elif step_id is None:
            snapshot_path = self.work_dir / 'snapshot.pt'
        else:
            snapshot_path = self.work_dir / 'snapshots' / f'snapshot_{step_id}.pt'
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"快照文件不存在: {snapshot_path}")
        
        # 加载快照
        with snapshot_path.open('rb') as f:
            payload = torch.load(f, map_location=self.device)
        
        # 处理预训练模型（仅Actor权重）
        if 'actor_state_dict' in payload and 'agent' not in payload:
            print("检测到预训练模型，仅加载Actor网络权重...")
            if hasattr(self.agent, 'actor'):
                # 加载权重并确保在指定设备
                self.agent.actor.load_state_dict(payload['actor_state_dict'])
                self.agent.actor.to(self.device)
                print("✓ Actor网络权重加载成功（已迁移到指定设备）")
            else:
                print("⚠️  Agent无actor属性，无法加载预训练权重")
        else:
            # 加载完整快照
            for k, v in payload.items():
                if k in self.__dict__:
                    self.__dict__[k] = v
                    # 确保Agent在指定设备
                    if k == 'agent':
                        self.agent.to(self.device)
            print(f"✓ 完整快照加载成功: {snapshot_path}")
            
            # 同步WandB步数（避免日志错位）
            if self.cfg.use_wandb and self.wandb_init:
                wandb.log({'global_step': self._global_step}, step=self._global_step)
                print(f"✓ WandB步数同步完成: {self._global_step}")


@hydra.main(config_path='piper/cfgs', config_name='config')
def main(cfgs):
    # 初始化工作空间
    workspace = Workspace(cfgs)
    
    # 确定快照路径
    snapshot_path = None
    if hasattr(cfgs, 'snapshot_path') and cfgs.snapshot_path:
        snapshot_path = cfgs.snapshot_path
    elif cfgs.load_from_id:
        snapshot_path = workspace.work_dir / 'snapshots' / f'snapshot_{cfgs.load_id}.pt'
    else:
        snapshot_path = workspace.work_dir / 'snapshot.pt'
    
    # 加载快照（如果存在）
    snapshot_path = Path(snapshot_path)
    if snapshot_path.exists():
        print(f'从快照恢复训练: {snapshot_path}')
        workspace.load_snapshot(snapshot_path=str(snapshot_path))
    else:
        print(f'未找到快照文件 {snapshot_path}，从头开始训练')
    
    # 执行评估或训练
    if hasattr(cfgs, 'eval_only') and cfgs.eval_only:
        print('进入仅评估模式...')
        workspace.eval()
    else:
        workspace.train()
    
    # 关闭WandB
    if cfgs.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
