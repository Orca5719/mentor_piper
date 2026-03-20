"""
Piper SpaceMouse 示范数据收集脚本
基于 hil-serl 的 record_demos.py 修改，适配 Piper 机械臂
"""

import os
import pickle as pkl
import datetime
import numpy as np
import time
from tqdm import tqdm

from piper.env import PiperEnv


def main():
    print("=" * 60)
    print("Piper SpaceMouse 示范数据收集")
    print("=" * 60)
    
    # 环境配置
    config = {
        'task_name': 'coffee_push',
        'use_sim': False,  # 使用真实机械臂
        'visualize': True,
        'print_reward': True,
        'debug_mode': False,
        'use_apriltag': True,
        'tag_size': 0.05,
        'enable_spacemouse': True,  # 启用 SpaceMouse
        'spacemouse_scale': 0.05,    # 控制速度
    }
    
    # 创建环境
    env = PiperEnv(**config)
    
    # 目标示范数量
    successes_needed = 10
    success_count = 0
    
    print(f"\n✓ 环境已创建")
    print(f"✓ 目标：收集 {successes_needed} 个成功示范")
    print(f"\n📝 操作说明：")
    print(f"  - 使用 SpaceMouse 控制机械臂")
    print(f"  - 完成 coffee push 任务将自动保存示范")
    print(f"  - 按 Ctrl+C 提前结束")
    print(f"\n🖱️ SpaceMouse 控制映射：")
    print(f"  - X/Y (平移) → 基座和肩部")
    print(f"  - Z (升降) → 臂的升降")
    print(f"  - Pitch/Yaw → 腕部")
    print(f"  - 按钮 0 → 夹爪开/关")
    print(f"\n{'='*60}\n")
    
    transitions = []
    trajectory = []
    returns = 0.0
    episode_num = 0
    
    try:
        pbar = tqdm(total=successes_needed, desc="收集示范")
        
        while success_count < successes_needed:
            # 重置环境
            obs, info = env.reset(), {}
            obs_dict = env.observation
            done = False
            episode_num += 1
            returns = 0.0
            trajectory = []
            
            print(f"\n--- Episode {episode_num} ---")
            print("开始示范，等待成功...")
            
            step_count = 0
            while not done:
                step_count += 1
                
                # 使用零动作（SpaceMouse 会接管）
                action = np.zeros(7, dtype=np.float32)
                action_dict = {"action": action}
                
                # 执行 step（SpaceMouse 会在内部接管）
                obs_dict = env.step(action_dict)
                
                reward = obs_dict["reward"]
                done = obs_dict["is_last"] or obs_dict["success"]
                success = obs_dict["success"]
                
                returns += reward
                
                # 检查是否使用了 SpaceMouse 干预
                spacemouse_action = None
                if env._spacemouse_controller is not None:
                    spacemouse_action, is_active = env._spacemouse_controller.get_action()
                
                # 记录转换
                transition = {
                    'observations': obs,
                    'actions': spacemouse_action if spacemouse_action is not None else action,
                    'next_observations': obs_dict["image"],
                    'rewards': reward,
                    'masks': 1.0 - float(done),
                    'dones': done,
                    'infos': {
                        'success': success,
                        'spacemouse_active': spacemouse_action is not None and np.any(np.abs(spacemouse_action) > 0.001)
                    }
                }
                trajectory.append(transition)
                obs = obs_dict["image"]
                
                if step_count % 10 == 0:
                    pbar.set_description(f"Ep {episode_num} | Step {step_count} | R: {returns:.2f}")
            
            # 如果成功，保存轨迹
            if success:
                print(f"✓ Episode {episode_num} 成功！保存示范...")
                for transition in trajectory:
                    transitions.append(transition)
                success_count += 1
                pbar.update(1)
            else:
                print(f"✗ Episode {episode_num} 失败，丢弃示范")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️ 用户中断，已收集 {success_count} 个成功示范")
    finally:
        # 保存示范数据
        if len(transitions) > 0:
            if not os.path.exists("./demo_data"):
                os.makedirs("./demo_data")
            
            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"./demo_data/piper_coffee_push_{success_count}_demos_{uuid}.pkl"
            
            with open(file_name, "wb") as f:
                pkl.dump(transitions, f)
            
            print(f"\n{'='*60}")
            print(f"✓ 示范数据已保存到: {file_name}")
            print(f"✓ 总共保存 {success_count} 个成功示范")
            print(f"✓ 总步数: {len(transitions)}")
            print(f"{'='*60}")
        else:
            print("\n没有收集到成功的示范数据")
        
        env.close()


if __name__ == "__main__":
    main()
