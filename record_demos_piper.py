"""
Piper 键盘控制示范数据收集脚本
使用 pygame 处理键盘输入
"""

import os
import pickle as pkl
import datetime
import numpy as np
import time
from tqdm import tqdm

import pygame
from pygame.locals import *

from piper.env import PiperEnv


def main():
    print("=" * 60)
    print("Piper 键盘控制示范数据收集")
    print("=" * 60)
    
    # 环境配置
    config = {
        'task_name': 'coffee_push',
        'use_sim': False,
        'visualize': True,
        'print_reward': True,
        'debug_mode': False,
        'use_apriltag': True,
        'tag_size': 0.05,
        'enable_spacemouse': False,
        'spacemouse_scale': 0.05,
    }
    
    # 创建环境
    env = PiperEnv(**config)
    
    # 初始化 pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Piper 键盘控制 - 点击此窗口后按方向键/WASD/空格")
    font = pygame.font.Font(None, 24)
    
    # 聚焦窗口以接收键盘事件
    os.environ['SDL_IGNORE_KEYBOARD_FOCUS'] = '0'
    
    # 目标示范数量
    successes_needed = 10
    success_count = 0
    
    # 夹爪状态
    gripper_open = True
    
    print(f"\n✓ 环境已创建")
    print(f"✓ 目标：收集 {successes_needed} 个成功示范")
    print(f"\n📝 操作说明：")
    print(f"  - 在 pygame 窗口中按方向键/WASD/E")
    print(f"  - 完成 coffee push 任务将自动保存示范")
    print(f"  - 关闭窗口或按 ESC 提前结束")
    print(f"\n⌨️ 键盘控制映射：")
    print(f"  ←/→ 方向键 → 左右移动 (关节0)")
    print(f"  ↑/↓ 方向键 → 前后伸出 (关节1)")
    print(f"  W/S 键     → 上下升降 (关节3)")
    print(f"  E 键       → 夹爪开/关切换")
    print(f"  空格       → 夹爪开/关切换 (备用)")
    print(f"\n{'='*60}\n")
    
    transitions = []
    trajectory = []
    episode_num = 0
    
    running = True
    clock = pygame.time.Clock()
    
    try:
        pbar = tqdm(total=successes_needed, desc="收集示范")
        
        while success_count < successes_needed and running:
            # 重置环境
            obs_dict = env.reset()
            obs = obs_dict["image"]
            done = False
            episode_num += 1
            episode_return = 0.0
            trajectory = []
            
            print(f"\n--- Episode {episode_num} ---")
            print("开始示范，等待成功...")
            
            step_count = 0
            
            while not done and running:
                step_count += 1
                
                # 持续检测按键状态
                keys = pygame.key.get_pressed()
                dx = dy = dz = 0
                
                # 处理 pygame 事件 - 事件驱动方式检测夹爪
                for event in pygame.event.get():
                    if event.type == QUIT:
                        running = False
                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            running = False
                        elif event.key == K_e or event.key == K_SPACE:
                            # E 或空格键切换夹爪
                            gripper_open = not gripper_open
                            print(f"夹爪: {'打开' if gripper_open else '关闭'}")
                
                # 持续检测按键（不是事件触发）
                if keys[K_LEFT]:
                    dx = -1
                if keys[K_RIGHT]:
                    dx = 1
                if keys[K_UP]:
                    dy = 1
                if keys[K_DOWN]:
                    dy = -1
                if keys[K_w]:
                    dz = 1
                if keys[K_s]:
                    dz = -1
                
                # 执行笛卡尔移动
                gripper_value = 1.0 if gripper_open else -1.0
                env.robot.move_end_effector_delta(dx=dx, dy=dy, dz=dz, gripper=gripper_value)
                
                # 调试：每10帧打印一次夹爪状态
                if step_count % 10 == 0:
                    print(f"[调试] 夹爪状态: {'打开' if gripper_open else '关闭'}, gripper_value={gripper_value}")
                
                # 获取图像和奖励
                obs_dict = env.step({"action": np.zeros(7, dtype=np.float32)})
                
                reward = obs_dict["reward"]
                done = obs_dict["is_last"] or obs_dict["success"]
                success = obs_dict["success"]
                
                episode_return += reward
                
                # 记录动作
                action = np.array([dx*0.5, dy*0.5, dz*0.5, 0, 0, 0, gripper_value], dtype=np.float32)
                
                # 记录转换
                transition = {
                    'observations': obs,
                    'actions': action,
                    'next_observations': obs_dict["image"],
                    'rewards': reward,
                    'masks': 1.0 - float(done),
                    'dones': done,
                    'infos': {'success': success}
                }
                trajectory.append(transition)
                obs = obs_dict["image"]
                
                # 更新 pygame 窗口显示
                screen.fill((30, 30, 30))
                
                # 显示状态
                gripper_text = font.render(f"夹爪: {'打开' if gripper_open else '关闭'}", True, (255, 255, 255))
                screen.blit(gripper_text, (20, 20))
                
                keys_text = font.render(f"方向: X={dx} Y={dy} Z={dz}", True, (255, 255, 255))
                screen.blit(keys_text, (20, 50))
                
                reward_text = font.render(f"累计奖励: {episode_return:.2f}", True, (255, 255, 0))
                screen.blit(reward_text, (20, 80))
                
                success_text = font.render(f"成功: {'是' if success else '否'}", True, (0, 255, 0) if success else (255, 100, 100))
                screen.blit(success_text, (20, 110))
                
                info_text = font.render("←→↑↓:移动 | W/S:升降 | E/空格:夹爪 | ESC:退出", True, (150, 150, 150))
                screen.blit(info_text, (20, 250))
                
                pygame.display.flip()
                
                if step_count % 10 == 0:
                    pbar.set_description(f"Ep {episode_num} | Step {step_count} | R: {episode_return:.2f}")
                
                clock.tick(30)  # 30 FPS
            
            # 保存成功轨迹
            if success:
                print(f"✓ Episode {episode_num} 成功！保存示范...")
                for t in trajectory:
                    transitions.append(t)
                success_count += 1
                pbar.update(1)
            elif running:
                print(f"✗ Episode {episode_num} 失败，丢弃示范")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️ 用户中断，已收集 {success_count} 个成功示范")
    finally:
        pygame.quit()
        
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
