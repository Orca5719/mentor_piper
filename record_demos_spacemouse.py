"""
Piper SpaceMouse 示范数据收集脚本
使用 3Dconnexion SpaceMouse 控制机械臂末端位置
"""

import os
import pickle as pkl
import datetime
import numpy as np
import time
from tqdm import tqdm

from piper.env import PiperEnv
from piper.spacemouse_controller import PiperSpaceMouseController


def main():
    print("=" * 60)
    print("Piper SpaceMouse 示范数据收集")
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
    
    # 检查是否在模拟模式
    print(f"\n机械臂状态: {'模拟模式' if env.robot.use_sim else '真实机械臂'}")
    if not env.robot.use_sim and env.robot.piper is None:
        print("⚠️ 警告：机械臂对象为 None，请检查 CAN 连接")
    elif not env.robot.use_sim and env.robot.piper is not None:
        print("✓ 真实机械臂已连接")
        
        # 测试直接控制机械臂关节
        print("\n测试机械臂控制...")
        try:
            import time
            initial_pos = env.robot.current_joint_pos.copy()
            print(f"当前关节位置: {initial_pos}")
            
            # 尝试移动第一个关节
            test_action = np.zeros(7)
            test_action[0] = 0.1  # 关节 0 移动 0.1 弧度
            
            print(f"发送测试动作: {test_action}")
            env.robot.step(test_action)
            time.sleep(0.5)
            
            new_pos = env.robot.current_joint_pos.copy()
            print(f"移动后关节位置: {new_pos}")
            
            if np.allclose(initial_pos, new_pos, atol=0.01):
                print("⚠️ 警告：关节位置未变化，机械臂可能未被使能或控制失败")
            else:
                print("✓ 机械臂控制正常")
        except Exception as e:
            print(f"⚠️ 测试控制失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✓ 模拟模式已启用")

    # 创建 SpaceMouse 控制器
    spacemouse = PiperSpaceMouseController(
        enable_spacemouse=True,
        scale_factor=0.5,   # 控制速度（降低以减少卡顿）
        deadzone=0.1        # 增加死区，过滤残余值
    )

    if not spacemouse.enable_spacemouse:
        print("\n✗ SpaceMouse 不可用，请检查：")
        print("  1. SpaceMouse 是否已连接")
        print("  2. easyhid 是否已安装 (pip install easyhid)")
        print("\n退出程序...")
        return

    # 目标示范数量
    successes_needed = 10
    success_count = 0

    print(f"\n✓ 环境已创建")
    print(f"✓ SpaceMouse 已连接")
    print(f"✓ 目标：收集 {successes_needed} 个成功示范")
    print(f"\n📝 操作说明：")
    print(f"  - 使用 SpaceMouse 控制机械臂末端位置")
    print(f"  - 完成 coffee push 任务将自动保存示范")
    print(f"  - 按 Ctrl+C 提前结束")
    print(f"\n🎮 SpaceMouse 控制映射：")
    print(f"  X 轴 (左右推帽) → 关节 0 (基座旋转)")
    print(f"  Y 轴 (前后推帽) → 关节 1,2 (臂的前伸)")
    print(f"  Z 轴 (上下推帽) → 关节 2,3 (臂的升降)")
    print(f"  Pitch (前后倾斜) → 关节 3,4 (腕部俯仰)")
    print(f"  Yaw (左右倾斜) → 关节 5 (腕部旋转)")
    print(f"  按钮 0 (左键) → 夹爪开/关")
    print(f"\n{'='*60}\n")

    transitions = []
    trajectory = []
    episode_num = 0

    try:
        pbar = tqdm(total=successes_needed, desc="收集示范")

        while success_count < successes_needed:
            # 重置环境
            obs_dict = env.reset()
            obs = obs_dict["image"]
            done = False
            episode_num += 1
            episode_return = 0.0
            trajectory = []

            print(f"\n--- Episode {episode_num} ---")
            print("开始示范，使用 SpaceMouse 控制机械臂...")

            step_count = 0
            while not done:
                step_count += 1

                # 读取 SpaceMouse 输入
                spacemouse_action, is_active = spacemouse.get_action()

                # 执行动作
                obs_dict = env.step({"action": spacemouse_action})

                reward = obs_dict["reward"]
                done = obs_dict["is_last"] or obs_dict["success"]
                success = obs_dict["success"]

                episode_return += reward

                # 记录转换
                transition = {
                    'observations': obs,
                    'actions': spacemouse_action,
                    'next_observations': obs_dict["image"],
                    'rewards': reward,
                    'masks': 1.0 - float(done),
                    'dones': done,
                    'infos': {'success': success}
                }
                trajectory.append(transition)
                obs = obs_dict["image"]

                # 更新进度
                if step_count % 10 == 0:
                    pbar.set_description(f"Ep {episode_num} | Step {step_count} | R: {episode_return:.2f}")

                time.sleep(0.03)

            # 保存成功轨迹
            if success:
                print(f"✓ Episode {episode_num} 成功！保存示范...")
                for t in trajectory:
                    transitions.append(t)
                success_count += 1
                pbar.update(1)
            else:
                print(f"✗ Episode {episode_num} 失败，丢弃示范")

    except KeyboardInterrupt:
        print(f"\n\n⚠️ 用户中断，已收集 {success_count} 个成功示范")
    finally:
        spacemouse.close()

        if len(transitions) > 0:
            if not os.path.exists("./demo_data"):
                os.makedirs("./demo_data")

            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"./demo_data/piper_coffee_push_{success_count}_demos_spacemouse_{uuid}.pkl"

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
