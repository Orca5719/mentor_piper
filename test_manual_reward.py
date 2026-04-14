"""
测试手动奖励和惩罚功能
运行此脚本测试键盘监听是否正常工作
"""

import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper.env import PiperEnv

def test_manual_reward():
    print("="*70)
    print("测试手动奖励和惩罚功能")
    print("="*70)
    
    # 创建环境（使用模拟模式）
    env = PiperEnv(
        task_name="piper_push",
        seed=42,
        action_repeat=1,
        size=(256, 256),
        use_sim=True,  # 使用模拟模式
        visualize=False,  # 不开启可视化
        print_reward=True,
        debug_mode=True
    )
    
    print("\n环境已创建，键盘监听应该已经启动")
    print("现在测试手动奖励和惩罚：")
    print("  - 按空格键：+10 奖励")
    print("  - 按 C 键：-10 惩罚")
    print("  - 按 Q 键：退出测试")
    print("\n请在终端中按键盘测试...")
    
    # 重置环境
    time_step = env.reset()
    
    print(f"\n初始状态:")
    print(f"  Reward: {time_step['reward']}")
    print(f"  Is First: {time_step['is_first']}")
    
    # 运行几步，看看手动奖励是否有效
    print("\n开始运行 10 步...")
    
    for step in range(10):
        # 创建一个随机 action
        action = {
            "action": (env.act_space["action"].sample()).astype(float)
        }
        
        # 执行 action
        time_step = env.step(action)
        
        print(f"Step {step+1}: Reward={time_step['reward']:.2f}, "
              f"Success={time_step['success']}")
        
        time.sleep(0.5)
    
    print("\n测试完成！")
    print("如果你在上面的步骤中按了空格或 C 键，应该能看到奖励/惩罚输出")
    
    # 关闭环境
    env.close()
    print("\n环境已关闭，键盘监听应该已停止")

if __name__ == "__main__":
    try:
        test_manual_reward()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n错误：{e}")
        import traceback
        traceback.print_exc()
