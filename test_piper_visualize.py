import sys
import numpy as np
from piper.env import make

def test_visualize():
    print("="*60)
    print(" Piper 环境可视化测试")
    print("="*60)
    
    print("\n正在初始化环境（可视化模式）...")
    print("将显示摄像头画面，按 'q' 键可以关闭窗口但继续测试")
    
    env = make(
        name="piper_push",
        frame_stack=3,
        action_repeat=2,
        seed=0,
        use_sim=True,
        visualize=True
    )
    
    print("\n✓ 环境初始化成功")
    print("\n开始测试 10 步...")
    
    try:
        # 重置环境
        time_step = env.reset()
        print(f"  重置成功，观察形状: {time_step.observation.shape}")
        
        # 测试 10 步
        for step in range(10):
            # 随机动作
            action = np.random.uniform(-1, 1, size=6).astype(np.float32)
            
            # 执行一步
            time_step = env.step(action)
            
            print(f"  第 {step+1}/10 步: reward={time_step.reward:.2f}, success={time_step.success}")
    
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭环境
        print("\n关闭环境...")
        env.close()
        print("✓ 环境已关闭")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print("\n提示：")
    print("- 在训练时，设置 piper/cfgs/config.yaml 中的 visualize: true")
    print("- 按 'q' 键可以关闭可视化窗口，训练会继续进行")
    print("- 可视化会轻微影响训练速度，长时间训练建议关闭")

if __name__ == "__main__":
    test_visualize()
