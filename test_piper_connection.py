import time
import sys

try:
    from piper_sdk import *
except ImportError:
    print("错误：piper_sdk 未安装！")
    print("请确保 piper_sdk 已正确安装在 Python 路径中")
    print("（你的 test_piper_camera.py 可以找到 SDK，说明 SDK 是可用的）")
    sys.exit(1)


def test_piper_connection():
    print("="*60)
    print(" Piper 机械臂连接测试")
    print("="*60)
    
    robot = None
    try:
        # 检查 CAN 接口
        print("\n[1/5] 检查 CAN 接口...")
        print("   请确保已执行：sudo ip link set can0 up type can bitrate 1000000")
        print("   如果还没执行，请先在另一个终端执行该命令")
        input("   按回车继续...")
        
        # 初始化机械臂（使用和 test_piper_camera.py 一样的方式）
        print("\n[2/5] 初始化机械臂...")
        print("   使用 C_PiperInterface...")
        
        robot = C_PiperInterface(
            can_name="can0",
            judge_flag=True,
            can_auto_init=False,
            dh_is_offset=1,
            start_sdk_joint_limit=False,
            start_sdk_gripper_limit=False,
            logger_level=LogLevel.WARNING,
            log_to_file=False
        )
        
        # 创建 CAN 总线
        print("   创建 CAN 总线...")
        robot.CreateCanBus(can_name="can0")
        
        # 连接端口
        print("   连接端口...")
        robot.ConnectPort()
        
        # 配置主从
        print("   配置主从...")
        robot.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.5)
        
        print("✓ 机械臂初始化成功")
        
        # 获取当前关节位置
        print("\n[3/5] 获取关节位置...")
        joint_pose = robot.GetArmJointPoseMsgs()
        current_joints = [
            joint_pose.joint_pos.J1,
            joint_pose.joint_pos.J2,
            joint_pose.joint_pos.J3,
            joint_pose.joint_pos.J4,
            joint_pose.joint_pos.J5,
            joint_pose.joint_pos.J6
        ]
        print(f"✓ 当前关节位置: [{current_joints[0]:.3f}, {current_joints[1]:.3f}, {current_joints[2]:.3f},")
        print(f"                          {current_joints[3]:.3f}, {current_joints[4]:.3f}, {current_joints[5]:.3f}]")
        
        # 获取末端位置
        print("\n[4/5] 获取末端位姿...")
        end_pose = robot.GetArmEndPoseMsgs()
        print(f"✓ 末端位置 (mm): X={end_pose.end_pose.X_axis:.1f}, Y={end_pose.end_pose.Y_axis:.1f}, Z={end_pose.end_pose.Z_axis:.1f}")
        print(f"✓ 末端姿态 (°): RX={end_pose.end_pose.RX_axis:.1f}, RY={end_pose.end_pose.RY_axis:.1f}, RZ={end_pose.end_pose.RZ_axis:.1f}")
        
        # 测试小幅度移动（安全起见）
        print("\n[5/5] 测试小幅度移动...")
        print("   ⚠️  警告：机械臂即将移动，请确保周围安全！")
        print("   机械臂将进行小幅度移动，然后回到原位")
        print("   移动速度: 50 (慢速)")
        response = input("   确认继续？(yes/no): ")
        
        if response.lower() != 'yes':
            print("   跳过移动测试")
        else:
            # 稍微移动第一个关节
            new_joints = current_joints.copy()
            new_joints[0] += 0.1  # 小幅度移动第一个关节
            print(f"   移动到: [{new_joints[0]:.3f}, {new_joints[1]:.3f}, ...]")
            
            robot.MoveJ(
                new_joints[0],
                new_joints[1],
                new_joints[2],
                new_joints[3],
                new_joints[4],
                new_joints[5],
                50,  # 速度
                1,   # 模式
                0    # 阻塞
            )
            time.sleep(2)
            
            # 回到原位
            print(f"   回到原位...")
            robot.MoveJ(
                current_joints[0],
                current_joints[1],
                current_joints[2],
                current_joints[3],
                current_joints[4],
                current_joints[5],
                50,
                1,
                0
            )
            time.sleep(2)
            print("✓ 移动测试成功")
        
        # 断开连接
        print("\n断开连接...")
        robot.DisconnectPort()
        print("✓ 断开连接成功")
        
        print("\n" + "="*60)
        print("✓ 所有测试通过！机械臂连接正常")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n常见问题解决：")
        print("1. CAN 接口未激活：执行 sudo ip link set can0 up type can bitrate 1000000")
        print("2. 示教模式未退出：短按示教按钮直到指示灯熄灭")
        print("3. 检查 piper_sdk 是否正确安装（参考 test_piper_camera.py 的路径查找方式）")
        
        return False
    finally:
        if robot is not None:
            try:
                robot.DisconnectPort()
            except:
                pass

if __name__ == '__main__':
    success = test_piper_connection()
    sys.exit(0 if success else 1)
