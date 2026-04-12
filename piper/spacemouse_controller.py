"""
Piper SpaceMouse Controller - 适配 Piper 机械臂的 SpaceMouse 控制
将 3Dconnexion SpaceMouse 的 6DOF 输入映射到 Piper 的关节空间

直接控制模式：直接使用 SpaceMouse 的值（超过死区才输出）
translation 和 rotation 分开处理，避免数据不同步问题
"""

import numpy as np
from typing import Optional, Tuple

try:
    from piper.pyspacemouse import SpaceNavigator, open, close, read
    SPACEMOUSE_AVAILABLE = True
except ImportError:
    print("警告：pyspacemouse 未安装，SpaceMouse 功能不可用")
    SPACEMOUSE_AVAILABLE = False
    
    # 定义空类型用于类型注解
    class SpaceNavigator:
        """pyspacemouse 未安装时的替代类型"""
        x = y = z = pitch = yaw = roll = 0
        buttons = []


class PiperSpaceMouseController:
    """
    将 SpaceMouse 的笛卡尔控制映射到 Piper 的关节控制
    
    SpaceMouse 输出：6 DOF (x, y, z, roll, pitch, yaw)
    Piper 需求：7 维动作 (6 关节 + 1 夹爪)
    
    直接控制模式：
    - 直接使用 SpaceMouse 的 x, y, z, pitch, yaw 值
    - translation (x,y,z) 和 rotation (pitch,yaw) 分开处理避免数据不同步
    - 只有当输入值超过死区时才有动作输出
    """
    
    def __init__(self,
                 enable_spacemouse=True,
                 scale_factor=0.3,
                 deadzone=0.08,
                 device_name=None):
        """
        初始化 SpaceMouse 控制器

        Args:
            enable_spacemouse: 是否启用 SpaceMouse
            scale_factor: 动作缩放因子，控制移动速度
            deadzone: 死区，小于此值的输入会被忽略（归一化后）
            device_name: SpaceMouse 设备名称，None 表示自动检测
        """
        self.enable_spacemouse = enable_spacemouse and SPACEMOUSE_AVAILABLE
        self.scale_factor = scale_factor
        self.deadzone = deadzone
        
        # 夹爪状态（用于保持夹爪开/闭状态）
        self.gripper_state = 1.0  # 1.0 = 打开, -1.0 = 闭合
        self._last_button_states = [0, 0]  # 上一帧按钮状态
        self._button_pressed_count = 0  # 按钮按下计数器（防抖）
        self._gripper_toggle_enabled = True  # 夹爪切换使能（防止快速重复切换）
        
        # 上一帧的 translation 和 rotation 值（分开存储避免不同步）
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._prev_z = 0.0
        self._prev_pitch = 0.0
        self._prev_yaw = 0.0
        
        # 数据有效性标记
        self._translation_valid = False
        self._rotation_valid = False
        self._last_translation_t = -1.0
        self._last_rotation_t = -1.0
        
        if self.enable_spacemouse:
            try:
                self.device = open(device=device_name)
                print(f"✓ SpaceMouse 已连接: {self.device}")
                # 打印按钮数量
                state = self.read_state()
                if state and hasattr(state, 'buttons'):
                    print(f"  按钮数量: {len(state.buttons)}")
                # 等待设备稳定
                import time
                time.sleep(0.5)
            except Exception as e:
                print(f"✗ SpaceMouse 初始化失败: {e}")
                self.enable_spacemouse = False
                self.device = None
        else:
            self.device = None

    def read_state(self) -> Optional[SpaceNavigator]:
        """
        读取 SpaceMouse 当前状态
        
        Returns:
            SpaceNavigator 状态元组，或 None（如果未启用/未连接）
        """
        if not self.enable_spacemouse or self.device is None:
            return None
        
        state = read()
        return state
    
    def _apply_deadzone(self, value: float) -> float:
        """应用死区"""
        if abs(value) < self.deadzone:
            return 0.0
        return value
    
    def _detect_button_press(self, spacemouse_state: SpaceNavigator):
        """检测按钮按下（带防抖）"""
        if spacemouse_state is None or not spacemouse_state.buttons:
            return
        
        # 获取当前按钮状态
        current_buttons = list(spacemouse_state.buttons) if spacemouse_state.buttons else [0, 0]
        
        # 检查按钮0是否按下（边沿检测）
        if current_buttons[0] and not self._last_button_states[0]:
            self._button_pressed_count += 1
        else:
            self._button_pressed_count = 0
        
        # 只有当按钮按下计数器达到阈值时才切换
        # 这样可以防止误触和抖动
        if self._button_pressed_count >= 3 and self._gripper_toggle_enabled:
            self.gripper_state = -self.gripper_state
            self._gripper_toggle_enabled = False
            print(f"夹爪切换: {'闭合' if self.gripper_state < 0 else '打开'}")
        
        # 当按钮释放时，重新允许切换
        if not current_buttons[0]:
            self._gripper_toggle_enabled = True
            self._button_pressed_count = 0
        
        self._last_button_states = current_buttons
    
    def map_to_piper_action(self, 
                           spacemouse_state: Optional[SpaceNavigator]) -> np.ndarray:
        """
        将 SpaceMouse 状态映射到 Piper 的动作空间
        
        直接控制模式：直接使用 SpaceMouse 的值，超过死区才有输出
        
        Args:
            spacemouse_state: SpaceNavigator 状态
            
        Returns:
            7 维动作数组 [j0, j1, j2, j3, j4, j5, gripper]
        """
        if spacemouse_state is None:
            return np.zeros(7, dtype=np.float32)

        # 处理夹爪按钮（带防抖）
        self._detect_button_press(spacemouse_state)
        
        # 直接使用 translation 值（x, y, z）
        x = spacemouse_state.x
        y = spacemouse_state.y
        z = spacemouse_state.z
        pitch = spacemouse_state.pitch
        yaw = spacemouse_state.yaw
        
        # 检测数据跳变（translation/rotation 数据不同步时会出现）
        if hasattr(spacemouse_state, 't') and spacemouse_state.t > 0:
            # 根据时间戳判断是 translation 还是 rotation 数据包
            if abs(x - self._prev_x) > 0.5 or abs(y - self._prev_y) > 0.5 or abs(z - self._prev_z) > 0.5:
                # 检测到大跳变，可能是 rotation 数据干扰了 translation
                x, y, z = self._prev_x, self._prev_y, self._prev_z
            if abs(pitch - self._prev_pitch) > 0.5 or abs(yaw - self._prev_yaw) > 0.5:
                # 检测到大跳变，可能是 translation 数据干扰了 rotation
                pitch, yaw = self._prev_pitch, self._prev_yaw
        
        # 保存当前值作为下一帧的参考
        self._prev_x = x
        self._prev_y = y
        self._prev_z = z
        self._prev_pitch = pitch
        self._prev_yaw = yaw

        # 应用死区（过滤掉微小抖动）
        x = self._apply_deadzone(x)
        y = self._apply_deadzone(y)
        z = self._apply_deadzone(z)
        pitch = self._apply_deadzone(pitch)
        yaw = self._apply_deadzone(yaw)
        
        # 映射到 6 个关节
        # translation 控制位置移动
        joint0 = x * self.scale_factor * 2  # 基座左右旋转
        joint1 = -y * self.scale_factor * 1.5  # 肩部上下
        joint2 = z * self.scale_factor * 1.5  # 肘部伸展
        
        # rotation 控制姿态调整
        joint3 = -pitch * self.scale_factor * 2  # 腕部俯仰
        joint4 = pitch * self.scale_factor * 1  # 腕部偏航（较小）
        joint5 = yaw * self.scale_factor * 2  # 腕部旋转
        
        # 夹爪控制
        gripper = self.gripper_state
        
        action = np.array([
            joint0, joint1, joint2, joint3, joint4, joint5, gripper
        ], dtype=np.float32)
        
        return action
    
    def get_action(self) -> Tuple[np.ndarray, bool]:
        """
        获取当前动作
        
        Returns:
            (action, is_active): 
                - action: 7 维动作数组
                - is_active: 是否有用户输入
        """
        state = self.read_state()
        action = self.map_to_piper_action(state)
        
        # 判断是否有有效输入（translation 或 rotation 超过死区）
        is_active = (np.abs(action[:6]) > 0.001).any()
        
        return action, is_active
    
    def close(self):
        """关闭 SpaceMouse 连接"""
        if self.device is not None:
            close()
            self.device = None
            print("SpaceMouse 已关闭")


def test_spacemouse():
    """测试 SpaceMouse 连接和控制"""
    print("=" * 60)
    print("Piper SpaceMouse 控制器测试 (直接控制模式)")
    print("=" * 60)
    
    controller = PiperSpaceMouseController(deadzone=0.08, scale_factor=0.3)
    
    if not controller.enable_spacemouse:
        print("\n✗ SpaceMouse 不可用")
        return
    
    print("\n✓ SpaceMouse 已连接，开始测试...")
    print("提示：移动 SpaceMouse 机械臂会动，松开手机械臂会停止")
    print("按左按钮切换夹爪开/闭")
    print("按 Ctrl+C 退出测试\n")
    
    try:
        import time
        while True:
            action, is_active = controller.get_action()
            
            if is_active:
                print(f"动作: [{', '.join([f'{a:+.3f}' for a in action])}]")
            else:
                print(".", end="", flush=True)
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n测试结束")
    finally:
        controller.close()


if __name__ == "__main__":
    test_spacemouse()
