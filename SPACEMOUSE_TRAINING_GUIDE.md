# Piper SpaceMouse 人工示范训练指南

本指南介绍如何使用 SpaceMouse 人工示范来加速 Piper 机械臂的强化学习训练。

## 📋 概述

### 解决的问题
- **关节打结**：自由探索时机械臂进入极端姿态
- **探索效率低**：随机探索需要大量 trial and error
- **需要精细控制**：coffee push 任务需要轻柔的力道控制

### 解决方案
1. **人工示范**：使用 SpaceMouse 收集高质量示范数据
2. **行为克隆**：从示范数据预训练策略
3. **RL 微调**：从预训练策略开始强化学习

## 🔧 安装依赖

### 1. SpaceMouse 驱动
```bash
pip install easyhid
```

### 2. 可选：3Dconnexion SpaceMouse 硬件
支持以下设备：
- SpaceNavigator
- SpaceMouse (USB, Compact, Wireless, Pro)
- SpaceExplorer
- SpacePilot

如果没有硬件，可以跳过示范收集，直接使用随机初始化。

## 📌 完整训练流程

### 步骤 1: 收集示范数据

```bash
cd /home/isee604/mentor_final?/mentor_piper
python record_demos_piper.py
```

**操作说明：**
- 使用 SpaceMouse 控制机械臂完成 coffee push 任务
- 每次成功完成后自动保存示范
- 目标：收集 5-10 个成功示范
- 按 Ctrl+C 提前结束并保存已收集的数据

**SpaceMouse 控制映射：**
```
X/Y (平移)   → 基座和肩部关节
Z (升降)      → 臂的升降
Pitch/Yaw     → 腕部关节
按钮 0        → 夹爪开/关
```

**输出：**
```
demo_data/piper_coffee_push_N_demos_YYYY-MM-DD_HH-MM-SS.pkl
```

### 步骤 2: BC 预训练

```bash
python train_bc_piper.py
```

**参数说明：**
- `--epochs`: 训练轮数（默认 100）
- `--batch_size`: 批大小（默认 16）
- `--lr`: 学习率（默认 1e-4）

**输出：**
```
bc_checkpoints/bc_policy_best.pt           # 最佳模型
bc_checkpoints/bc_policy_epoch10.pt        # 每 10 epoch 的 checkpoint
...
```

### 步骤 3: RL 微调（使用 BC 初始化）

**方式 A：仅加载 BC 权重**
```bash
# 在 config.yaml 中添加
bc_checkpoint_path: ./bc_checkpoints/bc_policy_best.pt

# 然后运行
python train_bc_rl_piper.py
```

**方式 B：加载示范数据到 replay buffer**
```bash
# 在 config.yaml 中添加
load_demo_to_buffer: true
demo_dir: ./demo_data

# 然后运行
python train_bc_rl_piper.py
```

**方式 C：同时使用 BC 权重和示范数据**
```bash
# 在 config.yaml 中添加
bc_checkpoint_path: ./bc_checkpoints/bc_policy_best.pt
load_demo_to_buffer: true
demo_dir: ./demo_data

# 然后运行
python train_bc_rl_piper.py
```

### 步骤 4: 训练时启用 SpaceMouse 干预（可选）

如果训练过程中发现策略有问题，可以手动干预：

```python
# 在 train_bc_rl_piper.py 的 setup() 中
self.train_env = piper_env.make(
    ...,
    enable_spacemouse=True,  # 启用 SpaceMouse
    spacemouse_scale=0.05,    # 控制速度
)
```

训练时：
- **SpaceMouse 闲置** → 策略自动控制
- **操作 SpaceMouse** → 人工接管控制
- 适合在危险动作时手动干预

## 📊 效果对比

### 纯随机探索
- ❌ 容易出现关节打结
- ❌ 需要大量探索
- ❌ 初期成功率很低

### SpaceMouse 人工示范
- ✅ 完全避免危险姿态
- ✅ 收集 5-10 个示范即可
- ✅ 示范质量高，策略起点好

### BC 预训练 + RL 微调
- ✅ 继承示范的安全特性
- ✅ RL 进一步优化性能
- ✅ 收敛速度快，样本效率高

## ⚙️ 配置文件说明

### config.yaml 添加项

```yaml
# SpaceMouse 人工示范配置
enable_spacemouse: false          # 是否启用 SpaceMouse（训练时通常设为 false）
spacemouse_scale: 0.05           # SpaceMouse 控制速度

# BC 预训练配置
bc_checkpoint_path: null          # BC checkpoint 路径（e.g., ./bc_checkpoints/bc_policy_best.pt）
load_demo_to_buffer: false        # 是否加载示范数据到 replay buffer
demo_dir: ./demo_data            # 示范数据目录
```

## 🐛 故障排除

### SpaceMouse 无法连接
```bash
# 检查设备是否识别
python -c "from piper.spacemouse_controller import PiperSpaceMouseController; PiperSpaceMouseController()"

# 检查权限
ls -la /dev/hidraw*
sudo chmod 666 /dev/hidraw*
```

### BC 训练损失不下降
- 检查示范数据质量（确保动作非零）
- 尝试调整学习率（1e-5 到 1e-3）
- 增加训练轮数

### RL 微调后性能下降
- BC 预训练和 RL 的动作空间可能不匹配
- 尝试降低 RL 学习率
- 增加 BC 权重加载的层数

## 📈 性能优化建议

1. **示范收集**
   - 确保示范多样性（不同的初始位置、路径）
   - 每个示范控制在 50-200 步
   - 避免过度依赖人工干预

2. **BC 训练**
   - 使用数据增强（随机裁剪、颜色抖动）
   - 早停策略（验证集损失不再下降）
   - 保存多个 checkpoint 供选择

3. **RL 微调**
   - 从较低的学习率开始（1e-5）
   - 逐步增加探索噪声
   - 定期评估并选择最佳 checkpoint

## 🔗 相关文件

- `piper/spacemouse_controller.py` - SpaceMouse 控制器
- `piper/env.py` - 集成 SpaceMouse 的环境
- `record_demos_piper.py` - 示范数据收集脚本
- `train_bc_piper.py` - BC 预训练脚本
- `train_bc_rl_piper.py` - BC + RL 训练脚本
- `train_piper.py` - 原始纯 RL 训练脚本

## 📝 示例命令

```bash
# 1. 测试 SpaceMouse
python -m piper.spacemouse_controller

# 2. 收集示范
python record_demos_piper.py

# 3. BC 预训练
python train_bc_piper.py

# 4. RL 微调（使用 BC）
python train_bc_rl_piper.py bc_checkpoint_path=./bc_checkpoints/bc_policy_best.pt

# 5. RL 微调（使用示范数据）
python train_bc_rl_piper.py load_demo_to_buffer=true demo_dir=./demo_data

# 6. 训练时允许 SpaceMouse 干预
python train_bc_rl_piper.py enable_spacemouse=true
```

## 🎯 预期效果

使用 SpaceMouse 人工示范 + BC 预训练：

- **关节打结问题**：几乎完全避免
- **初始成功率**：从 0% 提升到 40-60%
- **收敛速度**：快 2-3 倍
- **样本效率**：减少 60-80% 的训练步数

对于 coffee push 这种精细操作任务，人工示范的收益尤为明显！
