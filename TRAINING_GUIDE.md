# Piper 机械臂训练与验证流程指南

## 目录

1. [环境准备](#1-环境准备)
2. [手眼标定](#2-手眼标定)
3. [配置文件说明](#3-配置文件说明)
4. [推荐：手动收集数据与训练](#4-推荐手动收集数据与训练)
5. [自动训练](#5-自动训练)
6. [验证与评估](#6-验证与评估)
7. [手动奖励功能](#7-手动奖励功能)
8. [常见问题](#8-常见问题)
9. [文件结构](#9-文件结构)

---

## 1. 环境准备

### 1.1 硬件要求
- Piper 机械臂
- USB 摄像头
- 支持 CUDA 的 NVIDIA 显卡（推荐）
- 打印机（用于打印 AprilTag）

### 1.2 软件依赖
确保已安装以下依赖：
```bash
pip install torch torchvision hydra-core numpy opencv-python tqdm
```

### 1.3 AprilTag 准备
1. 打印 AprilTag（推荐使用 tag36h11 系列）
2. 将标签贴在需要抓取或推动的物体上
3. 确保标签尺寸与配置文件中的 `tag_size` 一致（默认 0.05 米）

---

## 2. 手眼标定

### 2.1 运行标定程序
```bash
python calibrate_hand_eye.py
```

### 2.2 标定步骤
1. 程序会自动初始化机械臂和相机
2. 查看相机画面，确保 AprilTag 可以被检测到
3. 按 `s` 键保存当前配置
4. 按 `q` 键退出

标定配置会保存到 `simple_hand_eye.json` 文件中。

---

## 3. 配置文件说明

配置文件位于 `piper/cfgs/config.yaml`

### 3.1 主要配置项

#### 实验基础设置
- `frame_stack`: 帧堆叠数量（默认 3）
- `action_repeat`: 动作重复次数（默认 2）
- `num_seed_frames`: 随机策略采样帧数
- `num_train_frames`: 总训练帧数

#### Piper 机械臂设置
- `use_sim`: 是否使用模拟环境（true=模拟，false=真实机械臂）
- `visualize`: 是否显示摄像头画面
- `use_apriltag`: 是否使用 AprilTag 检测物体位置
- `tag_size`: AprilTag 的物理大小（单位：米）
- `goal_pos`: 目标位置 [X, Y, Z]
- `obj_pos`: 物体初始位置 [X, Y, Z]

#### 训练参数
- `device`: 训练设备（默认 cuda:0）
- `lr`: 学习率
- `batch_size`: 训练批次大小
- `eval_every_frames`: 评估间隔帧数
- `num_eval_episodes`: 每次评估的 episode 数量

---

## 4. 推荐：手动收集数据与训练

### 4.1 为什么使用手动收集？
直接开始自动训练随机性太强，需要很长时间才能收敛。通过人工操作机械臂并给予奖励，可以快速收集高质量数据，大大降低训练成本。

### 4.2 启动手动收集
```bash
python manual_collect.py
```

### 4.3 操作说明
程序启动后会显示操作菜单，选择模式：
- **模式 1**：收集数据 + 同时训练（推荐）
- **模式 2**：仅收集数据（之后再训练）

### 4.4 键盘控制
| 按键 | 功能 |
|------|------|
| W/S | 前后移动 |
| A/D | 左右移动 |
| Q/E | 上下移动 |
| 1/2 | 夹爪开合 |
| 空格 | +10 奖励 |
| S | 保存模型和数据 |
| Q/ESC | 退出 |

### 4.5 使用技巧
1. 先练习操作几次，熟悉机械臂的运动
2. 当机械臂做出正确动作时，立即按空格键给予奖励
3. 每完成一个 Episode（250步），机械臂会自动复位
4. 定期按 S 键保存进度

### 4.6 收集多少数据？
- 入门：10-20 个 Episode
- 推荐：30-50 个 Episode
- 高质量：100+ 个 Episode

---

## 5. 自动训练

### 5.1 训练前检查清单
- [ ] 机械臂已通电并连接
- [ ] 相机已连接并正常工作
- [ ] AprilTag 已贴在物体上
- [ ] 手眼标定已完成（`simple_hand_eye.json` 存在）
- [ ] 配置文件已根据实际情况调整

### 5.2 启动训练
```bash
python train_piper.py
```

### 5.3 训练过程
- 程序会显示 tqdm 进度条
- 在训练过程中可以按空格键手动给奖励
- 定期会自动进行评估
- 模型会定期保存到 `snapshots/` 目录

---

## 6. 验证与评估

### 6.1 仅评估模式
如果已经有训练好的模型，可以只运行评估：

```bash
python train_piper.py eval_only=true
```

### 6.2 从指定快照恢复
```bash
python train_piper.py load_from_id=true load_id=10000
```

或指定快照路径：
```bash
python train_piper.py snapshot_path=/path/to/snapshot.pt
```

### 6.3 评估指标
- `episode_success_rate`: 成功率
- `episode_reward`: 平均奖励
- `episode_length`: 平均 episode 长度

---

## 7. 手动奖励功能

### 7.1 奖励按键
- **空格键**: 给予 +10.0 奖励

### 7.2 使用场景
当机械臂做出正确动作时，立即按下空格键给予奖励。奖励会在下一步计算时生效。

### 7.3 提示
- 按下空格键后会在控制台显示奖励信息
- tqdm 进度条会自动更新

---

## 8. 常见问题

### Q: 机械臂没有反应？
A: 检查：
1. CAN 接口是否正确连接
2. `use_sim` 是否设置为 false
3. 机械臂是否已通电

### Q: AprilTag 检测不到？
A: 检查：
1. 标签是否清晰
2. 光照是否充足
3. 标签尺寸是否与配置一致
4. 相机对焦是否正确

### Q: 如何中断训练？
A: 按 `Ctrl+C` 中断训练，模型会在保存点保存。

### Q: 手动收集数据时机械臂不动？
A: 确保按下的是正确的键，并且焦点在终端窗口上。

---

## 9. 文件结构

```
mentor/
├── train_piper.py              # 自动训练主程序
├── manual_collect.py           # 手动收集数据与训练（推荐）
├── manual_train.py             # 手动训练（旧版本）
├── calibrate_hand_eye.py       # 手眼标定程序
├── TRAINING_GUIDE.md           # 本指南文档
├── piper/
│   ├── env.py                  # 环境定义
│   ├── robot.py                # 机械臂控制
│   ├── cfgs/
│   │   └── config.yaml         # 配置文件
│   └── __init__.py
└── simple_hand_eye.json        # 手眼标定配置（生成）
```
