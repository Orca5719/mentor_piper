# Piper 机械臂真实世界训练完整指南

---

## 📋 目录
1. [快速开始（5步完成）](#快速开始5步完成)
2. [环境准备](#环境准备)
3. [硬件连接](#硬件连接)
4. [AprilTag 准备](#apriltag-准备)
5. [相机标定](#相机标定)
6. [手眼标定（关键！）](#手眼标定关键)
7. [目标位置标定](#目标位置标定)
8. [开始训练](#开始训练)
9. [常见问题](#常见问题)

---

## 🚀 快速开始（5步完成）

### **如果你已经完成了所有标定：**

```bash
# 1. 确认标定文件都在
ls camera_calibration.npz simple_hand_eye.json

# 2. 标定目标位置（可选，如果需要调整）
python calibrate_positions.py

# 3. 开始训练！
python train_piper.py
```

---

## 🛠️ 环境准备

### 1️⃣ 安装依赖

```bash
# 基础依赖
pip install numpy opencv-python pyyaml

# AprilTag 依赖
pip install pupil-apriltags

# 其他依赖（根据需要）
pip install hydra-core dm-control
```

### 2️⃣ 检查配置文件

打开 `piper/cfgs/config.yaml`，确认已设置为真实世界模式：

```yaml
# piper specific
use_sim: false              # ✅ 关闭模拟，使用真实机械臂
visualize: true             # ✅ 显示摄像头画面
print_reward: true          # ✅ 打印 reward 信息

# AprilTag 物体追踪设置
use_apriltag: true          # ✅ 启用 AprilTag
tag_size: 0.05              # AprilTag 大小（米），根据实际调整

# 目标位置设置（通过 calibrate_positions.py 标定）
goal_pos: [0.0, 0.75, 0.0]
obj_pos: [0.0, 0.6, 0.0]
```

---

## 🔌 硬件连接

### 1️⃣ 连接 Piper 机械臂

1. 用 USB 线连接机械臂到电脑
2. 确认机械臂已上电
3. 测试连接：

```bash
python test_piper_connection.py
```

### 2️⃣ 连接 RealSense 摄像头

1. 用 USB 线连接 RealSense 摄像头到电脑
2. 固定摄像头位置（建议在机械臂正前方）
3. 测试摄像头：

```bash
python test_piper_camera.py
```

---

## 🏷️ AprilTag 准备

### 1️⃣ 打印 AprilTag

1. 访问：https://april.eecs.umich.edu/software/apriltag
2. 下载 `tag36h11` 家族的标签（推荐 ID: 0）
3. 打印标签，大小约 **5cm × 5cm**（实际测量一下）
4. 用双面胶贴在咖啡杯侧面

### 2️⃣ 调整 `tag_size`

用尺子测量打印出的 AprilTag 实际大小，修改 `config.yaml`：

```yaml
tag_size: 0.05  # 如果实际是 5cm，就是 0.05 米
```

### 3️⃣ 测试 AprilTag 检测

```bash
python april_tag_tracker.py
```

你应该能看到：
- 摄像头画面
- AprilTag 被绿色框标记
- 红色中心点

按 `q` 退出，按 `s` 保存截图。

---

## 📷 相机标定

### 1️⃣ 打印棋盘格

搜索 "chessboard calibration printable"，打印一张：
- 内角点：9×6
- 每个格子大小：约 25mm × 25mm

### 2️⃣ 运行相机标定

```bash
python calibrate_camera.py
```

### 3️⃣ 采集图像

选择 `1` - 采集新图像：

1. 将棋盘格放在摄像头前
2. 按 `c` 键采集图像
3. 移动棋盘格到不同位置、不同角度
4. 重复采集 10-20 张图像
5. 按 `q` 键结束采集

### 4️⃣ 自动生成文件

标定完成后，会自动生成：
- ✅ `camera_calibration.npz` - 相机内参（**自动加载，不用管**）
- `camera_config.yaml` - 配置片段

---

## 👁️ 手眼标定（关键！）

### **为什么需要手眼标定？**

```
相机坐标系 → [手眼标定] → 机械臂坐标系
   (x,y,z)                    (x,y,z)
```

### **使用推荐工具：easy_hand_eye_calibration.py**

#### **步骤：**

```bash
python easy_hand_eye_calibration.py
```

#### **操作流程（超简单！）：**

```
程序会引导你采集 5 个点，每个点：

1. 把贴有 AprilTag 的咖啡杯放在一个位置
   ↓
2. 按示教按钮（灯亮）
   ↓
3. 用手把机械臂末端移到 AprilTag 中心
   ↓
4. 按示教按钮退出
   ↓
5. 按回车确认
   ↓
6. 重复 5 次！

完成后自动生成：simple_hand_eye.json（自动加载，不用管！）
```

#### **推荐的 5 个采集点：**

```
                    点 4 (稍远)
                        ↑
                        │
    点 3 (左侧) ←─────┼─────→ 点 2 (右侧)
                        │
                        ↓
                    点 1 (正前)
                    
                    点 5 (稍近)
```

---

## 🎯 目标位置标定

### **标定物体位置和目标位置**

#### **使用工具：calibrate_positions.py**

```bash
python calibrate_positions.py
```

#### **两种方式可选：**

| 方式 | 说明 |
|------|------|
| **1. 示教模式** | 用手移动机械臂到位置 |
| **2. AprilTag 模式** | 把物体放在位置，自动检测 |

#### **操作流程：**

```
1. 运行脚本
   ↓
2. 选择方式（推荐示教模式）
   ↓
3. 标定物体初始位置
   ├─ 把咖啡杯放在初始位置
   ├─ 用示教模式移机械臂到该点
   └─ 确认
   ↓
4. 标定目标位置
   ├─ 把咖啡杯放在目标位置
   ├─ 用示教模式移机械臂到该点
   └─ 确认
   ↓
5. 复制输出到 piper/cfgs/config.yaml
```

#### **配置文件更新示例：**

```yaml
goal_pos: [0.0, 0.75, 0.0]  # 替换为你的标定结果
obj_pos: [0.0, 0.6, 0.0]    # 替换为你的标定结果
```

---

## 🧪 测试所有功能

在开始训练前，先测试一下：

```bash
python test_piper_visualize.py
```

你应该看到：
- ✅ 机械臂连接成功
- ✅ 摄像头画面显示
- ✅ AprilTag 被检测到
- ✅ 物体位置正确显示（通过手眼标定转换）
- ✅ Reward 信息打印
- ✅ 可视化窗口显示

如果一切正常，按 `q` 退出。

---

## 🚀 开始训练！

### 1️⃣ 确认所有标定文件都在

```bash
# 检查这些文件是否存在
ls camera_calibration.npz      # ✅ 相机标定（自动加载）
ls simple_hand_eye.json        # ✅ 手眼标定（自动加载）
```

### 2️⃣ 确认配置文件

检查 `piper/cfgs/config.yaml`：

```yaml
use_sim: false          # 使用真实机械臂
use_apriltag: true      # 使用 AprilTag
visualize: true         # 显示画面
print_reward: true      # 打印 reward
goal_pos: [x, y, z]     # 你标定的目标位置
obj_pos: [x, y, z]      # 你标定的物体位置
```

### 3️⃣ 准备工作区

1. 确保咖啡杯在初始位置
2. 确保 AprilTag 在摄像头视野内
3. 确保周围环境安全
4. 准备好紧急停止按钮（如有）

### 4️⃣ 启动训练

```bash
python train_piper.py
```

### 5️⃣ 观察训练

你会看到：
- 📊 终端每步打印 reward
- 🎥 可视化窗口显示摄像头画面
- 📈 物体到目标的距离
- ✅ 成功状态

按 `q` 可以关闭可视化窗口（训练继续）。

### 6️⃣ 停止训练

按 `Ctrl+C` 停止训练。

---

## 📊 训练输出

### 终端输出示例

```
[Step   1] Reward:   0.12 | Episode Reward:   0.12 | Obj->Target: 0.1485m | Success: ❌
[Step   2] Reward:   0.15 | Episode Reward:   0.27 | Obj->Target: 0.1452m | Success: ❌
[Step   3] Reward:   0.20 | Episode Reward:   0.47 | Obj->Target: 0.1389m | Success: ❌
...
[Step  42] Reward:  10.00 | Episode Reward:  45.23 | Obj->Target: 0.0450m | Success: ✅
```

### 可视化窗口显示

- Step: 当前步数
- Reward: 当前步 reward
- Episode Reward: 本回合累计 reward
- Obj->Target: 物体到目标距离（绿色=成功范围内）
- Success: 是否成功

---

## 📁 标定文件总结

| 文件 | 是否自动加载 | 需要手动设置吗？ | 说明 |
|------|-------------|----------------|------|
| `camera_calibration.npz` | ✅ 自动 | ❌ 不需要 | 相机内参 |
| `simple_hand_eye.json` | ✅ 自动 | ❌ 不需要 | 手眼标定偏移量 |
| `config.yaml` 中的 goal_pos | ❌ 手动 | ⚠️ 需要标定 | 目标位置 |
| `config.yaml` 中的 obj_pos | ❌ 手动 | ⚠️ 需要标定 | 物体初始位置 |

---

## ❓ 常见问题

### Q1: 标定好了之后，还用手动设置参数吗？

**A:** 大部分不用！
- ✅ `camera_calibration.npz` - 自动加载
- ✅ `simple_hand_eye.json` - 自动加载
- ⚠️ `goal_pos` 和 `obj_pos` - 需要用 `calibrate_positions.py` 标定

### Q2: AprilTag 检测不到怎么办？

**A:** 检查以下几点：
1. 光照是否充足
2. AprilTag 是否平整
3. AprilTag 是否在摄像头视野内
4. 摄像头对焦是否清晰
5. 尝试更大的 AprilTag

### Q3: 物体位置不准确怎么办？

**A:** 
1. 重新运行相机标定
2. 重新运行手眼标定，采集更多点
3. 检查 `tag_size` 设置是否正确
4. 确认 AprilTag 粘贴位置正确

### Q4: 如何判断训练是否成功？

**A:** 观察：
1. Reward 是否逐渐增加
2. Obj->Target 距离是否逐渐减小
3. Success 标记是否频繁出现 ✅
4. 机械臂是否能稳定推动物体到目标

### Q5: 训练太慢怎么办？

**A:** 
1. 可以关闭 `visualize` 来加速
2. 可以关闭 `print_reward` 来加速
3. 调整 `action_repeat` 参数

---

## 🔧 故障排除

### 问题：机械臂连接失败

```
检查：
1. USB 线是否插好
2. 机械臂是否上电
3. 运行 test_piper_connection.py 测试
```

### 问题：摄像头无法初始化

```
检查：
1. USB 线是否插好
2. RealSense 驱动是否安装
3. 运行 test_piper_camera.py 测试
```

### 问题：AprilTag 初始化失败

```
检查：
1. pupil-apriltags 是否安装：pip install pupil-apriltags
2. camera_calibration.npz 是否存在
3. simple_hand_eye.json 是否存在
```

### 问题：手眼标定后物体位置还是不对

```
解决：
1. 重新运行 easy_hand_eye_calibration.py
2. 采集更多点（5-10 个）
3. 确保机械臂末端准确对准 AprilTag 中心
4. 检查 tag_size 设置
```

---

## 📚 所有工具文件

| 文件 | 用途 | 是否必须 |
|------|------|---------|
| `easy_hand_eye_calibration.py` | ✅ 手眼标定（推荐） | ⭐⭐⭐⭐⭐ |
| `calibrate_positions.py` | ✅ 目标位置标定 | ⭐⭐⭐⭐⭐ |
| `calibrate_camera.py` | 相机标定 | ⭐⭐⭐⭐⭐ |
| `april_tag_tracker.py` | AprilTag 检测测试 | ⭐⭐⭐ |
| `test_piper_visualize.py` | 完整功能测试 | ⭐⭐⭐⭐ |

---

## 🎉 总结

### **完整流程回顾：**

```
1. 硬件连接
   ↓
2. AprilTag 准备
   ↓
3. 相机标定 → camera_calibration.npz (自动加载)
   ↓
4. 手眼标定 → simple_hand_eye.json (自动加载)
   ↓
5. 目标位置标定 → 更新 config.yaml
   ↓
6. 开始训练！
```

### **你需要手动操作的：**
- ✅ 标定目标位置和物体位置（`calibrate_positions.py`）
- ✅ 把结果复制到 `config.yaml`

### **自动加载，不用管的：**
- ✅ `camera_calibration.npz` - 相机标定
- ✅ `simple_hand_eye.json` - 手眼标定

---

## � 祝你训练顺利！

有问题随时检查：
1. 硬件连接
2. 标定文件是否存在
3. 配置文件是否正确
4. 运行测试脚本验证
