"""
Piper 行为克隆（BC）预训练脚本
基于 hil-serl 的 train_bc.py 修改，适配 Piper 机械臂
"""

import os
import glob
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class DemoDataset(Dataset):
    """示范数据集"""
    
    def __init__(self, demo_files, obs_dim=(256, 256, 3), action_dim=7):
        """
        Args:
            demo_files: 示范文件路径列表
            obs_dim: 观测维度
            action_dim: 动作维度
        """
        self.transitions = []
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        print(f"加载示范数据...")
        for demo_file in demo_files:
            print(f"  - {demo_file}")
            with open(demo_file, 'rb') as f:
                transitions = pkl.load(f)
                self.transitions.extend(transitions)
        
        print(f"✓ 共加载 {len(self.transitions)} 个转换")
        
        # 统计信息
        actions = [t['actions'] for t in self.transitions]
        actions = np.array(actions)
        print(f"✓ 动作统计: mean={actions.mean():.4f}, std={actions.std():.4f}")
        print(f"✓ 动作范围: min={actions.min():.4f}, max={actions.max():.4f}")
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        transition = self.transitions[idx]
        obs = transition['observations']
        action = transition['actions']
        
        # 归一化观测到 [0, 1]
        obs_normalized = obs.astype(np.float32) / 255.0
        
        # 确保 action 是 numpy array
        if isinstance(action, list):
            action = np.array(action)
        
        # 调整维度：HWC -> CHW
        obs_tensor = torch.from_numpy(obs_normalized).permute(2, 0, 1)
        action_tensor = torch.from_numpy(action.astype(np.float32))
        
        return obs_tensor, action_tensor


class SimpleBCNetwork(nn.Module):
    """简单的行为克隆网络"""
    
    def __init__(self, obs_shape=(3, 256, 256), action_dim=7, hidden_dim=256):
        super(SimpleBCNetwork, self).__init__()
        
        # 简单的 CNN 特征提取
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 计算卷积后的特征维度
        conv_out_size = self._get_conv_out_size(obs_shape)
        
        # 全连接层
        self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def _get_conv_out_size(self, shape):
        """计算卷积层的输出尺寸"""
        x = torch.zeros(1, *shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))
    
    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # 输出到 [-1, 1]
        return action


def train_bc(demo_files, output_path, num_epochs=100, batch_size=32, learning_rate=1e-4):
    """
    训练 BC 策略
    
    Args:
        demo_files: 示范文件列表
        output_path: 输出路径
        num_epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
    """
    print("=" * 60)
    print("Piper 行为克隆（BC）预训练")
    print("=" * 60)
    
    # 创建数据集
    dataset = DemoDataset(demo_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 创建模型
    model = SimpleBCNetwork()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✓ 使用设备: {device}")
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    print(f"\n开始训练 ({num_epochs} epochs)...")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for obs, action in pbar:
            obs = obs.to(device)
            action = action.to(device)
            
            # 前向传播
            pred_action = model(obs)
            
            # 计算损失
            loss = F.mse_loss(pred_action, action)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, output_path)
            print(f"  → 保存最佳模型 (loss={best_loss:.6f})")
        
        # 每 10 个 epoch 保存一次 checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_path.replace('.pt', f'_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  → 保存 checkpoint: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ 训练完成！")
    print(f"✓ 最佳模型已保存到: {output_path}")
    print(f"✓ 最佳损失: {best_loss:.6f}")
    print(f"{'='*60}")


def main():
    # 配置
    demo_dir = "./demo_data"
    output_dir = "./bc_checkpoints"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有示范文件
    demo_files = glob.glob(os.path.join(demo_dir, "*.pkl"))
    
    if len(demo_files) == 0:
        print(f"错误：在 {demo_dir} 中没有找到示范数据文件")
        print(f"请先运行 record_demos_piper.py 收集示范数据")
        return
    
    print(f"找到 {len(demo_files)} 个示范文件:")
    for f in demo_files:
        print(f"  - {f}")
    
    # 输出路径
    output_path = os.path.join(output_dir, "bc_policy_best.pt")
    
    # 训练参数
    num_epochs = 100
    batch_size = 16
    learning_rate = 1e-4
    
    print(f"\n训练配置:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Output: {output_path}")
    
    # 开始训练
    train_bc(demo_files, output_path, num_epochs, batch_size, learning_rate)


if __name__ == "__main__":
    main()
