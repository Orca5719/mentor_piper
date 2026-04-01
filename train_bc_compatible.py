"""
MENTOR 兼容的行为克隆网络
与 agents/mentor.py 中的 Encoder 和 Actor 结构一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class MentorCompatibleBC(nn.Module):
    """
    与 MENTOR 兼容的 BC 网络
    使用与 MENTOR 相同的 Encoder + Actor 结构
    """
    
    def __init__(self, obs_shape=(3, 256, 256), action_dim=7, 
                 encoder_type='scratch', resnet_fix=True, 
                 pretrained_factor=1.0, hidden_dim=1024):
        """
        Args:
            obs_shape: 观测形状 (C, H, W)
            action_dim: 动作维度
            encoder_type: 'scratch' 或 'spawnnet'
            resnet_fix: 是否冻结预训练 ResNet
            pretrained_factor: 预训练特征的缩放因子
            hidden_dim: Actor 隐藏层维度
        """
        super(MentorCompatibleBC, self).__init__()
        
        self.encoder = _create_encoder(obs_shape, encoder_type, resnet_fix, pretrained_factor)
        self.actor = Actor(self.encoder.repr_dim, action_dim, hidden_dim)
        
    def forward(self, obs):
        """前向传播：观测 -> 动作"""
        # 编码观测
        obs = obs / 255.0 - 0.5  # 与 MENTOR 相同的归一化
        features = self.encoder(obs)
        
        # 动作
        action = self.actor(features)
        return action


def _create_encoder(obs_shape, encoder_type, resnet_fix, pretrained_factor):
    """创建与 MENTOR 兼容的编码器"""
    from agents.mentor import Encoder
    return Encoder(obs_shape, encoder_type, resnet_fix, pretrained_factor)


class Actor(nn.Module):
    """
    与 MENTOR 相同的 Actor 网络
    参考 agents/mentor.py 中的 Actor 类
    """
    def __init__(self, repr_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, action_dim))
        self.outputs = dict()
        self.apply(utils.weight_init)
    
    def forward(self, obs):
        mu = self.trunk(obs)
        mu = torch.tanh(mu)
        return mu


def train_bc_compatible(demo_files, output_path, num_epochs=100, 
                       batch_size=32, learning_rate=1e-4, 
                       encoder_type='scratch'):
    """
    训练与 MENTOR 兼容的 BC 策略
    
    Args:
        demo_files: 示范文件列表
        output_path: 输出路径
        num_epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        encoder_type: 编码器类型 ('scratch' 或 'spawnnet')
    """
    print("=" * 70)
    print("MENTOR 兼容的行为克隆（BC）预训练")
    print("=" * 70)
    
    # 导入必要模块
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from train_bc_piper import DemoDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # 创建数据集
    dataset = DemoDataset(demo_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 创建模型
    model = MentorCompatibleBC(encoder_type=encoder_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✓ 使用设备: {device}")
    print(f"✓ 编码器类型: {encoder_type}")
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
        
        # 保存最佳模型（包含 Encoder 和 Actor 分离的 state_dict）
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'actor_state_dict': model.actor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'encoder_type': encoder_type,
            }
            torch.save(checkpoint, output_path)
            print(f"  → 保存最佳模型 (loss={best_loss:.6f})")
        
        # 每 10 个 epoch 保存一次 checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_path.replace('.pt', f'_epoch{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"  → 保存 checkpoint: {checkpoint_path}")
    
    print(f"\n{'='*70}")
    print(f"✓ 训练完成！")
    print(f"✓ 最佳模型已保存到: {output_path}")
    print(f"✓ 最佳损失: {best_loss:.6f}")
    print(f"✓ 注意：此模型可直接加载到 MENTOR agent 中")
    print(f"{'='*70}")


if __name__ == "__main__":
    # 临时导入 utils（用于 weight_init）
    class Utils:
        @staticmethod
        def weight_init(m):
            """简单的权重初始化"""
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0.0)
    utils = Utils()
    
    # 测试代码
    print("MENTOR 兼容的 BC 网络模块已加载")
