import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
from pathlib import Path
import numpy as np
import torch
from dm_env import specs
from tqdm import tqdm
import sys
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'piper'))

torch.backends.cudnn.benchmark = True

# 补充缺失的 utils 模块核心函数
class utils:
    @staticmethod
    @contextmanager
    def eval_mode(agent):
        """临时将agent设为eval模式，退出后恢复train模式"""
        old_mode = agent.training
        agent.eval()
        try:
            yield
        finally:
            agent.train(old_mode)

# 适配256×256×9输入的Actor网络
class Actor(torch.nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        # 卷积网络（适配9通道、256×256输入）
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(9, 32, kernel_size=8, stride=4),  # (32, 62, 62)
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (64, 29, 29)
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (64, 27, 27)
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        
        # 动态计算卷积输出维度（适配256×256输入）
        with torch.no_grad():
            dummy = torch.randn(1, *obs_shape)
            conv_out = self.conv_layers(dummy).shape[1]
            print(f"卷积层输出维度: {conv_out} (256×256×9 → {conv_out})")
        
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(conv_out, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, action_shape[0])
        )
    
    def forward(self, x):
        # 归一化输入（uint8→[0,1]）
        x = x.float() / 255.0
        conv_out = self.conv_layers(x)
        mean = self.fc_layers(conv_out)
        # 返回简单的分布（仅均值，适配训练代码的dist.mean）
        return type('obj', (object,), {'mean': mean})

class Agent:
    def __init__(self, obs_shape, action_shape):
        self.actor = Actor(obs_shape, action_shape)
        self.training = True  # 适配eval_mode
    
    def train(self, mode=True):
        self.training = mode
        self.actor.train(mode)

def make_agent(obs_spec, action_spec, cfg):
    """适配训练代码的agent创建函数"""
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return Agent(obs_spec.shape, action_spec.shape)


class ManualDataTrainer:
    def __init__(self, data_file="spacemouse_data_256x256x9.npz"):  # 默认文件名对齐数据收集代码
        print("="*60)
        print("      手动数据预训练工具 (适配256×256×9输入)")
        print("="*60)
        print()
        
        self.data_file = data_file
        self.work_dir = Path.cwd()
        
        # 简化配置（无需yaml文件）
        print("⚠️  使用内置简化配置（无需yaml文件）")
        self.cfg = type('obj', (object,), {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'agent': type('obj', (object,), {})
        })
        
        self.device = torch.device(self.cfg.device)
        print(f"设备: {self.device}")
        print()
        
        self.agent = None
        self.data = None
    
    def load_data(self):
        print(f"正在加载数据: {self.data_file}")
        
        if not os.path.exists(self.data_file):
            print(f"✗ 数据文件不存在: {self.data_file}")
            return False
        
        try:
            self.data = np.load(self.data_file)
            print(f"✓ 数据加载成功")
            print(f"  观测数据: {self.data['observations'].shape} (预期: [N, 9, 256, 256])")
            print(f"  动作数据: {self.data['actions'].shape} (预期: [N, 4])")
            print(f"  奖励数据: {self.data['rewards'].shape}")
            print(f"  数据总量: {len(self.data['rewards'])} transitions")
            print()
            return True
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def init_mentor_agent(self):
        print("正在初始化 Mentor Agent...")
        
        # 核心修改：观测规格改为256×256×9
        obs_spec = specs.BoundedArray(
            (9, 256, 256),  # 适配256×256×9输入
            np.uint8,
            0,
            255,
            name='observation'
        )
        
        action_spec = specs.BoundedArray(
            (4,),
            np.float32,
            -1.0,
            1.0,
            'action'
        )
        
        self.agent = make_agent(obs_spec, action_spec, self.cfg.agent)
        # 将agent移到设备
        self.agent.actor.to(self.device)
        print("✓ Mentor Agent 初始化成功")
        print(f"  Actor网络结构: {self.agent.actor}")
        print()
    
    def train_with_behavior_cloning(self, num_epochs=100, batch_size=32):
        if self.data is None:
            print("✗ 没有数据可训练")
            return
        
        if self.agent is None:
            self.init_mentor_agent()
        
        print("="*60)
        print("开始行为克隆预训练 (256×256×9输入)")
        print("="*60)
        print()
        
        observations = self.data['observations']
        actions = self.data['actions']
        
        num_samples = len(observations)
        # 处理最后一批不足batch_size的情况
        num_batches = max(1, num_samples // batch_size)
        
        print(f"训练参数:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Samples per epoch: {num_samples}")
        print(f"  Batches per epoch: {num_batches}")
        print()
        
        best_loss = float('inf')
        
        optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=1e-4)
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(num_samples)
            
            epoch_loss = 0.0
            
            pbar = tqdm(range(num_batches), desc=f'Epoch {epoch + 1}/{num_epochs}')
            
            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)  # 避免越界
                if start_idx >= end_idx:
                    break
                
                batch_indices = indices[start_idx:end_idx]
                
                # 加载批次数据（uint8→保持原样，在模型内归一化）
                obs_batch = torch.tensor(observations[batch_indices], dtype=torch.uint8).to(self.device)
                act_batch = torch.tensor(actions[batch_indices], dtype=torch.float32).to(self.device)
                
                with utils.eval_mode(self.agent):
                    dist = self.agent.actor(obs_batch)
                    pred_actions = dist.mean
                
                # MSE损失（行为克隆核心）
                loss = torch.mean((pred_actions - act_batch) ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * (end_idx - start_idx)  # 加权求和
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = epoch_loss / num_samples  # 按样本数平均
            print(f'Epoch {epoch + 1} 完成, 平均 Loss: {avg_loss:.6f}')
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_snapshot('pretrained_snapshot_256x256x9.pt', best_loss=best_loss)
                print(f'✓ 保存最佳预训练模型 (Loss: {best_loss:.6f})')
            
            # 定期保存
            if (epoch + 1) % 20 == 0:
                self.save_snapshot(f'pretrained_epoch_{epoch + 1}_256x256x9.pt', best_loss=best_loss)
        
        print()
        print("="*60)
        print("预训练完成！")
        print(f"最佳 Loss: {best_loss:.6f}")
        print()
        print("使用方法:")
        print("  1. 将 pretrained_snapshot_256x256x9.pt 复制到工作目录")
        print("  2. 在 config.yaml 中设置:")
        print("     snapshot_path: 'pretrained_snapshot_256x256x9.pt'")
        print("  3. 运行: python train_piper.py")
        print("="*60)
    
    def save_snapshot(self, filename='pretrained_snapshot_256x256x9.pt', best_loss=None):
        save_path = self.work_dir / filename
        
        # 保存模型权重（仅保存必要部分，避免冗余）
        payload = {
            'agent': self.agent,
            'actor_state_dict': self.agent.actor.state_dict(),
            'optimizer_state_dict': torch.optim.Adam(self.agent.actor.parameters()).state_dict(),
            'best_loss': best_loss if best_loss is not None else float('inf'),
            '_global_step': 0,
            '_global_episode': 0,
            'input_shape': (9, 256, 256)  # 保存输入形状信息
        }
        
        torch.save(payload, save_path)
        print(f'✓ 预训练模型已保存到: {save_path}')


def main():
    print("\n请选择数据文件:")
    data_file = input("数据文件路径 (默认 spacemouse_data_256x256x9.npz): ").strip() or "spacemouse_data_256x256x9.npz"
    
    trainer = ManualDataTrainer(data_file=data_file)
    
    if not trainer.load_data():
        print("\n请先运行 manual_collect.py 收集数据！")
        return
    
    print("\n请选择训练模式:")
    print("1. 行为克隆预训练（推荐）")
    print("2. 退出")
    
    choice = input("\n请输入选项 (1-2): ").strip()
    
    if choice == '1':
        num_epochs = int(input("训练多少个 epoch? (默认 100): ") or "100")
        batch_size = int(input("Batch size? (默认 32): ") or "32")
        
        trainer.train_with_behavior_cloning(num_epochs=num_epochs, batch_size=batch_size)
    else:
        print("退出")


if __name__ == '__main__':
    main()