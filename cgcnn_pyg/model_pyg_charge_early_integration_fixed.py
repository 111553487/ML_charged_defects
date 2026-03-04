#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Geometric版本的CGCNN模型 - 早期电荷集成版本（修正版）
符合文献中的CGCNN架构要求：

1. 电荷信息在模型最开始就加入到原子特征中，参与整个卷积过程
2. 对于空位，在池化过程中，仅在引入空位的氧位点提取特征
3. 当晶胞中有多个不等价的O位点时，提取不同的O位点特征
4. 使用MAE（平均绝对误差）作为损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Batch
import pytorch_lightning as pl
import numpy as np


class CGCNNConvSimple(nn.Module):
    """简化的CGCNN卷积层"""
    
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # 节点更新网络
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.Softplus(),
            nn.Linear(node_dim, node_dim)
        )
        
        # 边更新网络
        self.edge_update_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, edge_dim),
            nn.Softplus(),
            nn.Linear(edge_dim, edge_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
        
        Returns:
            更新后的节点特征和边特征
        """
        row, col = edge_index
        
        # 更新边特征
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_attr_new = self.edge_update_net(edge_input)
        
        # 聚合消息
        messages = self.node_update(torch.cat([x[row], edge_attr_new], dim=1))
        
        # 按目标节点聚合
        x_new = torch.zeros_like(x)
        # 确保类型一致
        messages = messages.to(x_new.dtype)
        x_new.scatter_add_(0, col.unsqueeze(1).expand(-1, self.node_dim), messages)
        
        return x_new, edge_attr_new


class CGCNNPyGChargeEarlyFixed(nn.Module):
    """
    PyTorch Geometric版本的CGCNN模型 - 早期电荷集成（修正版）
    
    关键改进：
    1. 电荷信息在最开始就加入到原子特征中
    2. 实现选择性池化：仅在氧位点提取特征
    3. 支持多个不等价氧位点的特征提取
    4. 使用MAE作为损失函数
    """
    
    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_conv_layers: int = 3,
        dropout_ratio: float = 0.5,
        pooling: str = 'mean',
        use_charge: bool = True,
        charge_embed_dim: int = 16
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout_ratio = dropout_ratio
        self.pooling = pooling
        self.use_charge = use_charge
        self.charge_embed_dim = charge_embed_dim
        
        # 电荷嵌入层 - 在最开始就加入
        if self.use_charge:
            self.charge_embedding = nn.Linear(1, charge_embed_dim)
            # 原子特征 + 电荷嵌入 → embedding_dim
            self.atom_embedding = nn.Linear(num_atom_features + charge_embed_dim, embedding_dim)
        else:
            self.atom_embedding = nn.Linear(num_atom_features, embedding_dim)
        
        # 边特征嵌入
        self.bond_embedding = nn.Linear(num_bond_features, embedding_dim)
        
        # CGCNN卷积层
        self.conv_layers = nn.ModuleList([
            CGCNNConvSimple(embedding_dim, embedding_dim) 
            for _ in range(num_conv_layers)
        ])
        
        # 批归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(embedding_dim) 
            for _ in range(num_conv_layers)
        ])
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_dim, 1)
        )
        
        # 池化函数
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        前向传播 - 实现选择性池化
        
        步骤：
        1. 在卷积前加入电荷信息
        2. 进行多层卷积
        3. 仅在氧位点进行池化（选择性池化）
        4. 通过全连接层预测形成能
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 处理电荷信息 - 在最开始就加入
        if self.use_charge:
            # 处理charge张量
            if isinstance(data.charge, torch.Tensor):
                charge = data.charge.float()
            else:
                # 如果charge是列表或其他类型，转换为张量
                if isinstance(data.charge, list):
                    charge = torch.tensor(data.charge, dtype=torch.float, device=x.device)
                else:
                    # 单个值的情况
                    charge = torch.tensor([data.charge], dtype=torch.float, device=x.device)
            
            # 确保charge是1D张量 [batch_size]
            if charge.dim() == 0:
                charge = charge.unsqueeze(0)
            elif charge.dim() == 2:
                charge = charge.squeeze(-1)
            
            # 电荷嵌入
            charge_feat = self.charge_embedding(charge.unsqueeze(-1))  # [batch_size, charge_embed_dim]
            
            # 为每个原子添加电荷信息
            # 使用batch索引来确定每个原子属于哪个图
            charge_feat_expanded = charge_feat[batch]  # [num_atoms, charge_embed_dim]
            
            # 连接原子特征和电荷特征
            x = torch.cat([x, charge_feat_expanded], dim=1)  # [num_atoms, num_atom_features + charge_embed_dim]
        
        # 嵌入原子和边特征
        x = self.atom_embedding(x)  # [num_atoms, embedding_dim]
        edge_attr = self.bond_embedding(edge_attr)  # [num_edges, embedding_dim]
        
        # CGCNN卷积层
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_new, edge_attr = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x = F.softplus(x_new) + x  # 残差连接
        
        # 选择性池化：仅在氧位点提取特征
        # 检查是否有target_site_indices属性
        if hasattr(data, 'target_site_indices') and data.target_site_indices is not None:
            # 提取氧位点的特征
            oxygen_features = x[data.target_site_indices]
            oxygen_batch = batch[data.target_site_indices]
            graph_repr = self.pool(oxygen_features, oxygen_batch)
        else:
            # 如果没有指定氧位点，使用全局池化（回退方案）
            graph_repr = self.pool(x, batch)
        
        # 预测
        out = self.predictor(graph_repr)
        
        return out.squeeze(-1)


class CGCNNLightningChargeEarlyFixed(pl.LightningModule):
    """PyTorch Lightning版本的CGCNN模型 - 早期电荷集成（修正版）"""
    
    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_conv_layers: int = 3,
        dropout_ratio: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        milestones: list = None,
        gamma: float = 0.1,
        pooling: str = 'mean',
        use_charge: bool = True,
        charge_embed_dim: int = 16
    ):
        super().__init__()
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 模型
        self.model = CGCNNPyGChargeEarlyFixed(
            num_atom_features=num_atom_features,
            num_bond_features=num_bond_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            dropout_ratio=dropout_ratio,
            pooling=pooling,
            use_charge=use_charge,
            charge_embed_dim=charge_embed_dim
        )
        
        # 损失函数 - 使用MAE（平均绝对误差）
        self.criterion = nn.L1Loss()
        
        # 超参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.milestones = milestones or [100]
        self.gamma = gamma
        
        # 验证集预测值列表
        self.val_predictions = []
        self.val_targets = []
        self.val_charges = []
    
    def forward(self, data: Batch) -> torch.Tensor:
        return self.model(data)
    
    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """训练步骤 - 使用MAE（L1损失）作为损失函数"""
        predictions = self(batch)
        targets = batch.y
        
        loss = self.criterion(predictions, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """验证步骤 - 使用MAE（平均绝对误差）作为损失"""
        predictions = self(batch)
        targets = batch.y
        
        # 计算MAE损失（L1损失）
        loss = self.criterion(predictions, targets)
        
        # 保存预测值、目标值和电荷到列表
        self.val_predictions.extend(predictions.detach().cpu().numpy().flatten().tolist())
        self.val_targets.extend(targets.detach().cpu().numpy().flatten().tolist())
        self.val_charges.extend(batch.charge.detach().cpu().numpy().flatten().tolist())
        
        # 记录val_loss（现在是MAE值）
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_start(self):
        """验证 epoch 开始时清空预测值列表"""
        self.val_predictions = []
        self.val_targets = []
        self.val_charges = []
    
    def on_validation_epoch_end(self):
        """验证 epoch 结束时计算并打印该 epoch 的 MAE"""
        if len(self.val_predictions) > 0:
            predictions_arr = np.array(self.val_predictions)
            targets_arr = np.array(self.val_targets)
            mae_this_epoch = np.mean(np.abs(predictions_arr - targets_arr))
            
            # 在最后一个 epoch 打印详细信息
            if self.current_epoch == self.trainer.max_epochs - 1:
                print(f"\n[Epoch {self.current_epoch}] 验证集统计 (归一化空间):")
                print(f"  样本数: {len(self.val_predictions)}")
                print(f"  MAE: {mae_this_epoch:.6f}")
                print(f"  (这个 MAE 就是 val_loss 的值)")
    
    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """测试步骤 - 使用MAE（L1损失）作为损失函数"""
        predictions = self(batch)
        targets = batch.y
        
        loss = self.criterion(predictions, targets)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.milestones,
            gamma=self.gamma
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


if __name__ == "__main__":
    # 测试模型
    print("测试修正版CGCNN模型...")
    
    # 创建测试数据
    num_atoms = 10
    num_edges = 20
    num_atom_features = 92
    num_bond_features = 41
    
    # 创建模拟数据
    x = torch.randn(num_atoms, num_atom_features)
    edge_index = torch.randint(0, num_atoms, (2, num_edges))
    edge_attr = torch.randn(num_edges, num_bond_features)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    y = torch.randn(1)
    charge = torch.tensor([0])
    target_site_indices = torch.tensor([2, 5, 8])  # 模拟氧位点索引
    
    data = Batch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        batch=batch,
        charge=charge,
        target_site_indices=target_site_indices
    )
    
    # 创建模型
    model = CGCNNPyGChargeEarlyFixed(
        num_atom_features=num_atom_features,
        num_bond_features=num_bond_features,
        embedding_dim=64,
        hidden_dim=128,
        num_conv_layers=3,
        use_charge=True,
        charge_embed_dim=16
    )
    
    # 前向传播
    output = model(data)
    print(f"模型输出形状: {output.shape}")
    print(f"模型输出值: {output.item():.4f}")
    print("模型测试完成！")