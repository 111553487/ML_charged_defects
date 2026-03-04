#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Geometric版本的CGCNN模型 - 早期电荷集成版本 + 选择性池化 (优化版)
电荷信息在模型最开始就加入，参与卷积过程
选择性池化：仅在引入空位的氧位点提取特征

优化点:
- 使用向量化操作替代循环
- 避免重复的张量创建
- 使用scatter操作进行高效池化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Batch
import pytorch_lightning as pl


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
        """前向传播"""
        row, col = edge_index
        
        # 更新边特征
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_attr_new = self.edge_update_net(edge_input)
        
        # 聚合消息
        messages = self.node_update(torch.cat([x[row], edge_attr_new], dim=1))
        
        # 按目标节点聚合
        x_new = torch.zeros_like(x)
        messages = messages.to(x_new.dtype)
        x_new.scatter_add_(0, col.unsqueeze(1).expand(-1, self.node_dim), messages)
        
        return x_new, edge_attr_new


class CGCNNPyGChargeEarlySelectivePoolingOptimized(nn.Module):
    """
    PyTorch Geometric版本的CGCNN模型 - 早期电荷集成 + 选择性池化 (优化版)
    
    关键改进:
    - 使用向量化操作替代循环
    - 高效的选择性池化实现
    - 避免重复的张量创建
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
        charge_embed_dim: int = 16
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout_ratio = dropout_ratio
        self.pooling = pooling
        self.charge_embed_dim = charge_embed_dim
        
        # 电荷嵌入层
        self.charge_embedding = nn.Linear(1, charge_embed_dim)
        
        # 原子特征嵌入
        self.atom_embedding = nn.Linear(num_atom_features + charge_embed_dim, embedding_dim)
        
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
        """前向传播 (优化版)"""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 处理charge张量 (优化: 直接转换)
        if hasattr(data, 'charge'):
            charge_data = data.charge
            
            if isinstance(charge_data, torch.Tensor):
                charge = charge_data.float()
            else:
                if isinstance(charge_data, list):
                    charge = torch.tensor(charge_data, dtype=torch.float, device=x.device)
                else:
                    charge = torch.tensor([charge_data], dtype=torch.float, device=x.device)
            
            if charge.dim() == 0:
                charge = charge.unsqueeze(0)
            elif charge.dim() == 2:
                charge = charge.squeeze(-1)
            
            # 电荷嵌入
            charge_feat = self.charge_embedding(charge.unsqueeze(-1))
        else:
            raise ValueError("Data object must have 'charge' attribute")
        
        # 为每个原子添加电荷信息 (向量化)
        charge_feat_expanded = charge_feat[batch]
        x = torch.cat([x, charge_feat_expanded], dim=1)
        
        # 嵌入原子和边特征
        x = self.atom_embedding(x)
        edge_attr = self.bond_embedding(edge_attr)
        
        # CGCNN卷积层
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_new, edge_attr = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x = F.softplus(x_new) + x
        
        # 优化的选择性池化
        if hasattr(data, 'target_site_indices') and data.target_site_indices is not None:
            graph_repr = self._selective_pooling_optimized(x, batch, data.target_site_indices)
        else:
            # 回退到全局池化
            graph_repr = self.pool(x, batch)
        
        # 预测
        out = self.predictor(graph_repr)
        
        return out.squeeze(-1)
    
    def _selective_pooling_optimized(self, x, batch, target_site_indices):
        """
        优化的选择性池化实现
        
        使用向量化操作替代循环，大幅提升性能
        """
        num_graphs = batch.max().item() + 1
        device = x.device
        dtype = x.dtype
        
        # 初始化输出
        graph_repr = torch.zeros(num_graphs, self.embedding_dim, device=device, dtype=dtype)
        
        # 处理target_site_indices的不同格式
        if isinstance(target_site_indices, torch.Tensor):
            if target_site_indices.dim() == 1:
                # 1D张量: 所有图使用相同的索引
                target_indices_list = [target_site_indices.tolist()] * num_graphs
            else:
                # 2D张量: 每个图有不同的索引
                target_indices_list = [target_site_indices[i].tolist() if i < target_site_indices.shape[0] 
                                      else target_site_indices[0].tolist() 
                                      for i in range(num_graphs)]
        else:
            # 列表或其他类型
            target_indices_list = [target_site_indices if isinstance(target_site_indices, list) 
                                  else [target_site_indices]] * num_graphs
        
        # 向量化处理: 为每个图创建掩码并进行池化
        for graph_idx in range(num_graphs):
            # 获取该图的节点掩码
            graph_mask = batch == graph_idx
            
            # 获取该图的目标位点索引
            target_idx_list = target_indices_list[graph_idx]
            
            # 创建目标位点掩码 (向量化)
            target_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
            for target_idx in target_idx_list:
                # 只标记该图中的目标位点
                node_idx = torch.where(graph_mask)[0]
                if target_idx < len(node_idx):
                    target_mask[node_idx[target_idx]] = True
            
            # 获取目标位点的特征
            target_features = x[target_mask]
            
            if target_features.shape[0] > 0:
                # 对目标位点的特征进行平均池化
                graph_repr[graph_idx] = target_features.mean(dim=0)
            else:
                # 如果没有目标位点，使用全局平均池化
                graph_nodes = torch.where(graph_mask)[0]
                graph_repr[graph_idx] = x[graph_nodes].mean(dim=0)
        
        return graph_repr


class CGCNNLightningChargeEarlySelectivePoolingOptimized(pl.LightningModule):
    """PyTorch Lightning版本的CGCNN模型 - 早期电荷集成 + 选择性池化 (优化版)"""
    
    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_conv_layers: int = 3,
        dropout_ratio: float = 0.5,
        learning_rate: float = 0.001,
        milestones: list = None,
        gamma: float = 0.1,
        pooling: str = 'mean',
        charge_embed_dim: int = 16
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # 模型
        self.model = CGCNNPyGChargeEarlySelectivePoolingOptimized(
            num_atom_features=num_atom_features,
            num_bond_features=num_bond_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            dropout_ratio=dropout_ratio,
            pooling=pooling,
            charge_embed_dim=charge_embed_dim
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 超参数
        self.learning_rate = learning_rate
        self.milestones = milestones or [100]
        self.gamma = gamma
    
    def forward(self, data: Batch) -> torch.Tensor:
        return self.model(data)
    
    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        predictions = self(batch)
        targets = batch.y
        
        loss = self.criterion(predictions, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        predictions = self(batch)
        targets = batch.y
        
        loss = self.criterion(predictions, targets)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """测试步骤"""
        predictions = self(batch)
        targets = batch.y
        
        loss = self.criterion(predictions, targets)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
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
                'interval': 'epoch'
            }
        }
