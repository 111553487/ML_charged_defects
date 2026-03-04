#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Geometric版本的CGCNN模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple
import pytorch_lightning as pl


class CGCNNConv(MessagePassing):
    """
    CGCNN卷积层 - PyTorch Geometric版本
    """
    
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__(aggr='add')
        
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
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
        
        Returns:
            更新后的节点特征和边特征
        """
        # 更新边特征
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_attr_new = self.edge_update_net(edge_input)
        
        # 消息传递更新节点特征
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr_new, size=None)
        
        return x_new, edge_attr_new
    
    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        构造消息
        
        Args:
            x_j: 源节点特征
            edge_attr: 边特征
        
        Returns:
            消息
        """
        return self.node_update(torch.cat([x_j, edge_attr], dim=1))


class CGCNNPyG(nn.Module):
    """
    PyTorch Geometric版本的CGCNN模型
    """
    
    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_conv_layers: int = 3,
        dropout_ratio: float = 0.5,
        pooling: str = 'mean'
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout_ratio = dropout_ratio
        self.pooling = pooling
        
        # 原子特征嵌入
        self.atom_embedding = nn.Linear(num_atom_features, embedding_dim)
        
        # 边特征嵌入
        self.bond_embedding = nn.Linear(num_bond_features, embedding_dim)
        
        # CGCNN卷积层
        self.conv_layers = nn.ModuleList([
            CGCNNConv(embedding_dim, embedding_dim) 
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
        前向传播
        
        Args:
            data: PyTorch Geometric批次数据
        
        Returns:
            预测值 [batch_size * num_targets_per_graph, 1]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 嵌入原子和边特征
        x = self.atom_embedding(x)
        edge_attr = self.bond_embedding(edge_attr)
        
        # CGCNN卷积层
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_new, edge_attr = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x = F.softplus(x_new) + x  # 残差连接
        
        # 图级别池化
        graph_repr = self.pool(x, batch)
        
        # 预测
        out = self.predictor(graph_repr)
        
        return out.squeeze(-1)


class CGCNNLightningPyG(pl.LightningModule):
    """
    PyTorch Lightning版本的CGCNN模型
    """
    
    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_conv_layers: int = 3,
        dropout_ratio: float = 0.5,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        milestones: list = None,
        gamma: float = 0.1,
        pooling: str = 'mean'
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # 模型
        self.model = CGCNNPyG(
            num_atom_features=num_atom_features,
            num_bond_features=num_bond_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            dropout_ratio=dropout_ratio,
            pooling=pooling
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 超参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.milestones = milestones or [100]
        self.gamma = gamma
    
    def forward(self, data: Batch) -> torch.Tensor:
        return self.model(data)
    
    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        predictions = self(batch)
        
        # 处理多目标情况
        targets = []
        for i, num_targets in enumerate(batch.y.shape[0] if batch.y.dim() > 1 else [1] * batch.num_graphs):
            if batch.y.dim() > 1:
                targets.extend(batch.y[i].tolist())
            else:
                targets.append(batch.y[i].item())
        
        targets = torch.tensor(targets, device=self.device, dtype=torch.float)
        
        loss = self.criterion(predictions, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        predictions = self(batch)
        
        # 处理多目标情况
        targets = []
        for i, num_targets in enumerate(batch.y.shape[0] if batch.y.dim() > 1 else [1] * batch.num_graphs):
            if batch.y.dim() > 1:
                targets.extend(batch.y[i].tolist())
            else:
                targets.append(batch.y[i].item())
        
        targets = torch.tensor(targets, device=self.device, dtype=torch.float)
        
        loss = self.criterion(predictions, targets)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_loss_step', loss, on_step=True, on_epoch=False)
        self.log('val_loss_epoch', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """测试步骤"""
        predictions = self(batch)
        
        # 处理多目标情况
        targets = []
        for i, num_targets in enumerate(batch.y.shape[0] if batch.y.dim() > 1 else [1] * batch.num_graphs):
            if batch.y.dim() > 1:
                targets.extend(batch.y[i].tolist())
            else:
                targets.append(batch.y[i].item())
        
        targets = torch.tensor(targets, device=self.device, dtype=torch.float)
        
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
                'interval': 'epoch'
            }
        }