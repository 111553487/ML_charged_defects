#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Geometric版本的CGCNN模型 - 早期电荷集成版本 + 选择性池化
电荷信息在模型最开始就加入，参与卷积过程
选择性池化：仅在引入空位的氧位点提取特征
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


class CGCNNPyGChargeEarlySelectivePooling(nn.Module):
    """
    PyTorch Geometric版本的CGCNN模型 - 早期电荷集成 + 选择性池化
     
    关键改进：
    - 电荷信息在最开始就加入到原子特征中
    - 电荷嵌入后与原子特征连接
    - 电荷信息参与整个卷积过程
    - 选择性池化：仅在引入空位的氧位点提取特征
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
        
        # 电荷嵌入层 - 在最开始就加入
        self.charge_embedding = nn.Linear(1, charge_embed_dim)
        
        # 原子特征嵌入 - 加上电荷维度
        # 原子特征 + 电荷嵌入 → embedding_dim
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
        """前向传播"""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 处理charge张量
        # 从Batch对象中提取charge信息
        if hasattr(data, 'charge'):
            charge_data = data.charge
            
            # 如果charge是张量，直接使用
            if isinstance(charge_data, torch.Tensor):
                charge = charge_data.float()
            else:
                # 如果charge是列表或其他类型，转换为张量
                if isinstance(charge_data, list):
                    charge = torch.tensor(charge_data, dtype=torch.float, device=x.device)
                else:
                    # 单个值的情况
                    charge = torch.tensor([charge_data], dtype=torch.float, device=x.device)
            
            # 确保charge是1D张量 [batch_size]
            if charge.dim() == 0:
                charge = charge.unsqueeze(0)
            elif charge.dim() == 2:
                charge = charge.squeeze(-1)
            
            # 电荷嵌入
            charge_feat = self.charge_embedding(charge.unsqueeze(-1))  # [batch_size, charge_embed_dim]
        else:
            raise ValueError("Data object must have 'charge' attribute")
        
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
        
        # 选择性池化：仅在引入空位的氧位点提取特征
        if hasattr(data, 'target_site_indices') and data.target_site_indices is not None:
            # 获取目标位点索引
            target_indices = data.target_site_indices
            
            # 为每个图创建掩码
            # 需要处理批处理中的多个图
            num_graphs = batch.max().item() + 1
            
            # 初始化图表示为零
            graph_repr = torch.zeros(num_graphs, self.embedding_dim, device=x.device, dtype=x.dtype)
            
            # 对每个图进行选择性池化
            for graph_idx in range(num_graphs):
                # 获取该图中的所有节点
                graph_mask = batch == graph_idx
                graph_nodes = torch.where(graph_mask)[0]
                
                # 获取该图的目标位点索引
                if isinstance(target_indices, torch.Tensor):
                    if target_indices.dim() > 1:
                        # 如果是二维张量，取第graph_idx行
                        if graph_idx < target_indices.shape[0]:
                            target_idx_list = target_indices[graph_idx].tolist()
                        else:
                            target_idx_list = target_indices[0].tolist()
                    else:
                        target_idx_list = target_indices.tolist()
                else:
                    target_idx_list = target_indices if isinstance(target_indices, list) else [target_indices]
                
                # 创建该图中目标位点的掩码
                target_mask = torch.zeros(len(graph_nodes), dtype=torch.bool, device=x.device)
                for target_idx in target_idx_list:
                    if target_idx < len(graph_nodes):
                        target_mask[target_idx] = True
                
                # 获取目标位点的特征
                if target_mask.sum() > 0:
                    target_features = x[graph_nodes][target_mask]
                    # 对目标位点的特征进行平均池化
                    graph_repr[graph_idx] = target_features.mean(dim=0)
                else:
                    # 如果没有目标位点，使用全局平均池化
                    graph_repr[graph_idx] = x[graph_nodes].mean(dim=0)
        else:
            # 回退到全局池化
            graph_repr = self.pool(x, batch)
        
        # 预测
        out = self.predictor(graph_repr)
        
        return out.squeeze(-1)


class CGCNNLightningChargeEarlySelectivePooling(pl.LightningModule):
    """PyTorch Lightning版本的CGCNN模型 - 早期电荷集成 + 选择性池化"""
    
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
        self.model = CGCNNPyGChargeEarlySelectivePooling(
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
