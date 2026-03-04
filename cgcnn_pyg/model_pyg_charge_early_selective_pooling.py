#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Geometric版本的CGCNN模型 - 早期电荷集成 + 选择性池化版本
在原有早期电荷集成的基础上，添加选择性池化功能
仅在氧位点提取特征，符合原文逻辑

改进点：
- 保持原有的早期电荷集成逻辑
- 添加选择性池化：仅在target_site_indices指定的O位点提取特征
- 处理多个不等价O位点的情况
- 保持CGCNN的不变性
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
    - 保持原有的早期电荷集成逻辑
    - 添加选择性池化：仅在O位点提取特征
    - 处理多个不等价O位点
    - 符合原文逻辑
    """
    
    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_conv_layers: int = 5,
        dropout_ratio: float = 0.3,
        pooling: str = 'mean',
        use_charge: bool = True,
        charge_embed_dim: int = 16,
        use_selective_pooling: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout_ratio = dropout_ratio
        self.pooling = pooling
        self.use_charge = use_charge
        self.charge_embed_dim = charge_embed_dim
        self.use_selective_pooling = use_selective_pooling
        
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
        
        # 预测头 - 不再需要在这里加入电荷信息
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
        
        # 处理电荷信息 - 在最开始就加入（保持原有逻辑）
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
        
        # 池化：选择性池化或全局池化
        if self.use_selective_pooling:
            graph_repr = self._selective_pooling(x, batch, data)
        else:
            graph_repr = self.pool(x, batch)
        
        # 预测 - 直接使用池化后的图表示
        out = self.predictor(graph_repr)
        
        return out.squeeze(-1)
    
    def _selective_pooling(self, x, batch, data):
        """
        选择性池化：仅在氧位点提取特征
        
        这是符合原文逻辑的关键改进：
        "在池化过程中，仅在引入空位的氧位点提取特征"
        
        Args:
            x: 节点特征 [num_atoms, embedding_dim]
            batch: 批索引 [num_atoms]
            data: 包含target_site_indices的Data对象
        
        Returns:
            图级别表示 [batch_size, embedding_dim]
        """
        batch_size = batch.max().item() + 1
        graph_features = []
        
        # 为每个图进行选择性池化
        for batch_idx in range(batch_size):
            # 获取该图中的所有原子
            mask = batch == batch_idx
            graph_x = x[mask]  # [num_atoms_in_graph, embedding_dim]
            
            # 获取该图的氧位点索引
            if hasattr(data, 'target_site_indices') and data.target_site_indices is not None:
                site_indices = data.target_site_indices
                
                # 处理site_indices的不同格式
                if isinstance(site_indices, torch.Tensor):
                    site_indices = site_indices.tolist()
                
                # 提取氧位点的特征
                oxygen_features = []
                
                if isinstance(site_indices, list):
                    # 多个O位点的情况
                    for site_idx in site_indices:
                        if isinstance(site_idx, (int, torch.Tensor)):
                            site_idx = int(site_idx)
                            if site_idx < graph_x.shape[0]:
                                oxygen_features.append(graph_x[site_idx])
                elif isinstance(site_indices, int):
                    # 单个O位点的情况
                    if site_indices < graph_x.shape[0]:
                        oxygen_features.append(graph_x[site_indices])
                
                # 对O位点特征进行池化
                if len(oxygen_features) > 0:
                    oxygen_features = torch.stack(oxygen_features)  # [num_oxygen, embedding_dim]
                    # 对多个O位点的特征进行平均
                    graph_feat = oxygen_features.mean(dim=0)  # [embedding_dim]
                else:
                    # 如果没有找到O位点，回退到全局池化
                    graph_feat = graph_x.mean(dim=0)
            else:
                # 如果没有target_site_indices，回退到全局池化
                graph_feat = graph_x.mean(dim=0)
            
            graph_features.append(graph_feat)
        
        # 堆叠所有图的特征
        if len(graph_features) > 0:
            graph_repr = torch.stack(graph_features)  # [batch_size, embedding_dim]
        else:
            # 回退到全局池化
            graph_repr = self.pool(x, batch)
        
        return graph_repr


class CGCNNLightningChargeEarlySelectivePooling(pl.LightningModule):
    """PyTorch Lightning版本的CGCNN模型 - 早期电荷集成 + 选择性池化"""
    
    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_conv_layers: int = 5,
        dropout_ratio: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        milestones: list = None,
        gamma: float = 0.1,
        pooling: str = 'mean',
        use_charge: bool = True,
        charge_embed_dim: int = 16,
        use_selective_pooling: bool = True
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
            use_charge=use_charge,
            charge_embed_dim=charge_embed_dim,
            use_selective_pooling=use_selective_pooling
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
