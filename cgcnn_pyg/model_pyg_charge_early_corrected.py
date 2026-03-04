#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Geometric版本的CGCNN模型 - 早期电荷集成 + 空位感知特征提取（修正版）

关键改进：
1. 电荷信息在卷积过程中参与（早期集成）
2. 在空位处（O位点）提取特征，而不是全局池化
3. 支持多个不等价O位点的处理
4. 电荷状态添加到特征向量中，然后输入全连接层
5. 保留CGCNN的平移、旋转和宇称不变性
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
        messages = messages.to(x_new.dtype)
        x_new.scatter_add_(0, col.unsqueeze(1).expand(-1, self.node_dim), messages)
        
        return x_new, edge_attr_new


class CGCNNPyGChargeEarlyCorrected(nn.Module):
    """
    PyTorch Geometric版本的CGCNN模型 - 早期电荷集成 + 空位感知特征提取
    
    架构流程：
    1. 输入：晶胞结构 + 不等价O位点索引 + 电荷状态
    2. 电荷嵌入并加入原子特征（早期集成）
    3. CGCNN卷积层处理（电荷参与整个过程）
    4. 在O位点处提取特征（空位感知）
    5. 多O位点时分别提取后聚合
    6. 电荷状态添加到特征向量
    7. 输入全连接层预测
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
        
        # 步骤1：电荷嵌入层 - 在最开始就加入到原子特征中
        if self.use_charge:
            self.charge_embedding = nn.Linear(1, charge_embed_dim)
            # 原子特征 + 电荷嵌入 → embedding_dim
            self.atom_embedding = nn.Linear(num_atom_features + charge_embed_dim, embedding_dim)
        else:
            self.atom_embedding = nn.Linear(num_atom_features, embedding_dim)
        
        # 步骤2：边特征嵌入
        self.bond_embedding = nn.Linear(num_bond_features, embedding_dim)
        
        # 步骤3：CGCNN卷积层（电荷参与整个过程）
        self.conv_layers = nn.ModuleList([
            CGCNNConvSimple(embedding_dim, embedding_dim) 
            for _ in range(num_conv_layers)
        ])
        
        # 批归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(embedding_dim) 
            for _ in range(num_conv_layers)
        ])
        
        # 步骤4-5：在O位点处提取特征后输入全连接层
        # 预测头的输入维度 = embedding_dim（电荷已在卷积前添加）
        predictor_input_dim = embedding_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_dim, 1)
        )
        
        # 池化函数（备用）
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        前向传播
        
        流程：
        1. 处理电荷信息，加入原子特征
        2. 嵌入原子和边特征
        3. CGCNN卷积处理（电荷参与）
        4. 在O位点处提取特征
        5. 输入全连接层预测
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # ========== 步骤1：处理电荷信息 ==========
        if self.use_charge:
            # 处理charge张量
            if isinstance(data.charge, torch.Tensor):
                charge = data.charge.float()
            else:
                if isinstance(data.charge, list):
                    charge = torch.tensor(data.charge, dtype=torch.float, device=x.device)
                else:
                    charge = torch.tensor([data.charge], dtype=torch.float, device=x.device)
            
            # 确保charge是1D张量 [batch_size]
            if charge.dim() == 0:
                charge = charge.unsqueeze(0)
            elif charge.dim() == 2:
                charge = charge.squeeze(-1)
            
            # 电荷嵌入
            charge_feat = self.charge_embedding(charge.unsqueeze(-1))  # [batch_size, charge_embed_dim]
            
            # 为每个原子添加电荷信息
            charge_feat_expanded = charge_feat[batch]  # [num_atoms, charge_embed_dim]
            
            # 连接原子特征和电荷特征
            x = torch.cat([x, charge_feat_expanded], dim=1)  # [num_atoms, num_atom_features + charge_embed_dim]
        
        # ========== 步骤2：嵌入原子和边特征 ==========
        x = self.atom_embedding(x)  # [num_atoms, embedding_dim]
        edge_attr = self.bond_embedding(edge_attr)  # [num_edges, embedding_dim]
        
        # ========== 步骤3：CGCNN卷积层 ==========
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_new, edge_attr = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x = F.softplus(x_new) + x  # 残差连接
        
        # ========== 步骤4：在O位点处提取特征（空位感知） ==========
        if hasattr(data, 'target_site_indices') and data.target_site_indices is not None:
            vacancy_indices = data.target_site_indices
            
            if len(vacancy_indices) > 1:
                # 多个不等价O位点：分别提取特征后聚合
                vacancy_features_list = []
                for site_idx in vacancy_indices:
                    # 提取该O位点的特征
                    site_feature = x[site_idx:site_idx+1]  # [1, embedding_dim]
                    vacancy_features_list.append(site_feature)
                
                # 聚合多个O位点的特征（平均）
                graph_repr = torch.mean(torch.cat(vacancy_features_list, dim=0), dim=0, keepdim=True)
            else:
                # 单个O位点：直接提取
                site_idx = vacancy_indices[0]
                graph_repr = x[site_idx:site_idx+1]  # [1, embedding_dim]
        else:
            # 回退到全局池化（如果没有提供target_site_indices）
            graph_repr = self.pool(x, batch)
        
        # ========== 步骤5：输入全连接层预测 ==========
        out = self.predictor(graph_repr)
        
        return out.squeeze(-1)


class CGCNNLightningChargeEarlyCorrected(pl.LightningModule):
    """PyTorch Lightning版本的CGCNN模型 - 早期电荷集成 + 空位感知特征提取"""
    
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
        pooling: str = 'mean',
        use_charge: bool = True,
        charge_embed_dim: int = 16
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # 模型
        self.model = CGCNNPyGChargeEarlyCorrected(
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
