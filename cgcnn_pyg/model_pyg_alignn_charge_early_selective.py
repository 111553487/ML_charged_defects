#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于官方 ALIGNN 的 PyTorch Geometric 版本 - 早期电荷集成 + 选择性池化版本

ALIGNN (Atom Linearly Independent Geometric Network) 特点：
1. 基于官方 ALIGNN 实现（alignn/alignn/models/alignn.py）
2. 使用边门控图卷积（Edge-Gated Graph Convolution）
3. 更强的几何表达能力
4. 电荷信息在模型最开始就加入到原子特征中
5. 对于空位，在池化过程中，仅在引入空位的氧位点提取特征（选择性池化）
6. 使用 MAE（平均绝对误差）作为损失函数

关键修复（与 CGCNN 保持一致）：
- 使用 Softplus 激活函数（与 CGCNN 一致）
- 在卷积层后使用 BatchNorm
- 添加残差连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Batch
import pytorch_lightning as pl
import numpy as np


class EdgeGatedGraphConv(nn.Module):
    """边门控图卷积 - 官方 ALIGNN 聚合方式

    参考：arxiv:1711.07553 和 arxiv:2003.0098
    使用自适应加权聚合
    
    关键：使用 BatchNorm 稳定训练（官方 ALIGNN 实现）
    使用 Softplus 激活（与 CGCNN 一致）
    """

    def __init__(self, input_features: int, output_features: int, residual: bool = True):
        """初始化边门控卷积层"""
        super().__init__()
        self.residual = residual
        self.input_features = input_features
        self.output_features = output_features

        # 源节点门控
        self.src_gate = nn.Linear(input_features, output_features)
        # 目标节点门控
        self.dst_gate = nn.Linear(input_features, output_features)
        # 边门控
        self.edge_gate = nn.Linear(input_features, output_features)
        # 边特征的 BatchNorm（官方 ALIGNN 使用）
        self.bn_edges = nn.BatchNorm1d(output_features)

        # 源节点更新
        self.src_update = nn.Linear(input_features, output_features)
        # 目标节点更新
        self.dst_update = nn.Linear(input_features, output_features)
        # 节点特征的 BatchNorm（官方 ALIGNN 使用）
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(self, x, edge_index, edge_attr):
        """
        边门控图卷积前向传播 - 官方 ALIGNN 聚合方式

        h_i^{l+1} = Softplus(BN(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j / sum_{j->i} eta_{ij}))

        Args:
            x: 节点特征 [num_nodes, input_features]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, input_features]

        Returns:
            更新后的节点特征和边特征
        """
        row, col = edge_index
        num_nodes = x.size(0)

        # 计算边门控
        e_src = self.src_gate(x[row])
        e_dst = self.dst_gate(x[col])
        m = e_src + e_dst + self.edge_gate(edge_attr)

        # 边门控激活（sigmoid）
        sigma = torch.sigmoid(m)

        # 节点更新：h_i = sum_{j->i} sigma_{ij} * dst_update(x[j])
        h_dst = self.dst_update(x[col])
        h_weighted = h_dst * sigma

        # 聚合到目标节点（自适应加权聚合）
        h_sum = torch.zeros(num_nodes, self.output_features, device=x.device, dtype=x.dtype)
        h_sum.scatter_add_(0, col.unsqueeze(1).expand(-1, self.output_features), h_weighted)

        # 计算 sigma 的和用于归一化
        sigma_sum = torch.zeros(num_nodes, self.output_features, device=x.device, dtype=x.dtype)
        sigma_sum.scatter_add_(0, col.unsqueeze(1).expand(-1, self.output_features), sigma)

        # 按照 sigma 的和进行自适应归一化（ALIGNN 的核心）
        h_normalized = h_sum / (sigma_sum + 1e-6)

        # 最终节点更新 + BatchNorm + Softplus（与 CGCNN 一致）
        x_new = self.src_update(x) + h_normalized
        x_new = self.bn_nodes(x_new)
        x_new = F.softplus(x_new)

        # 边特征更新 + BatchNorm + Softplus
        y_new = self.bn_edges(m)
        y_new = F.softplus(y_new)

        # 残差连接（官方 ALIGNN 实现）
        if self.residual:
            x_new = x + x_new
            y_new = edge_attr + y_new

        return x_new, y_new


class ALIGNNConv(nn.Module):
    """ALIGNN 卷积层 - 简化版本

    官方 ALIGNN 在晶体图和线图上分别进行更新，但在 PyG 中我们只有一个图。
    因此使用单层 EdgeGatedGraphConv，但保持其强大的表达能力。
    """

    def __init__(self, in_features: int, out_features: int):
        """初始化 ALIGNN 卷积层"""
        super().__init__()
        self.conv = EdgeGatedGraphConv(in_features, out_features)

    def forward(self, x, edge_index, edge_attr):
        """
        ALIGNN 卷积前向传播

        Args:
            x: 节点特征
            edge_index: 边索引
            edge_attr: 边特征

        Returns:
            更新后的节点特征和边特征
        """
        x_new, edge_attr_new = self.conv(x, edge_index, edge_attr)
        return x_new, edge_attr_new


class ALIGNNPyGChargeEarlySelective(nn.Module):
    """
    基于官方 ALIGNN 的 PyTorch Geometric 模型 - 早期电荷集成 + 选择性池化

    关键特性：
    1. 基于官方 ALIGNN 实现（边门控图卷积）
    2. 电荷信息在最开始就加入到原子特征中
    3. 实现选择性池化：仅在氧位点提取特征
    4. 支持多个不等价氧位点的特征提取
    5. 使用 Softplus 激活函数和 BatchNorm（与 CGCNN 一致）
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        embedding_dim: int = 64,  # 与 CGCNN 一致
        hidden_dim: int = 128,  # 与 CGCNN 一致
        num_conv_layers: int = 3,  # 与 CGCNN 一致
        dropout_ratio: float = 0.5,  # 与 CGCNN 一致
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

        # 电荷嵌入层 - 在最开始就加入（线性嵌入，与 CGCNN 一致）
        if self.use_charge:
            self.charge_embedding = nn.Linear(1, charge_embed_dim)
            # 原子特征 + 电荷嵌入 → embedding_dim
            self.atom_embedding = nn.Linear(num_atom_features + charge_embed_dim, embedding_dim)
        else:
            self.atom_embedding = nn.Linear(num_atom_features, embedding_dim)

        # 边特征嵌入
        self.bond_embedding = nn.Linear(num_bond_features, embedding_dim)

        # ALIGNN 卷积层
        self.conv_layers = nn.ModuleList([
            ALIGNNConv(embedding_dim, embedding_dim)
            for _ in range(num_conv_layers)
        ])

        # 批归一化层（与 CGCNN 一致）
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(embedding_dim)
            for _ in range(num_conv_layers)
        ])

        # 预测头 - 与 CGCNN 一致
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
            # 处理 charge 张量
            if isinstance(data.charge, torch.Tensor):
                charge = data.charge.float()
            else:
                if isinstance(data.charge, list):
                    charge = torch.tensor(data.charge, dtype=torch.float, device=x.device)
                else:
                    charge = torch.tensor([data.charge], dtype=torch.float, device=x.device)

            # 确保 charge 是 1D 张量 [batch_size]
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

        # 嵌入原子和边特征
        x = self.atom_embedding(x)  # [num_atoms, embedding_dim]
        edge_attr = self.bond_embedding(edge_attr)  # [num_edges, embedding_dim]

        # ALIGNN 卷积层 + BatchNorm + 残差连接（与 CGCNN 一致）
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_new, edge_attr = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x = F.softplus(x_new) + x  # 残差连接

        # 选择性池化：仅在氧位点提取特征
        if hasattr(data, 'target_site_indices') and data.target_site_indices is not None:
            valid_indices = data.target_site_indices[data.target_site_indices < x.size(0)]

            if len(valid_indices) == 0:
                graph_repr = self.pool(x, batch)
            else:
                oxygen_features = x[valid_indices]
                oxygen_batch = batch[valid_indices]
                graph_repr = self.pool(oxygen_features, oxygen_batch)
        else:
            graph_repr = self.pool(x, batch)

        # 预测
        out = self.predictor(graph_repr)

        return out.squeeze(-1)


class ALIGNNLightningChargeEarlySelective(pl.LightningModule):
    """PyTorch Lightning 版本的 ALIGNN 模型 - 早期电荷集成 + 选择性池化"""

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
        pooling: str = 'mean',
        use_charge: bool = True,
        charge_embed_dim: int = 16,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 20,
        reduce_lr_threshold: float = 1e-4
    ):
        super().__init__()

        # 保存超参数
        self.save_hyperparameters()

        # 模型
        self.model = ALIGNNPyGChargeEarlySelective(
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

        # 损失函数 - 使用 MAE（平均绝对误差）
        self.criterion = nn.L1Loss()

        # 超参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_threshold = reduce_lr_threshold

        # 验证集预测值列表
        self.val_predictions = []
        self.val_targets = []
        self.val_charges = []

    def forward(self, data: Batch) -> torch.Tensor:
        return self.model(data)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """训练步骤 - 使用 MAE（L1 损失）作为损失函数"""
        predictions = self(batch)
        targets = batch.y

        loss = self.criterion(predictions, targets)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """验证步骤 - 使用 MAE（平均绝对误差）作为损失"""
        predictions = self(batch)
        targets = batch.y

        # 计算 MAE 损失（L1 损失）
        loss = self.criterion(predictions, targets)

        # 保存预测值、目标值和电荷到列表
        self.val_predictions.extend(predictions.detach().cpu().numpy().flatten().tolist())
        self.val_targets.extend(targets.detach().cpu().numpy().flatten().tolist())
        self.val_charges.extend(batch.charge.detach().cpu().numpy().flatten().tolist())

        # 记录 val_loss（现在是 MAE 值）
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
                print(f"  样本数：{len(self.val_predictions)}")
                print(f"  MAE: {mae_this_epoch:.6f}")
                print(f"  (这个 MAE 就是 val_loss 的值)")

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """测试步骤 - 使用 MAE（L1 损失）作为损失函数"""
        predictions = self(batch)
        targets = batch.y

        loss = self.criterion(predictions, targets)

        self.log('test_loss', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 使用 ReduceLROnPlateau 调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            threshold=self.reduce_lr_threshold,
            min_lr=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss'
            }
        }


if __name__ == "__main__":
    # 测试模型
    print("测试 ALIGNN 模型 - 早期电荷集成 + 选择性池化...")

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
    target_site_indices = torch.tensor([2, 5, 8])

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
    model = ALIGNNPyGChargeEarlySelective(
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
    print(f"模型输出形状：{output.shape}")
    print(f"模型输出值：{output.item():.4f}")
    print("模型测试完成！")
