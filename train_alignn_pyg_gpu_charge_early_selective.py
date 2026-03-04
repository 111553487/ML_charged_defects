#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完全GPU加速的PyTorch Geometric ALIGNN训练脚本 - 早期电荷集成 + 选择性池化版本
预测氧空位形成能

符合文献要求：
1. 电荷信息在神经网络最开始就加入到原子特征中
2. 对于空位，在池化过程中，仅在引入空位的氧位点提取特征（选择性池化）
3. 当晶胞中有多个不等价的O位点时，提取不同的O位点特征
4. 使用MAE（平均绝对误差）作为损失函数

特点：
- 完全GPU加速
- 电荷信息早期集成（在原子特征嵌入之前）
- 选择性池化（仅在氧位点提取特征）
- PyTorch Geometric 实现
- 支持多个电荷态
- 使用ALIGNN网络架构（比CGCNN更强大）

使用方法:
     python train_cgcnn_pyg_gpu_charge_early_selective_alignn.py --epochs 250 --batch-size 32
    python train_cgcnn_pyg_gpu_charge_early_selective_alignn.py --epochs 1000 --batch-size 500  # 完整训练
    python train_cgcnn_pyg_gpu_charge_early_selective_alignn.py --use-charge False  # 不使用电荷信息（消融研究）
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from monty.serialization import loadfn
import numpy as np
import pandas as pd

from cgcnn.featurizer import ElementFeaturizer, GaussianBasis
from cgcnn.parameters import BondFeaturizerParams
from cgcnn.normalizer import DefectDistributions, make_normalizer
from cgcnn_pyg.model_pyg_alignn_charge_early_selective import ALIGNNLightningChargeEarlySelective


class MaterialDataset(Dataset):
    """
    材料数据集 - 支持选择性池化
    
    特点：
    - 为每个材料的每个电荷态创建一个数据点
    - 包含原子特征、键特征和电荷信息
    - 包含氧位点索引（用于选择性池化）
    - 支持 PyTorch Geometric 的 Data 对象
    
    数据流：
        Material (charges=[0,1,2], target_vals=[0.5,0.8,1.2], target_site_indices=[2,5,8])
            ↓
        3 个 Data 对象 (每个对应一个电荷态)
            ↓
        Batch 对象 (包含 charge 和 target_site_indices 属性)
            ↓
        模型 (在最开始就使用电荷信息，并在氧位点进行选择性池化)
    """
    
    def __init__(self, materials, bond_featurizer, elem_featurizer):
        self.materials = materials
        self.bond_featurizer = bond_featurizer
        self.elem_featurizer = elem_featurizer
        self.data_list = []
        
        for material in materials:
            self._process_material(material)
    
    def _process_material(self, material):
        """
        处理单个材料
        
        步骤：
        1. 提取原子特征
        2. 计算键特征和键索引
        3. 获取氧位点索引（必须提供）
        4. 为每个电荷态创建一个 Data 对象
        """
        try:
            # 检查必要属性
            if not hasattr(material, 'structure'):
                raise ValueError(f"Material {material.formula} missing structure")
            if not hasattr(material, 'target_vals') or not hasattr(material, 'charges'):
                raise ValueError(f"Material {material.formula} missing target_vals or charges")
            
            # 原子特征
            atom_features = []
            for atom in material.structure:
                elem_feat = self.elem_featurizer.featurize(atom.specie)
                atom_features.append(elem_feat)
            
            if len(atom_features) == 0:
                raise ValueError(f"Material {material.formula} has no atoms")
            
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # 边特征
            bond_features, bond_indices = self.bond_featurizer.apply(material.structure)
            
            edge_indices = []
            edge_features = []
            
            for i, (bond_feat_row, bond_idx_row) in enumerate(zip(bond_features, bond_indices)):
                for j, (bond_feat, bond_idx) in enumerate(zip(bond_feat_row, bond_idx_row)):
                    edge_indices.append([i, bond_idx])
                    edge_features.append(bond_feat.tolist())
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, self.bond_featurizer.num_bond_features), dtype=torch.float)
            
            # 获取氧位点索引 - 优先使用材料提供的索引
            target_site_indices = None
            if hasattr(material, 'target_site_indices') and material.target_site_indices:
                target_site_indices = material.target_site_indices
            else:
                # 如果没有提供，找出所有氧原子的索引
                target_site_indices = []
                for idx, atom in enumerate(material.structure):
                    if atom.specie.symbol == 'O':
                        target_site_indices.append(idx)
            
            if not target_site_indices:
                # 如果仍然没有氧位点，跳过这个材料
                print(f"Warning: Material {material.formula} has no oxygen sites, skipping")
                return
            
            # 目标值 - 为每个目标值创建一个数据对象
            # 关键：每个电荷态对应一个数据点
            if len(material.target_vals) == 0:
                raise ValueError(f"Material {material.formula} has no target values")
            
            for target_val, charge in zip(material.target_vals, material.charges):
                # 创建 PyTorch Geometric Data 对象
                # 包含：原子特征、键特征、目标值、电荷信息、氧位点索引
                data = Data(
                    x=x.clone(),                                    # 原子特征 [num_atoms, num_features]
                    edge_index=edge_index.clone(),                  # 键索引 [2, num_edges]
                    edge_attr=edge_attr.clone(),                    # 键特征 [num_edges, num_features]
                    y=torch.tensor([target_val], dtype=torch.float),  # 目标值（形成能）
                    formula=material.formula,                       # 材料化学式
                    charge=charge,                                  # 电荷态
                    target_site_indices=torch.tensor(target_site_indices, dtype=torch.long)  # 氧位点索引
                )
                self.data_list.append(data)
        
        except Exception as e:
            print(f"Warning: Failed to process {material.formula}: {str(e)}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


class ValidationCallback(Callback):
    """自定义回调函数，在训练结束时保存验证结果"""
    
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.val_predictions = []
        self.val_targets = []
        self.val_charges = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """在每个验证批次结束时保存预测值"""
        predictions = pl_module(batch)
        targets = batch.y
        
        self.val_predictions.extend(predictions.detach().cpu().numpy().flatten().tolist())
        self.val_targets.extend(targets.detach().cpu().numpy().flatten().tolist())
        self.val_charges.extend(batch.charge.detach().cpu().numpy().flatten().tolist())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """在每个验证epoch结束时清空列表"""
        self.val_predictions = []
        self.val_targets = []
        self.val_charges = []
    
    def on_train_end(self, trainer, pl_module):
        """在训练结束时保存最终的验证结果"""
        if len(self.val_predictions) > 0:
            predictions_arr = np.array(self.val_predictions)
            targets_arr = np.array(self.val_targets)
            
            # 在归一化空间中计算 MAE 和 RMSE
            mae_normalized = np.mean(np.abs(predictions_arr - targets_arr))
            rmse_normalized = np.sqrt(np.mean((predictions_arr - targets_arr) ** 2))
            
            print(f"\n[训练完成] 最终验证集统计 (归一化空间)")
            print(f"验证集样本数: {len(self.val_predictions)}")
            print(f"[归一化空间] MAE: {mae_normalized:.6f}, RMSE: {rmse_normalized:.6f}")
            print(f"预测值 (前10个): {self.val_predictions[:10]}")
            print(f"目标值 (前10个): {self.val_targets[:10]}")
            print(f"电荷 (前10个):   {self.val_charges[:10]}")
            
            # 按电荷态统计（归一化空间）
            print(f"\n[归一化空间] 按电荷态的统计:")
            for q in [0, 1, 2]:
                mask = np.array(self.val_charges) == q
                if np.sum(mask) > 0:
                    mae_q = np.mean(np.abs(predictions_arr[mask] - targets_arr[mask]))
                    rmse_q = np.sqrt(np.mean((predictions_arr[mask] - targets_arr[mask]) ** 2))
                    print(f"  电荷 {q}: 样本数={int(np.sum(mask))}, MAE={mae_q:.6f}, RMSE={rmse_q:.6f}")
            
            # 保存验证结果到 JSON 文件（归一化空间）
            val_results = {
                'predictions_normalized': self.val_predictions,
                'targets_normalized': self.val_targets,
                'charges': self.val_charges,
                'mae_normalized': float(mae_normalized),
                'rmse_normalized': float(rmse_normalized),
                'num_samples': len(self.val_predictions),
                'space': 'normalized',
                'note': '这些是最后一个 epoch 的验证集数据，在归一化空间中',
                'model': 'ALIGNN with early charge integration and selective pooling'
            }
            
            val_results_path = self.output_dir / "validation_results.json"
            with open(val_results_path, 'w') as f:
                json.dump(val_results, f, indent=2)
            print(f"\n✓ 验证结果已保存到: {val_results_path}")
            print(f"  (归一化空间中的预测值、目标值和指标)")
            print(f"  注意: 这是最后一个 epoch 的验证集数据")


def main():
    parser = argparse.ArgumentParser(
        description="Train PyG ALIGNN model with early charge integration and selective pooling"
    )
    
    # 数据参数
    parser.add_argument("--data-dir", type=str, default="oxy_vac_data/materials_coreAlign", 
                        help="Data directory containing JSON material files")
    parser.add_argument("--max-materials", type=int, default=None, 
                        help="Max materials to load (for testing)")
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")

    # 模型参数
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Embedding dimension for atoms and bonds")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension in predictor head")
    parser.add_argument("--num-conv", type=int, default=3,
                        help="Number of convolutional layers (ALIGNN uses fewer layers than CGCNN)")
    parser.add_argument("--dropout-ratio", type=float, default=0.2,
                        help="Dropout ratio")

    # 电荷处理参数
    parser.add_argument("--use-charge", type=bool, default=True,
                        help="Whether to use charge information in the model")
    parser.add_argument("--charge-embed-dim", type=int, default=16,
                        help="Embedding dimension for charge (1 -> charge-embed-dim)")

    # 学习率调度器参数
    parser.add_argument("--reduce-lr-factor", type=float, default=0.5,
                        help="ReduceLROnPlateau learning rate decay factor")
    parser.add_argument("--reduce-lr-patience", type=int, default=20,
                        help="ReduceLROnPlateau patience (epochs without improvement)")
    parser.add_argument("--reduce-lr-threshold", type=float, default=1e-4,
                        help="ReduceLROnPlateau threshold for measuring improvement")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="results", 
                        help="Output directory for results and checkpoints")
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"pyg_gpu_charge_early_selective_alignn_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("ALIGNN训练脚本 - 早期电荷集成 + 选择性池化")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 加载材料
    print("=" * 80)
    print("[1/6] 加载材料")
    print("=" * 80)
    
    data_dir = Path(args.data_dir)
    materials = []
    json_files = sorted(data_dir.glob("*.json"))
    
    if args.max_materials:
        json_files = json_files[:args.max_materials]
    
    failed_count = 0
    success_count = 0
    
    for json_file in json_files:
        try:
            material = loadfn(json_file)
            materials.append(material)
            success_count += 1
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # 只打印前5个失败
                print(f"Failed to load {json_file.name}: {str(e)}")
    
    print(f"成功加载: {success_count} 个材料")
    print(f"加载失败: {failed_count} 个材料")
    print(f"总材料数: {len(materials)}")

    
    # 规范化
    print("\n" + "=" * 80)
    print("[2/6] 规范化数据")
    print("=" * 80)
    
    distributions = DefectDistributions.from_materials(materials)
    normalizer = make_normalizer(distributions)
    
    for m in materials:
        m.target_vals = normalizer.normed_target_vals(m.target_vals, m.charges)
    
    print(f"Normalizer: shift={normalizer.shift:.6f}, mean={normalizer.mean:.6f}, std={normalizer.std:.6f}")
    
    # 创建特征化器
    print("\n" + "=" * 80)
    print("[3/6] 创建特征化器")
    print("=" * 80)
    
    bond_params = BondFeaturizerParams()
    bond_featurizer = GaussianBasis(
        cutoff_radius=bond_params.cutoff_radius,
        max_num_neighbors=bond_params.max_num_neighbors,
        etas=bond_params.etas,
        R_offset=bond_params.R_offsets
    )
    elem_featurizer = ElementFeaturizer()
    
    print(f"键特征维度: {bond_featurizer.num_bond_features}")
    print(f"元素特征维度: {elem_featurizer.num_feature}")
    
    # 创建数据集
    print("\n" + "=" * 80)
    print("[4/6] 创建数据集")
    print("=" * 80)
    
    dataset = MaterialDataset(materials, bond_featurizer, elem_featurizer)
    print(f"总数据点: {len(dataset)}")
    print(f"每个材料平均目标值数: {len(dataset) / len(materials):.2f}")
    
    if len(dataset) == 0:
        print("错误: 没有创建数据点！检查材料加载。")
        sys.exit(1)
    
    # 分割数据并保存索引
    print("\n" + "=" * 80)
    print("[5/6] 分割数据集 (按电荷态分别分割)")
    print("=" * 80)
    
    # 按电荷态分组数据点
    charge_to_indices = {}
    for idx, data in enumerate(dataset.data_list):
        charge = data.charge.item() if isinstance(data.charge, torch.Tensor) else data.charge
        if charge not in charge_to_indices:
            charge_to_indices[charge] = []
        charge_to_indices[charge].append(idx)
    
    print(f"电荷态分布:")
    for charge in sorted(charge_to_indices.keys()):
        print(f"  q={charge}: {len(charge_to_indices[charge])} 个数据点")
    
    # 为每个电荷态分别进行 8:1:1 分割
    train_indices = []
    val_indices = []
    test_indices = []
    
    for charge in sorted(charge_to_indices.keys()):
        indices = charge_to_indices[charge]
        np.random.shuffle(indices)
        
        n_total = len(indices)
        n_train_q = int(0.8 * n_total)
        n_val_q = int(0.1 * n_total)
        n_test_q = n_total - n_train_q - n_val_q
        
        train_indices.extend(indices[:n_train_q])
        val_indices.extend(indices[n_train_q:n_train_q + n_val_q])
        test_indices.extend(indices[n_train_q + n_val_q:])
        
        print(f"  q={charge}: 训练 {n_train_q}, 验证 {n_val_q}, 测试 {n_test_q}")
    
    # 创建子集
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    
    # 保存分割索引以便后续绘制奇偶校验图
    train_indices_list = train_indices if isinstance(train_indices, list) else train_indices.tolist()
    val_indices_list = val_indices if isinstance(val_indices, list) else val_indices.tolist()
    test_indices_list = test_indices if isinstance(test_indices, list) else test_indices.tolist()
    
    split_info = {
        'train_indices': train_indices_list,
        'val_indices': val_indices_list,
        'test_indices': test_indices_list,
        'train_size': len(train_set),
        'val_size': len(val_set),
        'test_size': len(test_set),
        'split_method': 'stratified by charge state (q=0, q=1, q=2 separately)'
    }
    
    print(f"训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")

    
    # 创建数据加载器
    def collate_fn(batch):
        """自定义collate函数，正确处理target_site_indices"""
        # 使用PyG的默认批处理
        batched_data = Batch.from_data_list(batch)
        
        # 手动处理target_site_indices，将局部索引转换为全局索引
        if hasattr(batch[0], 'target_site_indices'):
            global_target_indices = []
            node_offset = 0
            
            for data in batch:
                # 将局部索引转换为全局索引
                local_indices = data.target_site_indices
                global_indices = local_indices + node_offset
                global_target_indices.append(global_indices)
                
                # 更新偏移量
                node_offset += data.x.size(0)
            
            # 连接所有全局索引
            batched_data.target_site_indices = torch.cat(global_target_indices, dim=0)
        
        return batched_data
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")
    
    # 创建模型
    print("\n" + "=" * 80)
    print("[6/6] 创建模型")
    print("=" * 80)
    
    print(f"模型配置:")
    print(f"  - 网络架构: ALIGNN (Atom Linearly Independent Geometric Network)")
    print(f"  - 嵌入维度: {args.embedding_dim}")
    print(f"  - 隐藏层维度: {args.hidden_dim}")
    print(f"  - 卷积层数: {args.num_conv}")
    print(f"  - Dropout 比率：{args.dropout_ratio} (不使用 Dropout)")
    print(f"  - 使用电荷：{args.use_charge}")
    if args.use_charge:
        print(f"  - 电荷嵌入维度: {args.charge_embed_dim}")
        print(f"  - 电荷集成: EARLY (在网络开始时)")
    print(f"  - 池化策略: SELECTIVE (仅在氧位点)")
    print(f"  - 损失函数: MAE (平均绝对误差)")
    print(f"  - 学习率调度器：ReduceLROnPlateau")
    print(f"    - factor: {args.reduce_lr_factor}")
    print(f"    - patience: {args.reduce_lr_patience}")
    print(f"    - threshold: {args.reduce_lr_threshold}")
    
    model = ALIGNNLightningChargeEarlySelective(
        num_atom_features=elem_featurizer.num_feature,
        num_bond_features=bond_featurizer.num_bond_features,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_conv_layers=args.num_conv,
        dropout_ratio=args.dropout_ratio,
        learning_rate=args.learning_rate,
        use_charge=args.use_charge,
        charge_embed_dim=args.charge_embed_dim,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_threshold=args.reduce_lr_threshold
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    
    # 训练
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="alignn-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        verbose=True
    )
    
    validation_callback = ValidationCallback(output_dir)
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, validation_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        precision="32"
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # 测试
    print("\n" + "=" * 80)
    print("测试模型")
    print("=" * 80)
    
    test_results = trainer.test(model, test_loader)
    print(f"测试结果: {test_results}")
    
    # 保存
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    
    model_path = output_dir / "model_final.ckpt"
    torch.save(model.state_dict(), model_path)
    print(f"模型保存到 {model_path}")
    
    normalizer_path = output_dir / "normalizer.json"
    normalizer.to_json_file(normalizer_path)
    print(f"Normalizer保存到 {normalizer_path}")
    
    # 保存分割索引信息
    split_info_path = output_dir / "split_indices.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"分割索引保存到 {split_info_path}")
    
    config_path = output_dir / "config.json"
    config = {
        "model_architecture": "ALIGNN",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_conv": args.num_conv,
        "dropout_ratio": args.dropout_ratio,
        "use_charge": args.use_charge,
        "charge_embed_dim": args.charge_embed_dim if args.use_charge else None,
        "charge_integration": "early" if args.use_charge else "none",
        "pooling_strategy": "selective (oxygen sites only)",
        "loss_function": "MAE (L1 Loss)",
        "total_materials": len(materials),
        "total_data_points": len(dataset),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n训练完成！结果保存到 {output_dir}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("训练总结")
    print("=" * 80)
    print(f"模型架构: ALIGNN (Atom Linearly Independent Geometric Network)")
    print(f"总材料数: {len(materials)}")
    print(f"总数据点: {len(dataset)}")
    print(f"训练集大小: {len(train_set)}")
    print(f"验证集大小: {len(val_set)}")
    print(f"测试集大小: {len(test_set)}")
    print(f"模型参数: {total_params:,}")
    print(f"电荷信息使用: {args.use_charge}")
    if args.use_charge:
        print(f"电荷嵌入维度: {args.charge_embed_dim}")
        print(f"电荷集成方式: EARLY (在卷积前加入)")
    print(f"池化策略: SELECTIVE (仅在氧位点提取特征)")
    print(f"损失函数: MAE (平均绝对误差)")
    print(f"学习率调度器：ReduceLROnPlateau")
    print(f"  - factor: {args.reduce_lr_factor}")
    print(f"  - patience: {args.reduce_lr_patience}")
    print(f"  - threshold: {args.reduce_lr_threshold}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
