#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Geometric版本的数据处理模块
"""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from cgcnn.material import Material
from cgcnn.featurizer import ElementFeaturizer, GaussianBasis


class MaterialToPyGTransform:
    """将Material对象转换为PyTorch Geometric Data对象"""
    
    def __init__(self, bond_featurizer: GaussianBasis, elem_featurizer: ElementFeaturizer):
        self.bond_featurizer = bond_featurizer
        self.elem_featurizer = elem_featurizer
    
    def __call__(self, material: Material) -> Data:
        """转换单个材料为PyG Data对象"""
        
        # 获取原子特征
        atom_features = []
        for atom in material.structure:
            elem_feat = self.elem_featurizer.featurize(atom.specie)
            atom_features.append(elem_feat)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # 使用bond_featurizer处理整个结构
        try:
            bond_features, bond_indices = self.bond_featurizer.apply(material.structure)
            
            # 构建边索引和边特征
            edge_indices = []
            edge_features = []
            
            for i, (bond_feat_row, bond_idx_row) in enumerate(zip(bond_features, bond_indices)):
                for j, (bond_feat, bond_idx) in enumerate(zip(bond_feat_row, bond_idx_row)):
                    edge_indices.append([i, bond_idx])
                    edge_features.append(bond_feat.tolist())
            
            # 转换为张量
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                # 如果没有边，创建空的边索引和特征
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, self.bond_featurizer.num_bond_features), dtype=torch.float)
                
        except Exception as e:
            print(f"Warning: Failed to process bonds for {material.formula}: {e}")
            # 创建空的边索引和特征
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.bond_featurizer.num_bond_features), dtype=torch.float)
        
        # 目标值
        targets = []
        charges = []
        for target_val, charge in zip(material.target_vals, material.charges):
            targets.append(target_val)
            charges.append(charge)
        
        y = torch.tensor(targets, dtype=torch.float)
        charge = torch.tensor(charges, dtype=torch.long)
        
        # 创建PyG Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            charge=charge,
            formula=material.formula,
            num_nodes=len(material.structure)
        )
        
        return data


class MaterialDatasetPyG(Dataset):
    """PyTorch Geometric版本的材料数据集"""
    
    def __init__(
        self, 
        materials: List[Material], 
        bond_featurizer: GaussianBasis,
        elem_featurizer: ElementFeaturizer,
        transform: Optional[BaseTransform] = None,
        pre_transform: Optional[BaseTransform] = None
    ):
        self.materials = materials
        self.bond_featurizer = bond_featurizer
        self.elem_featurizer = elem_featurizer
        
        # 创建转换器
        self.material_transform = MaterialToPyGTransform(bond_featurizer, elem_featurizer)
        
        super().__init__(transform=transform, pre_transform=pre_transform)
    
    def len(self) -> int:
        return len(self.materials)
    
    def get(self, idx: int) -> Data:
        """获取单个数据点"""
        material = self.materials[idx]
        data = self.material_transform(material)
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data


class PyGDataModule:
    """PyTorch Geometric数据模块"""
    
    def __init__(
        self,
        materials: List[Material],
        bond_featurizer: GaussianBasis,
        elem_featurizer: ElementFeaturizer,
        batch_size: int = 32,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False
    ):
        self.materials = materials
        self.bond_featurizer = bond_featurizer
        self.elem_featurizer = elem_featurizer
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        # 分割数据集
        self._split_dataset()
    
    def _split_dataset(self):
        """分割数据集"""
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(self.materials))
        
        n_val = int(len(self.materials) * self.val_ratio)
        n_test = int(len(self.materials) * self.test_ratio)
        n_train = len(self.materials) - n_val - n_test
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        self.train_materials = [self.materials[i] for i in train_indices]
        self.val_materials = [self.materials[i] for i in val_indices]
        self.test_materials = [self.materials[i] for i in test_indices]
        
        print(f"Dataset split:")
        print(f"  Train: {len(self.train_materials)} samples")
        print(f"  Val:   {len(self.val_materials)} samples")
        print(f"  Test:  {len(self.test_materials)} samples")
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        dataset = MaterialDatasetPyG(
            self.train_materials, 
            self.bond_featurizer, 
            self.elem_featurizer
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True if len(dataset) > self.batch_size else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        dataset = MaterialDatasetPyG(
            self.val_materials, 
            self.bond_featurizer, 
            self.elem_featurizer
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        dataset = MaterialDatasetPyG(
            self.test_materials, 
            self.bond_featurizer, 
            self.elem_featurizer
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )


def collate_materials_pyg(batch: List[Data]) -> Data:
    """
    自定义的批处理函数，用于处理不同大小的图
    PyTorch Geometric的默认DataLoader已经处理了这个问题
    """
    # PyG的DataLoader会自动调用Batch.from_data_list
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)