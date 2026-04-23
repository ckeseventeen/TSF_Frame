"""
基础数据集模块 / Base dataset module

定义时序预测数据集的抽象基类，负责将 DataFrame 转换为滑动窗口序列。
Defines the abstract base class for time-series forecasting datasets,
converting DataFrames into sliding-window sequences.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    时序数据集抽象基类 / Abstract base class for time-series datasets

    将 pandas DataFrame 按滑动窗口切分为 (输入序列, 预测目标) 对，
    供 PyTorch DataLoader 使用。
    Slices a pandas DataFrame into (input sequence, prediction target) pairs
    via a sliding window, for use with PyTorch DataLoader.

    Args:
        data: 原始时序 DataFrame / Raw time-series DataFrame
        seq_len: 输入序列长度 / Input sequence length
        label_len: 标签序列长度（Transformer 解码器输入）/ Label length (for Transformer decoder)
        pred_len: 预测长度 / Prediction horizon
        target_cols: 目标列名列表 / Target column names
        feature_cols: 特征列名列表（默认使用全部列）/ Feature columns (defaults to all)
    """
    def __init__(self, data: pd.DataFrame, seq_len: int, label_len: int,
                 pred_len: int, target_cols: list, feature_cols: Optional[list] = None):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_cols = target_cols
        # feature_cols 为空时默认使用所有列作为特征
        # Default to all columns as features when feature_cols is not specified
        self.feature_cols = feature_cols if feature_cols else data.columns.tolist()

        # 预计算列索引,避免 __getitem__ 热路径重复查找 / Precompute column indices for hot-path speed
        # target_indices : data_values 中目标列的列号,供 __getitem__ 切片 y
        # feature_indices: data_values 中特征列的列号,供 __getitem__ 切片 X
        self.target_indices = [data.columns.get_loc(col) for col in target_cols]
        self.feature_indices = [data.columns.get_loc(col) for col in self.feature_cols]

        # 一次性转为 float32 numpy 数组,避免每次 __getitem__ 都做类型转换
        # Convert once to float32 ndarray; avoids per-sample dtype conversion in DataLoader
        self.data_values = data.values.astype(np.float32)

        # 校验数据长度是否满足序列窗口要求 / Validate data length against sequence window
        min_required = self.seq_len + self.pred_len
        if len(self.data) < min_required:
            raise ValueError(
                f"数据长度不足: 需要至少 {min_required} 行 "
                f"(seq_len={self.seq_len} + pred_len={self.pred_len})，"
                f"实际只有 {len(self.data)} 行。"
            )
    
    def __len__(self) -> int:
        """可用样本数 = 数据长度 - 输入窗口 - 预测窗口 + 1 / Available samples = data_len - seq_len - pred_len + 1"""
        return len(self.data) - self.seq_len - self.pred_len + 1

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """子类实现：按索引返回 (输入序列, 预测目标) / Subclass: return (input_seq, target) by index"""
        pass

    def get_data_stats(self) -> dict:
        """返回数据集统计摘要 / Return dataset statistics summary"""
        return {
            'num_samples': len(self),
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'num_features': len(self.feature_cols),
            'num_targets': len(self.target_cols),
            'data_mean': np.mean(self.data_values, axis=0).tolist(),
            'data_std': np.std(self.data_values, axis=0).tolist()
        }
