"""数据加载子包 / Data loading subpackage.

当前仅包含 datasets (PyTorch Dataset 封装 + 公开数据集加载)。
特征工程已迁出至 tsf_frame.features.
"""

from .datasets.base_dataset import BaseDataset

__all__ = ['BaseDataset']
