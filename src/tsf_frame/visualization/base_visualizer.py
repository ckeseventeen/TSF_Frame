from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ── 中文字体配置 ──────────────────────────────────────────────────────────────
# 按优先级尝试常见中文字体，确保在 Windows / Linux / macOS 均能显示中文
_CN_FONTS = [
    'SimHei',           # Windows 黑体
    'Microsoft YaHei',  # Windows 微软雅黑
    'PingFang SC',      # macOS
    'WenQuanYi Micro Hei',  # Linux
    'Noto Sans CJK SC', # Linux/通用
    'DejaVu Sans',      # 兜底（不支持中文但不报错）
]
matplotlib.rcParams['font.sans-serif'] = _CN_FONTS
matplotlib.rcParams['axes.unicode_minus'] = False   # 修复负号显示为方块的问题
# ─────────────────────────────────────────────────────────────────────────────


class BaseVisualizer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.save_dir = config.get('save_dir', './experiments/results')
        self.save_plots = config.get('save_plots', True)
        self.show_plots = config.get('show_plots', False)
        self.figure_size = config.get('figure_size', (12, 8))
        self.dpi = config.get('dpi', 100)
        
        if self.save_plots:
            os.makedirs(self.save_dir, exist_ok=True)
    
    @abstractmethod
    def plot_predictions(self, y_true: pd.DataFrame, y_pred: pd.DataFrame, 
                        title: str = 'Predictions vs Actual', **kwargs) -> Any:
        pass
    
    @abstractmethod
    def plot_metrics(self, metrics: Dict[str, float], 
                     title: str = 'Performance Metrics', **kwargs) -> Any:
        pass
    
    @abstractmethod
    def plot_training_history(self, history: Dict[str, list], 
                              title: str = 'Training History', **kwargs) -> Any:
        pass
    
    def get_save_path(self, filename: str) -> str:
        return os.path.join(self.save_dir, filename)
