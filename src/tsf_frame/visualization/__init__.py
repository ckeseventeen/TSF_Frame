"""TSF_Frame 可视化模块 / Visualization toolkit.

公开:
    PredictionPlotter — 项目统一的画图工具 (原子方法 + 复合工具)
    DEFAULT_PALETTE   — 通用调色板
    _CN_FONTS         — 中文字体列表 (import 即触发 matplotlib rcParams 全局生效)
"""
from .base_visualizer import _CN_FONTS, BaseVisualizer
from .prediction_plotter import DEFAULT_PALETTE, PredictionPlotter

__all__ = [
    'PredictionPlotter',
    'DEFAULT_PALETTE',
    'BaseVisualizer',  # 保留以防外部继承,不再推荐使用
    '_CN_FONTS',
]
