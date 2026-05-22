"""
tsf_frame.models.tabular — 表格/横截面回归模型.

Transformer 系列的 5 个时序 DL 模型在 L=1 横截面任务下退化无意义;
本子包提供合适的替代——MLP, 接受 (B, F) 2D 输入直接回归.
"""

from .mlp_model import MLPModel

__all__ = ['MLPModel']
