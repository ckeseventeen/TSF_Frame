"""
MLPModel — 横截面 / 表格回归 / Cross-section & tabular regression
====================================================================

为什么需要 MLP 而不是用框架现有的 5 个时序 DL 模型?

  框架现有的 LSTM / Transformer / Autoformer / iTransformer / TimesNet
  都是为 L > 1 时序设计的:
    - LSTM 在 L=1 时只跑一次 cell, 循环优势归零
    - Transformer self-attention 在 L=1 时矩阵是 1×1 ≡ 恒等, attention 失效
    - Autoformer SeriesDecomp(MovingAvg) 在 L 太短时退化
    - TimesNet FFT 周期检测 L<8 无意义
  → 横截面任务(每个 (B, F) 样本就一个时间步)上, 它们都不合适.

  MLP 是表格数据的工业标准 DL baseline:
    - 多层 Linear + ReLU + Dropout, 支持 MC Dropout 概率预测
    - 支持 (B, F) 2D 输入(纯横截面)和 (B, L, C) 3D 输入(时序+降维)
    - 不需要 RevIN (instance norm 在 L=1 上无意义), 默认关
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ..transformer.transformer_models import _DLBaseModel


class MLPModel(_DLBaseModel):
    """
    MLP / Multi-Layer Perceptron for cross-section and tabular regression.

    支持两种输入形态 / Supports two input shapes:
      - 2D ``(B, F)`` — 纯横截面回归 (income/spending vs age/industry/education ...)
      - 3D ``(B, L, C)`` — 时序输入按 ``mlp_reduce`` 降到 2D 再回归

    Config:
        input_size:    特征数 (= F 或 C, 由 mlp_reduce 决定)
        output_size:   输出维度 (单目标 1, 多目标 = num_targets)
        hidden_size:   隐藏层宽度, 默认 128
        num_layers:    Linear+ReLU+Dropout 层数, 默认 2 (即两个 hidden)
        dropout:       Dropout 率, 默认 0.1; >0 时支持 MC Dropout 概率预测
        mlp_reduce:    3D → 2D 降维策略, 仅 3D 输入时生效:
                       - 'last'    : 取最后一步 (B, L, C) → (B, C) (默认, 等价 RNN 末位 pooling)
                       - 'mean'    : 沿 L 维平均 (B, L, C) → (B, C)
                       - 'flatten' : (B, L, C) → (B, L*C); 此时 input_size 必须 = L*C
        learning_rate: Adam lr, 默认 1e-3
        use_revin:     横截面任务默认 False (instance norm 在 L=1 上无意义);
                       若 use_revin=True 且输入是 3D, 走标准 RevIN 路径.

    Example::

        # 横截面: 8 列特征预测收入
        cfg = {'input_size': 8, 'output_size': 1, 'hidden_size': 64,
               'num_layers': 3, 'dropout': 0.1}
        model = MLPModel(cfg)
        model.fit((X_2d, y_1d), val_data=(X_val, y_val))
        prob = model.predict_probabilistic(X_test)   # MC Dropout 区间
    """

    VALID_REDUCE = ('last', 'mean', 'flatten')

    def __init__(self, config: Dict[str, Any]):
        # MLP 在横截面上 RevIN 无意义, 默认关 (用户传 use_revin=True 仍可启用走 3D 路径)
        config = {'use_revin': False, **config}
        super().__init__(config)
        self.model_name = 'mlp'
        self.input_size = int(config.get('input_size', 1))
        self.output_size = int(config.get('output_size', 1))
        self.hidden_size = int(config.get('hidden_size', 128))
        self.num_layers = int(config.get('num_layers', 2))
        self.dropout_rate = float(config.get('dropout', 0.1))
        self.mlp_reduce = str(config.get('mlp_reduce', 'last'))
        if self.mlp_reduce not in self.VALID_REDUCE:
            raise ValueError(
                f"MLPModel: mlp_reduce={self.mlp_reduce!r}, 必须是 {self.VALID_REDUCE}"
            )

        # 构造 Linear → ReLU → Dropout 堆叠, num_layers 表示隐藏层数
        # / Build hidden stack
        layers = []
        in_dim = self.input_size
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))
            in_dim = self.hidden_size
        layers.append(nn.Linear(in_dim, self.output_size))
        self.mlp = nn.Sequential(*layers)

        # 可选 RevIN (默认关, 见 __init__ 顶部)
        # / Optional RevIN — off by default for cross-section
        self._init_revin(num_features=self.input_size)

        self.criterion = nn.MSELoss()
        # to(device) 必须在 optimizer 创建之前
        self.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config.get('learning_rate', 1e-3),
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: ``(B, F)`` 2D 或 ``(B, L, C)`` 3D.
               3D 时按 ``mlp_reduce`` 降维到 (B, F).

        Returns:
            ``(B, output_size)``
        """
        if x.dim() == 3:
            if self.revin is not None:
                # 3D + RevIN: 走标准 norm/denorm 路径 (与其他 DL 模型一致)
                x = self._maybe_revin_norm(x)
            if self.mlp_reduce == 'last':
                x_2d = x[:, -1, :]                              # (B, C)
            elif self.mlp_reduce == 'mean':
                x_2d = x.mean(dim=1)                            # (B, C)
            else:  # flatten
                x_2d = x.contiguous().view(x.size(0), -1)       # (B, L*C)
        elif x.dim() == 2:
            x_2d = x
        else:
            raise ValueError(
                f"MLPModel.forward: 不支持的输入维度 {x.dim()}D, 期望 2D 或 3D."
            )

        out = self.mlp(x_2d)
        # RevIN 反归一化只在 3D 路径下有意义 (单目标默认 channel 0)
        if x.dim() == 3 and self.revin is not None:
            out = self._maybe_revin_denorm_target(out)
        return out


# ─── 注册到 DL_MODEL_REGISTRY ─────────────────────────────────────────────
# 延迟 import 避免循环依赖 — 由 models/__init__.py 触发
def _register():
    from ..transformer.transformer_models import DL_MODEL_REGISTRY
    DL_MODEL_REGISTRY['mlp'] = MLPModel


_register()
