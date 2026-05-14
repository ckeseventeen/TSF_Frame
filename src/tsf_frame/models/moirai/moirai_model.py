import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
from ..base_model import BaseModel, ProbabilisticPrediction
# 复用 transformer_models 的训练循环 + MC Dropout 推理, 避免重复造轮子.
# 此前 MoiraiModel 自己写了一份 fit/_predict_probabilistic, 与 _dl_fit /
# _mc_dropout_predict 几乎完全相同, 已重构为继承 _DLBaseModel.
from ..transformer.transformer_models import _DLBaseModel


class PretrainedMoiraiModel(BaseModel):
    """
    基于 Salesforce 官方 uni2ts 库的预训练 Moirai 零样本模型.
    
    采用 Wrapper 设计模式:
    - 兼容 BaseModel 的 fit / predict 接口.
    - fit() 函数不进行反向传播, 仅用来计算残差以兼容框架的经验置信区间.
    - predict() 自动进行 tensor 转换并调用官方预训练权重进行 Zero-shot 推理.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'moirai_zeroshot'
        self.size = config.get('moirai_size', 'small')  # 'small', 'base', 'large'
        
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            import os
            
            repo_id = f"Salesforce/moirai-1.0-R-{self.size}"
            pred_len = config.get('pred_len', 1)
            seq_len = config.get('seq_len', 24)
            # 默认假设输入的列全是预测目标列 (multi_dim)
            # 如果你有 covariates，需要修改 feat_dynamic_real_dim
            features_dim = config.get('num_features', 1)
            # resolve() 可以把路径转为绝对路径，避免相对路径带来的干扰
            _project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            cache_dir = os.path.join(_project_root, 'pretrained_models')
            os.makedirs(cache_dir, exist_ok=True)
            self.pretrained_model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(repo_id, cache_dir=cache_dir),
                prediction_length=pred_len,
                context_length=seq_len,
                patch_size='auto',
                num_samples=100,
                target_dim=features_dim,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            
            self.pretrained_model.to(self.device)
            # MoiraiForecast 返回的是分布预测器
            self.predictor = self.pretrained_model.create_predictor(batch_size=config.get('batch_size', 16))
            self._has_uni2ts = True
        except ImportError:
            self.pretrained_model = None
            self._has_uni2ts = False
            import logging
            logging.getLogger(__name__).warning(
                "无法导入 'uni2ts' 库，Moirai 零样本模型将无法工作。"
                "请运行: pip install uni2ts"
            )
            
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """BaseModel 要求实现 forward，零样本 Wrapper 直接透传官方调用."""
        if not self._has_uni2ts:
            raise RuntimeError("Missing uni2ts library.")
        return self.pretrained_model(x)

    def fit(self, train_data: Any, val_data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        零样本模型无需训练。并且由于我们重写了 probabilistic_predict 使用 Moirai 原生概率分布，
        此处不再需要计算残差，直接标记为已训练即可。
        """
        if not self._has_uni2ts:
            raise RuntimeError("Missing uni2ts library.")
            
        self._is_fitted = True
        return {'train_loss': [0.0], 'val_loss': [0.0]}

    def predict(self, test_data: Any, **kwargs) -> np.ndarray:
        """将 3D numpy array 输入转为 Moirai 所需格式进行零样本推理."""
        if not self._has_uni2ts:
            raise RuntimeError("Missing uni2ts library. Cannot predict.")
            
        X_test = test_data  # 期望 shape: (N, seq_len, features)
        N, seq_len, num_features = X_test.shape
        pred_len = self.config.get('pred_len', 1)
        
        preds = []
        # uni2ts 需要特别的数据迭代器或者直接通过内部模块调用，这里我们提供一个张量直接推理的简化实现
        # 注意: 官方 MoiraiForecast 推理通常需要传入 GluonTS 格式的 dataset
        # 为了兼容 Numpy, 我们直接提取其内部的 pytorch 模型调用逻辑
        self.pretrained_model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            # Moirai 输入通常需要 past_target, past_observed_values 等
            # 这里为适应 TSF_Frame 给出通用封装:
            # (官方接口较为复杂，此处提供简化张量推理，若使用 predictor 则需要构造 GluonTS ListDataset)
            # 针对张量：如果模型是 Module, 我们用内部方法. Moirai 默认处理一维时间序列(多变量需 flattening 或独立)
            
            # 简化版：使用 GluonTS ListDataset 兼容官方 predictor
            from gluonts.dataset.common import ListDataset
            import pandas as pd
            
            # 将 (N, seq_len, features) 转化为 N 个独立的 time series dict
            ds_list = []
            for i in range(N):
                # GluonTS expects shape (features, seq_len) or (seq_len,) for 1D
                target = X_test[i].T
                if num_features == 1:
                    target = target.flatten()
                
                ds_list.append({
                    "start": pd.Timestamp("2000-01-01"), # 任意起始时间
                    "target": target
                })
            dataset = ListDataset(ds_list, freq="M", one_dim_target=(num_features==1))
            
            # 覆盖 prediction_length
            self.pretrained_model.prediction_length = pred_len
            predictor = self.pretrained_model.create_predictor(batch_size=self.config.get('batch_size', 16))
            
            # 推理得到分布预测对象列表
            forecasts = list(predictor.predict(dataset))
            
            # 取 mean (shape: [pred_len, features])
            for f in forecasts:
                # 若是多变量，取平均或指定 target
                mean_pred = f.mean  # (pred_len, features) 或 (pred_len,)
                if mean_pred.ndim == 2:
                    # 我们框架预期返回单步或者多步的 target (默认最后一列或者指定列)
                    target_idx = self.config.get('target_idx', 0)
                    mean_pred = mean_pred[:, target_idx]
                preds.append(mean_pred)
                
        # 结果拼接为 (N, pred_len) 或 (N, 1)
        y_pred = np.array(preds) 
        return y_pred

    def probabilistic_predict(self, X_test: Any, quantiles: list = [0.1, 0.9], **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        重写 BaseModel 的 probabilistic_predict，直接提取 Moirai 原生的概率分布分位数，
        不再依赖残差计算。
        """
        if not self._has_uni2ts:
            raise RuntimeError("Missing uni2ts library. Cannot predict.")
            
        N, seq_len, num_features = X_test.shape
        pred_len = self.config.get('pred_len', 1)
        
        preds_mean, preds_lower, preds_upper = [], [], []
        
        self.pretrained_model.eval()
        with torch.no_grad():
            from gluonts.dataset.common import ListDataset
            import pandas as pd
            
            ds_list = []
            for i in range(N):
                target = X_test[i].T
                if num_features == 1:
                    target = target.flatten()
                
                ds_list.append({
                    "start": pd.Timestamp("2000-01-01"),
                    "target": target
                })
            dataset = ListDataset(ds_list, freq="M", one_dim_target=(num_features==1))
            
            self.pretrained_model.prediction_length = pred_len
            predictor = self.pretrained_model.create_predictor(batch_size=self.config.get('batch_size', 16))
            
            forecasts = list(predictor.predict(dataset))
            
            for f in forecasts:
                mean_pred = f.mean
                # 提取用户指定的分位数
                lower_pred = f.quantile(quantiles[0])
                upper_pred = f.quantile(quantiles[1])
                
                if mean_pred.ndim == 2:
                    target_idx = self.config.get('target_idx', 0)
                    mean_pred = mean_pred[:, target_idx]
                    lower_pred = lower_pred[:, target_idx]
                    upper_pred = upper_pred[:, target_idx]
                    
                preds_mean.append(mean_pred)
                preds_lower.append(lower_pred)
                preds_upper.append(upper_pred)
                
        y_pred = np.array(preds_mean)
        y_lower = np.array(preds_lower)
        y_upper = np.array(preds_upper)
        
        return y_pred, y_lower, y_upper


class PatchEmbedding(nn.Module):
    def __init__(self, input_size: int, d_model: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.proj = nn.Conv1d(input_size, d_model, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs):
    q = (q * freqs.cos()) + (rotate_half(q) * freqs.sin())
    k = (k * freqs.cos()) + (rotate_half(k) * freqs.sin())
    return q, k


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, freqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        if freqs is not None:
            freqs = freqs[None, None, :, :]
            q, k = apply_rotary_pos_emb(q, k, freqs)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MoiraiBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, freqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), freqs)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class MoiraiModel(_DLBaseModel):
    """
    Salesforce Moirai 风格的零样本 / 通用时序基础模型.

    继承 ``_DLBaseModel``: ``fit`` / ``predict`` / ``_predict_probabilistic``
    全部走 transformer_models 的共享实现 (``_dl_fit`` / ``_dl_predict`` /
    ``_mc_dropout_predict``), 子类只需实现 ``__init__`` 和 ``forward``.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'moirai'

        self.input_size = config.get('input_size', 1)
        self.d_model = config.get('d_model', 512)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 2048)
        self.dropout = config.get('dropout', 0.1)
        self.patch_size = config.get('patch_size', 16)
        self.output_size = config.get('output_size', 1)

        self.patch_embedding = PatchEmbedding(self.input_size, self.d_model, self.patch_size)
        self.rope = RotaryPositionalEmbedding(self.d_model)

        self.blocks = nn.ModuleList([
            MoiraiBlock(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.output_size)

        self.criterion = nn.MSELoss()
        # 把所有参数搬到目标设备, 必须在 optimizer 创建之前 (与 transformer 系列一致)
        self.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=config.get('learning_rate', 0.0001),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.patch_embedding(x)
        freqs = self.rope(x)
        for block in self.blocks:
            x = block(x, freqs)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

    # fit / predict / _predict_probabilistic 全部继承自 _DLBaseModel.
    # / Inherited from _DLBaseModel.


MOIRAI_REGISTRY = {
    'moirai': MoiraiModel,
    'moirai_zeroshot': PretrainedMoiraiModel
}


def get_moirai_model(model_name: str, config: Dict[str, Any]) -> BaseModel:
    if model_name not in MOIRAI_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MOIRAI_REGISTRY.keys())}")
    return MOIRAI_REGISTRY[model_name](config)

