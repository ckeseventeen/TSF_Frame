import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..base_model import BaseModel, ProbabilisticPrediction
# 复用 transformer_models 的训练循环 + MC Dropout 推理, 避免重复造轮子.
# 此前 MoiraiModel 自己写了一份 fit/_predict_probabilistic, 与 _dl_fit /
# _mc_dropout_predict 几乎完全相同, 已重构为继承 _DLBaseModel.
from ..transformer.transformer_models import _DLBaseModel


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
    'moirai': MoiraiModel
}


def get_moirai_model(model_name: str, config: Dict[str, Any]) -> BaseModel:
    if model_name not in MOIRAI_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MOIRAI_REGISTRY.keys())}")
    return MOIRAI_REGISTRY[model_name](config)
