"""
深度学习时序预测模型 / Deep learning time-series forecasting models

提供五种模型，所有模型共享 _dl_fit/_dl_predict 训练/推理函数，
以及 _mc_dropout_predict MC Dropout 概率预测。
Provides five models, all sharing _dl_fit/_dl_predict for training/inference,
and _mc_dropout_predict for MC Dropout probabilistic prediction.

  - LSTMModel       : LSTM，含独立 Dropout 层以支持 MC Dropout / LSTM with standalone Dropout for MC Dropout
  - TransformerModel: 标准 Transformer Encoder / Standard Transformer Encoder
  - Autoformer      : NeurIPS 2021，序列分解 + FFT 自相关注意力 / Series decomposition + FFT auto-correlation
  - iTransformer    : ICLR 2024，逆向 Transformer，变量维度注意力 / Inverted Transformer, variate-wise attention
  - TimesNet        : ICLR 2023，1D→2D 时序建模 + 多尺度卷积 / 1D-to-2D temporal modeling + multi-scale convolution
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..base_model import BaseModel, ProbabilisticPrediction


# ─── 共享工具 ─────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    正弦位置编码 / Sinusoidal positional encoding

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return self.dropout(x + self.pe[:, :x.size(1)])


def _dl_fit(model: 'BaseModel', train_data, val_data, config: dict) -> dict:
    """
    所有深度学习模型共享的训练循环 / Shared training loop for DL models.

    流程 / Workflow:
      1. 从 config 读取 epoch 数、batch 大小、device
      2. 一次性把全部训练数据上传到 device(适合中小数据集;超大数据应改 DataLoader)
         Upload all training data to device at once (suited for small/mid data).
      3. 每个 epoch:
         - 随机打乱索引(防止样本顺序偏差),按 batch_size 切片
         - 前向 → 计算 MSE loss → 反向 → 梯度裁剪(norm<=1.0) → 优化器一步
           Shuffle, forward, backprop, gradient clipping (max norm=1.0), step.
         - loss 按 batch 数平均(num_batches = ceil(N/B)),而非按 sample 平均,
           以保持与 validation loss 的同量纲可比。
      4. 若提供 val_data,在 eval 模式下无梯度计算验证集 loss
      5. 记录到 history,打印进度

    约束 / Constraints:
      - 模型必须已定义 self.optimizer 和 self.criterion 属性
      - 梯度裁剪阈值硬编码为 1.0(常见稳定默认值;如需可改为 config 驱动)

    Args:
        model:      待训练模型,需要 optimizer/criterion 属性 / Model with optimizer and criterion.
        train_data: (X_train, y_train) numpy 数组元组 / Training tuple.
        val_data:   可选验证元组 / Optional validation tuple.
        config:     配置字典 / Config dict with train_epochs/batch_size/device.

    Returns:
        训练历史 {'train_loss': [...], 'val_loss': [...]}
        Training history dict.
    """
    epochs = config.get('train_epochs', 100)
    batch_size = config.get('batch_size', 32)
    device = config.get('device', 'cpu')

    X_train, y_train = train_data
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)

    history = {'train_loss': [], 'val_loss': []}

    n_train = len(X_train)
    for epoch in range(epochs):
        model.train()
        # total_loss 现在累计 "loss * batch_size" — 即未取平均的样本损失之和;
        # 末轮按总样本数 n_train 取平均, 与 val_loss (一次性 forward 全部 X_val
        # 走 MSELoss / PinballLoss 默认 reduction='mean') 真正同量纲.
        # 旧实现 `total_loss += loss.item(); avg = total_loss / num_batches`
        # 在 n_train % batch_size != 0 时, 尾批样本数偏小却被等权计入, 导致训练
        # loss 估计有偏 (尾批权重虚高). 现改为样本加权平均.
        # / Sample-weighted loss average: aligns with val_loss scale and removes
        #   the tail-batch over-weighting bias.
        total_loss = 0.0
        # 打乱样本顺序,避免同一 mini-batch 里样本始终相邻带来的偏差
        # Shuffle sample order to remove position bias across epochs.
        indices = torch.randperm(n_train)

        for i in range(0, n_train, batch_size):
            batch_idx = indices[i: i + batch_size]
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            bsz = batch_x.size(0)

            model.optimizer.zero_grad()
            outputs = model(batch_x)
            loss = model.criterion(outputs, batch_y)
            loss.backward()
            # 梯度裁剪:按 L2 范数限制总梯度 <= 1.0,防止梯度爆炸(Transformer/RNN 常见)
            # Clip gradient L2 norm to prevent explosion (common in Transformer/RNN).
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.optimizer.step()
            # 关键: 乘 bsz 把"批均损失"还原成"样本损失之和"
            total_loss += loss.item() * bsz

        # 按总样本数取平均, 与 val_loss (默认 reduction='mean') 同量纲
        # / Per-sample average — matches val_loss scale
        avg_loss = total_loss / n_train
        history['train_loss'].append(avg_loss)

        if val_data is not None:
            model.eval()
            X_val, y_val = val_data
            X_val = torch.FloatTensor(X_val).to(device)
            y_val = torch.FloatTensor(y_val).to(device)
            with torch.no_grad():
                val_loss = model.criterion(model(X_val), y_val).item()
            history['val_loss'].append(val_loss)
            print(f'Epoch {epoch+1}/{epochs}  Train: {avg_loss:.4f}  Val: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{epochs}  Train: {avg_loss:.4f}')

    return history


def _dl_predict(model: 'BaseModel', test_data, device: str) -> np.ndarray:
    """
    所有深度学习模型共享的点预测推理 / Shared deterministic inference for DL models.

    切换到 eval 模式(关闭 Dropout、BN 使用 running stats),torch.no_grad
    避免构建计算图节省显存,最后转回 numpy 方便与下游 sklearn/pandas 交互。
    Switches to eval mode, disables autograd, returns numpy for downstream use.

    Args:
        model:     已训练的深度学习模型 / Trained DL model.
        test_data: np.ndarray 或 (X, y) 元组 / Array or tuple.
        device:    设备标识 / Device string ('cpu'/'cuda').

    Returns:
        预测数组,形状 (N, output_size)。
    """
    model.eval()
    X = test_data[0] if isinstance(test_data, tuple) else test_data
    X = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        return model(X).cpu().numpy()


def _mc_dropout_predict(
    model: 'BaseModel', test_data, device: str,
    num_samples: int, confidence_level: float
) -> ProbabilisticPrediction:
    """
    MC Dropout 概率预测 / Monte-Carlo Dropout probabilistic prediction.

    原理 / Principle:
      以 Dropout 作为贝叶斯近似,推理时保持 Dropout 开启,对同一输入
      重复采样 num_samples 次得到预测分布,再用经验分位数构造置信区间。
      Treat Dropout as Bayesian approximation; sample predictions multiple
      times and build confidence intervals from empirical quantiles.

    步骤 / Steps:
      1. 调用 model.train() 启用所有 Dropout 层(nn.Dropout/Dropout2d 会
         在 train 模式下生效,eval 模式下会被关闭)
         Enable Dropout via train() mode.
      2. torch.no_grad() 禁用梯度,循环采样 num_samples 次
         Sample num_samples times without gradients.
      3. 计算: mean = S 次采样的均值
                std  = S 次采样的标准差
                lower/upper = (alpha/2, 1-alpha/2) 分位数
                alpha = 1 - confidence_level (如 0.05 对应 95% CI)
         Compute mean, std, and empirical quantiles.

    注意 / Note:
      - LSTMModel 的独立 dropout_layer 保证单层 LSTM 也有随机性
        (PyTorch LSTM 参数 dropout 只在 num_layers > 1 时生效)
        LSTMModel adds an extra Dropout to ensure MC Dropout works with 1 layer.
      - preds 数组形状 (S, N, output_size),S 可能占用较大显存,
        大数据集推理时需注意 num_samples 设置
        preds shape is (S, N, output_size); watch memory for large S/N.

    Args:
        model:            已训练模型 / Trained model.
        test_data:        测试数据 / Test data.
        device:           设备 / Device.
        num_samples:      采样次数 S / Number of samples.
        confidence_level: 置信水平(如 0.95) / Confidence level (e.g. 0.95).

    Returns:
        ProbabilisticPrediction(mean, lower, upper, std, samples)
    """
    X = test_data[0] if isinstance(test_data, tuple) else test_data
    X = torch.FloatTensor(X).to(device)
    model.train()  # 激活 Dropout / Activate Dropout layers
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            preds.append(model(X).cpu().numpy())
    preds = np.array(preds)            # (S, N, output_size)
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    # 双侧分位数: alpha/2 与 1 - alpha/2 / Two-tailed quantiles
    alpha = 1 - confidence_level
    lower = np.percentile(preds, alpha / 2 * 100, axis=0)
    upper = np.percentile(preds, (1 - alpha / 2) * 100, axis=0)
    return ProbabilisticPrediction(mean=mean, lower=lower, upper=upper, std=std, samples=preds)


# ─── DL 模型公用基类 ──────────────────────────────────────────────────────────

class _DLBaseModel(BaseModel):
    """
    深度学习模型公用中间基类 / Shared intermediate base for DL models.

    抽走 5 个 Transformer 家族模型完全一致的训练/推理样板:
      - fit:                 _dl_fit 训练循环
      - predict:             _dl_predict 点预测
      - _predict_probabilistic: MC Dropout(默认) 或 fallback 到点预测

    使用方式 / Usage:
        class MyModel(_DLBaseModel):
            def __init__(self, config): ...
            def forward(self, x): ...
        # 无需再写 fit/predict/_predict_probabilistic

    如需分位数等非 MC Dropout 概率方法,子类 override _predict_probabilistic 即可
    (例: DLinear 仅 override predict 和 _predict_probabilistic,复用 fit)。
    Override _predict_probabilistic in subclass if a different probabilistic
    method is needed (e.g. DLinear uses quantile regression and overrides
    predict / _predict_probabilistic while reusing fit).
    """

    def fit(self, train_data, val_data=None, **kwargs) -> dict:
        return _dl_fit(self, train_data, val_data, self.config)

    def predict(self, test_data, **kwargs) -> np.ndarray:
        return _dl_predict(self, test_data, self.device)

    def _predict_probabilistic(self, test_data, **kwargs) -> ProbabilisticPrediction:
        if self.probabilistic_method == 'mc_dropout':
            return _mc_dropout_predict(
                self, test_data, self.device, self.num_samples, self.confidence_level
            )
        return ProbabilisticPrediction(mean=self.predict(test_data))


# ─── LSTM ─────────────────────────────────────────────────────────────────────

class LSTMModel(_DLBaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'lstm'
        self.input_size = config.get('input_size', 1)
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout_rate = config.get('dropout', 0.2)
        self.output_size = config.get('output_size', 1)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            # PyTorch 在单层时不允许 dropout，这里正确处理 这里是层间dropout
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
        )
        # 独立 Dropout 层：单层 LSTM 时仍可进行 MC Dropout 采样
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.criterion = nn.MSELoss()
        # 把所有参数搬到目标设备, 必须在 optimizer 创建之前
        # 否则 optimizer 绑定的是 CPU 参数引用, 即使后续 .to(cuda) 也指不回来
        # / Move params to device BEFORE optimizer; optimizer binds to tensor references.
        self.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config.get('learning_rate', 0.001)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout_layer(out[:, -1, :]))


# ─── Transformer ──────────────────────────────────────────────────────────────

#encoder_only
class TransformerModel(_DLBaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'transformer'
        self.input_size = config.get('input_size', 1)
        self.d_model = config.get('d_model', 64)            # 512 → 64
        self.nhead = config.get('nhead', 4)                  # 8 → 4
        self.num_layers = config.get('num_layers', 2)        # 6 → 2
        self.dim_feedforward = config.get('dim_feedforward', 128)  # 2048 → 128
        self.dropout_rate = config.get('dropout', 0.1)
        self.output_size = config.get('output_size', 1)

        # 加一个断言，d_model必须能被nhead整除
        assert self.d_model % self.nhead == 0, \
            f"d_model({self.d_model}) 必须能被 nhead({self.nhead}) 整除"

        self.input_proj = nn.Linear(self.input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )
        self.decoder = nn.Linear(self.d_model, self.output_size)

        self.criterion = nn.MSELoss()
        # 把所有参数搬到目标设备, 必须在 optimizer 创建之前 (见 LSTMModel 注释)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config.get('learning_rate', 0.001)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播 / Forward pass

        Args:
            x: 输入张量,形状 (batch_size, seq_len, input_size)
               Input tensor of shape (B, T, C_in).

        Returns:
            预测张量,形状 (batch_size, output_size),取最后一步的 hidden 过线性层。
            Output tensor of shape (B, C_out), decoded from the last time step.

        Note (encoder-only, no causal mask):
            本类是 *encoder-only* + 末位 pooling (取 ``x[:, -1, :]`` 过 decoder)。
            历史窗口在 forward 入口已经全部可见, 编码器内部应该走 **双向自注意力**
            充分混合上下文; 因此**不传** ``mask=causal_mask``:
              * 输入窗口的全部时间步对模型来说都是已知历史, 不存在"未来泄漏"
              * 加上三角 mask 后, 第 1 步看不到第 2..T 步, 削弱了表达能力,
                间接污染末位 hidden state
              * Causal mask 只在 *autoregressive decoder* 场景下需要 (本类不是)
        """
        x = self.pos_encoder(self.input_proj(x))
        # 双向自注意力: 不传 mask, 让历史窗口的所有时间步互相可见
        # / Bidirectional self-attention over the already-known history
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])


# ─── Autoformer ───────────────────────────────────────────────────────────────
# 论文: NeurIPS 2021 — Wu et al.
# 核心: 序列分解（趋势+季节）+ FFT 自相关注意力

class MovingAvg(nn.Module):
    """移动平均提取趋势分量，自动对称填充以保持序列长度。"""
    def __init__(self, kernel_size: int):
        super().__init__()
        # 保证 kernel_size 为奇数
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].expand(-1, pad, -1)
        end = x[:, -1:, :].expand(-1, pad, -1)
        x_padded = torch.cat([front, x, end], dim=1)
        return self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L, D)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelationLayer(nn.Module):
    """
    自相关注意力（Auto-Correlation）。
    使用 FFT 计算 Q-K 跨时间步相关性，选取 top-k 时延聚合 V。
    """
    def __init__(self, d_model: int, n_heads: int, factor: int = 1, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须整除 n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor   # 控制top-k时延数量的系数
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Auto-Correlation 前向计算 / Auto-Correlation forward pass.

        核心步骤 / Key steps:
          1. Q/K/V 多头投影,并 permute 到 (B, H, L, Dh) 便于在时间维做 FFT
             Multi-head projection then reshape to (B, H, L, Dh).
          2. 用 FFT 快速计算 Q 与 K 在所有时延 τ 上的互相关:
                corr[τ] = iFFT(FFT(Q) * conj(FFT(K)))
             物理含义: corr[τ] 越大表示序列在时延 τ 上的周期性越强。
             FFT-based cross-correlation: corr[τ] = iFFT(Q_f * conj(K_f)).
          3. 对通道维(Dh)取均值得到 (B, H, L) 的标量相关谱,
             取 top-k 个最强时延作为候选周期(top_k = factor * ln(L+1))。
             Select top-k delays from the correlation spectrum.
          4. 对 top-k 权重做 softmax 归一化,构造权重向量。
          5. 按 top-k 时延对 V 做循环移位(gather 索引 (t - delay) mod L),
             然后加权求和得到注意力输出:
                out[t] = Σ_i softmax(weights)[i] * V[(t - delays[i]) mod L]
             Roll V by each delay and sum weighted contributions.
          6. 合并多头 → 输出投影 → Dropout。

        Args:
            q, k, v: (B, L, D) 三个张量 / Query, key, value tensors of shape (B, L, D).

        Returns:
            注意力输出,形状 (B, L, D) / Attention output (B, L, D).
        """
        B, L, D = q.shape
        H, Dh = self.n_heads, self.d_head

        # 多头投影 + 时间维前置,为后续 FFT 做准备
        # Multi-head projection; put L in dim=2 for FFT.
        Q = self.q_proj(q).view(B, L, H, Dh).permute(0, 2, 1, 3)  # (B,H,L,Dh)
        K = self.k_proj(k).view(B, L, H, Dh).permute(0, 2, 1, 3)
        V = self.v_proj(v).view(B, L, H, Dh).permute(0, 2, 1, 3)

        # FFT 自相关: 频域相乘再反变换 == 时域卷积 / circular correlation
        # FFT cross-correlation: corr[b,h,τ,d] = Σ_t Q[t]*K[t-τ]
        Q_f = torch.fft.rfft(Q, n=L, dim=2)   # (B,H,L//2+1,Dh)
        K_f = torch.fft.rfft(K, n=L, dim=2)
        corr = torch.fft.irfft(Q_f * torch.conj(K_f), n=L, dim=2)  # (B,H,L,Dh)

        # 选 top-k 时延: 数量随 ln(L) 增长,避免长序列下 top_k 爆炸
        # Number of top delays grows as ln(L) to avoid blow-up on long sequences.
        top_k = max(1, int(self.factor * math.log(L + 1)))
        corr_mean = corr.mean(dim=-1)                         # (B,H,L) 按通道平均
        weights, delays = torch.topk(corr_mean, top_k, dim=2) # (B,H,top_k) 最强 k 个时延
        weights = torch.softmax(weights, dim=-1)               # 归一化作为聚合权重

        # 时延聚合: 把 V 按每个时延做循环移位,再按权重加权求和
        # rolled[b,h,t] = V[b,h,(t-delay)%L]; sum over top_k delays.
        out = torch.zeros_like(V)
        t_idx = torch.arange(L, device=Q.device).view(1, 1, L, 1).expand(B, H, L, Dh)
        for i in range(top_k):
            d_i = delays[:, :, i].view(B, H, 1, 1).expand(B, H, L, Dh)
            # 循环索引: (t - d_i) mod L,注意 d_i 已在 [0, L) 不必再 mod,
            # 但显式 `d_i % L` 防御 factor 调大导致边缘越界。
            # Circular index with defensive modulo.
            gather_idx = (t_idx - d_i % L + L) % L
            rolled = torch.gather(V, 2, gather_idx)
            out += weights[:, :, i].view(B, H, 1, 1) * rolled

        # 合并多头: (B,H,L,Dh) → (B,L,D)
        # Merge heads back.
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.dropout(self.out_proj(out))


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 kernel_size: int, factor: int = 1, dropout: float = 0.1):
        super().__init__()
        self.auto_corr = AutoCorrelationLayer(d_model, n_heads, factor, dropout)
        self.decomp1 = SeriesDecomp(kernel_size)   # 第一次序列分解
        self.decomp2 = SeriesDecomp(kernel_size)   # 第二次序列分解
        # 前馈网络FFN，与Transformer一致
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.auto_corr(x, x, x)
        x, _ = self.decomp1(x + self.dropout(attn))
        ff_out = self.ff(x)
        x, _ = self.decomp2(x + self.dropout(ff_out))
        return x


class Autoformer(_DLBaseModel):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation
    NeurIPS 2021 — Wu et al.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'autoformer'
        self.input_size = config.get('input_size', 1)
        self.d_model = config.get('d_model', 256)
        self.n_heads = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 3)
        self.d_ff = config.get('dim_feedforward', 1024)
        self.dropout_rate = config.get('dropout', 0.1)
        self.output_size = config.get('output_size', 1)
        self.kernel_size = config.get('moving_avg_kernel', 25)
        self.factor = config.get('autocorr_factor', 1)

        self.input_proj = nn.Linear(self.input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout_rate)
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(
                self.d_model, self.n_heads, self.d_ff,
                self.kernel_size, self.factor, self.dropout_rate
            )
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.d_model)
        self.projection = nn.Linear(self.d_model, self.output_size)

        self.criterion = nn.MSELoss()
        # 把所有参数搬到目标设备, 必须在 optimizer 创建之前 (见 LSTMModel 注释)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config.get('learning_rate', 0.001)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.pos_encoder(self.input_proj(x))
        for layer in self.encoder_layers:
            x = layer(x)
        return self.projection(self.norm(x[:, -1, :]))


# ─── iTransformer ─────────────────────────────────────────────────────────────
# 论文: ICLR 2024 — Liu et al.
# 核心: 将每个变量的时序作为 token，在变量维度做注意力

class iTransformer(_DLBaseModel):
    """
    iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    ICLR 2024 — Liu et al.

    架构: 将 (B, L, N) 的输入转置为 (B, N, L)，把每个变量的 L 步历史
    投影为 d_model 维嵌入，在变量（N）维度做 Transformer 注意力，
    最后将所有变量的嵌入展平后映射到 output_size。
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'itransformer'
        self.input_size = config.get('input_size', 1)   # N 变量数
        self.seq_len = config.get('seq_len', 96)         # L 回看长度
        self.d_model = config.get('d_model', 256)
        self.n_heads = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 3)
        self.d_ff = config.get('dim_feedforward', 512)
        self.dropout_rate = config.get('dropout', 0.1)
        self.output_size = config.get('output_size', 1)

        # 每个变量的时序嵌入: L → d_model
        self.variate_proj = nn.Linear(self.seq_len, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_rate, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.d_model)

        # 将所有变量嵌入展平后映射到预测输出
        self.output_proj = nn.Linear(self.input_size * self.d_model, self.output_size)

        self.criterion = nn.MSELoss()
        # 把所有参数搬到目标设备, 必须在 optimizer 创建之前 (见 LSTMModel 注释)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config.get('learning_rate', 0.001)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (B, L, N)
        B, L, N = x.shape
        # 逆转：每个变量的 L 步历史 → 嵌入
        x_inv = x.permute(0, 2, 1)          # (B, N, L)
        x_emb = self.variate_proj(x_inv)    # (B, N, d_model)
        # 在 N 变量维度做注意力
        x_enc = self.norm(self.transformer(x_emb))  # (B, N, d_model)
        # 展平所有变量嵌入，投影到输出
        out = x_enc.contiguous().view(B, -1)         # (B, N * d_model)
        return self.output_proj(out)                  # (B, output_size)


# ─── TimesNet ─────────────────────────────────────────────────────────────────
# 论文: ICLR 2023 — Wu et al.
# 核心: FFT 检测主要周期，将 1D 时序重塑为 2D 图像，用多尺度 2D 卷积建模

class TimesBlock2D(nn.Module):
    """
    多尺度 2D 卷积块，用于同时捕捉周期内（intra-period）和周期间（inter-period）变化。
    输入/输出形状: (B, D, T_rows, T_cols)
    每个卷积分支含 Dropout2d，确保 MC Dropout 采样时各分支均有随机性。
    """
    def __init__(self, d_model: int, kernel_sizes: Tuple[int, ...] = (3, 5), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=k, padding=k // 2),
                nn.GELU(),
                nn.Dropout2d(p=dropout),   # 整通道随机置零，增强 MC Dropout 多样性
            )
            for k in kernel_sizes
        ])
        # 合并所有分支
        self.fuse = nn.Conv2d(d_model * len(kernel_sizes), d_model, kernel_size=1)
        # GroupNorm 代替 BatchNorm2d：不依赖 batch 统计，train/eval 行为一致，
        # 与 MC Dropout（需要 model.train()）不冲突。
        # num_groups=1 等价于 InstanceNorm，对任意 d_model 均可整除。
        self.norm = nn.GroupNorm(num_groups=1, num_channels=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [b(x) for b in self.branches]
        return self.norm(self.fuse(torch.cat(branches, dim=1)))


class TimesNet(_DLBaseModel):
    """
    TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
    ICLR 2023 — Wu et al.

    流程:
      1. FFT 分析找出 top-k 主要周期
      2. 按每个周期将 1D 序列重塑为 (周期数, 周期长) 的 2D 图像
      3. 用 TimesBlock2D 进行多尺度 2D 卷积
      4. 所有周期结果加权平均后累加到原序列（残差连接）
      5. 取最后时间步映射到预测输出
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'timesnet'
        self.input_size = config.get('input_size', 1)
        self.seq_len = config.get('seq_len', 96)
        self.d_model = config.get('d_model', 64)
        self.top_k = config.get('top_k_periods', 5)
        self.num_layers = config.get('num_layers', 3)
        self.dropout_rate = config.get('dropout', 0.1)
        self.output_size = config.get('output_size', 1)

        self.input_proj = nn.Linear(self.input_size, self.d_model)
        self.times_blocks = nn.ModuleList([
            TimesBlock2D(self.d_model, dropout=self.dropout_rate) for _ in range(self.num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.num_layers)
        ])
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.output_proj = nn.Linear(self.d_model, self.output_size)

        self.criterion = nn.MSELoss()
        # 把所有参数搬到目标设备, 必须在 optimizer 创建之前 (见 LSTMModel 注释)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config.get('learning_rate', 0.001)
        )

    def _detect_periods(self, x: torch.Tensor) -> list:
        """
        用 FFT 振幅谱检测 top-k 主要周期 / Detect top-k dominant periods via FFT.

        算法 / Algorithm:
          1. 对时序(沿通道均值后的一维信号)做实数 FFT,得到 (L//2+1,) 的振幅谱
             Real FFT on channel-averaged signal; amplitude spectrum has L//2+1 bins.
          2. 去除直流分量(amps[0]=0),否则整体均值会主导频率选择
             Zero out DC component to avoid mean dominance.
          3. 取振幅最大的 top_k 个频率下标
             Pick top-k frequencies by amplitude.
          4. 频率 f(下标从 0 计,故用 idx+1)→ 周期 p = L / f,夹紧到 >= 2
             Convert frequency index to period length; clamp to >= 2.

        Args:
            x: (B, L, D) 时序张量 / Time series tensor.

        Returns:
            top-k 主要周期长度列表 / List of top-k period lengths.
        """
        B, L, D = x.shape
        # 对通道做均值 → 一维序列做 rFFT → 批次维再取均值,得到 (L//2+1,) 振幅
        # Average channels -> rFFT -> average batch -> amplitude spectrum.
        amps = torch.fft.rfft(x.mean(-1), dim=1).abs().mean(0)  # (L//2+1,)
        amps[0] = 0  # 去除直流分量 / Remove DC component
        top_k = min(self.top_k, L // 2)
        _, freq_idx = torch.topk(amps[1:], top_k)
        # 频率 f → 周期 L/f: idx+1 是因为上面切片 amps[1:] 把 index 0 留给了 f=1
        # Period = L / frequency; +1 offset accounts for the amps[1:] slice.
        periods = [max(2, L // (idx.item() + 1)) for idx in freq_idx]
        return periods

    def _reshape_1d_to_2d(self, x: torch.Tensor, period: int):
        """
        将 1D 序列按 period 重塑为 2D 图像 / Reshape 1D series into 2D image.

        目的: 把"同相位的时间步"对齐到 2D 图像的同一列,让 2D 卷积能同时
        捕捉 intra-period(周期内,列方向) 和 inter-period(周期间,行方向) 的依赖。
        Align time steps of the same phase into columns so 2D conv captures
        intra-period (within-column) and inter-period (across-row) patterns.

        不足 period 整数倍时用零填充尾部,填充长度由调用方记录,
        反变换 _reshape_2d_to_1d 时截断还原。
        Zero-pads the tail to fit period; pad_len is returned for later truncation.

        Args:
            x:      (B, L, D)
            period: 周期长度 / Period length
        Returns:
            (x_2d, pad_len): x_2d 形状 (B, D, rows, period),pad_len 为尾部填充长度
        """
        B, L, D = x.shape
        rows = math.ceil(L / period)
        pad_len = rows * period - L
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        x_2d = x.reshape(B, rows, period, D).permute(0, 3, 1, 2)  # (B, D, rows, period)
        return x_2d, pad_len

    def _reshape_2d_to_1d(self, x_2d: torch.Tensor, L: int, pad_len: int) -> torch.Tensor:
        """
        2D 图像还原为 1D 序列 / Reshape 2D image back to 1D series.

        去掉 _reshape_1d_to_2d 阶段尾部填充的 pad_len 个元素,保证
        输出长度与原始输入 L 一致。
        Truncates the pad_len elements appended during the 1D->2D reshape.

        Args:
            x_2d:    (B, D, rows, period)
            L:       原始序列长度 / Original sequence length.
            pad_len: 之前填充的长度 / Padding length used during forward reshape.

        Returns:
            (B, L, D) 1D 序列张量 / 1D series tensor.
        """
        B, D, rows, period = x_2d.shape
        x_1d = x_2d.permute(0, 2, 3, 1).reshape(B, rows * period, D)
        return x_1d[:, :L, :]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        TimesNet 前向 / TimesNet forward pass.

        完整流程 / Full pipeline:
          1. 输入投影: (B, L, C) → (B, L, d_model)
          2. FFT 检测主要周期 top-k(对 projected 特征 detach 避免梯度经过周期选择)
             Detect top-k dominant periods (detached from autograd).
          3. 逐层(TimesBlock2D):
             - 对每个候选周期 p: 1D→2D 重塑 → 2D 卷积 → 2D→1D 还原
             - 所有周期贡献求均值,与输入做残差连接后 LayerNorm
             Per layer: for each period, reshape to 2D, apply 2D conv, reshape back;
             average contributions across periods, residual + LayerNorm.
          4. 取最后一个时间步 → Dropout → 线性映射到 output_size

        注意 / Note:
          - `x.detach()` 传入 _detect_periods 防止 FFT 操作回传梯度
            (周期检测应是固定的"元操作",而非学习得到)
          - `contributions` 若为空会导致除零,但 _detect_periods 保证 top_k >= 1
            (periods 列表至少含 1 个元素),无需额外保护
        """
        # x: (B, L, C) 输入,C 是原始变量数
        x = self.input_proj(x)          # → (B, L, d_model)
        L = x.size(1)
        periods = self._detect_periods(x.detach())

        for block, ln in zip(self.times_blocks, self.layer_norms):
            contributions = []
            for p in periods:
                # 每个周期独立走 2D 卷积 / Per-period 2D conv
                x_2d, pad_len = self._reshape_1d_to_2d(x, p)
                x_2d = block(x_2d)
                x_1d = self._reshape_2d_to_1d(x_2d, L, pad_len)
                contributions.append(x_1d)
            # 各周期贡献均等加权 + 残差 + LayerNorm
            # Equal-weight average across periods + residual + LayerNorm.
            x = ln(x + sum(contributions) / len(contributions))

        # 取最后时间步 → Dropout → 映射输出 / Last step → Dropout → output projection
        return self.output_proj(self.dropout_layer(x[:, -1, :]))


# ─── DLinear ──────────────────────────────────────────────────────────────────
# 论文: AAAI 2023 — Zeng et al.
# "Are Transformers Effective for Time Series Forecasting?"
# 核心: 序列分解（MovingAvg）+ 各分量独立线性层 + 相加
# 概率预测: 分位数回归（Pinball Loss），一次前向输出多个分位数


class PinballLoss(nn.Module):
    """
    Pinball Loss（分位数损失）/ Quantile regression loss.

    L_q(y, ŷ) = q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)

    对每个分位数 q ∈ quantiles 分别计算后取均值，
    驱动模型同时学习多个分位数的条件分布。

    Args:
        quantiles: 分位数列表，如 [0.025, 0.5, 0.975]
    """
    def __init__(self, quantiles: list):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds:  (B, H * Q) — 模型输出，H 个时间步 × Q 个分位数交织
            target: (B, H)     — 真实值

        流程:
          1. preds reshape → (B, H, Q)
          2. target unsqueeze → (B, H, 1) 广播对齐
          3. 逐分位数计算 pinball loss 后取全局均值
        """
        B, H = target.shape
        Q = len(self.quantiles)
        preds = preds.view(B, H, Q)          # (B, H, Q)
        target = target.unsqueeze(-1)         # (B, H, 1) → 广播到 (B, H, Q)
        errors = target - preds               # (B, H, Q)
        loss = torch.max(
            self.quantiles * errors,
            (self.quantiles - 1) * errors
        )                                     # (B, H, Q)
        return loss.mean()


class DLinear(_DLBaseModel):
    """
    DLinear: Decomposition Linear
    AAAI 2023 — Zeng et al.

    点预测架构 / Point forecast architecture:
      seasonal, trend = SeriesDecomp(x)
      pred = Linear_Seasonal(seasonal) + Linear_Trend(trend)

    概率预测 / Probabilistic prediction:
      probabilistic_method = 'quantile' 时，输出层扩展为 output_size * Q，
      一次前向同时预测 Q 个分位数，损失函数切换为 PinballLoss。
      置信区间由最低/最高分位数构成，中位数（q=0.5）作为点预测 mean。

    配置示例 / Config example:
      config = {
          'seq_len': 24,
          'output_size': 1,
          'input_size': 5,
          'moving_avg_kernel': 5,
          'probabilistic_method': 'quantile',          # 启用分位数回归
          'quantiles': [0.025, 0.1, 0.5, 0.9, 0.975], # 可自定义
          'confidence_level': 0.95,                    # 决定取哪两端分位数作 CI
      }

    Channel-independent:
      N 个变量各自独立映射（permute 实现），不建模变量间相关性。
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'dlinear'
        self.input_size  = config.get('input_size', 1)
        self.seq_len     = config.get('seq_len', 96)
        self.output_size = config.get('output_size', 1)
        self.kernel_size = config.get('moving_avg_kernel', 25)

        # Fail-fast: 当前实现是 channel-independent + 取第 0 个变量做目标,
        # 多变量输入 (input_size > 1) 时会**静默丢弃**后 N-1 个变量的预测,
        # 极易让用户排查"预测对不上"。明确拒绝, 而不是悄悄吞掉。
        # 真要多目标? 请在外层用 MultiTargetMonitor 组合多个独立 DLinear 实例,
        # 或者改 forward 返回 out.permute(0, 2, 1).reshape(B, -1) 走多输出路径
        # (本类不在 P0 范围内做这件事).
        # / Reject multi-variate input explicitly to prevent silent drop.
        if self.input_size != 1:
            raise ValueError(
                f"DLinear 当前实现仅支持单变量输入 (input_size=1), "
                f"传入 input_size={self.input_size} 会导致除第 0 个变量外的预测被静默丢弃. "
                f"多目标场景请用 MultiTargetMonitor 组合多个 DLinear 实例, "
                f"或改用支持多变量输出的模型 (LSTM/Transformer)."
            )

        # 分位数配置 / Quantile config
        # 默认与 confidence_level=0.95 对应的 [0.025, 0.5, 0.975]
        default_quantiles = self._default_quantiles(
            config.get('confidence_level', 0.95)
        )
        self.quantiles = config.get('quantiles', default_quantiles)
        self.Q = len(self.quantiles)

        # 是否启用分位数回归 / Whether to use quantile regression

        self.use_quantile = (self.probabilistic_method  == 'quantile')
        # 输出维度: 点预测 = output_size，分位数模式 = output_size * Q
        # Output dim: point = H, quantile = H * Q
        out_dim = self.output_size * self.Q if self.use_quantile else self.output_size

        self.decomp          = SeriesDecomp(self.kernel_size)
        self.linear_trend    = nn.Linear(self.seq_len, out_dim)
        self.linear_seasonal = nn.Linear(self.seq_len, out_dim)

        # 损失函数: 分位数模式用 PinballLoss，否则 MSE
        # Loss: PinballLoss for quantile mode, MSE otherwise.
        if self.use_quantile:
            self.criterion = PinballLoss(self.quantiles)
        else:
            self.criterion = nn.MSELoss()

        # 把所有参数搬到目标设备, 必须在 optimizer 创建之前 (见 LSTMModel 注释)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config.get('learning_rate', 0.001)
        )

    @staticmethod
    def _default_quantiles(confidence_level: float) -> list:
        """
        由置信水平推导默认分位数三元组 / Derive default quantile triple from CI level.
        例: confidence_level=0.95 → [0.025, 0.5, 0.975]
        """
        alpha = 1 - confidence_level
        return [round(alpha / 2, 4), 0.5, round(1 - alpha / 2, 4)]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (B, L, N)

        Returns:
            点预测模式: (B, output_size)
            分位数模式: (B, output_size * Q)，调用方负责 reshape 为 (B, H, Q)
        """
        seasonal, trend = self.decomp(x)                          # 各 (B, L, N)

        # (B, L, N) → permute → (B, N, L) → Linear → (B, N, out_dim)
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1))
        trend_out    = self.linear_trend(trend.permute(0, 2, 1))

        out = seasonal_out + trend_out                             # (B, N, out_dim)

        # __init__ 已经断言 input_size == 1, 这里 N=1, [:, 0, :] 等价取唯一变量
        # / N==1 by __init__ guard; this slice picks the single variate
        return out[:, 0, :].contiguous()                           # (B, out_dim)

    # fit 沿用 _DLBaseModel.fit (_dl_fit 训练循环)。
    # 分位数模式下 y_train 形状仍为 (N, output_size),PinballLoss 内部自动处理维度对齐。
    # Reuse _DLBaseModel.fit; PinballLoss handles dim alignment internally.

    def predict(self, test_data, **kwargs) -> np.ndarray:
        """
        点预测: 分位数模式下返回中位数（q=0.5 对应的分位数）。
        Point prediction: returns median quantile in quantile mode.
        """
        raw = _dl_predict(self, test_data, self.device)  # (N, out_dim)
        if not self.use_quantile:
            return raw
        # 找 q=0.5 最近的分位数索引作为点预测
        median_idx = int(np.argmin(np.abs(np.array(self.quantiles) - 0.5)))
        N = raw.shape[0]
        # raw: (N, H*Q) → (N, H, Q) → 取 median_idx → (N, H)
        return raw.reshape(N, self.output_size, self.Q)[:, :, median_idx]

    def _predict_probabilistic(self, test_data, **kwargs) -> ProbabilisticPrediction:
        """
        分位数概率预测 / Quantile probabilistic prediction.

        返回:
          mean  = 中位数分位数预测（q≈0.5）
          lower = alpha/2 分位数（如 q=0.025）
          upper = 1-alpha/2 分位数（如 q=0.975）
          std   = (upper - lower) / (2 * 1.96)，高斯近似标准差，供参考
          samples = 所有分位数预测，形状 (Q, N, H)

        若未启用分位数模式，退化为确定性预测并给出警告。
        Falls back to deterministic if quantile mode is not enabled.
        """
        if not self.use_quantile:
            import warnings
            warnings.warn(
                "DLinear: probabilistic_method='quantile' 未设置，"
                "退化为确定性预测。请在 config 中设置 probabilistic_method='quantile'。",
                UserWarning
            )
            mean = self.predict(test_data)
            return ProbabilisticPrediction(mean=mean)

        raw = _dl_predict(self, test_data, self.device)   # (N, H*Q)
        N = raw.shape[0]
        all_quantiles = raw.reshape(N, self.output_size, self.Q)  # (N, H, Q)

        alpha = 1 - self.confidence_level
        q_arr = np.array(self.quantiles)

        # 找最接近各目标分位数的索引 / Find closest quantile indices
        lower_idx  = int(np.argmin(np.abs(q_arr - alpha / 2)))
        upper_idx  = int(np.argmin(np.abs(q_arr - (1 - alpha / 2))))
        median_idx = int(np.argmin(np.abs(q_arr - 0.5)))

        mean  = all_quantiles[:, :, median_idx]   # (N, H)
        lower = all_quantiles[:, :, lower_idx]    # (N, H)
        upper = all_quantiles[:, :, upper_idx]    # (N, H)
        # 高斯近似 std，仅供参考（非严格统计量）
        std = (upper - lower) / (2 * 1.96)

        # samples: (Q, N, H)，与 mc_dropout 的 (S, N, H) 接口对齐
        samples = all_quantiles.transpose(2, 0, 1)

        return ProbabilisticPrediction(
            mean=mean, lower=lower, upper=upper, std=std, samples=samples
        )




# ─── 模型注册表 ──────────────────────────────────────────────────────────────

DL_MODEL_REGISTRY = {
    'lstm': LSTMModel,
    'transformer': TransformerModel,
    'autoformer': Autoformer,
    'itransformer': iTransformer,
    'timesnet': TimesNet,
    'dlinear': DLinear,
}


def get_dl_model(model_name: str, config: Dict[str, Any]) -> BaseModel:
    """工厂函数：按名称创建深度学习模型实例 / Factory: create DL model instance by name"""
    if model_name not in DL_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list(DL_MODEL_REGISTRY.keys())}"
        )
    return DL_MODEL_REGISTRY[model_name](config)











