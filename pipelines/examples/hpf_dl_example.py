"""
住房公积金（HPF）业务预测 - 深度学习版完整示例
HPF business forecasting - Deep Learning edition

与 hpf_example.py 的差异 / Differences from hpf_example.py:
  - 模型: LSTM / Transformer / Autoformer / iTransformer / TimesNet (DL_MODEL_REGISTRY)
        代替 XGBoost / LightGBM / Ridge
  - 输入: 3D 序列 (N, seq_len, n_features) 代替 2D 矩阵 (N, n_features)
  - 概率方法: MC Dropout 代替残差法
        (Dropout 开启下多次采样得到分布,再取分位数构造置信区间)
  - 外推: DL 模型可通过隐藏状态/注意力外推,无需做差分再累加

工作流 / Workflow:
  1. 生成模拟公积金月度数据
  2. 数据验证 + 业务适配器预处理(zscore 归一化对 DL 模型至关重要)
  3. 特征工程(时间/滞后/滚动/差分)
  4. 2D→3D 序列构造 (N, seq_len, n_features)
  5. 多个 DL 模型训练对比(LSTM/Transformer/Autoformer/iTransformer/TimesNet)
  6. MC Dropout 概率预测(95% 置信区间)
  7. 业务指标评估 + 可视化
  8. 政策情景分析

运行方式 / Run:
    python examples/hpf_dl_example.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
for _p in (_HERE.parent, *_HERE.parents):
    if (_p / 'configs').is_dir() and (_p / 'src').is_dir():
        for _q in (_p, _p / 'src'):
            if str(_q) not in sys.path:
                sys.path.insert(0, str(_q))
        break

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # 无界面环境下也能运行(须在 import pyplot 之前设置)
import matplotlib.pyplot as plt
# 统一画图入口 (import 即激活中文字体 rcParams)
from tsf_frame.visualization import PredictionPlotter

from configs.hpf.hpf_config import HPFConfig
from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.features.mixed_feature_handler import MixedFeatureHandler
from tsf_frame.utils.metrics import MetricsCalculator
from tsf_frame.utils.logger import get_logger
from tsf_frame.models.transformer.transformer_models import get_dl_model
from tsf_frame.models.base_model import ProbabilisticPrediction


# ─── 1. 数据生成 ──────────────────────────────────────────────────────────────
# 与 hpf_example.py 完全一致,确保两个示例可直接对比
# Identical to hpf_example.py so both examples can be compared directly.

def generate_hpf_data(years: int = 50, start_year: int = 2012) -> pd.DataFrame:
    """
    生成模拟月度公积金数据,包含:
      - 长期增长趋势(政策驱动)
      - 年度季节性(年末提取高峰、年初缴存高峰)
      - 政策冲击(2015 年降低缴存比例、2020 年疫情影响)
      - 随机噪声
    """
    np.random.seed(42)
    n_months = years * 12
    dates = pd.date_range(start=f'{start_year}-01-01', periods=n_months, freq='MS')

    t = np.arange(n_months)
    month_idx = dates.month.values

    # ── 月缴存额(亿元)
    deposit_trend = 80 + 1.2 * t
    deposit_seasonal = (
        10 * np.sin(2 * np.pi * (month_idx - 3) / 12)
        + 5 * np.sin(2 * np.pi * (month_idx - 1) / 6)
    )
    deposit_policy = np.zeros(n_months)
    deposit_policy[t >= (2015 - start_year) * 12] -= 8
    deposit_policy[t >= (2020 - start_year) * 12] -= 15
    deposit_policy[t >= (2021 - start_year) * 12] += 10
    deposit_noise = np.random.normal(0, 4, n_months)
    monthly_deposit = np.maximum(
        deposit_trend + deposit_seasonal + deposit_policy + deposit_noise, 10
    )

    # ── 月提取额(亿元)
    withdrawal_trend = 60 + 0.9 * t
    withdrawal_seasonal = (
        20 * np.sin(2 * np.pi * (month_idx - 12) / 12)
        + 8 * np.cos(2 * np.pi * (month_idx - 6) / 12)
    )
    withdrawal_noise = np.random.normal(0, 5, n_months)
    monthly_withdrawal = np.maximum(
        withdrawal_trend + withdrawal_seasonal + withdrawal_noise, 5
    )

    # ── 贷款余额(亿元)—— 累积性质
    loan_growth = 500 + 15 * t + 2 * t ** 1.2 / 10
    loan_noise = np.random.normal(0, 20, n_months)
    loan_balance = loan_growth + loan_noise

    # ── 缴存人数(万人)
    depositor_trend = 200 + 0.5 * t
    depositor_noise = np.random.normal(0, 3, n_months)
    depositor_count = np.maximum(depositor_trend + depositor_noise, 50)

    data = pd.DataFrame({
        'monthly_deposit': monthly_deposit,
        'monthly_withdrawal': monthly_withdrawal,
        'loan_balance': loan_balance,
        'depositor_count': depositor_count,
    }, index=dates)

    return data


# ─── 2. 数据处理与转换 ──────────────────────────────────────────────────────────
# 深度学习模型直接使用原始多元时间序列构建滑动窗口，无需手动构建 lag / diff 等传统特征
# DL models directly use raw multivariate time series for sliding windows.


# ─── 3. DL 模型训练与评估 ─────────────────────────────────────────────────────

def train_and_evaluate_dl(
    model_name: str,
    model_config: dict,
    X_train_seq: np.ndarray, y_train_seq: np.ndarray,
    X_val_seq: np.ndarray, y_val_seq: np.ndarray,
    X_test_seq: np.ndarray, y_test_seq: np.ndarray,
    adapter: HPFAdapter,
    metadata: dict,
    logger,
    target_col: str,
    use_diff: bool = False,
) -> dict:
    """
    训练单个深度学习模型并返回评估结果 / Train a single DL model and return evaluation.

    流程 / Pipeline:
      1. (可选) DiffTransform 把 y 变成一阶差分, X 同步对齐 [1:]
      2. get_dl_model 工厂函数创建模型(自动 to(device))
      3. model.fit((X_train, y_train), (X_val, y_val))
      4. model.predict(X_test) 点预测 (差分空间) → inverse_transform 累加还原
      5. model.predict_probabilistic(X_test) MC Dropout 区间 → 同样累加还原
      6. 反归一化 → MetricsCalculator.calculate_all(MAE/MAPE/R2/...)

    Args:
        model_name:  'lstm' / 'transformer' / 'autoformer' / 'itransformer' /
                     'timesnet' / 'dlinear'
        model_config: DL 模型配置字典
        X_*_seq:     3D 序列张量 (N, seq_len, F)
        y_*_seq:     (N, 1) 目标序列
        adapter:     HPFAdapter, 用于反归一化
        metadata:    预处理元数据
        target_col:  目标列名
        use_diff:    True 时把目标改为 y_diff[i]=y[i]-y[i-1] 训练, 推理后累加还原.
                     anchor 自动取 y_val_seq[-1] (归一化空间下最贴近测试集的真值).
                     适用: LSTM/Transformer/Autoformer/iTransformer/TimesNet 这些
                     **无外推能力**的 DL 模型 — 它们的 LayerNorm + softmax + 末位
                     Linear 都是反外推机制, 长趋势数据测试集外推时必崩.
                     不适用: DLinear (自带 SeriesDecomp 趋势外推, 叠加 diff 反而
                     破坏架构归纳偏置).
                     / Differencing target so non-extrapolating DL models work
                       on data with strong trend (HPF deposit/loan_balance).

    Returns:
        dict: {'model', 'y_pred', 'prob_result', 'metrics', 'history'}
              y_pred / prob_result 都已经在**原 (归一化) 水平值空间**, 即
              已经做过 inverse_transform; 下游反归一化继续走 _denormalize.
    """
    from tsf_frame.utils.target_transforms import DiffTransform

    tag = '（差分目标）' if use_diff else ''
    logger.info(f'  Training {model_name}{tag}...')

    # ── 训练侧: 可选差分 ──────────────────────────────────────────────────
    if use_diff:
        tt = DiffTransform()
        # transform 同时对齐 X (差分后 y 少 1 行, X 也 [1:])
        y_train_t, X_train_t = tt.transform(y_train_seq, X_train_seq)
        y_val_t,   X_val_t   = tt.transform(y_val_seq,   X_val_seq)
        # anchor: 测试集开始前最后一个真实水平 (归一化空间)
        # 用 y_val_seq[-1] 比 y_train_seq[-1] 更紧邻测试集, 误差更小.
        anchor = float(y_val_seq.flatten()[-1])
    else:
        tt = None
        X_train_t, y_train_t = X_train_seq, y_train_seq
        X_val_t,   y_val_t   = X_val_seq,   y_val_seq
        anchor = None

    model = get_dl_model(model_name, model_config)
    model = model.to(model_config['device'])

    history = model.fit((X_train_t, y_train_t), (X_val_t, y_val_t))

    # ── 推理侧: 模型输出在差分空间, 累加还原回水平值空间 ─────────────────
    y_pred_raw = model.predict(X_test_seq).flatten()
    prob_raw = model.predict_probabilistic(X_test_seq)
    if tt is not None:
        y_pred = tt.inverse_transform(y_pred_raw, anchor=anchor)
        if prob_raw.lower is not None and prob_raw.upper is not None:
            prob_result = ProbabilisticPrediction(
                mean=y_pred,
                lower=tt.inverse_transform(prob_raw.lower.flatten(), anchor=anchor),
                upper=tt.inverse_transform(prob_raw.upper.flatten(), anchor=anchor),
            )
        else:
            prob_result = ProbabilisticPrediction(mean=y_pred)
    else:
        y_pred = y_pred_raw
        prob_result = prob_raw

    # ── 反归一化到真实量纲后计算指标 ──────────────────────────────────────
    y_test_flat = y_test_seq.flatten()
    y_test_df = pd.DataFrame(y_test_flat.reshape(-1, 1), columns=[target_col])
    y_pred_df = pd.DataFrame(y_pred.reshape(-1, 1), columns=[target_col])
    y_test_orig = adapter._denormalize(y_test_df, metadata)[target_col].values
    y_pred_orig = adapter._denormalize(y_pred_df, metadata)[target_col].values
    metrics = MetricsCalculator.calculate_all(y_test_orig, y_pred_orig)

    logger.info(
        f'    MAE={metrics["MAE"]:.4f}  MAPE={metrics["MAPE"]:.2%}  '
        f'R2={metrics["R2"]:.4f}  final_train_loss={history["train_loss"][-1]:.5f}'
    )

    return {
        'model': model,
        'y_pred': y_pred,
        'prob_result': prob_result,
        'metrics': metrics,
        'history': history,
    }


# ─── 4. 可视化 ────────────────────────────────────────────────────────────────

#: 模块级 plotter 单例
_PLOTTER = PredictionPlotter(figsize=(14, 10), dpi=120)


def _yunit(target_col: str) -> str:
    if any(k in target_col for k in ('deposit', 'withdrawal', 'loan')):
        return '亿元'
    return '万人'


def plot_forecast_comparison(
    y_true: np.ndarray,
    results: dict,
    dates,
    target_col: str,
    save_path: str,
):
    """DL 多模型预测对比 + MC Dropout 概率区间."""
    models_pred = {
        f'{name} (MAPE={res["metrics"]["MAPE"]:.2%})': res['y_pred']
        for name, res in results.items()
    }
    best_name = min(results, key=lambda k: results[k]['metrics']['MAPE'])
    best = results[best_name]
    best_interval = None
    prob = best.get('prob_result')
    if prob is not None and prob.lower is not None and prob.upper is not None:
        best_interval = (
            best['y_pred'],
            prob.lower.flatten(),
            prob.upper.flatten(),
        )
    _PLOTTER.forecast_comparison_fig(
        x=dates, y_true=y_true,
        models_pred=models_pred,
        best_interval=best_interval,
        target_label=f'{target_col} 深度学习', ylabel=_yunit(target_col),
        save_path=save_path,
    )


def plot_metrics_comparison(results: dict, save_path: str):
    """DL 模型 MAPE / R² 柱状对比."""
    models = list(results.keys())
    _PLOTTER.metrics_bars_fig(
        groups={
            'MAPE(越低越好)': (
                models, [results[m]['MAPE'] for m in models], '{:.2%}'),
            'R² 决定系数(越高越好)': (
                models, [results[m]['R2'] for m in models], '{:.4f}'),
        },
        suptitle='公积金预测深度学习模型性能对比',
        figsize=(13, 5),
        save_path=save_path,
    )


def plot_training_curves(results: dict, save_path: str):
    """训练 / 验证 Loss 曲线 (1x2, log 轴, 多模型 overlay)."""
    fig = _PLOTTER.training_curves_fig(
        # results[model] 形如 {'history': {'train_loss': [...], 'val_loss': [...]}, ...}
        results={name: res['history'] for name, res in results.items()},
        log_scale=True,
        ylabel='MSE Loss',
        save_path=None,             # 先暂不保存, 加 suptitle 后再统一保存
        close=False,
    )
    fig.suptitle('深度学习模型训练过程', fontsize=14)
    fig.tight_layout()
    _PLOTTER.save(fig, save_path)
    plt.close(fig)


def plot_policy_scenario(
    base_forecast: pd.DataFrame,
    policy_scenarios: dict,
    dates,
    target_col: str,
    save_path: str,
):
    """政策情景分析图 (基准 + 多情景虚线)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    _PLOTTER.lines_compare(
        ax,
        x=dates,
        baseline=base_forecast[target_col].values,
        baseline_label='基准预测(无政策调整)',
        series={name: adj[target_col].values
                for name, adj in policy_scenarios.items()},
        title=f'公积金 {target_col} 政策情景分析(DL 基准)',
        ylabel=_yunit(target_col),
    )
    fig.tight_layout()
    _PLOTTER.save(fig, save_path)
    plt.close(fig)


# ─── 5. 主流程 ────────────────────────────────────────────────────────────────

def main():
    # 随机种子统一,保证可复现
    # Fix all RNGs for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 设备选择: 优先使用 GPU
    # Prefer GPU when available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 输出目录 (统一布局: logs/{runs,outputs/...})
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, 'logs', 'outputs', 'hpf_dl')
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(
        'hpf_dl_example',
        log_dir=os.path.join(project_root, 'logs', 'runs'),
    )

    logger.info('=' * 65)
    logger.info('  住房公积金(HPF)业务预测 - 深度学习版')
    logger.info('=' * 65)
    logger.info(f'  Device: {device}')

    # ── 配置 ──────────────────────────────────────────────────────────────
    config = HPFConfig()
    config.data.target_columns = ['monthly_deposit']
    config.model.pred_len = 1
    # DL 模型专属调参 / DL-specific hyperparameters
    config.model.seq_len = 12           # 回看 12 个月,匹配年周期
    config.model.train_epochs = 80       # 小数据集 epoch 不宜过大
    config.model.batch_size = 16
    config.model.dropout = 0.2
    config.model.hidden_size = 64

    # ── Step 1: 生成数据 ──────────────────────────────────────────────────
    logger.info('\n[Step 1] 生成模拟公积金月度数据...')
    raw_data = generate_hpf_data()
    logger.info(f'  数据范围: {raw_data.index[0].date()} ~ {raw_data.index[-1].date()}')
    logger.info(f'  数据形状: {raw_data.shape}')

    # ── Step 2: 数据验证 ──────────────────────────────────────────────────
    logger.info('\n[Step 2] 数据验证...')
    adapter = HPFAdapter(config.to_adapter_config())
    valid, msg = adapter.validate_data(raw_data)
    logger.info(f'  验证结果: {"通过" if valid else "失败"} — {msg}')
    if not valid:
        logger.error('数据验证失败,终止运行')
        return

    # ── Step 3: 预处理 ────────────────────────────────────────────────────
    logger.info('\n[Step 3] 业务适配器预处理(异常值处理 + zscore 归一化)...')
    logger.info('  注意: DL 模型对输入量纲敏感,zscore 归一化必不可少')
    processed_data, metadata = adapter.preprocess(raw_data)
    logger.info(f'  处理后形状: {processed_data.shape}')

    # ── Step 4 & 5: 序列构造 ──────────────────────────────────────────────────
    logger.info('\n[Step 4 & 5] 构建 3D 滑动窗口序列 (N, seq_len, n_features)...')
    target_col = config.data.target_columns[0]
    seq_len = config.model.seq_len
    
    # 获取所有数值列作为时序特征
    time_varying_cols = list(processed_data.select_dtypes(include=[np.number]).columns)
    # 确保 target_col 在第一列，这样 DLinear 用 num_targets=1 取前 1 列就正好是 target
    if target_col in time_varying_cols:
        time_varying_cols.remove(target_col)
    time_varying_cols = [target_col] + time_varying_cols

    handler = MixedFeatureHandler(
        time_varying_cols=time_varying_cols,
        static_cols=[],
        target_col=target_col,
        seq_len=seq_len,
        pred_len=1
    )
    handler.fit(processed_data)
    X_seq, y_seq = handler.create_sequences(processed_data)
    
    # create_sequences 会吃掉最前面的 seq_len 个时间步
    seq_dates = processed_data.index[seq_len:]

    logger.info(f'  特征顺序: {time_varying_cols}')
    logger.info(f'  3D 序列形状: X={X_seq.shape}, y={y_seq.shape}')
    logger.info(f'  输入特征维度 input_size = {X_seq.shape[-1]}')

    # ── Step 6: 训练/验证/测试划分 ────────────────────────────────────────
    logger.info('\n[Step 6] 划分训练/验证/测试集...')
    n_test = max(12, int(len(X_seq) * config.data.test_size))
    n_val = max(6, int(len(X_seq) * config.data.val_size))
    n_train = len(X_seq) - n_test - n_val

    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]
    X_val = X_seq[n_train: n_train + n_val]
    y_val = y_seq[n_train: n_train + n_val]
    X_test = X_seq[n_train + n_val:]
    y_test = y_seq[n_train + n_val:]
    test_dates = seq_dates[n_train + n_val:]

    logger.info(f'  训练: {n_train}条  验证: {n_val}条  测试: {n_test}条')

    # ── Step 7: DL 模型训练对比 ───────────────────────────────────────────
    logger.info('\n[Step 7] 深度学习模型训练与评估...')

    # 构造 DL 模型通用配置
    # Shared DL model config
    dl_config = {
        'input_size': X_seq.shape[-1],       # 特征数(与工程后的列数一致)
        'output_size': 1,                     # 单步预测
        'seq_len': seq_len,
        'pred_len': 1,
        'device': device,
        # 训练参数
        'train_epochs': config.model.train_epochs,
        'batch_size': config.model.batch_size,
        'learning_rate': 1e-3,
        # 网络结构(LSTM 专用)
        'hidden_size': config.model.hidden_size,
        'num_layers': 2,
        'dropout': config.model.dropout,
        # Transformer 系列共享
        'd_model': 64,
        'nhead': 4,
        'dim_feedforward': 128,
        # 概率预测: MC Dropout
        'probabilistic': True,
        'probabilistic_method': 'mc_dropout',
        'num_samples': 50,                    # MC Dropout 采样次数
        'confidence_level': 0.95,
        'use_revin': True,                    # 显式开启 RevIN (可逆实例归一化)
        # RevIN (ICLR 2022) 已经在 _DLBaseModel **默认开启** (use_revin=True),
        # 长趋势 HPF 数据上 RevIN 把 5 个 Transformer 系列 MAPE 从 6-20% 降到 ~0.5%.
    }

    models_to_compare = ['lstm', 'transformer', 'autoformer',
                         'itransformer', 'timesnet','dlinear']

    results = {}
    for model_name in models_to_compare:
        try:
            # 针对不同模型做必要的配置微调
            # Per-model config tweaks where defaults don't fit
            cfg = {**dl_config}
            if model_name == 'dlinear':
                cfg['probabilistic_method'] = 'quantile'
                cfg['quantiles'] = [0.025, 0.5, 0.975]
                # DLinear 用 quantile 回归; 单目标场景设 num_targets=1
                # 避免 channel-independent 默认预测所有 input_size 个变量
                cfg['num_targets'] = 1
                cfg['pred_len'] = 1
            if model_name == 'timesnet':
                # TimesNet 对 seq_len 的 FFT 有最小长度要求,数据不足时跳过
                # TimesNet needs seq_len long enough for FFT period detection
                pass

            # 外推策略选择 / Extrapolation strategy:
            #   RevIN (use_revin) 在 X 侧用 input window 自己的 (mean, std) 实现
            #   "反归一化外推", 与 use_diff (Y 侧目标差分) 互斥:
            #     - 同开会导致 RevIN 用 X(level) 量纲的 std/mean 给 Y(diff) 反归一化,
            #       量纲错位, 实测 MAPE 飙到 2000%+.
            #     - use_revin=True 时, 强制 use_diff=False, 让所有 DL 模型走 RevIN
            #       路径, 模型形状统一, 也避免下游反差分逻辑分支.
            # 默认 True 与 _DLBaseModel._init_revin 默认值保持一致.
            # / RevIN and use_diff are mutually exclusive — default mirrors _DLBaseModel.
            use_diff = (not cfg.get('use_revin', True)) and (model_name not in {'dlinear', 'itransformer'})
            res = train_and_evaluate_dl(
                model_name, cfg,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                adapter, metadata, logger,
                target_col=target_col,
                use_diff=use_diff,
            )
            results[model_name] = res
        except Exception as e:
            logger.warning(f'  跳过 {model_name}: {e}')

    if not results:
        logger.error('所有 DL 模型均训练失败')
        return

    # ── Step 8: 业务指标评估 ──────────────────────────────────────────────
    logger.info('\n[Step 8] 业务指标评估...')
    best_name = min(results, key=lambda k: results[k]['metrics']['MAPE'])
    best_pred = results[best_name]['y_pred']

    y_test_flat = y_test.flatten()
    y_test_df = pd.DataFrame(
        y_test_flat.reshape(-1, 1), columns=[target_col], index=test_dates
    )
    y_pred_df = pd.DataFrame(
        best_pred.reshape(-1, 1), columns=[target_col], index=test_dates
    )
    y_test_orig = adapter._denormalize(y_test_df, metadata)
    y_pred_orig = adapter._denormalize(y_pred_df, metadata)

    biz_metrics = adapter.get_business_metrics(y_test_orig, y_pred_orig)
    logger.info(f'  最优模型: {best_name}')
    for k, v in biz_metrics.items():
        logger.info(f'    {k}: {v:.4f}')

    # ── Step 9: 可视化 ────────────────────────────────────────────────────
    logger.info('\n[Step 9] 生成可视化图表...')

    # 反归一化各模型预测结果 + 置信区间
    # Denormalize predictions + CI for plotting
    plot_results = {}
    for mn, res in results.items():
        pred_df = pd.DataFrame(
            res['y_pred'].reshape(-1, 1), columns=[target_col], index=test_dates
        )
        pred_orig = adapter._denormalize(pred_df, metadata)

        prob = res['prob_result']
        if prob.lower is not None:
            lower_df = pd.DataFrame(prob.lower.reshape(-1, 1), columns=[target_col])
            upper_df = pd.DataFrame(prob.upper.reshape(-1, 1), columns=[target_col])
            lower_orig = adapter._denormalize(lower_df, metadata)[target_col].values
            upper_orig = adapter._denormalize(upper_df, metadata)[target_col].values
            prob_orig = ProbabilisticPrediction(
                mean=pred_orig[target_col].values,
                lower=lower_orig,
                upper=upper_orig,
            )
        else:
            prob_orig = ProbabilisticPrediction(mean=pred_orig[target_col].values)

        plot_results[mn] = {
            'y_pred': pred_orig[target_col].values,
            'prob_result': prob_orig,
            'metrics': res['metrics'],
            'history': res['history'],
        }

    plot_forecast_comparison(
        y_test_orig[target_col].values, plot_results,
        test_dates, target_col,
        save_path=os.path.join(output_dir, 'hpf_dl_forecast_comparison.png'),
    )
    plot_metrics_comparison(
        {k: v['metrics'] for k, v in results.items()},
        save_path=os.path.join(output_dir, 'hpf_dl_metrics_comparison.png'),
    )
    plot_training_curves(
        plot_results,
        save_path=os.path.join(output_dir, 'hpf_dl_training_curves.png'),
    )

    # ── Step 10: 政策情景分析 ─────────────────────────────────────────────
    logger.info('\n[Step 10] 政策情景分析...')
    base_forecast_df = y_pred_orig.copy()

    policy_scenarios = {
        '上调缴存比例+2pp(+5%效应)': adapter.get_policy_adjusted_forecast(
            base_forecast_df, {target_col: 0.05}
        ),
        '下调缴存比例-2pp(-5%效应)': adapter.get_policy_adjusted_forecast(
            base_forecast_df, {target_col: -0.05}
        ),
        '引进新企业缴存(+10%效应)': adapter.get_policy_adjusted_forecast(
            base_forecast_df, {target_col: 0.10}
        ),
    }

    plot_policy_scenario(
        base_forecast_df, policy_scenarios, test_dates, target_col,
        save_path=os.path.join(output_dir, 'hpf_dl_policy_scenario.png'),
    )
    for scenario, adjusted in policy_scenarios.items():
        diff_pct = (adjusted[target_col].mean() / base_forecast_df[target_col].mean() - 1) * 100
        logger.info(f'  {scenario}: 平均变化 {diff_pct:+.1f}%')

    # ── 汇总 ──────────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 65)
    logger.info('  DL 预测结果汇总')
    logger.info('=' * 65)
    logger.info(f'  目标变量    : {target_col}')
    logger.info(f'  测试集范围  : {test_dates[0].date()} ~ {test_dates[-1].date()}')
    logger.info(f'  参与对比模型: {list(results.keys())}')
    logger.info(f'  最优模型    : {best_name}')
    logger.info(f'  最优 MAPE   : {results[best_name]["metrics"]["MAPE"]:.2%}')
    logger.info(f'  最优 R2     : {results[best_name]["metrics"]["R2"]:.4f}')
    logger.info(f'\n  输出文件保存至: {output_dir}')
    logger.info('=' * 65)


if __name__ == '__main__':
    main()
