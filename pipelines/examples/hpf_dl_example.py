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
from tsf_frame.features.engineering import create_feature_engineer
from tsf_frame.utils.metrics import MetricsCalculator
from tsf_frame.utils.logger import get_logger
from tsf_frame.models.transformer.transformer_models import get_dl_model
from tsf_frame.models.base_model import ProbabilisticPrediction


# ─── 1. 数据生成 ──────────────────────────────────────────────────────────────
# 与 hpf_example.py 完全一致,确保两个示例可直接对比
# Identical to hpf_example.py so both examples can be compared directly.

def generate_hpf_data(years: int = 12, start_year: int = 2012) -> pd.DataFrame:
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


# ─── 2. 特征工程 ──────────────────────────────────────────────────────────────

def build_ml_features(
    data: pd.DataFrame,
    target_col: str,
    feature_config: dict,
) -> tuple:
    """
    特征工程,返回 (X_2d, y, dates, feature_cols)。
    Feature engineering returns 2D matrix; 3D reshape is done in build_dl_sequences.

    X_2d 形状: (N, n_features)
    y    形状: (N,)
    """
    engineer = create_feature_engineer(
        feature_types=['time', 'lag', 'rolling', 'difference'],
        config=feature_config,
    )
    df_feat = engineer.fit_transform(data)
    df_feat = df_feat.dropna()

    feature_cols = [c for c in df_feat.columns if c != target_col]
    X = df_feat[feature_cols].values.astype(np.float32)
    y = df_feat[target_col].values.astype(np.float32)
    dates = df_feat.index

    return X, y, dates, feature_cols


def build_dl_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple:
    """
    将 2D 特征矩阵 (N, F) 转为 3D 序列 (N', seq_len, F) 供 DL 模型使用。
    Convert 2D feature matrix (N, F) into 3D sequences (N', seq_len, F) for DL models.

    每条样本: 用前 seq_len 步的特征预测第 seq_len 步的目标。
    Each sample: use past seq_len feature steps to predict target at step seq_len.

    Returns:
        X_seq: (N - seq_len, seq_len, F)
        y_seq: (N - seq_len, 1)
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len: i])
        y_seq.append(y[i])
    X_seq = np.asarray(X_seq, dtype=np.float32)
    # 目标 reshape 为 (N, 1) 以匹配 DL 模型的 output_size=1
    # Reshape y to (N, 1) to match DL model output_size=1
    y_seq = np.asarray(y_seq, dtype=np.float32).reshape(-1, 1)
    return X_seq, y_seq


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
) -> dict:
    """
    训练单个深度学习模型并返回评估结果 / Train a single DL model and return evaluation.

    流程 / Pipeline:
      1. get_dl_model 工厂函数创建模型(自动 to(device))
         Create model via factory and move to device.
      2. model.fit((X_train, y_train), (X_val, y_val))
         训练(内部循环 epoch,梯度裁剪 1.0,按 batch 平均 loss)
         Training loop with gradient clipping and batch-averaged loss.
      3. model.predict(X_test) 点预测 / Point prediction.
      4. model.predict_probabilistic(X_test) MC Dropout 分位数区间
         MC Dropout quantile-based confidence intervals.
      5. 反归一化 → MetricsCalculator.calculate_all(MAE/MAPE/R2/...)
         Denormalize and compute metrics.

    Args:
        model_name:  'lstm' / 'transformer' / 'autoformer' / 'itransformer' / 'timesnet'
        model_config: DL 模型配置字典(已注入 input_size/device/probabilistic 等)
        X_*_seq:     3D 序列张量 (N, seq_len, F) / 3D input tensors
        y_*_seq:     (N, 1) 目标序列 / Target sequences (N, 1)
        adapter:     HPFAdapter 实例,用于反归一化 / For denormalization
        metadata:    预处理元数据 / Preprocessing metadata
        target_col:  目标列名 / Target column name

    Returns:
        dict: {'model', 'y_pred', 'prob_result', 'metrics'}
    """
    logger.info(f'  Training {model_name}...')

    model = get_dl_model(model_name, model_config)
    # DL 模型需要显式 to(device),BaseModel 构造器只缓存 device 字符串
    # DL models need explicit .to(device); BaseModel only stores the string
    model = model.to(model_config['device'])

    # 训练(带验证集监控)
    # Train with validation monitoring
    history = model.fit((X_train_seq, y_train_seq), (X_val_seq, y_val_seq))

    # 点预测(eval 模式,Dropout 关闭)
    # Point prediction in eval mode
    y_pred = model.predict(X_test_seq).flatten()

    # 概率预测: MC Dropout 保持 Dropout 开启,多次采样取分位数
    # Probabilistic prediction via MC Dropout sampling
    prob_result = model.predict_probabilistic(X_test_seq)

    # 反归一化到真实量纲后计算指标
    # Denormalize to original scale for metric computation
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
    raw_data = generate_hpf_data(years=30)
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

    # ── Step 4: 特征工程 ──────────────────────────────────────────────────
    logger.info('\n[Step 4] 特征工程...')
    feature_cfg = config.to_feature_config()
    target_col = config.data.target_columns[0]

    X_2d, y_1d, feat_dates, feature_cols = build_ml_features(
        processed_data, target_col, feature_config=feature_cfg,
    )
    logger.info(f'  特征数量: {len(feature_cols)}')
    logger.info(f'  2D 样本数量: {len(X_2d)}')

    # ── Step 5: 2D→3D 序列构造 ───────────────────────────────────────────
    logger.info('\n[Step 5] 2D 矩阵 → 3D 序列 (N, seq_len, n_features)...')
    seq_len = config.model.seq_len
    X_seq, y_seq = build_dl_sequences(X_2d, y_1d, seq_len)
    seq_dates = feat_dates[seq_len:]  # 丢弃前 seq_len 个点的预测目标
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
                # DLinear 用 quantile 回归，output_size 保持 1（预测步数）
            if model_name == 'timesnet':
                # TimesNet 对 seq_len 的 FFT 有最小长度要求,数据不足时跳过
                # TimesNet needs seq_len long enough for FFT period detection
                pass

            res = train_and_evaluate_dl(
                model_name, cfg,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                adapter, metadata, logger,
                target_col=target_col,
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
