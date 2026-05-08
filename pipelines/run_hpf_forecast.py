"""
住房公积金（HPF）业务预测完整示例

工作流:
  1. 生成模拟公积金月度数据
  2. 数据验证 + 业务适配器预处理
  3. 特征工程（时间/滞后/滚动/差分）
  4. 多模型训练对比（XGBoost / LightGBM / Ridge）
  5. 概率预测（置信区间）
  6. 业务指标评估 + 可视化
  7. 政策情景分析

运行方式:
    python examples/hpf_example.py
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
import matplotlib
matplotlib.use('Agg')  # 无界面环境下也能运行（须在 import pyplot 之前设置）
import matplotlib.pyplot as plt
# 统一画图入口 (import 即激活中文字体 rcParams)
from tsf_frame.visualization import PredictionPlotter
matplotlib.rcParams['axes.unicode_minus'] = False
from datetime import datetime

from configs.hpf.hpf_config import HPFConfig
from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.features.engineering import create_feature_engineer
from tsf_frame.utils.metrics import MetricsCalculator
from tsf_frame.utils.logger import get_logger

# ─── 1. 数据生成 ──────────────────────────────────────────────────────────────

def generate_hpf_data(years: int = 12, start_year: int = 2012) -> pd.DataFrame:
    """
    生成模拟月度公积金数据，包含:
      - 长期增长趋势（政策驱动）
      - 年度季节性（年末提取高峰、年初缴存高峰）
      - 政策冲击（2015年降低缴存比例、2020年疫情影响）
      - 随机噪声
    """
    np.random.seed(42)
    n_months = years * 12
    dates = pd.date_range(start=f'{start_year}-01-01', periods=n_months, freq='MS')

    t = np.arange(n_months)
    month_idx = dates.month.values  # 1~12

    # ── 月缴存额（亿元）
    deposit_trend = 80 + 1.2 * t                              # 长期增长
    deposit_seasonal = (
        10 * np.sin(2 * np.pi * (month_idx - 3) / 12)         # 年度周期
        + 5 * np.sin(2 * np.pi * (month_idx - 1) / 6)         # 半年周期
    )
    deposit_policy = np.zeros(n_months)
    deposit_policy[t >= (2015 - start_year) * 12] -= 8         # 2015 降缴存比例
    deposit_policy[t >= (2020 - start_year) * 12] -= 15        # 2020 疫情冲击
    deposit_policy[t >= (2021 - start_year) * 12] += 10        # 2021 恢复
    deposit_noise = np.random.normal(0, 4, n_months)
    monthly_deposit = np.maximum(deposit_trend + deposit_seasonal + deposit_policy + deposit_noise, 10)

    # ── 月提取额（亿元）
    withdrawal_trend = 60 + 0.9 * t
    withdrawal_seasonal = (
        20 * np.sin(2 * np.pi * (month_idx - 12) / 12)        # 年末提取高峰
        + 8 * np.cos(2 * np.pi * (month_idx - 6) / 12)
    )
    withdrawal_noise = np.random.normal(0, 5, n_months)
    monthly_withdrawal = np.maximum(
        withdrawal_trend + withdrawal_seasonal + withdrawal_noise, 5
    )

    # ── 贷款余额（亿元）—— 累积性质
    loan_growth = 500 + 15 * t + 2 * t ** 1.2 / 10
    loan_noise = np.random.normal(0, 20, n_months)
    loan_balance = loan_growth + loan_noise

    # ── 缴存人数（万人）
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


# ─── 2. 特征工程辅助 ──────────────────────────────────────────────────────────

def build_ml_features(
    data: pd.DataFrame,
    target_col: str,
    seq_len: int,
    feature_config: dict,
) -> tuple:
    """
    特征工程 + 滑动窗口，返回 (X, y, dates)。
    X 形状: (N, n_features)  ← 机器学习模型所需的 2D 输入
    y 形状: (N,)
    """
    # 特征工程
    engineer = create_feature_engineer(
        feature_types=['time', 'lag', 'rolling', 'difference'],
        config=feature_config,
    )
    df_feat = engineer.fit_transform(data)
    df_feat = df_feat.dropna()

    # 选取特征列（排除目标列中非当前目标的列）
    feature_cols = [c for c in df_feat.columns if c != target_col]
    X = df_feat[feature_cols].values
    y = df_feat[target_col].values
    dates = df_feat.index

    return X, y, dates, feature_cols


def build_dl_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple:
    """
    将 2D 特征矩阵转为 3D 序列 (N, seq_len, n_features) 供 LSTM/Transformer 使用。
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len: i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# ─── 3. 模型训练与评估 ────────────────────────────────────────────────────────

def train_and_evaluate(
    model_name: str,
    model_config: dict,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    adapter: HPFAdapter,
    metadata: dict,
    logger,
    target_col: str = 'monthly_deposit',
    use_diff: bool = False,
    last_y_before_test: float = None,
) -> dict:
    """训练单个模型并返回评估结果（指标在反归一化后的真实量纲上计算）。

    use_diff=True 时对目标做一阶差分训练，预测后通过 cumsum 还原水平值，
    解决树模型无法外推趋势的问题。

    差分还原公式 / Difference reconstruction formula:
        训练阶段: y_diff[i] = y[i+1] - y[i]
        预测阶段: y_pred[t] = anchor + Σ_{k=0}^{t} y_pred_diff[k]
        其中 anchor 取验证集最后一个已知水平值 last_y_before_test。
        Training: y_diff[i] = y[i+1] - y[i]
        Inference: y_pred[t] = anchor + cumsum(y_pred_diff)[t]
        anchor = last known level value (last point in validation set).

    为什么树模型需要差分 / Why tree models need differencing:
        树模型预测被限制在训练集值域内,无法外推趋势。对目标做差分后,
        模型学习的是"变化量"而非"绝对水平",累积回来后可突破训练集值域。
        Tree predictions are bounded by training value range and cannot
        extrapolate trends; differencing lets them learn changes instead.
    """
    from tsf_frame.models.classical.ml_models import get_ml_model
    from tsf_frame.models.base_model import ProbabilisticPrediction

    tag = '（差分）' if use_diff else ''
    logger.info(f'  Training {model_name}{tag}...')
    model = get_ml_model(model_name, model_config)

    if use_diff:
        # 训练集差分：y_diff[i] = y[i+1] - y[i]，X 对齐到 y[i+1]
        # 当期预测，用今天的特征预测今天的结果；X_train_fit = X_train[1:] 对齐到 y[i+1]
        # Training-time differencing: y_diff[i] = y[i+1] - y[i], X aligned to y[i+1]
        y_train_diff = np.diff(y_train)
        X_train_fit = X_train[1:]
        model.fit((X_train_fit, y_train_diff))

        # 预测差分，再用测试集前的最后已知水平值累加还原
        # anchor + cumsum(diff) 还原水平值序列
        # Predict deltas, then reconstruct levels via anchor + cumsum(deltas)
        y_pred_diff = model.predict(X_test).flatten()
        anchor = last_y_before_test if last_y_before_test is not None else y_train[-1]
        y_pred = anchor + np.cumsum(y_pred_diff)

        # 概率区间：同样在差分空间预测，累加还原上下界
        # 注意: 累加后的区间宽度会随预测步数增长(不确定性累积),符合业务直觉
        # Confidence intervals in diff-space are also cumsum-reconstructed;
        # interval width grows with horizon (cumulative uncertainty).
        prob_diff = model.predict_probabilistic(X_test)
        if prob_diff.lower is not None:
            prob_result = ProbabilisticPrediction(
                mean=y_pred,
                lower=anchor + np.cumsum(prob_diff.lower.flatten()),
                upper=anchor + np.cumsum(prob_diff.upper.flatten()),
            )
        else:
            prob_result = ProbabilisticPrediction(mean=y_pred)
    else:
        model.fit((X_train, y_train))
        y_pred = model.predict(X_test).flatten()
        prob_result = model.predict_probabilistic(X_test)

    # 反归一化到真实量纲后再计算指标
    y_test_df = pd.DataFrame(y_test.reshape(-1, 1), columns=[target_col])
    y_pred_df = pd.DataFrame(y_pred.reshape(-1, 1), columns=[target_col])
    y_test_orig = adapter._denormalize(y_test_df, metadata)[target_col].values
    y_pred_orig = adapter._denormalize(y_pred_df, metadata)[target_col].values
    metrics = MetricsCalculator.calculate_all(y_test_orig, y_pred_orig)

    logger.info(f'    MAE={metrics["MAE"]:.4f}  MAPE={metrics["MAPE"]:.2%}  R2={metrics["R2"]:.4f}')

    return {
        'model': model,
        'y_pred': y_pred,
        'prob_result': prob_result,
        'metrics': metrics,
    }


# ─── 4. 可视化 ────────────────────────────────────────────────────────────────

#: 模块级单例 plotter (复用,避免重复 init)
_PLOTTER = PredictionPlotter(figsize=(14, 10), dpi=120)


def _yunit(target_col: str) -> str:
    """根据目标列推断 y 轴单位."""
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
    """对比多个模型预测结果 (调用统一 PredictionPlotter)."""
    # 多模型预测
    models_pred = {
        f'{name} (MAPE={res["metrics"]["MAPE"]:.2%})': res['y_pred']
        for name, res in results.items()
    }
    # 最优模型 + 概率区间
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
        target_label=target_col, ylabel=_yunit(target_col),
        save_path=save_path,
    )


def plot_metrics_comparison(results: dict, save_path: str):
    """MAPE / R² 对比柱状图 (统一 PredictionPlotter.metrics_bars_fig)."""
    models = list(results.keys())
    _PLOTTER.metrics_bars_fig(
        groups={
            'MAPE(越低越好)': (
                models, [results[m]['MAPE'] for m in models], '{:.2%}'),
            'R² 决定系数(越高越好)': (
                models, [results[m]['R2'] for m in models], '{:.4f}'),
        },
        suptitle='公积金预测模型性能对比',
        figsize=(12, 5),
        save_path=save_path,
    )


def plot_policy_scenario(
    base_forecast: pd.DataFrame,
    policy_scenarios: dict,
    dates,
    target_col: str,
    save_path: str,
):
    """政策情景分析图 (单图: 基准 + 多情景虚线对比)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    _PLOTTER.lines_compare(
        ax,
        x=dates,
        baseline=base_forecast[target_col].values,
        baseline_label='基准预测(无政策调整)',
        series={name: adj[target_col].values
                for name, adj in policy_scenarios.items()},
        title=f'公积金 {target_col} 政策情景分析',
        ylabel=_yunit(target_col),
    )
    fig.tight_layout()
    _PLOTTER.save(fig, save_path)
    plt.close(fig)


# ─── 5. 主流程 ────────────────────────────────────────────────────────────────

def main():
    # 输出目录 (统一布局: logs/{runs,outputs/...})
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'logs', 'outputs', 'hpf')
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(
        'hpf_example',
        log_dir=os.path.join(project_root, 'logs', 'runs'),
    )

    logger.info('=' * 65)
    logger.info('  住房公积金（HPF）业务预测示例')
    logger.info('=' * 65)

    # ── 配置 ──────────────────────────────────────────────────────────────
    config = HPFConfig()
    config.data.target_columns = ['monthly_deposit']   # 本示例预测月缴存额
    config.model.model_name = 'xgboost'                # 主模型
    config.model.pred_len = 1

    # ── Step 1: 生成数据 ──────────────────────────────────────────────────
    logger.info('\n[Step 1] 生成模拟公积金月度数据...')
    raw_data = generate_hpf_data(years=12)
    logger.info(f'  数据范围: {raw_data.index[0].date()} ~ {raw_data.index[-1].date()}')
    logger.info(f'  数据形状: {raw_data.shape}')
    logger.info(f'  前5行:\n{raw_data.head()}')

    # ── Step 2: 数据验证 ──────────────────────────────────────────────────
    logger.info('\n[Step 2] 数据验证...')
    adapter = HPFAdapter(config.to_adapter_config())
    valid, msg = adapter.validate_data(raw_data)
    logger.info(f'  验证结果: {"通过" if valid else "失败"} — {msg}')
    if not valid:
        logger.error('数据验证失败，终止运行')
        return

    # ── Step 3: 预处理 ────────────────────────────────────────────────────
    logger.info('\n[Step 3] 业务适配器预处理（异常值处理 + 归一化）...')
    processed_data, metadata = adapter.preprocess(raw_data)
    logger.info(f'  处理后形状: {processed_data.shape}')
    if metadata['outliers_info']:
        for col, info in metadata['outliers_info'].items():
            logger.info(f'  {col}: 处理了 {info["outlier_count"]} 个异常值')

    # ── Step 4: 特征工程 ──────────────────────────────────────────────────
    logger.info('\n[Step 4] 特征工程...')
    feature_cfg = config.to_feature_config()
    target_col = config.data.target_columns[0]

    X, y, feat_dates, feature_cols = build_ml_features(
        processed_data, target_col,
        seq_len=config.model.seq_len,
        feature_config=feature_cfg,
    )
    logger.info(f'  特征数量: {len(feature_cols)}')
    logger.info(f'  样本数量: {len(X)}')

    # ── Step 5: 训练/测试划分 ─────────────────────────────────────────────
    logger.info('\n[Step 5] 划分训练/测试集...')
    n_test = max(12, int(len(X) * config.data.test_size))
    n_val = max(6, int(len(X) * config.data.val_size))
    n_train = len(X) - n_test - n_val

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train: n_train + n_val]
    y_val = y[n_train: n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    test_dates = feat_dates[n_train + n_val:]

    logger.info(f'  训练: {n_train}条  验证: {n_val}条  测试: {n_test}条')

    # ── Step 6: 多模型训练对比 ────────────────────────────────────────────
    logger.info('\n[Step 6] 多模型训练与评估...')
    model_config = config.to_model_config()
    model_config['probabilistic'] = True
    model_config['probabilistic_method'] = 'residual'

    models_to_compare = {
        'xgboost': {**model_config, 'model_name': 'xgboost'},
        'lightgbm': {**model_config, 'model_name': 'lightgbm'},
        'ridge': {**model_config, 'model_name': 'ridge'},
    }

    # 树模型用差分方案解决外推问题；线性模型直接预测水平值
    tree_models = {'xgboost', 'lightgbm', 'gradient_boosting', 'random_forest'}

    results = {}
    for model_name, cfg in models_to_compare.items():
        try:
            use_diff = model_name in tree_models
            res = train_and_evaluate(
                model_name, cfg,
                X_train, y_train,
                X_test, y_test,
                adapter, metadata, logger,
                target_col=target_col,
                use_diff=use_diff,
                last_y_before_test=float(y_val[-1]),
            )
            results[model_name] = res
        except ImportError as e:
            logger.warning(f'  跳过 {model_name}: {e}')

    if not results:
        logger.error('所有模型均训练失败')
        return

    # 给所有模型注入特征名，使 get_feature_importance() 能显示真实列名
    for res in results.values():
        res['model'].feature_names = feature_cols

    # ── Step 7: 业务指标评估 ──────────────────────────────────────────────
    logger.info('\n[Step 7] 业务指标评估...')
    best_name = min(results, key=lambda k: results[k]['metrics']['MAPE'])
    best_pred = results[best_name]['y_pred']

    # 反归一化还原真实量纲
    y_test_df = pd.DataFrame(
        y_test.reshape(-1, 1), columns=[target_col], index=test_dates
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

    # ── Step 8: 特征重要性 ────────────────────────────────────────────────
    logger.info('\n[Step 8] 特征重要性（XGBoost）...')
    if 'xgboost' in results:
        feat_imp = results['xgboost']['model'].get_feature_importance()
        if feat_imp is not None:
            logger.info(f'  Top 10 重要特征:\n{feat_imp.head(10).to_string(index=False)}')

            # 特征重要性 Top-10
            _PLOTTER.feature_importance_fig(
                names=feat_imp['feature'].tolist(),
                values=feat_imp['importance'].tolist(),
                top_n=10,
                title='XGBoost 特征重要性 Top-10',
                xlabel='Importance Score',
                save_path=os.path.join(output_dir, 'hpf_feature_importance.png'),
            )

    # ── Step 9: 可视化 ────────────────────────────────────────────────────
    logger.info('\n[Step 9] 生成可视化图表...')

    # 反归一化各模型预测结果用于绘图
    plot_results = {}
    for mn, res in results.items():
        pred_df = pd.DataFrame(
            res['y_pred'].reshape(-1, 1), columns=[target_col], index=test_dates
        )
        pred_orig = adapter._denormalize(pred_df, metadata)

        # 同步反归一化概率区间
        prob = res['prob_result']
        if prob.lower is not None:
            lower_df = pd.DataFrame(prob.lower.reshape(-1, 1), columns=[target_col])
            upper_df = pd.DataFrame(prob.upper.reshape(-1, 1), columns=[target_col])
            lower_orig = adapter._denormalize(lower_df, metadata)[target_col].values
            upper_orig = adapter._denormalize(upper_df, metadata)[target_col].values
            from tsf_frame.models.base_model import ProbabilisticPrediction
            prob_orig = ProbabilisticPrediction(
                mean=pred_orig[target_col].values,
                lower=lower_orig,
                upper=upper_orig,
            )
        else:
            from tsf_frame.models.base_model import ProbabilisticPrediction
            prob_orig = ProbabilisticPrediction(mean=pred_orig[target_col].values)

        plot_results[mn] = {
            'y_pred': pred_orig[target_col].values,
            'prob_result': prob_orig,
            'metrics': res['metrics'],
        }

    plot_forecast_comparison(
        y_test_orig[target_col].values, plot_results,
        test_dates, target_col,
        save_path=os.path.join(output_dir, 'hpf_forecast_comparison.png'),
    )
    plot_metrics_comparison(
        {k: v['metrics'] for k, v in results.items()},
        save_path=os.path.join(output_dir, 'hpf_metrics_comparison.png'),
    )

    # ── Step 10: 政策情景分析 ─────────────────────────────────────────────
    logger.info('\n[Step 10] 政策情景分析...')
    base_forecast_df = y_pred_orig.copy()

    policy_scenarios = {
        '上调缴存比例+2pp（+5%效应）': adapter.get_policy_adjusted_forecast(
            base_forecast_df, {target_col: 0.05}
        ),
        '下调缴存比例-2pp（-5%效应）': adapter.get_policy_adjusted_forecast(
            base_forecast_df, {target_col: -0.05}
        ),
        '引进新企业缴存（+10%效应）': adapter.get_policy_adjusted_forecast(
            base_forecast_df, {target_col: 0.10}
        ),
    }

    plot_policy_scenario(
        base_forecast_df, policy_scenarios, test_dates,
        target_col,
        save_path=os.path.join(output_dir, 'hpf_policy_scenario.png'),
    )
    for scenario, adjusted in policy_scenarios.items():
        diff_pct = (adjusted[target_col].mean() / base_forecast_df[target_col].mean() - 1) * 100
        logger.info(f'  {scenario}: 平均变化 {diff_pct:+.1f}%')

    # ── 汇总 ──────────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 65)
    logger.info('  预测结果汇总')
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
