"""
住房公积金（HPF）业务预测 - Moirai 零样本 (Zero-shot) 预测示例
HPF business forecasting - Moirai Zero-shot edition

工作流 / Workflow:
  1. 生成模拟公积金月度数据
  2. 数据验证 + 业务适配器预处理 (zscore 归一化非常重要)
  3. 特征工程 (2D→3D 序列构造)
  4. 使用 PretrainedMoiraiModel 进行零样本预测 (不需要梯度下降训练)
  5. 业务指标评估 + 可视化

运行方式 / Run:
    python pipelines/examples/hpf_moirai_zeroshot_example.py
    
注意：运行本示例需要安装 uni2ts 和 gluonts：
    pip install uni2ts gluonts
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
matplotlib.use('Agg')


from tsf_frame.visualization import PredictionPlotter
from configs.hpf.hpf_config import HPFConfig
from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.features.mixed_feature_handler import MixedFeatureHandler
from tsf_frame.utils.metrics import MetricsCalculator
from tsf_frame.utils.logger import get_logger

# 导入 moirai 模型工厂函数
from tsf_frame.models.moirai.moirai_model import get_moirai_model
from tsf_frame.models.base_model import ProbabilisticPrediction

# ─── 1. 数据生成 ──────────────────────────────────────────────────────────────
def generate_hpf_data(years: int = 20, start_year: int = 2012) -> pd.DataFrame:
    np.random.seed(42)
    n_months = years * 12
    dates = pd.date_range(start=f'{start_year}-01-01', periods=n_months, freq='MS')
    t = np.arange(n_months)
    month_idx = dates.month.values

    # 长期增长趋势 + 年度季节性 + 噪声
    deposit_trend = 80 + 1.2 * t
    deposit_seasonal = 10 * np.sin(2 * np.pi * (month_idx - 3) / 12)
    deposit_noise = np.random.normal(0, 4, n_months)
    monthly_deposit = np.maximum(deposit_trend + deposit_seasonal + deposit_noise, 10)

    data = pd.DataFrame({
        'monthly_deposit': monthly_deposit,
    }, index=dates)

    return data

# ─── 2. 可视化辅助函数 ────────────────────────────────────────────────────────
_PLOTTER = PredictionPlotter(figsize=(14, 10), dpi=120)

def plot_forecast_comparison(y_true, y_pred, y_lower, y_upper, dates, target_col, save_path):
    models_pred = {
        'Moirai Zero-shot': y_pred
    }
    _PLOTTER.forecast_comparison_fig(
        x=dates, y_true=y_true,
        models_pred=models_pred,
        best_interval=(y_pred, y_lower, y_upper) if y_lower is not None else None,
        target_label=f'{target_col} Moirai预测', ylabel='亿元',
        save_path=save_path,
    )

# ─── 3. 主流程 ────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, 'logs', 'outputs', 'hpf_moirai')
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger('hpf_moirai_example', log_dir=os.path.join(project_root, 'logs', 'runs'))
    
    logger.info('=' * 65)
    logger.info('  住房公积金(HPF)业务预测 - Moirai 零样本版')
    logger.info('=' * 65)

    try:
        import uni2ts
        import gluonts
    except ImportError:
        logger.error("运行本示例需要 uni2ts 和 gluonts 库。请执行: pip install uni2ts gluonts")
        return

    # 配置
    config = HPFConfig()
    config.data.target_columns = ['monthly_deposit']
    seq_len = 24  # Moirai 需要更长的上下文来理解周期
    pred_len = 1
    target_col = config.data.target_columns[0]

    # 生成数据并预处理
    logger.info('[Step 1] 生成数据与预处理...')
    raw_data = generate_hpf_data()
    adapter = HPFAdapter(config.to_adapter_config())
    processed_data, metadata = adapter.preprocess(raw_data)

    # 构造序列
    logger.info('[Step 2] 构建 3D 滑动窗口序列...')
    handler = MixedFeatureHandler(
        time_varying_cols=[target_col], static_cols=[],
        target_col=target_col, seq_len=seq_len, pred_len=pred_len
    )
    handler.fit(processed_data)
    X_seq, y_seq = handler.create_sequences(processed_data)
    seq_dates = processed_data.index[seq_len:]

    # 划分数据集 (测试集留最后12个月)
    n_test = 12
    n_val = 12
    n_train = len(X_seq) - n_test - n_val

    X_train, y_train = X_seq[:n_train], y_seq[:n_train]
    X_test, y_test = X_seq[-n_test:], y_seq[-n_test:]
    test_dates = seq_dates[-n_test:]

    # 实例化并运行 Moirai 零样本模型
    logger.info('\n[Step 3] 加载 Moirai 零样本预训练模型 (初次运行可能需要下载权重)...')
    moirai_config = {
        'moirai_size': 'small', # 可选 small, base, large
        'batch_size': 16,
        'pred_len': pred_len,
        'device': device,
        'target_idx': 0
    }
    
    model = get_moirai_model('moirai_zeroshot', moirai_config)
    
    logger.info('[Step 4] 模型已在内部跳过 fit() (因使用了 Moirai 原生概率预测)...')
    model.fit((X_train, y_train))
    
    logger.info('[Step 5] 运行 probabilistic_predict() 提取原生概率分布分位数 (P10~P90)...')
    y_pred_raw, y_lower_raw, y_upper_raw = model.probabilistic_predict(X_test, quantiles=[0.1, 0.9])
    
    y_pred = y_pred_raw.flatten()
    y_lower = y_lower_raw.flatten()
    y_upper = y_upper_raw.flatten()

    # 反归一化评估
    logger.info('\n[Step 6] 业务评估与可视化...')
    y_test_df = pd.DataFrame(y_test.flatten().reshape(-1, 1), columns=[target_col], index=test_dates)
    y_pred_df = pd.DataFrame(y_pred.reshape(-1, 1), columns=[target_col], index=test_dates)
    y_lower_df = pd.DataFrame(y_lower.reshape(-1, 1), columns=[target_col], index=test_dates)
    y_upper_df = pd.DataFrame(y_upper.reshape(-1, 1), columns=[target_col], index=test_dates)
    
    y_test_orig = adapter._denormalize(y_test_df, metadata)[target_col].values
    y_pred_orig = adapter._denormalize(y_pred_df, metadata)[target_col].values
    y_lower_orig = adapter._denormalize(y_lower_df, metadata)[target_col].values
    y_upper_orig = adapter._denormalize(y_upper_df, metadata)[target_col].values
    
    metrics = MetricsCalculator.calculate_all(y_test_orig, y_pred_orig)
    logger.info(f'  MAE={metrics["MAE"]:.4f}  MAPE={metrics["MAPE"]:.2%}  R2={metrics["R2"]:.4f}')

    # 保存图表
    save_path = os.path.join(output_dir, 'hpf_moirai_zeroshot_forecast.png')
    plot_forecast_comparison(y_test_orig, y_pred_orig, y_lower_orig, y_upper_orig, test_dates, target_col, save_path)
    logger.info(f'带有 10%~90% 分位数置信区间的预测图表已保存至: {save_path}')

if __name__ == '__main__':
    main()
