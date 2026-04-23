import sys
import os
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
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from tsf_frame.models.classical.ml_models import get_ml_model
from tsf_frame.models.transformer.transformer_models import get_dl_model
from tsf_frame.data.datasets.public_datasets import get_dataset
from tsf_frame.utils.metrics import MetricsCalculator


def generate_sample_data():
    np.random.seed(42)
    
    n_samples = 200
    dates = [datetime.now() - timedelta(days=n_samples - i) for i in range(n_samples)]
    
    trend = np.linspace(100, 400, n_samples)
    seasonality = 50 * np.sin(np.linspace(0, 8 * np.pi, n_samples))
    noise = np.random.normal(0, 15, n_samples)
    
    values = trend + seasonality + noise
    features = np.random.randn(n_samples, 5)
    
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        **{f'feature_{i}': features[:, i] for i in range(5)}
    })
    df.set_index('date', inplace=True)
    
    return df


def prepare_sliding_window_data(data, seq_len=12, pred_len=1):
    X, y = [], []
    values = data.values
    
    for i in range(len(values) - seq_len - pred_len + 1):
        X.append(values[i:i + seq_len].flatten())
        y.append(values[i + seq_len:i + seq_len + pred_len, 0])
    
    return np.array(X), np.array(y)


def plot_probabilistic_predictions(dates, y_true, pred_result, title):
    plt.figure(figsize=(12, 6))
    
    y_true = np.array(y_true).flatten()
    pred_mean = np.array(pred_result.mean).flatten()
    
    plt.plot(dates, y_true, label='真实值', color='blue', alpha=0.7)
    plt.plot(dates, pred_mean, label='预测均值', color='red', linewidth=2)
    
    if pred_result.lower is not None and pred_result.upper is not None:
        pred_lower = np.array(pred_result.lower).flatten()
        pred_upper = np.array(pred_result.upper).flatten()
        plt.fill_between(dates, pred_lower, pred_upper, 
                        alpha=0.3, color='red', label='95% 置信区间')
    
    plt.xlabel('日期')
    plt.ylabel('值')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {title.replace(' ', '_')}.png")


def test_ml_probabilistic():
    print("=" * 80)
    print("测试机器学习模型概率预测（残差分布方法）")
    print("=" * 80)
    
    print("\n[1. 加载数据...")
    data = generate_sample_data()
    print(f"  数据形状: {data.shape}")
    
    print("\n2. 准备滑动窗口数据...")
    seq_len = 12
    X, y = prepare_sliding_window_data(data, seq_len=seq_len)
    print(f"  X 形状: {X.shape}")
    print(f"  y 形状: {y.shape}")
    
    print("\n3. 划分数据集...")
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_test = data.index[train_size + seq_len: train_size + seq_len + len(y_test)]
    
    print("\n4. 创建并训练模型（启用概率预测）...")
    model_config = {
        'probabilistic': True,
        'probabilistic_method': 'residual',
        'confidence_level': 0.95,
        'n_estimators': 100,
        'max_depth': 10,
        'random_seed': 42
    }
    
    model = get_ml_model('random_forest', model_config)
    print("  训练中...")
    history = model.fit((X_train, y_train), (X_test, y_test))
    print("  ✓ 训练完成")
    
    print("\n5. 进行概率预测...")
    pred_result = model.predict_probabilistic(X_test)
    
    print("  预测结果:")
    print(f"    均值形状: {pred_result.mean.shape}")
    if pred_result.lower is not None:
        print(f"    下界形状: {pred_result.lower.shape}")
        print(f"    上界形状: {pred_result.upper.shape}")
    
    print("\n6. 评估预测...")
    mse = MetricsCalculator.mse(y_test, pred_result.mean)
    mae = MetricsCalculator.mae(y_test, pred_result.mean)
    mape = MetricsCalculator.mape(y_test, pred_result.mean)
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    print("\n7. 可视化结果...")
    plot_probabilistic_predictions(
        dates_test, y_test, pred_result, 
        "机器学习模型概率预测 - 残差分布方法"
    )
    
    print("\n8. 前5个预测样本:")
    for i in range(min(5, len(y_test))):
        print(f"  样本 {i+1}:")
        print(f"    真实值: {y_test[i].item():.2f}")
        print(f"    预测值: {pred_result.mean[i].item():.2f}")
        if pred_result.lower is not None:
            print(f"    置信区间: [{pred_result.lower[i].item():.2f}, {pred_result.upper[i].item():.2f}]")


def test_dl_probabilistic():
    print("\n" + "=" * 80)
    print("测试深度学习模型概率预测（MC Dropout方法）")
    print("=" * 80)
    
    print("\n1. 加载数据...")
    data = generate_sample_data()
    print(f"  数据形状: {data.shape}")
    
    print("\n2. 准备滑动窗口数据...")
    seq_len = 12
    n_features = data.shape[1]
    X, y = prepare_sliding_window_data(data, seq_len=seq_len)
    
    X = X.reshape(X.shape[0], seq_len, n_features)
    print(f"  X 形状: {X.shape}")
    print(f"  y 形状: {y.shape}")
    
    print("\n3. 划分数据集...")
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_test = data.index[train_size + seq_len: train_size + seq_len + len(y_test)]
    
    print("\n4. 创建并训练模型（启用概率预测）...")
    model_config = {
        'probabilistic': True,
        'probabilistic_method': 'mc_dropout',
        'num_samples': 50,
        'confidence_level': 0.95,
        'input_size': n_features,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.2,
        'output_size': 1,
        'learning_rate': 0.001,
        'train_epochs': 50,
        'batch_size': 16
    }
    
    model = get_dl_model('lstm', model_config)
    print("  训练中...")
    history = model.fit((X_train, y_train), (X_test, y_test))
    print("  ✓ 训练完成")
    
    print("\n5. 进行概率预测...")
    pred_result = model.predict_probabilistic(X_test)
    
    print("  预测结果:")
    print(f"    均值形状: {pred_result.mean.shape}")
    if pred_result.lower is not None:
        print(f"    下界形状: {pred_result.lower.shape}")
        print(f"    上界形状: {pred_result.upper.shape}")
    if pred_result.std is not None:
        print(f"    标准差形状: {pred_result.std.shape}")
    if pred_result.samples is not None:
        print(f"    采样形状: {pred_result.samples.shape}")
    
    print("\n6. 评估预测...")
    mse = MetricsCalculator.mse(y_test, pred_result.mean)
    mae = MetricsCalculator.mae(y_test, pred_result.mean)
    mape = MetricsCalculator.mape(y_test, pred_result.mean)
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    print("\n7. 可视化结果...")
    plot_probabilistic_predictions(
        dates_test, y_test, pred_result, 
        "深度学习模型概率预测 - MC Dropout方法"
    )
    
    print("\n8. 前5个预测样本:")
    for i in range(min(5, len(y_test))):
        print(f"  样本 {i+1}:")
        print(f"    真实值: {y_test[i].item():.2f}")
        print(f"    预测值: {pred_result.mean[i].item():.2f}")
        if pred_result.lower is not None:
            print(f"    置信区间: [{pred_result.lower[i].item():.2f}, {pred_result.upper[i].item():.2f}]")
        if pred_result.std is not None:
            print(f"    标准差: {pred_result.std[i].item():.4f}")


def main():
    print("=" * 80)
    print("TSF_Frame 概率预测示例")
    print("=" * 80)
    
    print("\n混合策略说明:")
    print("  - 机器学习模型: 残差分布方法")
    print("  - 深度学习模型: MC Dropout方法")
    print("  - 可配置: 通过配置文件切换精确/概率预测")
    
    try:
        test_ml_probabilistic()
    except Exception as e:
        print(f"\n机器学习测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_dl_probabilistic()
    except Exception as e:
        print(f"\n深度学习测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("示例运行完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
