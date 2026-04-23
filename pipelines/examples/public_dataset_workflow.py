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

from tsf_frame.data.datasets.public_datasets import get_dataset, list_available_datasets
from tsf_frame.models.classical.ml_models import get_ml_model
from tsf_frame.utils.metrics import MetricsCalculator


def main():
    print("=" * 80)
    print("公开数据集训练和推理流程测试")
    print("=" * 80)
    
    # 1. 列出可用数据集
    print("\n1. 列出可用数据集")
    datasets = list_available_datasets()
    for category, dataset_list in datasets.items():
        print(f"\n{category.upper()}:")
        for ds in dataset_list:
            print(f"  - {ds}")
    
    # 2. 加载数据集
    print("\n2. 加载Darts数据集")
    dataset = get_dataset('darts', {'dataset_name': 'air_passengers'})
    data = dataset.load()
    
    print(f"\n数据集信息:")
    print(f"  名称: {dataset.get_metadata()['dataset_name']}")
    print(f"  形状: {dataset.get_metadata()['shape']}")
    print(f"  列: {dataset.get_metadata()['columns']}")
    print(f"\n前5行数据:")
    print(data.head())
    
    # 3. 数据预处理
    print("\n3. 数据预处理")
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 4. 准备训练数据
    print("\n4. 准备训练数据")
    
    seq_len = 12
    X, y = [], []
    
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i:i + seq_len].flatten())
        y.append(scaled_data[i + seq_len, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"  X形状: {X.shape}")
    print(f"  y形状: {y.shape}")
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\n  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 5. 训练模型
    print("\n5. 训练模型")
    
    model_config = {
        'n_estimators': 500,
        'max_depth': 100,
        'random_seed': 42
    }
    
    model = get_ml_model('random_forest', model_config)
    print("  开始训练 RandomForest 模型...")
    
    history = model.fit((X_train, y_train), (X_test, y_test))
    print("  训练完成!")
    
    # 6. 推理预测
    print("\n6. 推理预测")
    
    y_pred = model.predict(X_test)
    print(f"  预测完成，形状: {y_pred.shape}")
    
    # 7. 计算评估指标
    print("\n7. 计算评估指标")
    
    y_test_original = []
    y_pred_original = []
    
    for i in range(len(y_test)):
        temp_test = np.zeros((1, data.shape[1]))
        temp_pred = np.zeros((1, data.shape[1]))
        temp_test[0, 0] = y_test[i]
        if len(y_pred.shape) == 1:
            temp_pred[0, 0] = y_pred[i]
        else:
            temp_pred[0, 0] = y_pred[i, 0]
        y_test_original.append(scaler.inverse_transform(temp_test)[0, 0])
        y_pred_original.append(scaler.inverse_transform(temp_pred)[0, 0])
    
    y_test_original = np.array(y_test_original)
    y_pred_original = np.array(y_pred_original)
    
    mse = MetricsCalculator.mse(y_test_original, y_pred_original)
    mae = MetricsCalculator.mae(y_test_original, y_pred_original)
    rmse = MetricsCalculator.rmse(y_test_original, y_pred_original)
    mape = MetricsCalculator.mape(y_test_original, y_pred_original)
    
    print(f"\n  评估结果:")
    print(f"    MSE:  {mse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAPE: {mape:.4f}%")
    
    print(f"\n  预测示例 (前5个):")
    for i in range(min(5, len(y_test_original))):
        print(f"    真实值: {y_test_original[i]:.2f}, 预测值: {y_pred_original[i]:.2f}")
    
    print("\n" + "=" * 80)
    print("流程测试成功!")
    print("=" * 80)


if __name__ == '__main__':
    main()
