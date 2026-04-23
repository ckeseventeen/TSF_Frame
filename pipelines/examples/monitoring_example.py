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
from datetime import datetime, timedelta

from tsf_frame.monitoring import ModelMonitor


def generate_sample_data():
    np.random.seed(42)
    
    n_samples = 200
    dates = [datetime.now() - timedelta(days=n_samples - i) for i in range(n_samples)]
    
    trend = np.linspace(100, 400, n_samples)
    seasonality = 50 * np.sin(np.linspace(0, 8 * np.pi, n_samples))
    noise = np.random.normal(0, 15, n_samples)
    
    values = trend + seasonality + noise
    features = np.random.randn(n_samples, 5)
    
    return dates, features, values


def main():
    print("=" * 80)
    print("模型监测模块使用示例")
    print("=" * 80)
    
    print("\n[步骤 1] 生成示例数据")
    dates, features, values = generate_sample_data()
    print(f"  样本数量: {len(dates)}")
    print(f"  特征维度: {features.shape[1]}")
    
    print("\n[步骤 2] 创建模型监测器")
    monitor = ModelMonitor(
        model_id='sales_forecast_model',
        performance_baseline={'mape': 5.0, 'mae': 10.0},
        reference_data=features[:50],
        alert_thresholds={'mape': 7.0, 'mae': 15.0},
        retrain_thresholds={'mape': 10.0, 'mae': 20.0},
        window_size=30
    )
    print("  ✓ 监测器创建成功")
    
    print("\n[步骤 3] 模拟正常预测 (前100个样本)")
    print("  记录预测结果...")
    for i in range(100):
        date = dates[i]
        feat = features[i]
        true_value = values[i]
        pred_value = true_value + np.random.normal(0, 5, 1)[0]
        
        monitor.record_prediction(
            timestamp=date,
            features=feat,
            prediction=pred_value,
            actual_value=true_value
        )
    
    status = monitor.check_status()
    print(f"  当前状态:")
    print(f"    性能指标: {status.performance_metrics}")
    print(f"    数据漂移: {status.data_drift_detected}")
    print(f"    概念漂移: {status.concept_drift_detected}")
    print(f"    是否需要重训: {status.needs_retraining}")
    print(f"    告警级别: {status.alert_level}")
    
    print("\n[步骤 4] 模拟性能下降 (中间50个样本)")
    print("  引入较大预测误差...")
    for i in range(100, 150):
        date = dates[i]
        feat = features[i]
        true_value = values[i]
        pred_value = true_value + np.random.normal(0, 30, 1)[0]
        
        monitor.record_prediction(
            timestamp=date,
            features=feat,
            prediction=pred_value,
            actual_value=true_value
        )
    
    status = monitor.check_status()
    print(f"  当前状态:")
    print(f"    性能指标: {status.performance_metrics}")
    print(f"    数据漂移: {status.data_drift_detected}")
    print(f"    概念漂移: {status.concept_drift_detected}")
    print(f"    是否需要重训: {status.needs_retraining}")
    print(f"    告警级别: {status.alert_level}")
    print(f"    建议: {status.recommendations}")
    
    print("\n[步骤 5] 模拟数据漂移 (最后50个样本)")
    print("  改变数据分布...")
    drifted_features = features[150:] + np.random.normal(5, 1, features[150:].shape)
    for i in range(150, 200):
        date = dates[i]
        feat = drifted_features[i - 150]
        true_value = values[i]
        pred_value = true_value + np.random.normal(0, 8, 1)[0]
        
        monitor.record_prediction(
            timestamp=date,
            features=feat,
            prediction=pred_value,
            actual_value=true_value
        )
    
    status = monitor.check_status()
    print(f"  当前状态:")
    print(f"    性能指标: {status.performance_metrics}")
    print(f"    数据漂移: {status.data_drift_detected}")
    print(f"    概念漂移: {status.concept_drift_detected}")
    print(f"    是否需要重训: {status.needs_retraining}")
    print(f"    告警级别: {status.alert_level}")
    print(f"    建议: {status.recommendations}")
    
    print("\n[步骤 6] 查看告警信息")
    alerts = monitor.get_alerts(unacknowledged_only=True)
    print(f"  未确认告警数量: {len(alerts)}")
    for alert in alerts[:3]:
        print(f"    - [{alert.level.value}] {alert.message}")
    
    print("\n[步骤 7] 查看性能历史")
    perf_history = monitor.get_performance_history(limit=5)
    print(f"  最近5次性能记录:")
    for record in perf_history:
        print(f"    {record['timestamp'].strftime('%Y-%m-%d')}: {record['metrics']}")
    
    if status.needs_retraining:
        print("\n[步骤 8] 触发重新训练")
        monitor.trigger_retraining()
        print("  ✓ 重新训练已触发")
    
    print("\n" + "=" * 80)
    print("示例运行完成！")
    print("=" * 80)
    print("\n监测模块功能总结:")
    print("  ✓ 性能监测 - 持续跟踪预测准确性")
    print("  ✓ 数据漂移检测 - 检测输入数据分布变化")
    print("  ✓ 概念漂移检测 - 检测模型与数据关系变化")
    print("  ✓ 告警管理 - 多级告警机制")
    print("  ✓ 重训触发 - 自动判断是否需要重新训练")
    print("=" * 80)


if __name__ == '__main__':
    main()
