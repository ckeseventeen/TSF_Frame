# TSF_Frame 用户指南

> 面向**使用者**。按 section 顺序读或直接跳到你关心的部分。
> 想理解"为什么这么设计"请看 [ARCHITECTURE.md](ARCHITECTURE.md)；想扩展新业务/新模型请看 [EXTENDING.md](EXTENDING.md)。

---

## 目录

1. [安装](#1-安装)
2. [30 秒跑通 HPF pipeline](#2-30-秒跑通-hpf-pipeline)
3. [数据准备](#3-数据准备)
4. [业务适配器（HPFAdapter）](#4-业务适配器hpfadapter)
5. [特征工程](#5-特征工程)
6. [经典机器学习模型](#6-经典机器学习模型)
7. [深度学习模型](#7-深度学习模型)
8. [概率预测](#8-概率预测)
9. [公共数据集](#9-公共数据集)
10. [可视化](#10-可视化)
11. [监控 —— 快速上手](#11-监控--快速上手)
12. [监控 —— 完整流程与 FAQ](#12-监控--完整流程与-faq)
13. [命令速查表](#13-命令速查表)

---

## 1. 安装

### 环境要求

- Python 3.8+
- Windows / Linux / macOS
- 建议新建虚拟环境

### 安装步骤

```bash
git clone <repo-url> TSF_Frame
cd TSF_Frame

# 方式 A：开发模式安装（推荐）
pip install -e .
pip install -r requirements.txt

# 方式 B：不安装，直接跑（pipelines/* 自带 sys.path bootstrap）
pip install -r requirements.txt
```

### 验证

```bash
python -c "import tsf_frame; print(tsf_frame.__version__)"
# → 0.2.0

pytest tests/
# → 7 passed
```

---

## 2. 30 秒跑通 HPF pipeline

```bash
python pipelines/run_hpf_forecast.py
```

控制台输出类似：

```
[HPF] 生成 144 条月度数据 (2012-01 ~ 2023-12)
[HPF] XGBoost: MAPE=0.93%  R²=0.9509
[HPF] 报表 → experiments/results/hpf_forecast_YYYYMMDD_HHMMSS.csv
```

要同时跑监控演示：

```bash
python pipelines/examples/hpf_monitoring_example.py
```

会生成：
- `logs/hpf_monitor.db` —— SQLite 持久化
- `logs/hpf_alerts.log` —— WARNING+ 告警文件
- `logs/hpf_reports/hpf_report_*.png` —— 静态报表

---

## 3. 数据准备

### 期望的 DataFrame 格式

```python
import pandas as pd

df = pd.DataFrame({
    'date':               pd.date_range('2012-01-31', periods=144, freq='ME'),
    'monthly_deposit':    [...],   # 缴存额
    'monthly_withdrawal': [...],   # 提取额
    'loan_balance':       [...],   # 贷款余额
    'depositor_count':    [...],   # 缴存人数
    # 其它字段按业务添加
})
```

要求：

- `date` 列必须是 `datetime64[ns]`
- 频率为月度（`freq='ME'` 或 `'MS'`）
- 非负业务量列不允许出现负值（`HPFAdapter.validate_data` 会拒绝）

### 数据存放

```
data/
├── raw/        # 原始数据（gitignore）
├── processed/  # 清洗后的数据（gitignore）
└── hpf/        # HPF 业务数据（gitignore）
```

---

## 4. 业务适配器（HPFAdapter）

适配器是"业务防腐层"，负责校验、预处理、反变换、业务指标。

### 基本用法

```python
from configs.hpf import HPFConfig
from tsf_frame.business.hpf_adapter import HPFAdapter

cfg = HPFConfig()
adapter = HPFAdapter(cfg.to_adapter_config())   # 注意 .to_adapter_config()

# 1) 校验
ok, msg = adapter.validate_data(df)
assert ok, msg

# 2) 预处理（异常值处理 + 归一化）
processed_df, meta = adapter.preprocess(df)

# 3) 反归一化（预测完毕后）
pred_df = adapter.postprocess(y_pred, meta)    # 返回 DataFrame

# 4) 业务指标
metrics = adapter.get_business_metrics(y_true_df, y_pred_df)
# → {'monthly_deposit_mape': 0.0093, 'monthly_deposit_direction_accuracy': 0.87, ...}
```

### 常用配置

```python
cfg = HPFConfig()
cfg.data.target_columns  = ['monthly_deposit']       # 预测目标
cfg.feature.use_time     = True                      # 启用时间特征
cfg.feature.lags         = [1, 3, 6, 12]             # 滞后阶数
cfg.feature.rolling_windows = [3, 6, 12]             # 滚动窗口
cfg.model.model_name     = 'xgboost'
cfg.model.probabilistic  = True
cfg.model.probabilistic_method = 'residual'          # residual / mc_dropout / quantile
```

### 政策情景分析（HPF 特有）

```python
scenarios = adapter.get_policy_adjusted_forecast(
    base_forecast=y_pred,
    shocks={'2024-06': -0.08, '2024-12': +0.05},   # ±8%、+5%
)
```

---

## 5. 特征工程

所有特征工程类遵循 sklearn 的 fit → transform 模式，保证不窥视未来。

### 单个工程器

```python
from tsf_frame.features.engineering import (
    TimeFeatureEngineer, LagFeatureEngineer,
    RollingFeatureEngineer, DifferenceFeatureEngineer,
)

# target_cols 是列表；省略时自动选所有数值列
lag = LagFeatureEngineer({'target_cols': ['monthly_deposit'], 'lags': [1, 3, 6]})
lag.fit(train_df)
train_feat = lag.transform(train_df)
test_feat  = lag.transform(test_df)            # 用训练期 fit 的参数

# TimeFeatureEngineer 需要 DatetimeIndex 或通过 time_col 指定时间列
time_eng = TimeFeatureEngineer({'time_col': 'date', 'features': ['month', 'quarter', 'year']})
```

### 组合工程器（推荐）

工厂函数 `create_feature_engineer(feature_types, config)`：第一个参数是启用的特征类型列表，第二个参数是嵌套的子配置字典（键名固定为 `{type}_config`）。

```python
from tsf_frame.features.engineering import create_feature_engineer

feature_config = {
    'time_config':       {'features': ['month', 'quarter', 'year']},
    'lag_config':        {'target_cols': ['monthly_deposit'], 'lags': [1, 3, 6, 12]},
    'rolling_config':    {'target_cols': ['monthly_deposit'],
                          'windows': [3, 6, 12], 'stats': ['mean', 'std']},
    'difference_config': {'target_cols': ['monthly_deposit'], 'periods': [1, 12]},
}

engineer = create_feature_engineer(
    feature_types=['time', 'lag', 'rolling', 'difference'],
    config=feature_config,
)

# 要让 time 特征工作，df 必须有 DatetimeIndex 或 time_col 指定的列
df_with_idx = df.set_index('date')
df_feat = engineer.fit_transform(df_with_idx).dropna()

X = df_feat.drop(columns=['monthly_deposit'])
y = df_feat['monthly_deposit']
```

### 特征选择

`get_selector(name, config_dict)` —— config_dict 按 kwargs 展开给选择器构造函数。

```python
from tsf_frame.features import get_selector, get_reducer

selector = get_selector('kbest', {'k': 20})
selector.fit(X_train, y_train)
X_train_sel = selector.transform(X_train)

# 降维
reducer = get_reducer('pca', {'n_components': 10})
X_train_pca = reducer.fit_transform(X_train_sel)
```

可选选择器：`kbest` / `rfe` / `model_based` / `lasso` / `correlation` / `variance`。
可选降维器：`pca`。

---

## 6. 经典机器学习模型

11 个模型统一 `BaseModel` 接口，可互换。

```python
from tsf_frame.models.classical.ml_models import get_ml_model, MODEL_REGISTRY

print(list(MODEL_REGISTRY.keys()))
# ['linear', 'ridge', 'lasso', 'random_forest', 'gbdt',
#  'xgboost', 'lightgbm', 'catboost', 'svr', 'knn', 'decision_tree']

model = get_ml_model('xgboost', {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'probabilistic': True,                # 启用概率预测
    'probabilistic_method': 'residual',
})

# 注意：BaseMLModel.fit 接收元组 (X, y)，不是分开的两个参数
model.fit((X_train.values, y_train.values),
          val_data=(X_val.values, y_val.values))
y_pred = model.predict(X_test.values)
prob   = model.predict_probabilistic(X_test.values)
print(prob.mean.shape, prob.lower.shape, prob.upper.shape)
```

### XGBoost 的特殊处理

框架为 XGBoost 自动使用**一阶差分 + cumsum 还原**策略，避免树模型无法外推的问题。详见 [EXTENDING.md](EXTENDING.md) 技术决策章节。

---

## 7. 深度学习模型

```python
from tsf_frame.models.transformer import get_dl_model, DL_MODEL_REGISTRY

print(list(DL_MODEL_REGISTRY.keys()))
# ['lstm', 'transformer', 'informer', 'autoformer', 'itransformer', 'timesnet']

model = get_dl_model('autoformer', {
    'input_size':  10,
    'output_size': 1,
    'seq_len':     36,
    'pred_len':    12,
    'd_model':     64,
    'n_heads':     4,
    'e_layers':    2,
    'epochs':      50,
    'batch_size':  16,
    'learning_rate': 1e-3,
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
})

model.fit(train_loader, val_loader)
prob = model.predict_probabilistic(test_loader)   # MC Dropout 概率预测
```

### 月度稀疏数据提醒

12 年 × 12 月 = 144 条样本对 Transformer 系列是**极少**的。推荐优先考虑：

1. XGBoost / LightGBM（强特征）
2. LSTM（小模型）
3. Autoformer / TimesNet（只有在加更多外生变量时）

---

## 8. 概率预测

三种策略均返回统一的 `ProbabilisticPrediction`：

```python
from tsf_frame.models.base_model import ProbabilisticPrediction

prob: ProbabilisticPrediction = model.predict_probabilistic(X_test)

prob.mean        # np.ndarray  点预测
prob.lower       # 95% 置信下界
prob.upper       # 95% 置信上界
prob.std         # 标准差
prob.confidence  # float, 默认 0.95
prob.to_dict()   # 可序列化字典
```

### 策略选择

| 方法 | 适用模型 | 参数 | 特点 |
|------|---------|------|------|
| `residual` | 所有模型 | `probabilistic_method='residual'` | 用验证残差经验分布，简单稳健 |
| `mc_dropout` | 深度学习 | 要求模型含 Dropout 层 | 贝叶斯近似，能捕捉模型不确定性 |
| `quantile` | XGBoost/LightGBM | 底层走 quantile regression | 精确，训练 3 个分位数 |

### 可视化置信区间

```python
import matplotlib.pyplot as plt
plt.plot(ts, prob.mean, label='预测')
plt.fill_between(ts, prob.lower, prob.upper, alpha=0.3, label='95% CI')
plt.legend()
```

---

## 9. 公共数据集

内置 `BaseDataset` + 几个公共数据集（ETT / Electricity / Weather / Traffic / ILI）。

```python
from tsf_frame.data.datasets.public_datasets import ETTDataset

ds = ETTDataset(
    root_path='data/raw/ETT/',
    data_path='ETTh1.csv',
    flag='train',
    seq_len=96,
    pred_len=24,
)
print(len(ds), ds[0][0].shape)
```

完整示例：`pipelines/examples/public_dataset_workflow.py`

---

## 10. 可视化

`visualization/` 下的类均在 `base_visualizer.py` 顶部统一配置了中文字体（`_CN_FONTS`），避免在用户代码里重复设定。

```python
from tsf_frame.visualization.base_visualizer import _CN_FONTS
import matplotlib
matplotlib.rcParams['font.sans-serif'] = _CN_FONTS    # 复用
matplotlib.rcParams['axes.unicode_minus'] = False
```

常用绘图能力：

- `PredictionVisualizer` —— 真实 vs 预测 + 置信带
- `FeatureImportanceVisualizer` —— 树模型 feature_importances_
- `TrainingCurveVisualizer` —— epoch loss 曲线
- `MonitoringReportVisualizer` —— 8 子图的监控报表

---

## 11. 监控 —— 快速上手

```python
from configs.hpf import HPFConfig
from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.monitoring import HPFMonitor

cfg = HPFConfig()
adapter = HPFAdapter(cfg.to_adapter_config())

monitor = HPFMonitor(
    model_id='xgboost_deposit_v1',
    hpf_adapter=adapter,
    config=cfg.monitoring,
    reference_data=X_train,                     # 训练期特征作漂移基线
    performance_baseline={'mape': val_mape},    # 验证集 MAPE 作性能基线
)

# 逐条记录预测
for ts, (_, feat_row), y_true in zip(test_ts, X_test.iterrows(), y_test):
    feat_df = feat_row.to_frame().T
    prob = model.predict_probabilistic(feat_df.values)
    monitor.record_prediction(
        timestamp=ts,
        features=feat_df,
        prediction=prob,                         # ProbabilisticPrediction
        actual_value=y_true,
        target_col='monthly_deposit',
    )

# 聚合状态 + 必要时自动生成报表
status = monitor.check_status()
print(status.alert_level, status.recommendations)

report_path = monitor.generate_report()
print(f'Report saved → {report_path}')
```

运行 `pipelines/examples/hpf_monitoring_example.py` 看完整演示。

---

## 12. 监控 —— 完整流程与 FAQ

### 12.1 监控的 5 条主线

1. **业务规则**（R1-R5，`HPFBusinessRuleChecker`）
   - R1_NON_NEGATIVE：预测出现负值 → CRITICAL
   - R2_SUDDEN_CHANGE：月度变化 > 30% → WARNING
   - R3_MISSING_RATE：目标列缺失率 > 10% → WARNING
   - R4_FREQ_CONTINUITY：月频不连续 → ERROR
   - R5_OUTLIER_ECHO：预测值位于 3σ 之外 → WARNING

2. **性能监控**（`PerformanceMonitor`）
   - 滑窗内的 MAE / RMSE / MAPE / R²
   - 超过 `mape_warning` / `mape_critical` 触发告警

3. **数据漂移**（`DataDriftDetector`）
   - 特征维度的 PSI / KS
   - 默认 PSI warn=0.1 / crit=0.25

4. **概念漂移**（`ConceptDriftDetector`）
   - 残差趋势偏移、分布变化

5. **业务指标**（`HPFAdapter.get_business_metrics` 滑窗）
   - MAPE / direction_accuracy / yoy_mae / qoq_mae
   - `direction_accuracy < 0.6` 触发 WARNING

### 12.2 告警通道

P0 版本使用**日志 + 文件**双通道。在 `HPFMonitoringConfig` 中配置：

```python
config.log_dir          = './logs'
config.log_name         = 'hpf_monitor'
config.alert_log_file   = './logs/hpf_alerts.log'     # 仅 WARNING+
config.sqlite_path      = './logs/hpf_monitor.db'
config.report_on_alert  = True                        # WARNING+ 时自动出图
config.report_dir       = './logs/hpf_reports'
```

告警同时写 3 处：
- 主日志（所有级别）
- `hpf_alerts.log`（仅 WARNING+）
- SQLite `alerts` 表（所有级别）

### 12.3 SQLite 结构

```bash
sqlite3 logs/hpf_monitor.db ".tables"
# predictions  metrics_snapshot  alerts

sqlite3 logs/hpf_monitor.db "SELECT level, rule_id, COUNT(*) FROM alerts GROUP BY level, rule_id;"
```

三张表的字段详见 [ARCHITECTURE.md](ARCHITECTURE.md) 第 5 节。

### 12.4 冷启动降级

累积样本 `< cold_start_months`（默认 6 月）时，只跑业务规则，不做 PSI / error_trend。避免样本不足误报。

### 12.5 静态报表

8 个子图的 PNG：

1. 预测 vs 实际时序 + 95% CI
2. 业务指标趋势（MAPE / direction_accuracy）
3. YoY / QoQ 误差柱图
4. 规则违反计数堆叠柱图
5. PSI 漂移热力图
6. 残差分布直方图
7. 告警时间线散点
8. 政策情景 overlay（如果有 `get_policy_adjusted_forecast` 输出）

输出到 `{report_dir}/hpf_report_{model_id}_{YYYYMMDD_HHMMSS}.png`。

### 12.6 FAQ

**Q: 我能把监控接到 Airflow 里吗？**
A: 可以。`HPFMonitor.record_prediction` 是幂等的，在任务里 `for ts in batch: monitor.record_prediction(...)` 即可。SQLite WAL 模式支持并发读。

**Q: 重启后 deque 窗口会丢？**
A: 内存 deque 会，但 SQLite 里的 predictions / metrics_snapshot / alerts 全部持久化。`HPFMonitoringReport` 从 SQLite 读取，不依赖内存窗口。如果你需要重启后滑窗延续，可以在 `__init__` 后调用 `store.query_predictions(...)` 重填 deque（当前 P0 未实现，属 P1）。

**Q: 告警想推到钉钉 / 企业微信 / 邮件？**
A: P0 只做日志 + 文件。要接外部通道，给 `AlertManager` 添加一个 handler：
```python
def my_handler(alert):
    requests.post('https://...', json={'text': alert.message})
monitor.model_monitor.alert_manager.add_handler(my_handler)
```

**Q: 阈值怎么调？**
A: 修改 `HPFMonitoringConfig` 的字段。所有阈值（MAPE / PSI / direction_accuracy / sudden_change_ratio / missing_rate_threshold）都是配置项，不需要改代码。

---

## 13. 命令速查表

```bash
# 安装
pip install -e .

# 端到端 HPF 预测
python pipelines/run_hpf_forecast.py

# 通用训练器（可换模型）
python pipelines/train_model.py --model xgboost --dataset hpf

# 各模块演示
python pipelines/examples/feature_engineering_example.py
python pipelines/examples/probabilistic_example.py
python pipelines/examples/hpf_dl_example.py
python pipelines/examples/hpf_monitoring_example.py
python pipelines/examples/public_dataset_workflow.py

# 测试
pytest tests/
pytest tests/ -v
pytest tests/test_metrics.py::test_perfect_prediction

# 查 SQLite 监控库
sqlite3 logs/hpf_monitor.db ".tables"
sqlite3 logs/hpf_monitor.db "SELECT level, COUNT(*) FROM alerts GROUP BY level;"
```

---

## 下一步

- 改配置不改代码 → 读 [ARCHITECTURE.md 第 6 节](ARCHITECTURE.md#6-配置层)
- 添加新业务 / 新模型 → 读 [EXTENDING.md](EXTENDING.md)
- 踩到 bug 或阈值不合适 → [EXTENDING.md 第 6 节](EXTENDING.md#6-关键技术决策)
