# TSF_Frame

通用时序预测框架，当前业务层聚焦住房公积金（HPF）月度指标预测。src-layout、配置化、监控闭环、可二次开发。

```
┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐
│  业务适配层 │ → │  特征工程层  │ → │  模型层    │ → │  监控与告警  │
│  (防腐层)   │   │              │   │            │   │              │
└─────────────┘   └──────────────┘   └────────────┘   └──────────────┘
     HPFAdapter      Time/Lag/Roll       17 models       HPFMonitor
                     Diff / Select       (ML + DL)       SQLite + PNG
```

---

## 快速开始

```bash
git clone <repo-url> TSF_Frame
cd TSF_Frame
pip install -e .
pip install -r requirements.txt

python pipelines/run_hpf_forecast.py
```

输出：

```
[HPF] 生成 144 条月度数据 (2012-01 ~ 2023-12)
[HPF] XGBoost: MAPE=0.93%  R²=0.9509
```

要看监控演示：

```bash
python pipelines/examples/hpf_monitoring_example.py
```

生成 `logs/hpf_monitor.db`（SQLite）、`logs/hpf_alerts.log`、`logs/hpf_reports/*.png`。

---

## 核心能力

- **17 个模型** —— 11 个经典 ML（XGBoost / LightGBM / CatBoost / Ridge / SVR / ...）+ 6 个深度学习（LSTM / Transformer / Informer / Autoformer / iTransformer / TimesNet）
- **概率预测** —— 残差分布 / MC Dropout / 分位数回归，统一 `ProbabilisticPrediction` 接口
- **特征工程** —— 时间 / 滞后 / 滚动 / 扩展 / 差分 + KBest/RFE/Lasso/PCA，全部 fit→transform 因果
- **业务防腐层** —— `HPFAdapter` 把非负、季节、政策情景、YoY/QoQ 等 HPF 规则集中封装
- **监控闭环** —— `HPFMonitor` = 性能 + 数据/概念漂移 + 业务规则（R1-R5）+ SQLite 持久化 + 8 子图静态报表
- **中文字体** —— `visualization/base_visualizer.py` 顶部统一配 `_CN_FONTS`，报表不再出方框

---

## 项目结构

```
TSF_Frame/
├── src/tsf_frame/              # 框架源码（发布包）
│   ├── business/               # BaseBusinessAdapter / HPFAdapter
│   ├── features/               # engineering / selector / mixed
│   ├── models/                 # classical / transformer / moirai
│   ├── monitoring/             # ModelMonitor + HPFMonitor + SQLite
│   ├── visualization/ utils/
│   └── data/datasets/
├── configs/                    # 顶层配置包（BaseConfig / HPFConfig）
├── pipelines/                  # 可执行脚本（run_hpf_forecast.py、examples/）
├── tests/                      # pytest（7 cases）
├── data/ experiments/ logs/    # 运行产物（gitignore）
├── docs/                       # ← 你在这里
├── setup.py  requirements.txt
```

---

## 常用命令

```bash
# 安装
pip install -e .

# 端到端预测
python pipelines/run_hpf_forecast.py

# 通用训练器
python pipelines/train_model.py --model xgboost

# 监控演示
python pipelines/examples/hpf_monitoring_example.py

# 其它独立示例
python pipelines/examples/feature_engineering_example.py
python pipelines/examples/probabilistic_example.py
python pipelines/examples/hpf_dl_example.py
python pipelines/examples/public_dataset_workflow.py

# 测试
pytest tests/
```

---

## 文档

| 文档 | 面向读者 | 内容 |
|------|---------|------|
| [docs/GUIDE.md](docs/GUIDE.md) | 使用者 | 安装、数据、适配器、特征、模型、概率、可视化、监控、命令速查 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构师 | 设计原则、目录分层、核心抽象、数据流、监控架构、打包策略 |
| [docs/EXTENDING.md](docs/EXTENDING.md) | 二次开发者 | 新业务 / 新模型 / 新规则的扩展流程、代码规范、关键技术决策、bug 记录 |

---

## 最小代码示例

```python
from configs.hpf import HPFConfig
from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.features.engineering import create_feature_engineer
from tsf_frame.models.classical.ml_models import get_ml_model
from tsf_frame.monitoring import HPFMonitor

cfg = HPFConfig()
adapter = HPFAdapter(cfg.to_adapter_config())

# 预处理
df_ok, _ = adapter.validate_data(df), None
processed, meta = adapter.preprocess(df)

# 特征（注意：时间特征需要 DatetimeIndex）
eng = create_feature_engineer(
    feature_types=['time', 'lag', 'rolling'],
    config={
        'time_config':    {'features': ['month', 'quarter']},
        'lag_config':     {'target_cols': ['monthly_deposit'], 'lags': [1, 3, 6, 12]},
        'rolling_config': {'target_cols': ['monthly_deposit'], 'windows': [3, 12], 'stats': ['mean']},
    },
)
df_feat = eng.fit_transform(processed.set_index('date')).dropna()
X = df_feat.drop(columns=['monthly_deposit'])
y = df_feat['monthly_deposit']

# 模型（BaseMLModel.fit 接收元组 (X, y)）
model = get_ml_model('xgboost', {'probabilistic': True, 'probabilistic_method': 'residual'})
model.fit((X_train.values, y_train.values),
          val_data=(X_val.values, y_val.values))
prob = model.predict_probabilistic(X_test.values)

# 监控
monitor = HPFMonitor('xgb_v1', adapter, cfg.monitoring, reference_data=X_train.values)
for ts, (_, x_row), y_true in zip(test_ts, X_test.iterrows(), y_test):
    monitor.record_prediction(ts, x_row.to_frame().T, prob, y_true, 'monthly_deposit')
monitor.check_status()
monitor.generate_report()
```

---

## 当前状态

- Python 3.8+ / Windows · Linux · macOS
- 版本：0.2.0
- 测试：7 passed
- HPF baseline：XGBoost MAPE 0.93% · R² 0.9509（12 年模拟月度数据）

---

## License

MIT
