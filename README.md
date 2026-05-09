# TSF_Frame

**通用时序预测框架** · 当前业务层聚焦住房公积金(HPF)月度指标预测 · src-layout · 配置化 · 监控闭环 · 可二次开发

```
┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────────┐
│  业务适配层  │ → │  特征工程层   │ → │  模型层     │ → │  监控/告警/可视化  │
│  (防腐层)    │   │              │   │  17 models │   │   pluggable      │
│  HPFAdapter │   │  Time/Lag    │   │  ML + DL   │   │  ModelMonitor +  │
│             │   │  Roll/Diff   │   │            │   │  规则/漂移/重训   │
└─────────────┘   └──────────────┘   └────────────┘   └──────────────────┘
```

---

## 30 秒跑通

```bash
git clone <repo-url> TSF_Frame
cd TSF_Frame
pip install -e .

# 端到端 HPF 预测
python pipelines/run_hpf_forecast.py
# → logs/outputs/hpf/*.png  (4 张对比图)
# → 控制台输出 MAPE/R² 等指标

# 监控演示(规则 + 漂移 + 告警 + SQLite + PNG 报表)
python pipelines/examples/hpf_monitoring_example.py
# → logs/monitor/hpf_monitor.db   SQLite 三表
# → logs/monitor/hpf_alerts.log   告警(WARNING+)
# → logs/reports/hpf/*.png         报表
```

---

## 核心能力

- **17 个模型** — 11 ML(`xgboost`/`lightgbm`/`catboost`/`ridge`/`svr`/...)+ 6 DL(`lstm`/`transformer`/`autoformer`/`itransformer`/`timesnet`/`dlinear`)
- **概率预测** — 残差分布 / MC Dropout / 分位数回归,统一 `ProbabilisticPrediction` 接口
- **特征工程** — 时间/滞后/滚动/扩展/差分 + KBest/RFE/Lasso/PCA,**严格 fit→transform 因果**
- **业务防腐层** — `HPFAdapter` 把非负、季节、政策情景、YoY/QoQ 业务规则集中封装
- **监控闭环** — `ModelMonitor` 组合性能/数据漂移/概念漂移/规则引擎/重训触发器/告警/持久化(全部可插拔)
- **多目标 / 多步监控** — `MultiTargetMonitor`(温度+湿度并发)、`MultiHorizonMonitor`(未来 12 个月分桶 MAPE)
- **统一画图** — `PredictionPlotter` 7 个原子方法 + 4 个复合工具,所有项目图风格统一
- **运行时数据持久化** — `SQLiteStore` / `JsonlStore` / `InMemoryStore` 三选一,完整 schema

---

## 项目结构

```
TSF_Frame/
├── src/tsf_frame/              # 框架源码 (发布包)
│   ├── business/               # BaseBusinessAdapter / HPFAdapter
│   ├── features/               # engineering / selector
│   ├── models/                 # classical / transformer / moirai
│   ├── monitoring/             # 完整监控栈 (12 文件)
│   ├── visualization/          # PredictionPlotter (统一画图)
│   ├── data/datasets/
│   └── utils/                  # logger / metrics
├── configs/                    # 顶层配置包 (BaseConfig + HPFConfig)
├── pipelines/                  # 入口脚本
│   ├── run_hpf_forecast.py     # HPF 端到端
│   ├── train_model.py          # 通用 CLI 训练器
│   └── examples/               # 单模块演示
├── tests/                      # pytest (54 个测试)
├── docs/                       # 文档
├── logs/                       # 运行产物 (gitignore)
│   ├── runs/                   # 运行日志
│   ├── monitor/                # SQLite + alerts.log
│   ├── reports/                # 报表 PNG
│   └── outputs/                # 训练产出
├── setup.py
└── requirements.txt
```

---

## 常用命令

```bash
# 安装
pip install -e .

# 端到端 HPF 预测 (4 张对比图)
python pipelines/run_hpf_forecast.py

# HPF 监控完整演示 (规则 + 漂移 + 告警 + 报表)
python pipelines/examples/hpf_monitoring_example.py

# 通用训练器
python pipelines/train_model.py --model ridge   --dataset air_passengers
python pipelines/train_model.py --model xgboost --dataset synthetic
python pipelines/train_model.py --model lstm    --dataset air_passengers --epochs 20

# 单模块演示
python pipelines/examples/feature_engineering_example.py
python pipelines/examples/probabilistic_example.py
python pipelines/examples/hpf_dl_example.py
python pipelines/examples/public_dataset_workflow.py

# 测试
pytest tests/             # 54 passed
pytest tests/ -v
```

---

## 最小代码示例

```python
from configs.hpf import HPFConfig
from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.features.engineering import create_feature_engineer
from tsf_frame.models.classical.ml_models import get_ml_model
from tsf_frame.monitoring import ModelMonitor, RuleEngine

cfg = HPFConfig()
adapter = HPFAdapter(cfg.to_adapter_config())

# 1. 数据预处理
ok, msg = adapter.validate_data(df);  assert ok, msg
processed, meta = adapter.preprocess(df)

# 2. 特征工程
eng = create_feature_engineer(
    feature_types=['time', 'lag', 'rolling'],
    config={
        'time_config':    {'features': ['month', 'quarter']},
        'lag_config':     {'target_cols': ['monthly_deposit'], 'lags': [1, 3, 6, 12]},
        'rolling_config': {'target_cols': ['monthly_deposit'],
                           'windows': [3, 12], 'stats': ['mean']},
    },
)
df_feat = eng.fit_transform(processed.set_index('date')).dropna()
X, y = df_feat.drop(columns=['monthly_deposit']), df_feat['monthly_deposit']

# 3. 训练 (BaseMLModel.fit 接收元组 (X, y))
model = get_ml_model('xgboost', {
    'probabilistic': True, 'probabilistic_method': 'residual',
})
model.fit((X_train.values, y_train.values),
          val_data=(X_val.values, y_val.values))
prob = model.predict_probabilistic(X_test.values)

# 4. 监控
monitor = ModelMonitor(
    model_id='xgb_deposit_v1',
    rule_engine=RuleEngine(rule_ids=['R1_NON_NEGATIVE', 'R2_SUDDEN_CHANGE']),
)
for ts, y_p, y_t in zip(test_ts, prob.mean, y_test):
    monitor.record_prediction(timestamp=ts, prediction=y_p, actual=y_t)
status = monitor.check_status()
print(status.alert_level, status.recommendations)
```

---

## 文档

| 文档 | 面向读者 | 内容 |
|------|---------|------|
| [docs/使用指南.md](docs/使用指南.md) | 使用者 | 安装、数据准备、特征/模型/监控完整 API 用法、命令速查 |
| [docs/开发指南.md](docs/开发指南.md) | 二次开发者 | 设计原则、核心抽象、扩展点(新模型/规则/store/告警通道)、bug 记录 |
| [docs/HPF生产部署蓝图.md](docs/HPF生产部署蓝图.md) | 业务团队 | 20 个公积金需求项目化的目录布局/开发 SOP/部署/运维方案 |

---

## 当前状态

- Python 3.8+ · Windows / Linux / macOS
- 版本 0.2.0 · 测试 **54 passed**
- HPF baseline (Ridge): MAPE ~1% · R² ~0.99 (12 年模拟月度数据)

## License

MIT
