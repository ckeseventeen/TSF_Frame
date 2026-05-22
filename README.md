# TSF_Frame

**通用时序预测框架** · 当前业务层聚焦住房公积金(HPF)月度指标预测 · src-layout · 配置化 · 监控闭环 · 可二次开发

```
┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────────┐
│  业务适配层  │ → │  特征工程层   │ → │  模型层     │ → │  监控/告警/可视化  │
│  (防腐层)    │   │              │   │  18 models │   │   pluggable      │
│  HPFAdapter │   │  Time/Lag    │   │ ML+DL+MLP  │   │  ModelMonitor +  │
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
python pipelines/examples/run_hpf_forecast.py
# → logs/outputs/hpf/*.png  (4 张对比图)
# → 控制台输出 MAPE/R² 等指标

# 监控演示(规则 + 漂移 + 告警 + SQLite + PNG 报表)
python pipelines/examples/hpf_monitoring_example.py
# → logs/monitor/hpf_monitor.db   SQLite 三表
# → logs/monitor/hpf_alerts.log   告警(WARNING+)
# → logs/reports/hpf/*.png         报表

# 生产推理 wrapper 演示(train → save → load → predict)
python pipelines/examples/inference_demo.py
# → logs/models/inference_demo_ridge.pkl   端到端 artifact
```

---

## 核心能力

- **18 个模型** — 11 ML(`xgboost`/`lightgbm`/`catboost`/`ridge`/`svr`/...)+ 6 DL 时序(`lstm`/`transformer`/`autoformer`/`itransformer`/`timesnet`/`dlinear`)+ `MLPModel` 横截面表格 + Moirai 零样本
- **概率预测** — 残差分布(支持 train / val OOS / **K-fold CV OOF**)/ MC Dropout / 分位数回归,统一 `ProbabilisticPrediction` 接口
- **RevIN 默认开** — 6 个 DL 模型都内置 Reversible Instance Normalization,长趋势数据 MAPE 从 6-20% 降到 ~0.5%(ICLR 2022)
- **早停 + LR scheduling** — `_dl_fit` 可选 `early_stop_patience` / `lr_scheduler='plateau'|'cosine'`,自动节省 60-80% epoch
- **特征工程** — 时间/滞后/滚动/扩展/差分 + KBest/RFE/Lasso/PCA,**严格 fit→transform 因果**
- **业务防腐层** — `HPFAdapter` 把非负、季节、政策情景、YoY/QoQ 业务规则集中封装
- **监控闭环** — `ModelMonitor` 组合性能/数据漂移/概念漂移/规则引擎/重训触发器/告警/持久化(全部可插拔)
- **多目标 / 多步监控** — `MultiTargetMonitor`(温度+湿度并发)、`MultiHorizonMonitor`(未来 12 个月分桶 MAPE)
- **生产推理 wrapper** — `tsf_frame.deployment.InferenceRunner` 把 5 个训练时状态打包成单一 artifact,推理一行 `.predict(latest_df)` 出预测,自动 anti-data-leakage
- **推理协议 API** — `MixedFeatureHandler.min_required_rows` / `required_source_columns` 给 SQL 调用方做 defensive check
- **统一画图** — `PredictionPlotter` 7 个原子方法 + 4 个复合工具,所有项目图风格统一
- **运行时数据持久化** — `SQLiteStore` / `JsonlStore` / `InMemoryStore` 三选一,完整 schema

---

## 项目结构

```
TSF_Frame/
├── src/tsf_frame/              # 框架源码 (发布包)
│   ├── business/               # BaseBusinessAdapter / HPFAdapter
│   ├── features/               # engineering / selector / mixed_feature_handler
│   ├── models/                 # classical / transformer / tabular(MLP) / moirai
│   ├── monitoring/             # 完整监控栈 (12 文件)
│   ├── deployment/             # InferenceRunner (端到端 save/load/predict)
│   ├── visualization/          # PredictionPlotter (统一画图)
│   ├── data/datasets/
│   └── utils/                  # logger / metrics
├── configs/                    # 顶层配置包 (BaseConfig + HPFConfig)
│   └── hpf/                    # HPFConfig + task_registry + sql_templates/
├── pipelines/                  # 入口脚本
│   ├── run_monthly_controller.py  # 月度跑批主控 (遍历 TASKS)
│   ├── production_loader.py    # 数据接入 (csv/hive) + 结果落库 (csv/mysql)
│   ├── train_model.py          # 通用 CLI 训练器
│   ├── job_train.py / job_inference.py  # 单任务训练/推理
│   └── examples/               # 单模块演示 + run_hpf_forecast 端到端
├── tests/                      # pytest (100 个测试)
├── docs/                       # 文档
├── logs/                       # 运行产物 (gitignore)
│   ├── runs/                   # 运行日志
│   ├── monitor/                # SQLite + alerts.log
│   ├── reports/                # 报表 PNG
│   ├── models/                 # 训练好的 artifact (InferenceRunner.save)
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
python pipelines/examples/run_hpf_forecast.py

# HPF 监控完整演示 (规则 + 漂移 + 告警 + 报表)
python pipelines/examples/hpf_monitoring_example.py

# 6 个 DL 模型对比 (含 RevIN + 早停 + LR scheduling)
python pipelines/examples/hpf_dl_example.py

# 生产推理 wrapper (train → save → reload → predict)
python pipelines/examples/inference_demo.py

# 月度跑批主控 (遍历 TASKS,csv 兜底 / hive 生产)
python pipelines/run_monthly_controller.py

# 通用训练器
python pipelines/train_model.py --model ridge   --dataset air_passengers
python pipelines/train_model.py --model xgboost --dataset synthetic
python pipelines/train_model.py --model lstm    --dataset air_passengers --epochs 20

# 其他单模块演示
python pipelines/examples/feature_engineering_example.py
python pipelines/examples/probabilistic_example.py
python pipelines/examples/panel_income_spending_example.py   # 面板数据 + DLinear 多目标
python pipelines/examples/public_dataset_workflow.py

# 测试
pytest tests/             # 147 passed
pytest tests/ -v
pytest tests/test_improvements_tier_ab.py    # 16 个生产化改进项单测
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

## 生产推理 (InferenceRunner)

把训练时分散在多个对象里的状态打包成单一 artifact,推理一行调用,自动 anti-data-leakage:

```python
from tsf_frame.deployment import InferenceRunner

# 训练完: 打包 5 个状态(adapter / feat_eng / mfh / model / model_config)落盘
runner = InferenceRunner(
    adapter=adapter, feat_eng=feat_eng, mfh=mfh,
    model=model, model_config=model.config, target_col='monthly_deposit',
)
runner.save('logs/models/hpf_ridge_v1.pkl')

# 推理(可同进程也可新进程):
runner = InferenceRunner.load('logs/models/hpf_ridge_v1.pkl', model_cls=RidgeModel)
prob = runner.predict(latest_24_months_df, target_col='monthly_deposit')
print(f'预测: {prob.mean[0]:.2f}, 95% CI: [{prob.lower[0]:.2f}, {prob.upper[0]:.2f}]')
```

`load` 自动设 `adapter._is_fitted=True`、`predict` 入口 assert `len(latest_df) >= min_required_rows`、全链路 `fit=False` — 调用方想错也错不了。

---

## 当前状态

- Python 3.8+ · Windows / Linux / macOS
- 版本 0.2.0 · 测试 **147 passed**(含 16 个 Tier A+B + 47 个 registry/multi_horizon/edge_cases 改进项单测)
- HPF baseline:
  - ML (Ridge): MAPE ~1% · R² ~0.95 (12 年模拟月度数据)
  - DL (RevIN + 早停): 6 个 Transformer 系列模型 test MAPE ~0.5%

## Moirai 零样本大模型 (Zero-Shot Foundation Model)

**Moirai** 是由 Salesforce AI Research 提出的大型时间序列预测基础模型。它基于掩码编码器（Masked Encoder）架构，并在海量数据集 (LOTSA data) 上进行了预训练，拥有极其强大的开箱即用能力。

* **论文参考**: [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592) (Woo et al., 2024)

### 架构与原生概率预测
不同于传统的深度学习模型输出单点预测，Moirai 是一个**天然的概率预测模型**。它的最后一层会输出一个混合概率分布（Mixture Distribution）。
在 `TSF_Frame` 框架中，我们对其进行了深度集成封装（见 `PretrainedMoiraiModel`）：
- **跳过残差训练**：不再依赖 `BaseModel` 笨拙的静态残差计算。
- **动态置信区间**：直接通过 `probabilistic_predict` 接口提取 Moirai 基于上下文动态生成的 10% 和 90% 分位数（Quantiles）。
- **完全零样本**：整个过程无需反向传播（Zero-shot），即插即用。

### 核心家族与参数量
| 模型名称 | 参数量 | HuggingFace 仓库 | TSF_Frame 配置项 |
| :---: | :---: | :---: | :---: |
| Moirai-1.0-R-Small | 14M | [Salesforce/moirai-1.0-R-small](https://huggingface.co/Salesforce/moirai-1.0-R-small) | `'moirai_size': 'small'` |
| Moirai-1.0-R-Base | 91M | [Salesforce/moirai-1.0-R-base](https://huggingface.co/Salesforce/moirai-1.0-R-base) | `'moirai_size': 'base'` |
| Moirai-1.0-R-Large | 311M | [Salesforce/moirai-1.0-R-large](https://huggingface.co/Salesforce/moirai-1.0-R-large) | `'moirai_size': 'large'` |

> 💡 **权重缓存机制**: 框架已配置本地缓存。首次运行时模型将被自动下载并缓存在项目根目录的 `pretrained_models/` 文件夹下，不再占用系统 C 盘空间。

### 运行示例
使用 Moirai 前，请确保你的 Python 版本 >= 3.10，并安装了最新的依赖（见 `requirements.txt`）：
```bash
pip install uni2ts>=2.0.0 gluonts>=0.14.0 safetensors
```
运行完整的业务演示脚本：
```bash
python pipelines/examples/hpf_moirai_zeroshot_example.py
```

---

## License

MIT
