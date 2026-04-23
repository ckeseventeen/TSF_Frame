# TSF_Frame 架构说明

> 面向**架构师 / 二次开发者**。解释"为什么是这样的结构"，以及各层如何协作。
> 如果你只想跑起来，请直接看 [GUIDE.md](GUIDE.md)。

---

## 1. 设计目标

TSF_Frame 是一个通用时序预测框架，当前业务层聚焦住房公积金（HPF）月度指标预测，但所有通用能力（模型 / 特征 / 监控 / 可视化）都与业务解耦，可移植到其它行业。

四条硬性原则：

1. **业务与框架隔离** —— `tsf_frame.*`（通用）与 `configs.*` + `business.hpf_*`（HPF 特例）分离。换行业 = 加一个 `BusinessAdapter` + 一份 `Config` dataclass，不改框架代码。
2. **src-layout 强制隔离** —— 源码在 `src/tsf_frame/`，测试不会意外读到源目录里的残余 import；`pip install -e .` 是唯一的安装路径。
3. **配置即代码** —— 所有超参/阈值用 `@dataclass` 描述，`to_adapter_config()` 再转 dict 给运行时。可读 + 可 diff + IDE 补全。
4. **有监控才算闭环** —— 不只出预测数，还要能回答"这个模型今天是否还健康"。`HPFMonitor` 把性能/漂移/业务规则/持久化/报表串成一条链路。

---

## 2. 目录分层

```
TSF_Frame/
├── src/tsf_frame/              # 框架核心（发布包）
│   ├── business/               # 业务防腐层：BaseBusinessAdapter / HPFAdapter
│   ├── data/datasets/          # PyTorch 风格 Dataset
│   ├── features/               # 特征工程 + 选择 + 降维 + 混合特征
│   ├── models/                 # BaseModel + classical/transformer/moirai
│   ├── monitoring/             # 通用监控 + HPF 专属监控
│   ├── visualization/          # 预测可视化 + 监控报表（matplotlib）
│   └── utils/                  # logger / metrics
│
├── configs/                    # 独立顶层包（不在 src/ 下，语义即"配置总线"）
│   ├── base_config.py          # BaseConfig（框架级）
│   └── hpf/                    # HPFConfig + HPFMonitoringConfig（业务级）
│
├── pipelines/                  # 可执行入口（不是库，而是"运行脚本"）
│   ├── run_hpf_forecast.py     # HPF 端到端预测 pipeline
│   ├── train_model.py          # 通用训练器（argparse）
│   └── examples/               # 各模块的独立演示
│
├── tests/                      # pytest（conftest 注入 src/ 到 sys.path）
├── data/                       # 数据存储（gitignore，仅保留 .gitkeep）
│   ├── raw/  processed/  hpf/
├── experiments/  logs/         # 运行产物（gitignore）
├── docs/                       # 本目录
├── setup.py                    # 混合 package_dir（src/ + configs/ 同时打包）
└── requirements.txt
```

### 为什么 `configs/` 不在 `src/tsf_frame/` 里？

`configs/` 描述"某一次跑这个框架要用的参数"，而不是"框架本身的能力"。把它放进 `tsf_frame` 会让"换业务"变成"改框架内部"。独立顶层包的好处：

- 新业务只需在 `configs/` 下加一个子包（例如 `configs/stock/`），不污染 `tsf_frame`
- `setup.py` 用 `package_dir={'': 'src'}` + 额外 `packages=['configs', 'configs.hpf']` 把两个顶层包一起发布
- 导入侧看起来干净：`from configs.hpf import HPFConfig` / `from tsf_frame.business import HPFAdapter`

---

## 3. 核心抽象

框架用 3 个抽象基类把所有可变点收敛了。

### 3.1 `BaseModel` (`tsf_frame.models.base_model`)

```python
class BaseModel(ABC, nn.Module):
    def __init__(self, config: Dict[str, Any]): ...
    @abstractmethod
    def fit(self, train_data, val_data=None, **kw) -> Dict[str, Any]: ...
    @abstractmethod
    def predict(self, test_data, **kw) -> np.ndarray: ...
    @abstractmethod
    def predict_probabilistic(self, test_data, **kw) -> ProbabilisticPrediction: ...
    def save_model(self, path: str): ...
    def load_model(self, path: str): ...
```

概率预测的返回类型是 `ProbabilisticPrediction(mean, lower, upper, std, confidence)`，它是统一的数据载体 —— 无论底层走残差分布、MC Dropout 还是分位数回归，监控/可视化/业务层只看这个对象。

实现映射：
- `models/classical/ml_models.py` —— 11 个 sklearn/XGB/LGB/CatBoost 模型
- `models/transformer/transformer_models.py` —— LSTM / Transformer / Informer / Autoformer / iTransformer / TimesNet
- `models/moirai/` —— 零样本基础模型

### 3.2 `BaseBusinessAdapter` (`tsf_frame.business.base_adapter`)

```python
class BaseBusinessAdapter(ABC):
    def __init__(self, config: Dict[str, Any]): ...
    @abstractmethod
    def preprocess(self, data: pd.DataFrame, **kw) -> Tuple[pd.DataFrame, Dict]: ...
    @abstractmethod
    def postprocess(self, predictions: np.ndarray, metadata: Dict, **kw) -> pd.DataFrame: ...
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]: ...
    def get_business_metrics(self, y_true, y_pred) -> Dict[str, float]: ...
```

"业务"这一层是**防腐层**：
- 业务规则（非负、单调、缺失上限）集中在这里校验
- 归一化/反归一化细节（HPF 建议 zscore，工业/股票可能要 minmax）不泄漏到模型
- 业务指标（HPF 的 YoY / QoQ / direction_accuracy）不污染通用 `MetricsCalculator`

### 3.3 `BaseMonitor` (`tsf_frame.monitoring.base_monitor`)

```python
class BaseMonitor(ABC):
    @abstractmethod
    def record_prediction(self, timestamp, features, prediction, actual_value=None, ...): ...
    @abstractmethod
    def check_status(self) -> MonitoringStatus: ...
    @abstractmethod
    def reset(self): ...
```

通用 `ModelMonitor` 已组合了 `PerformanceMonitor + DataDriftDetector + ConceptDriftDetector + AlertManager + RetrainingTrigger`。HPF 专属的 `HPFMonitor` 再**组合** `ModelMonitor + HPFBusinessRuleChecker + SQLiteStore + HPFMonitoringReport`，形成完整闭环。

---

## 4. 端到端数据流

以 HPF 月度缴存预测为例（`pipelines/run_hpf_forecast.py`）：

```
原始 DataFrame (date, monthly_deposit, monthly_withdrawal, loan_balance, ...)
        │
        ▼  HPFAdapter.validate_data()           业务规则 & schema 检查
        │
        ▼  HPFAdapter.preprocess()              异常值处理 + zscore 归一化 → metadata
        │
        ▼  create_feature_engineer(...)         Time + Lag + Rolling + Difference
        │                                       全部 fit→transform，严格因果
        │
        ▼  train/val/test 时间顺序切分           不用 shuffle，不用随机 CV
        │
        ▼  get_ml_model('xgboost', cfg)         BaseModel 子类
        │       .fit(X_tr, y_tr, val_data=...)
        │       .predict_probabilistic(X_te)    → ProbabilisticPrediction
        │
        ▼  HPFAdapter.postprocess(pred, meta)   反归一化 + 裁剪非负 + 政策情景
        │
        ▼  MetricsCalculator.calculate_all      MAE/RMSE/MAPE/R²
        │  adapter.get_business_metrics         direction/YoY/QoQ
        │
        ▼  HPFMonitor.record_prediction(...)    写 SQLite + 业务规则检查
        │  HPFMonitor.check_status()            触发告警（日志 + 文件 + DB）
        │                                       必要时自动生成 PNG 报表
        ▼
输出: experiments/results/*.csv + logs/hpf_monitor.db + logs/hpf_reports/*.png
```

---

## 5. 监控子系统架构

```
┌──────────────────────── HPFMonitor ────────────────────────┐
│                                                            │
│  record_prediction(ts, features, pred, actual, target):    │
│    ├─► HPFBusinessRuleChecker.check()    R1..R5 规则       │
│    ├─► SQLiteStore.insert_prediction()   持久化            │
│    ├─► ModelMonitor.record_prediction()                    │
│    │     ├─► PerformanceMonitor  MAE/MSE/RMSE/MAPE/R²      │
│    │     ├─► DataDriftDetector   PSI / KS                  │
│    │     ├─► ConceptDriftDetector  残差趋势                │
│    │     └─► AlertManager  ───┐                            │
│    └─► 业务滑窗 (deque[window_months]) 供 YoY/QoQ 计算     │
│                                                            │
│  check_status() -> MonitoringStatus:                       │
│    ├─► 聚合 ModelMonitor.check_status()                    │
│    ├─► adapter.get_business_metrics(滑窗 y_true, y_pred)   │
│    ├─► SQLiteStore.insert_metrics_snapshot()               │
│    └─► 若 alert_level ≥ WARNING & report_on_alert          │
│           → HPFMonitoringReport.generate() (8-grid PNG)    │
│                                                            │
│  AlertManager.handler ──► 日志 + hpf_alerts.log + alerts 表│
└────────────────────────────────────────────────────────────┘
```

### SQLite schema（`monitoring/sqlite_store.py`）

三张表，一对索引。`journal_mode=WAL` 支持并发。

```sql
predictions      (id, model_id, ts, target_col, y_pred, y_lower, y_upper, y_actual)
metrics_snapshot (id, model_id, ts, metric_name, metric_value, window_months)
alerts           (alert_id, model_id, ts, level, rule_id, message, payload, acknowledged)
```

### 冷启动降级

`len(actual_history) < cold_start_months` 时，只跑业务规则（R1-R5），跳过 PSI / error_trend —— 样本不足会误报漂移。

---

## 6. 配置层

两级 dataclass：

```
BaseConfig                           # 框架级通用（model_name, seed, device…）
└── HPFConfig                        # 业务级，组合以下子配置
    ├── data:       HPFDataConfig
    ├── feature:    HPFFeatureConfig
    ├── model:      HPFModelConfig
    └── monitoring: HPFMonitoringConfig
```

运行时需要 `dict` 的地方用 `to_adapter_config()` / `to_dict()` 转换：

```python
cfg = HPFConfig()
adapter = HPFAdapter(cfg.to_adapter_config())   # dict → 运行时
```

好处：
- `HPFConfig()` 有完整 IDE 补全 / 默认值 / 类型校验
- 转成 dict 后才进框架，框架只认 dict，不依赖任何业务 dataclass
- `yaml` 可读可 diff 可 PR review

---

## 7. 导入与打包策略

### src-layout 的理由

```
ROOT/
├── src/tsf_frame/              # 发布包
├── configs/                    # 也发布，但不在 src/
├── pipelines/                  # 不发布，可执行脚本
└── tests/                      # 不发布
```

好处：
1. `pytest` / `ipython` 在 ROOT 运行时**不会**直接读到 `tsf_frame/` 源目录（因为那里没有，必须走 `src/tsf_frame/`）。强制走"已安装的包"路径，消除"本地能跑、装完就坏"的经典坑。
2. 运行 `pipelines/*.py` 时用 pathlib 向上找到 ROOT，把 ROOT 和 `ROOT/src` 同时插入 `sys.path`，使得"不 `pip install`"也能直接跑（对数据科学场景友好）。

pipeline 头部的 bootstrap：

```python
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
for _p in (_HERE.parent, *_HERE.parents):
    if (_p / 'configs').is_dir() and (_p / 'src').is_dir():
        for _q in (_p, _p / 'src'):
            if str(_q) not in sys.path:
                sys.path.insert(0, str(_q))
        break
```

### `setup.py` 的混合 package_dir

```python
src_packages    = find_packages(where='src')        # → tsf_frame, tsf_frame.business, ...
config_packages = find_packages(where='.', include=['configs', 'configs.*'])

setup(
    package_dir={'': 'src', 'configs': 'configs'},
    packages=src_packages + config_packages,
    ...
)
```

这一招让两个分别位于 `src/` 和 `./configs/` 的顶层包在 `pip install -e .` 后都可导入。

---

## 8. 测试策略

`tests/conftest.py` 往 `sys.path` 注入 ROOT + `ROOT/src`，所以：

```bash
pytest tests/           # 不需要先 pip install -e .
```

当前测试分 3 类：

| 文件 | 目的 |
|------|------|
| `test_metrics.py` | `MetricsCalculator.calculate_all` 契约（完美预测 → 误差为 0 / R²=1） |
| `test_data_leakage.py` | 特征工程 fit→transform 模式下不能"窥视未来" |
| `test_adapters.py` | `HPFAdapter` 能被 `HPFConfig.to_adapter_config()` 正确驱动 |

---

## 9. 设计权衡 & FAQ

**Q1: 为什么深度学习模型不拆成独立子包？**
保持 `tsf_frame.models` 下 `classical/transformer/moirai` 三类的对称性；调用侧统一走 `get_ml_model(name, cfg)` / `get_dl_model(name, cfg)`。

**Q2: 为什么没有 `orchestration/` 或 `workflow/`？**
P0 阶段保持简单：`pipelines/*.py` 就是 orchestration。等真要上 Airflow/Prefect 再抽象。

**Q3: 监控为什么绑死 SQLite 而不用更强的 DB？**
目标是"单机、零依赖、能重启回放"。`SQLiteStore` 接口清晰，未来切 Postgres 只要换实现。

**Q4: 业务规则放到 HPFMonitor 而不是 HPFAdapter？**
`validate_data` 在 Adapter 里做入口校验；`HPFBusinessRuleChecker` 在 Monitor 里做**运行时**校验（看预测结果、上下文、残差），两者职责不同。

**Q5: TimesNet 为什么用 GroupNorm 而不是 BatchNorm2d？**
MC Dropout 推理时需要保持 Dropout 层处于 train 模式，但又不想让 BN 统计量漂掉。GroupNorm 不依赖 batch 统计，天然适配。详见 [EXTENDING.md](EXTENDING.md) 第 6 节。

**Q6: XGBoost 为什么特别处理？**
XGBoost 是树模型，无法外推趋势。pipelines 里对目标做一次一阶差分、训练完再 cumsum 还原，避免预测长期序列时"贴顶 / 贴底"。详见 [EXTENDING.md](EXTENDING.md) 第 6 节。

---

## 扩展阅读

- 新手上路 → [GUIDE.md](GUIDE.md)
- 添加新模型 / 新业务 / 新监控规则 → [EXTENDING.md](EXTENDING.md)
