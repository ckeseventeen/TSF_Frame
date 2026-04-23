# TSF_Frame 扩展指南

> 面向**二次开发者**。怎么加一个新业务 / 新模型 / 新监控规则，以及框架里那些"不自然"的决策为什么要这么做。

---

## 目录

1. [添加一个新业务适配器](#1-添加一个新业务适配器)
2. [添加一个新模型](#2-添加一个新模型)
3. [添加一个自定义监控规则](#3-添加一个自定义监控规则)
4. [代码规范](#4-代码规范)
5. [测试要求](#5-测试要求)
6. [关键技术决策](#6-关键技术决策)
7. [已知 bug 修复记录](#7-已知-bug-修复记录)

---

## 1. 添加一个新业务适配器

以"股票月度收益预测"为例，复用所有框架能力、仅新增业务层。

### 步骤 1：新增配置

```
configs/
└── stock/
    ├── __init__.py
    ├── stock_config.py          # StockConfig + 子配置
    └── stock_monitoring_config.py
```

```python
# configs/stock/stock_config.py
from dataclasses import dataclass, field
from typing import List
from configs.base_config import BaseConfig
from .stock_monitoring_config import StockMonitoringConfig

@dataclass
class StockDataConfig:
    target_columns: List[str] = field(default_factory=lambda: ['close'])
    freq: str = 'B'                       # 交易日

@dataclass
class StockConfig(BaseConfig):
    data:       StockDataConfig       = field(default_factory=StockDataConfig)
    monitoring: StockMonitoringConfig = field(default_factory=StockMonitoringConfig)

    def to_adapter_config(self) -> dict:
        return {
            'normalization': 'minmax',    # 股票价格用 minmax 更合适
            'target_columns': self.data.target_columns,
            # ...
        }
```

### 步骤 2：新增 Adapter

```
src/tsf_frame/business/
└── stock_adapter.py
```

```python
# src/tsf_frame/business/stock_adapter.py
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from .base_adapter import BaseBusinessAdapter

class StockAdapter(BaseBusinessAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.business_type = 'stock'
        # ...

    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        # 股票业务的规则：不能有价格 ≤ 0、交易日频率必须对齐、...
        ...

    def preprocess(self, data: pd.DataFrame, **kw) -> Tuple[pd.DataFrame, Dict]:
        ...

    def postprocess(self, predictions: np.ndarray, metadata: Dict, **kw) -> pd.DataFrame:
        ...

    def get_business_metrics(self, y_true, y_pred) -> Dict[str, float]:
        # 股票的业务指标：Sharpe、MDD、方向准确率、...
        ...
```

记得在 `src/tsf_frame/business/__init__.py` 导出。

### 步骤 3：新增 pipeline

```
pipelines/
└── run_stock_forecast.py     # 复制 run_hpf_forecast.py 改行业特定部分
```

头部保留 pathlib bootstrap；正文把 `HPFConfig` → `StockConfig`、`HPFAdapter` → `StockAdapter`。

### 步骤 4：新增测试

```python
# tests/test_stock_adapter.py
def test_stock_adapter_validate_rejects_negative_price():
    df = tiny_stock_df.copy()
    df.loc[0, 'close'] = -1.0
    ok, msg = StockAdapter(StockConfig().to_adapter_config()).validate_data(df)
    assert not ok
```

整个过程**不改任何 `tsf_frame/*` 的既有文件**。

---

## 2. 添加一个新模型

### 2.1 经典 ML 模型

`src/tsf_frame/models/classical/ml_models.py` 里挂到 `MODEL_REGISTRY`：

```python
class MyModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._est = SomeSklearnEstimator(**{k: v for k, v in config.items() if k in SK_PARAMS})

    def fit(self, train_data, val_data=None, **kw):
        X, y = train_data
        self._est.fit(X, y)
        # 用验证残差拟合概率分布（所有经典模型都这样）
        if self.config.get('probabilistic'):
            val_pred = self._est.predict(val_data[0])
            self._fit_residuals(val_data[1], val_pred)
        return {'status': 'done'}

    def predict(self, test_data, **kw) -> np.ndarray:
        return self._est.predict(test_data)

    def predict_probabilistic(self, test_data, **kw):
        mean = self.predict(test_data)
        lower, upper = self._get_residual_interval(mean)
        return ProbabilisticPrediction(mean, lower, upper, ...)

MODEL_REGISTRY['my_model'] = MyModel
```

### 2.2 深度学习模型

`src/tsf_frame/models/transformer/transformer_models.py`，继承 `BaseModel` 同时 `nn.Module`，实现 `forward`、`fit`、`predict`、`predict_probabilistic`。MC Dropout 概率预测的关键：

```python
def predict_probabilistic(self, test_data, n_samples=50, **kw):
    self.train()                              # 打开 Dropout
    preds = np.stack([self.predict(test_data) for _ in range(n_samples)], axis=0)
    self.eval()
    mean, std = preds.mean(0), preds.std(0)
    return ProbabilisticPrediction(mean, mean - 1.96*std, mean + 1.96*std, std)
```

**坑**：如果模型里有 BatchNorm，推理时打开 `train()` 会让 BN 统计量漂掉。替换为 GroupNorm / LayerNorm（见第 6 节）。

---

## 3. 添加一个自定义监控规则

### 方式 A：直接加到 `HPFBusinessRuleChecker`

`src/tsf_frame/monitoring/hpf_business_rules.py` 里 `check()` 末尾 yield：

```python
# R6: 同比方向错误
if prev_yoy is not None:
    pred_yoy_sign = np.sign(prediction[-1] - prev_yoy)
    actual_yoy_sign = np.sign(last_actual - prev_yoy)
    if pred_yoy_sign != actual_yoy_sign and actual_yoy_sign != 0:
        yield RuleViolation('R6_YOY_DIR', 'WARNING',
            '同比方向预测错误', value=float(pred_yoy_sign))
```

`HPFMonitoringConfig` 加对应阈值；告警级别矩阵（ARCHITECTURE.md §4）补充一行。

### 方式 B：自定义一个 Checker

如果是新业务（例如 `StockBusinessRuleChecker`），直接仿照 `HPFBusinessRuleChecker` 写一个新文件，在自己的 `StockMonitor` 里组合即可，不影响 HPF。

---

## 4. 代码规范

- **类型注解**：所有 public 函数签名必须有类型
- **docstring**：中英双语，说明参数、返回、副作用
- **导入路径**：始终用绝对路径（`from tsf_frame.xxx import ...`），禁止 `import *`
- **命名**：类 PascalCase、函数/变量 snake_case、常量 UPPER_CASE
- **避免在库代码中 print**：用 `get_logger(name, log_dir=...)` 获取 logger
- **绝不在库代码中调 `matplotlib.pyplot`**：pipelines / examples 可以，库模块应接收 `ax` 或返回 fig

---

## 5. 测试要求

### 最小要求

每加一个新 adapter / 新模型，至少写两条测试：

1. **契约测试**：最小输入→非空输出，结构符合预期
2. **边界测试**：空数据、NaN、单行、全相同值

### 跑测试

```bash
pytest tests/                                  # 全部
pytest tests/ -v                               # 详细
pytest tests/test_metrics.py -k perfect        # 按名字
pytest --co tests/                             # 只看收集到的 case
```

### 防止数据泄漏（关键）

所有特征工程类必须遵守：

- `fit` 只在训练集上调用
- `transform` 对训练/验证/测试用同一套 fit 后的参数
- 不使用未来信息（rolling 只看过去，lag 只看过去）

对应测试在 `tests/test_data_leakage.py`。

---

## 6. 关键技术决策

### 6.1 XGBoost 一阶差分 + cumsum 还原

**问题**：XGBoost（以及其它树模型）的预测被训练集目标的范围 bounding，对长期趋势序列会"贴顶 / 贴底"。

**解决**：训练前对目标做 `Δy = y - y.shift(1)`，学习增量；推理时 `y_hat[t] = y_hat[t-1] + Δy_hat[t]` 累加还原。

**启用**：在 `pipelines/run_hpf_forecast.py` 的 XGBoost 训练路径中默认开启。代码片段：

```python
dy_train = y_train.diff().dropna()
model.fit(X_train.iloc[1:], dy_train)
dy_pred = model.predict(X_test)
y_pred = y_train.iloc[-1] + np.cumsum(dy_pred)
```

### 6.2 TimesNet MC Dropout 用 GroupNorm

**问题**：TimesNet 原始实现用 BatchNorm2d。MC Dropout 推理时需要 `model.train()` 保留 Dropout 层的随机性，但 BN 会因此污染 running stats。

**解决**：把 `nn.BatchNorm2d` 替换成 `nn.GroupNorm(num_groups=8, num_channels=C)`。GroupNorm 不依赖 batch 统计量，`train()` / `eval()` 行为一致。

文件：`src/tsf_frame/models/transformer/transformer_models.py` 的 `TimesNetBlock`。

### 6.3 src-layout

见 [ARCHITECTURE.md §7](ARCHITECTURE.md#7-导入与打包策略)。

### 6.4 配置 dataclass → dict 边界

Adapter / Model 只认 dict，不认 `HPFConfig`。这样：
- 从 yaml / json / argparse 进来的任何 dict-like 都能用
- 框架不绑死某个业务的 dataclass

**约定**：dataclass 提供 `to_adapter_config()` / `to_dict()`；跨业务/跨模块传参**一律用 dict**。

### 6.5 监控组合而不是继承

`HPFMonitor` 没有继承 `ModelMonitor`，而是**在构造函数里持有**一个 `ModelMonitor` 实例。原因：

- `ModelMonitor` 的能力是通用的（性能、漂移、告警、重训）
- `HPFMonitor` 要**叠加**业务规则、业务指标、SQLite 持久化
- 组合比继承更容易单测（可以 mock `ModelMonitor`）

---

## 7. 已知 bug 修复记录

历史上踩过的坑与修复方式。继续踩到类似问题时可参考。

| Bug | 表现 | 修复 |
|-----|------|------|
| DL 模型 fake 实现 | Autoformer/iTransformer/TimesNet 只有壳子 | 重写为真实 DL 实现（2025-Q4） |
| XGBoost 外推失败 | 预测长期趋势贴顶 | 一阶差分 + cumsum（§6.1） |
| TimesNet MC Dropout 漂统计 | BN running stats 被污染 | 改 GroupNorm（§6.2） |
| MC Dropout 推理时未开 train() | 概率预测退化为点预测 | `model.train()` + `torch.no_grad()` 混用 |
| 特征工程 transform 漏训练参数 | 测试集归一化用到测试集统计 | 严格 fit → transform 分离 + `test_data_leakage.py` |
| `configs/` 导入 setup.py 漏掉 | `pip install -e .` 后 `from configs.hpf import *` 报错 | `package_dir={'': 'src', 'configs': 'configs'}` + `find_packages(where='.', include=['configs', 'configs.*'])` |
| pipeline 头部 sys.path 正则残留 `)` | `SyntaxError: unmatched ')'` | 后处理脚本把多余 `)` 删掉，全量 8 文件统一 |

---

## 下一步

- 想理解整体设计 → [ARCHITECTURE.md](ARCHITECTURE.md)
- 想直接使用 → [GUIDE.md](GUIDE.md)
