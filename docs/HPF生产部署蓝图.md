# 公积金 (HPF) 生产化预测项目蓝图

> **目标:** 在 `TSF_Frame` 之上,落地 **20 个公积金业务预测需求**,每个需求多个指标。
> 跑得起来 → 监控起来 → 运维起来。**不含前端看板**(纯后台批处理 + 监控审计)。

---

## 目录

1. [项目分层与现状分析](#一项目分层与现状分析)
2. [业务项目完整目录结构](#二业务项目完整目录结构)
3. [20 个需求清单与组织](#三20-个需求清单与组织)
4. [核心文件作用速查](#四核心文件作用速查)
5. [开发 SOP — 加新需求 10 步](#五开发-sop--加新需求-10-步)
6. [部署方案(3 档)](#六部署方案3-档)
7. [生产运维手册](#七生产运维手册)
8. [最小可运行步骤(MVP)](#八最小可运行步骤mvp)
9. [常见问题与排查](#九常见问题与排查)

---

## 一、项目分层与现状分析

### 1.1 TSF_Frame 提供什么(底层框架)

```
┌──────────────────────────────────────────────────────────────┐
│  业务项目 (TSF_HPF_Production) ← 你要建的这个               │
│  20 个需求 × 月度跑批 × 监控告警 × 运维                       │
├──────────────────────────────────────────────────────────────┤
│  TSF_Frame (现状, 通用框架, 已 pip install -e . 装好)        │
│  ├── data/             数据集 + 特征工程                      │
│  ├── features/         特征生成 + 选择                        │
│  ├── models/           ML(11 个) + DL(LSTM/Transformer)      │
│  ├── business/         HPFAdapter (业务规则、季节、政策)      │
│  ├── monitoring/       完整监控(性能/漂移/规则/重训/告警/报表)│
│  ├── visualization/    PredictionPlotter (统一画图)           │
│  └── utils/            logger / metrics                        │
└──────────────────────────────────────────────────────────────┘
```

**TSF_Frame 已经具备的能力:**
- ✅ 11 种 ML 模型 + Transformer 系列(分类性预测 + 回归)
- ✅ HPFAdapter 业务适配(MAPE / direction / YoY / QoQ 业务指标)
- ✅ 完整监控栈:`ModelMonitor` / `MultiHorizonMonitor` / `MultiTargetMonitor`
- ✅ 异步真值回填 (`settle_actual` 按 `target_ts` 对齐)
- ✅ 三种持久化(In-memory / SQLite / JSONL),pluggable
- ✅ 告警通道(Console / Logging / File / Callback / Store),pluggable
- ✅ 重训触发器(性能/漂移/时间/样本量驱动)
- ✅ 月度跑批正确(target_ts 对齐 + cold-start 降级)

**TSF_Frame 不直接提供:**
- ❌ 业务侧 SQL/Hive 数据接入
- ❌ 多需求的项目级编排(20 个需求各自怎么落地)
- ❌ 调度(cron / Airflow)
- ❌ 容器/K8s 部署
- ❌ 钉钉/邮件等具体告警渠道
- ❌ 前端看板(本文不涉及)

**业务项目要补的就是这些"上层胶水"。**

### 1.2 项目分工

| 层 | 谁负责 | 改动频率 |
|------|--------|---------|
| `tsf_frame` 包(已稳定) | 框架维护者 | 低,只做 bug fix / 通用增强 |
| 业务项目脚手架(`tsf_hpf_production`) | 你 + 团队 | 每加一个需求一次 |
| 20 个 `requirements/R0X_*` 子目录 | 业务开发 | 频繁 |
| 调度 / 部署配置 | DevOps | 偶尔 |

---

## 二、业务项目完整目录结构

```
tsf_hpf_production/                        ← 业务项目根目录
│
├── README.md                              项目说明 (5 分钟启动指南)
├── pyproject.toml                         项目元数据 + 依赖
├── requirements.txt                       生产依赖 (pip 兼容)
├── requirements-dev.txt                   开发/测试依赖
├── .env.example                           环境变量模板 (DB 连接串等)
├── .gitignore
│
├── configs/                               ★ 配置目录 (yaml + py 双轨)
│   ├── __init__.py
│   ├── base.yaml                          全局基础配置 (项目名/默认 horizon/...)
│   ├── monitoring.yaml                    监控统一参数 (阈值/告警通道/store)
│   ├── data_sources.yaml                  数据源连接 (Hive/Oracle/CSV)
│   ├── env/
│   │   ├── dev.yaml                       开发环境覆盖
│   │   ├── staging.yaml
│   │   └── prod.yaml
│   └── requirements/                      ⭐ 20 个需求各自的 yaml
│       ├── R01_monthly_deposit.yaml
│       ├── R02_monthly_withdrawal.yaml
│       ├── R03_loan_balance_personal.yaml
│       ├── R04_loan_balance_corp.yaml
│       ├── R05_depositor_count.yaml
│       ├── R06_withdrawer_count.yaml
│       ├── R07_loan_issued_amount.yaml
│       ├── R08_loan_issued_count.yaml
│       ├── R09_loan_to_deposit_ratio.yaml
│       ├── R10_liquidity_index.yaml
│       ├── R11_policy_scenario.yaml
│       ├── R12_regional_distribution.yaml
│       ├── R13_industry_distribution.yaml
│       ├── R14_age_segment.yaml
│       ├── R15_income_segment.yaml
│       ├── R16_housing_demand.yaml
│       ├── R17_prepayment_rate.yaml
│       ├── R18_npl_rate.yaml
│       ├── R19_rate_sensitivity.yaml
│       └── R20_risk_score.yaml
│
├── data/                                  ★ 数据目录 (在 .gitignore 中, 仅本地/挂载)
│   ├── raw/                               原始数据 (从 Hive dump 下来或外部接入)
│   │   ├── R01/2024-01.csv
│   │   ├── R01/2024-02.csv
│   │   └── ...
│   ├── processed/                         预处理后数据
│   │   ├── R01_features.parquet
│   │   └── ...
│   └── reference/                         训练期参考 (漂移检测 baseline)
│       ├── R01_train_X.parquet
│       ├── R01_train_y.parquet
│       └── ...
│
├── src/                                   ★ 业务代码 (src-layout)
│   └── tsf_hpf/
│       ├── __init__.py
│       │
│       ├── core/                          通用脚手架 (所有需求共用)
│       │   ├── __init__.py
│       │   ├── pipeline.py                ForecastPipeline 基类 (train/predict/settle)
│       │   ├── registry.py                Requirement 注册表 (按 ID 取实现)
│       │   ├── config_loader.py           yaml → 强类型配置对象
│       │   └── monitor_factory.py         按 yaml 装配监控栈
│       │
│       ├── infra/                         基础设施 (DB/调度/告警渠道)
│       │   ├── __init__.py
│       │   ├── db/
│       │   │   ├── hive.py                Hive 连接器 (SQL → DataFrame)
│       │   │   ├── oracle.py              Oracle 连接器 (可选)
│       │   │   └── csv.py                 CSV 兜底接入
│       │   ├── alert_channels/
│       │   │   ├── dingtalk.py            钉钉 webhook (实现 AlertChannel ABC)
│       │   │   ├── email.py               SMTP 邮件
│       │   │   └── wecom.py               企业微信
│       │   └── locks.py                   分布式锁 (防同需求并发跑批)
│       │
│       ├── requirements/                  ⭐ 20 个需求的具体实现
│       │   ├── __init__.py                自动注册所有 R0X 需求
│       │   ├── base.py                    RequirementBase 接口
│       │   ├── _template/                 ⭐ 复制模板新加需求
│       │   │   ├── __init__.py
│       │   │   ├── config.py
│       │   │   ├── data_loader.py
│       │   │   ├── features.py
│       │   │   ├── model.py
│       │   │   ├── rules.py
│       │   │   └── postprocess.py
│       │   ├── R01_monthly_deposit/
│       │   │   ├── __init__.py            登记 register_requirement('R01', ...)
│       │   │   ├── config.py              R01 专属配置 (target='monthly_deposit', ...)
│       │   │   ├── data_loader.py         R01 数据 SQL/Parquet 加载
│       │   │   ├── features.py            R01 特征工程
│       │   │   ├── model.py               R01 模型训练逻辑 (一般 XGBoost)
│       │   │   ├── rules.py               R01 业务硬约束 (非负、突变 < 30%)
│       │   │   └── postprocess.py         R01 后处理 (反归一化、截断)
│       │   ├── R02_monthly_withdrawal/    (同样结构,7 个文件)
│       │   ├── R03_loan_balance_personal/
│       │   ├── ...
│       │   └── R20_risk_score/
│       │
│       └── postprocessing/                通用后处理(供需求引用)
│           ├── ensemble.py                多模型平均 / 加权
│           └── policy_adjuster.py         政策情景调整 (复用 HPFAdapter)
│
├── pipelines/                             ★ 主入口脚本 (被调度/手动执行)
│   ├── __init__.py
│   ├── train_one.py                       python -m pipelines.train_one --req R01
│   ├── train_all.py                       批训所有 (按需求列表并行/串行)
│   ├── predict_one.py                     python -m pipelines.predict_one --req R01 --date 2026-05
│   ├── predict_all.py                     全部 20 需求月度跑批
│   ├── settle_one.py                      回填上月真值 --req R01
│   ├── settle_all.py
│   ├── monitor_check.py                   周期性 check_status, 触发告警
│   └── monitor_report.py                  生成监控报表 PNG
│
├── tests/                                 ★ 测试
│   ├── conftest.py                        共享 fixture (tiny dataset / mock DB)
│   ├── unit/
│   │   ├── test_config_loader.py
│   │   ├── test_pipeline_base.py
│   │   ├── test_R01_features.py
│   │   ├── test_R01_rules.py
│   │   └── ...
│   ├── integration/
│   │   ├── test_train_R01_e2e.py          端到端: train → save model → load → predict
│   │   ├── test_predict_R01_e2e.py
│   │   ├── test_settle_R01_e2e.py
│   │   └── test_monitoring_full_loop.py   record → settle → check_status
│   └── smoke/
│       └── test_all_20_requirements_train.py   ⭐ CI 快速冒烟 20 需求
│
├── ops/                                   ★ 运维脚本 (运行时辅助)
│   ├── setup.sh                           一键初始化 (建 venv, pip install)
│   ├── crontab.example                    cron 配置示例 (单机部署用)
│   ├── healthcheck.sh                     健康检查 (供 nagios/zabbix 调用)
│   ├── backup.sh                          监控 SQLite + 模型权重备份
│   ├── rollback.sh                        模型版本回滚
│   └── airflow_dags/                      Airflow DAG 定义 (企业级部署)
│       ├── tsf_hpf_train_monthly_dag.py
│       ├── tsf_hpf_predict_monthly_dag.py
│       ├── tsf_hpf_settle_monthly_dag.py
│       └── tsf_hpf_monitor_daily_dag.py
│
├── deployments/                           ★ 部署配置
│   ├── docker/
│   │   ├── Dockerfile                     业务镜像 (基于 python:3.10-slim)
│   │   └── docker-compose.yaml            单机 Docker 部署
│   └── k8s/
│       ├── namespace.yaml
│       ├── configmap.yaml                 yaml 配置注入
│       ├── secret.yaml                    DB 密码等敏感信息
│       ├── pvc.yaml                       /data /logs 持久化卷
│       ├── train-cronjob.yaml             月度 1 日训练任务
│       ├── predict-cronjob.yaml           月度 5 日预测任务
│       ├── settle-cronjob.yaml            月度 6 日真值回填
│       └── monitor-cronjob.yaml           每日监控
│
├── docs/                                  项目文档
│   ├── HPF生产部署蓝图.md                 本文档 (你正在看)
│   ├── REQUIREMENTS.md                    20 个需求详细规格
│   ├── DEVELOPMENT.md                     开发指南
│   ├── DEPLOYMENT.md                      部署指南
│   └── OPERATIONS.md                      运维手册 (告警响应 SOP)
│
└── logs/                                  ★ 运行时输出 (.gitignore)
    ├── runs/                              所有运行日志 (按 timestamp)
    │   ├── train_R01_20260501_030001.log
    │   ├── predict_all_20260505_060001.log
    │   └── ...
    ├── monitor/
    │   ├── monitor.db                     ⭐ 全部 20 需求共用的 SQLite
    │   └── alerts.log                     WARNING+ 告警归集
    ├── reports/                           监控报表 PNG
    │   ├── R01/2026-05.png
    │   ├── R02/2026-05.png
    │   └── ...
    └── outputs/                           训练产出
        ├── R01/
        │   ├── model_20260501.pkl         月度版本化模型
        │   ├── metrics_20260501.json      训练时验证集指标
        │   ├── feature_importance_20260501.csv
        │   └── train_X_baseline.parquet   训练期特征 (drift 参考)
        └── ...
```

---

## 三、20 个需求清单与组织

### 3.1 需求清单(示例,按业务侧实际调整)

| ID | 需求名称 | 主预测指标 | 多指标(horizons) | 频率 | 数据源 |
|----|---------|----------|-----------------|------|--------|
| R01 | 月度缴存额预测 | `monthly_deposit` | h=1,3,6,12 | 月 | Hive: ods.hpf_deposit |
| R02 | 月度提取额预测 | `monthly_withdrawal` | h=1,3,6,12 | 月 | Hive: ods.hpf_withdraw |
| R03 | 个人住房贷款余额 | `loan_balance_personal` | h=1,6,12 | 月 | Hive: ods.hpf_loan |
| R04 | 单位贷款余额 | `loan_balance_corp` | h=1,6,12 | 月 | Hive: ods.hpf_loan |
| R05 | 月度缴存人数 | `depositor_count` | h=1,3,6,12 | 月 | Hive: ods.hpf_deposit |
| R06 | 月度提取人数 | `withdrawer_count` | h=1,3,6,12 | 月 | Hive: ods.hpf_withdraw |
| R07 | 贷款发放金额 | `loan_issued_amount` | h=1,3,6 | 月 | Hive: ods.hpf_loan_issue |
| R08 | 贷款发放笔数 | `loan_issued_count` | h=1,3,6 | 月 | Hive: ods.hpf_loan_issue |
| R09 | 贷存比 | `loan_to_deposit_ratio` | h=1,3,6 | 月 | 派生(R03/R01) |
| R10 | 流动性指数 | `liquidity_index` | h=1,3 | 月 | 派生 |
| R11 | 政策情景预测 | `monthly_deposit` (在政策下) | h=12 | 半年 | 业务输入 + R01 模型 |
| R12 | 区域分布预测 | 各市 `deposit_per_city` 多目标 | h=1,6 | 月 | Hive: ods.hpf_region |
| R13 | 行业分布预测 | 各行业 `deposit_per_industry` 多目标 | h=1,6 | 月 | Hive: ods.hpf_industry |
| R14 | 年龄段分布 | 各年龄段缴存 | h=12 | 季 | Hive: ods.hpf_age |
| R15 | 收入分层 | 各收入层缴存 | h=12 | 季 | Hive: ods.hpf_income |
| R16 | 住房需求预测 | `housing_demand_index` | h=6,12 | 季 | 派生 + 外部宏观 |
| R17 | 提前还款率 | `prepayment_rate` | h=1,6 | 月 | Hive: ods.hpf_prepay |
| R18 | 不良贷款率 | `npl_rate` | h=1,6 | 月 | Hive: ods.hpf_loan |
| R19 | 利率敏感度 | 多指标响应矩阵 | h=12 | 季 | 派生 + 政策 |
| R20 | 综合风险评分 | `risk_score` | h=1,6 | 月 | 集成 R09/R10/R17/R18 |

### 3.2 多指标的两种组织方式

**方式 A — 单需求多 horizon (用 `MultiHorizonMonitor`):**
- R01 月度缴存额: 同一 target,12 个 horizon → 用 `MultiHorizonMonitor`

**方式 B — 单需求多 target (用 `MultiTargetMonitor`):**
- R12 区域分布: 同时预测 31 个省的 deposit → 用 `MultiTargetMonitor(targets=['北京', '上海', ...])`

**方式 C — 组合(常见):**
- R12 区域分布 × 12 horizon: 每个省一个 `MultiHorizonMonitor`,组合在 R12 配置里

```python
# requirements/R12_regional_distribution/config.py
HORIZONS = [1, 3, 6, 12]
PROVINCES = ['北京', '上海', '广东', ...]
# pipeline 内部为每个省建一个 MultiHorizonMonitor,共 31 个
```

---

## 四、核心文件作用速查

### 4.1 配置层 (`configs/`)

| 文件 | 作用 | 谁改 |
|------|------|------|
| `base.yaml` | 项目级默认值 (project_name, default_horizons, log_level) | 项目初始化时 |
| `monitoring.yaml` | 全局监控阈值/告警通道/SQLite 路径 | 偶尔调阈值 |
| `data_sources.yaml` | DB 连接串 (env 变量引用,不存明文) | 数据源变更时 |
| `env/{dev,staging,prod}.yaml` | 环境覆盖 (dev 用 mock 数据, prod 用 Hive) | 环境切换时 |
| `requirements/R0X_*.yaml` | **每个需求独立的核心配置** | 每加一个需求 |

**`requirements/R01_monthly_deposit.yaml` 示例:**
```yaml
id: R01
name: 月度缴存额预测
description: 全量个人月度缴存总额, 含直接缴存与单位代缴

data:
  source: hive
  table: ods.hpf_deposit_monthly
  date_col: stat_month
  target_col: monthly_deposit
  feature_cols: [employees, avg_salary, deposit_rate, ...]

split:
  train_until: '2024-12'
  val_months: 6
  test_months: 12

model:
  type: xgboost            # ml_models 注册表的名字
  params:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05

horizons: [1, 3, 6, 12]    # 多步预测

monitoring:
  window_size: 36          # 队列保留 3 年
  metric_window: 12        # 指标算近 12 期
  thresholds:
    mape_warning: 0.10
    mape_critical: 0.15
  rules: [R1_NON_NEGATIVE, R2_SUDDEN_CHANGE]
  rule_params:
    R2_SUDDEN_CHANGE:
      max_ratio: 0.30

retraining:
  cooldown_hours: 720      # 30 天内不二次重训
  on_critical_alert: true  # CRITICAL 告警自动触发
```

### 4.2 业务核心 (`src/tsf_hpf/`)

| 文件 | 作用 | 谁改 |
|------|------|------|
| `core/pipeline.py` | `ForecastPipeline` 基类: train/predict/settle 通用骨架 | 框架升级时 |
| `core/registry.py` | `register_requirement('R01', cls)` 注册表,按 ID 取实例 | 不动 |
| `core/config_loader.py` | yaml → `RequirementConfig` 强类型对象 (含校验) | 不动 |
| `core/monitor_factory.py` | 按配置装配 `ModelMonitor` + 监控栈 | 不动 |
| `infra/db/hive.py` | Hive 连接(基于 PyHive / impyla),返回 DataFrame | 数据源变化时 |
| `infra/alert_channels/dingtalk.py` | 钉钉机器人,实现 `AlertChannel.send()` | 接入新渠道时 |
| `requirements/base.py` | `RequirementBase` 接口 (load/preprocess/features/model/rules/postprocess) | 不动 |
| `requirements/_template/` | 新需求的复制模板 | 不动 |
| `requirements/R0X_xxx/config.py` | 该需求的强类型配置 (默认从 yaml 加载) | 该需求开发时 |
| `requirements/R0X_xxx/data_loader.py` | 该需求的 SQL/数据加载 | 该需求开发时 |
| `requirements/R0X_xxx/features.py` | 该需求专属特征工程 (lag/rolling/seasonal) | 该需求开发时 |
| `requirements/R0X_xxx/model.py` | 该需求模型训练 (一般直接调 tsf_frame.models) | 该需求开发时 |
| `requirements/R0X_xxx/rules.py` | 该需求专属业务规则 (`@register_rule('R_R01_*')`) | 该需求开发时 |
| `requirements/R0X_xxx/postprocess.py` | 反归一化/截断/平滑 | 该需求开发时 |

### 4.3 入口脚本 (`pipelines/`)

| 文件 | 用途 | 调用频率 |
|------|------|---------|
| `train_one.py --req R01` | 单需求训练,产模型文件 | 每月 1 次 (cron) |
| `train_all.py` | 全部 20 需求批量训 (并发/串行可选) | 每月 1 次 |
| `predict_one.py --req R01 --date 2026-05` | 单需求月度预测,落库 | 每月 1 次 |
| `predict_all.py` | 全部 20 需求月度跑批 | 每月 1 次 |
| `settle_one.py --req R01 --date 2026-05` | 回填某月真值,触发指标更新 | 每月 1 次 |
| `settle_all.py` | 全部需求批量回填 | 每月 1 次 |
| `monitor_check.py` | 跑一遍 `check_status()`,持久化快照 + 触发告警 | 每日 1 次 |
| `monitor_report.py --req R01` | 生成报表 PNG (近 90 天) | 每月 1 次 / 出 CRITICAL 时 |

### 4.4 测试 (`tests/`)

| 子目录 | 作用 |
|--------|------|
| `unit/` | 单文件单元测试 (config / features / rules) |
| `integration/` | 跨文件 e2e (一个完整需求 train+predict+settle 走通) |
| `smoke/` | CI 快速冒烟 (用 tiny mock data 跑通 20 需求,< 5 min) |

### 4.5 运维 (`ops/`)

| 文件 | 作用 |
|------|------|
| `setup.sh` | 新机器一键安装 (`bash ops/setup.sh`) |
| `crontab.example` | 单机部署的 cron 模板 |
| `airflow_dags/` | 企业级 Airflow DAG (依赖 + 重试 + 通知) |
| `healthcheck.sh` | 供监控系统(zabbix/nagios)定期调用,exit code 反映健康度 |
| `backup.sh` | 备份 monitor.db + outputs/*/model_*.pkl 到 OSS/HDFS |
| `rollback.sh` | 模型回滚到指定月份 (`bash ops/rollback.sh R01 2026-04`) |

### 4.6 部署 (`deployments/`)

| 文件 | 作用 |
|------|------|
| `docker/Dockerfile` | 业务镜像 (Python 3.10 + 项目代码 + 依赖) |
| `k8s/train-cronjob.yaml` | K8s CronJob: 每月 1 日 03:00 跑训练 |
| `k8s/predict-cronjob.yaml` | 每月 5 日 06:00 跑预测 |
| `k8s/settle-cronjob.yaml` | 每月 6 日 09:00 跑真值回填 |
| `k8s/monitor-cronjob.yaml` | 每日 09:00 跑监控检查 |
| `k8s/pvc.yaml` | `/data` 和 `/logs` 持久化卷 |

---

## 五、开发 SOP — 加新需求 10 步

每个需求约 **半天工作量**,熟练后 2 小时。

### Step 1: 复制模板
```bash
cp -r src/tsf_hpf/requirements/_template src/tsf_hpf/requirements/R05_depositor_count
cp configs/requirements/R01_monthly_deposit.yaml configs/requirements/R05_depositor_count.yaml
```

### Step 2: 改 yaml 配置
编辑 `configs/requirements/R05_depositor_count.yaml`,改:
- `id: R05`
- `name: 月度缴存人数预测`
- `data.target_col: depositor_count`
- `data.feature_cols: [...]`(本需求实际可用的列)
- `model.params`(按数据规模调,n_estimators / max_depth)
- `horizons` / `metric_window` / 阈值

### Step 3: 实现数据加载 `data_loader.py`
```python
# requirements/R05_depositor_count/data_loader.py
from ...infra.db.hive import HiveClient

def load_data(start: str, end: str, hive: HiveClient) -> pd.DataFrame:
    sql = f"""
    SELECT stat_month AS date,
           depositor_count,
           total_units,
           avg_deposit_per_person,
           ...
    FROM ods.hpf_deposit_monthly
    WHERE stat_month BETWEEN '{start}' AND '{end}'
    ORDER BY stat_month
    """
    return hive.query(sql)
```

### Step 4: 实现特征工程 `features.py`
大部分场景直接复用 `tsf_frame.features.engineering`:
```python
from tsf_frame.features.engineering import create_feature_engineer

def build_features(df, cfg):
    fe = create_feature_engineer(
        feature_types=['time', 'lag', 'rolling'],
        config={
            'time_config': {'features': ['month', 'quarter', 'is_year_end']},
            'lag_config': {'target_col': cfg.target_col, 'lags': [1, 3, 12]},
            'rolling_config': {'target_col': cfg.target_col,
                               'windows': [3, 6, 12], 'stats': ['mean', 'std']},
        },
    )
    return fe.fit_transform(df)
```

### Step 5: 实现模型 `model.py`
```python
from tsf_frame.models.classical.ml_models import get_ml_model

def train_model(X_train, y_train, X_val, y_val, cfg):
    model = get_ml_model(cfg.model.type, cfg.model.params)
    history = model.fit((X_train, y_train), (X_val, y_val))
    return model, history

def predict(model, X):
    return model.predict_probabilistic(X)   # 含 lower/upper
```

### Step 6: 实现业务规则 `rules.py`
```python
from tsf_frame.monitoring import register_rule, RuleViolation, AlertLevel

@register_rule('R_R05_NON_NEGATIVE')
def rule_R05_non_negative(*, prediction, **_):
    arr = np.atleast_1d(prediction)
    if np.any(arr < 0):
        return [RuleViolation(
            rule_id='R_R05_NON_NEGATIVE',
            severity=AlertLevel.CRITICAL,
            message=f'缴存人数预测出现负值: {arr.min():.0f}',
        )]
    return []

@register_rule('R_R05_NOT_EXCEED_TOTAL')
def rule_R05_not_exceed_total(*, prediction, context=None, **_):
    """缴存人数不能超过总缴存单位数."""
    total_units = (context or {}).get('total_units')
    if total_units and prediction[0] > total_units:
        return [RuleViolation(
            rule_id='R_R05_NOT_EXCEED_TOTAL',
            severity=AlertLevel.ERROR,
            message=f'预测缴存人数 {prediction[0]:.0f} 超过总单位数 {total_units}',
        )]
    return []
```

### Step 7: 实现后处理 `postprocess.py`
```python
def postprocess(prediction, cfg):
    # 反归一化
    prediction = prediction * cfg.scale + cfg.offset
    # 非负截断
    prediction = np.maximum(prediction, 0)
    # 取整(人数)
    return np.round(prediction).astype(int)
```

### Step 8: 注册需求 `__init__.py`
```python
# requirements/R05_depositor_count/__init__.py
from ...core.registry import register_requirement
from .config import R05_CONFIG
from .data_loader import load_data
from .features import build_features
from .model import train_model, predict
from .rules import rule_R05_non_negative, rule_R05_not_exceed_total
from .postprocess import postprocess

register_requirement(
    id='R05',
    config=R05_CONFIG,
    handlers={
        'load_data': load_data,
        'build_features': build_features,
        'train_model': train_model,
        'predict': predict,
        'postprocess': postprocess,
    },
)
```

### Step 9: 写测试
```bash
cp tests/integration/test_train_R01_e2e.py tests/integration/test_train_R05_e2e.py
# 改 R01 → R05, 跑通
pytest tests/integration/test_train_R05_e2e.py -v
```

### Step 10: 跑通 + 提 PR
```bash
# 训练
python -m pipelines.train_one --req R05 --env dev

# 预测
python -m pipelines.predict_one --req R05 --date 2026-05 --env dev

# 监控检查
python -m pipelines.monitor_check --req R05 --env dev

# 全 20 需求冒烟测试
pytest tests/smoke/ -v

# 提交 PR
git add . && git commit -m "feat(R05): 添加月度缴存人数预测需求"
git push -u origin feature/R05_depositor_count
```

---

## 六、部署方案(3 档)

### 6.1 单机 cron(开发/测试/小规模生产)

**适合:** 一台 Linux 服务器,< 10 GB 数据,运维人手 1 人。

**配置:**
```bash
# crontab -e
# 每月 1 日 03:00 训练所有需求
0 3 1 * * cd /opt/tsf_hpf && python -m pipelines.train_all >> logs/runs/cron.log 2>&1

# 每月 5 日 06:00 全部需求月度预测
0 6 5 * * cd /opt/tsf_hpf && python -m pipelines.predict_all >> logs/runs/cron.log 2>&1

# 每月 6 日 09:00 真值回填(等上月真值在 DWD 出齐)
0 9 6 * * cd /opt/tsf_hpf && python -m pipelines.settle_all >> logs/runs/cron.log 2>&1

# 每日 09:00 监控检查
0 9 * * * cd /opt/tsf_hpf && python -m pipelines.monitor_check >> logs/runs/cron.log 2>&1
```

**安装:** `bash ops/setup.sh` 一键搞定 venv + pip install + 必要目录。

### 6.2 Docker(中等规模)

**适合:** 多台机器,容器化运行,DevOps 倾向 docker。

**Dockerfile:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .
CMD ["python", "-m", "pipelines.predict_all"]
```

**docker-compose:**
```yaml
services:
  tsf-hpf-train:
    image: tsf-hpf:latest
    command: python -m pipelines.train_all
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - ENV=prod
    # 用 ofelia / 主机 cron 定时拉起
```

### 6.3 Kubernetes + Airflow(企业级)

**适合:** 多团队、多环境、强 SLA、复杂依赖。

**K8s CronJob 示例:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: tsf-hpf-predict-monthly
spec:
  schedule: "0 6 5 * *"      # 每月 5 日 06:00
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: predict
            image: registry/tsf-hpf:v1.2.0
            command: ["python", "-m", "pipelines.predict_all"]
            envFrom:
              - secretRef: { name: hpf-db-secret }
            volumeMounts:
              - { name: data, mountPath: /app/data }
              - { name: logs, mountPath: /app/logs }
          volumes:
            - name: data
              persistentVolumeClaim: { claimName: hpf-data-pvc }
            - name: logs
              persistentVolumeClaim: { claimName: hpf-logs-pvc }
          restartPolicy: OnFailure
```

**Airflow DAG 示例(简化):**
```python
# ops/airflow_dags/tsf_hpf_predict_monthly_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    'tsf_hpf_predict_monthly',
    schedule_interval='0 6 5 * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:
    for req_id in [f'R{i:02d}' for i in range(1, 21)]:
        BashOperator(
            task_id=f'predict_{req_id}',
            bash_command=f'python -m pipelines.predict_one --req {req_id} '
                         f'--date {{ ds }}',
            retries=2,
        )
```

---

## 七、生产运维手册

### 7.1 监控告警栈(已经在框架里,只需配置)

```yaml
# configs/monitoring.yaml
store:
  type: sqlite
  path: ./logs/monitor/monitor.db

channels:
  - type: file
    path: ./logs/monitor/alerts.log
    min_level: warning
  - type: logging
    name: tsf_hpf
    min_level: info
  - type: dingtalk            # 自定义实现, infra/alert_channels/dingtalk.py
    webhook: ${DINGTALK_WEBHOOK}
    min_level: error
```

**告警分级响应:**
| 级别 | 谁响应 | 响应时限 | 动作 |
|------|--------|--------|------|
| INFO | 无人 | — | 仅入库审计 |
| WARNING | 值班开发 | 当日 | 看一眼日志 |
| ERROR | 值班开发 + 业务方 | 4 小时 | 排查根因 |
| CRITICAL | 全团队 + 业务负责人 | 1 小时 | **暂停下游使用 + 排查 + 决定回滚** |

### 7.2 日常巡检 SOP

**每日 09:00:**
```bash
# 1. 自动: monitor_check 已通过 cron 触发
# 2. 人工: 看 alerts.log
tail -50 /var/log/tsf_hpf/alerts.log

# 3. 看 SQLite 告警计数
sqlite3 logs/monitor/monitor.db \
  "SELECT level, COUNT(*) FROM alerts WHERE ts > datetime('now', '-1 day') GROUP BY level;"

# 4. 看待回填记录(说明真值还没到的目标月)
python -m pipelines.monitor_check --pending-only
```

**每月跑批后(5 日傍晚):**
```bash
# 1. 跑通性
grep -E 'ERROR|CRITICAL' logs/runs/predict_all_$(date +%Y%m05)*.log

# 2. 各需求 MAPE 是否在阈值内
python -m pipelines.monitor_report --month $(date +%Y-%m) --output /tmp/report.md

# 3. 出图(肉眼复核)
python -m pipelines.monitor_report --req R01 --gen-png
```

### 7.3 模型回滚

**触发条件:** CRITICAL 告警 + 调查发现新模型坏了。

**步骤:**
```bash
# 1. 停止下游使用
# (业务方暂停读取 logs/outputs/R01/predict_*.csv)

# 2. 回滚模型
bash ops/rollback.sh R01 2026-04
# 内部: cp logs/outputs/R01/model_20260401.pkl → model_current.pkl

# 3. 重跑预测
python -m pipelines.predict_one --req R01 --date 2026-05 --force

# 4. 通知业务方恢复使用
```

### 7.4 重训触发

**自动重训(框架已支持):**
- `RetrainingTrigger` 检测到 CRITICAL 漂移 → `should_retrain=True`
- `monitor_check.py` 看到决策后,**写入待重训队列**(不直接重训,需人工确认)

**人工重训:**
```bash
# 单需求重训
python -m pipelines.train_one --req R01 --reason "concept drift detected"

# 重训后通知监控器
python -c "
from tsf_hpf.core import get_monitor
mon = get_monitor('R01')
mon.record_retraining()
"
```

### 7.5 备份策略

**每日凌晨 02:00 自动:**
```bash
# ops/backup.sh
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR=/backup/tsf_hpf/$DATE

mkdir -p $BACKUP_DIR

# SQLite 监控数据(只读拷贝)
sqlite3 logs/monitor/monitor.db ".backup $BACKUP_DIR/monitor.db"

# 模型权重(月底之后保留至少 3 个月)
tar czf $BACKUP_DIR/models.tar.gz logs/outputs/*/model_*.pkl

# 同步到 OSS/HDFS
aliyun oss cp -r $BACKUP_DIR oss://hpf-backup/

# 清理 30 天前本地备份
find /backup/tsf_hpf -mtime +30 -type d -exec rm -rf {} \;
```

---

## 八、最小可运行步骤(MVP)

> 目标:**1 小时内**,从空目录到跑通第 1 个需求(R01 月度缴存额),证明全栈通。

### 阶段 1:项目初始化(15 分钟)

```bash
# 1. 创建项目
mkdir tsf_hpf_production && cd tsf_hpf_production
git init

# 2. 安装 TSF_Frame (本地开发模式)
pip install -e ../TSF_Frame   # 假设在同级目录

# 3. 创建最小目录结构
mkdir -p configs/requirements src/tsf_hpf/{core,infra,requirements} \
         pipelines tests data/raw logs/{runs,monitor,reports,outputs}

# 4. 复制本文档作为根 README 起点
cp ../TSF_Frame/docs/HPF生产部署蓝图.md docs/

# 5. 初始化基础脚手架
touch src/tsf_hpf/__init__.py
touch src/tsf_hpf/core/__init__.py
touch pipelines/__init__.py
```

### 阶段 2:实现 R01(30 分钟)

仅做 R01,验证全链路 work,然后再批量复制。

```bash
# 1. 准备一份测试数据(先用合成,后续接 Hive)
python -c "
import pandas as pd, numpy as np
n = 60
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=n, freq='MS'),
    'monthly_deposit': np.linspace(100, 300, n) + np.random.normal(0, 5, n),
    'employees': np.linspace(1000, 1500, n),
    'avg_salary': np.linspace(8000, 12000, n),
})
df.to_csv('data/raw/R01_synthetic.csv', index=False)
"

# 2. 写最小 yaml 配置 (configs/requirements/R01_monthly_deposit.yaml)
# (按上文示例填充)

# 3. 写最小训练脚本 (pipelines/train_one.py)
# 内部直接复用 TSF_Frame 的 ml_models, 不用做太复杂

# 4. 跑通
python -m pipelines.train_one --req R01 --data data/raw/R01_synthetic.csv

# 输出预期:
# logs/outputs/R01/model_20260501.pkl   (模型文件)
# logs/outputs/R01/metrics_20260501.json (验证集指标)
# logs/runs/train_R01_20260501_*.log     (训练日志)
```

### 阶段 3:挂监控(15 分钟)

```bash
# 1. 写最小监控脚本 (pipelines/monitor_check.py)
#    内部:
#    - 加载 R01 模型
#    - 装 ModelMonitor (ML store + RuleEngine + ConsoleChannel + FileChannel)
#    - 用最近一个月数据 record_prediction
#    - check_status(), 打印告警

# 2. 跑通
python -m pipelines.monitor_check --req R01

# 输出预期:
# logs/monitor/monitor.db (SQLite 自动建表)
# logs/monitor/alerts.log (若有告警)
# 控制台打印 status.alert_level
```

### 阶段 4:验证 + 复制扩展(剩余时间)

```bash
# 1. 验证 R01 端到端没问题, 提一个 PR
git add . && git commit -m "feat: bootstrap with R01 monthly_deposit"

# 2. 后续每加一个需求 = 复制 R01 → 改 5 个文件 (config / data_loader / rules / yaml / 测试)
# 第 2 个需求大约 2 小时
# 第 5 个之后熟练, 1 小时
# 20 个需求总工时: 约 5 个工作日 (1 周)
```

---

## 九、常见问题与排查

### Q1: 月度跑批,本月真值要等 1 个月才到,监控怎么算 MAPE?
**A:** 用 `settle_actual(target_ts, y_actual)`。预测时 `actual=None`,真值到达时调 settle。`PerformanceMonitor` 用 `target_ts` 对齐,不会错位。

### Q2: 20 个需求共用一个 SQLite,会不会锁死?
**A:** SQLite WAL 模式 + 短连接(框架已配置)。月度跑批 20 需求并发写完全够用。日均写入 < 1 万行的话,SQLite 撑得住几百万。如果未来要更高并发,把 store 切到 PostgreSQL(实现一个 `PgStore(MetricStore)` + `@register_store('postgres')`,代码无需改)。

### Q3: 一个需求出 CRITICAL 告警,会不会影响其他 19 个?
**A:** 不会。每个需求的 `ModelMonitor` 实例独立,告警通道虽然共用但有 `model_id` 字段隔离。CRITICAL 只触发对应 model_id 的重训决策。

### Q4: 政策情景预测(R11)怎么做?
**A:** 不是新模型,是基于 R01 模型 + `HPFAdapter.get_policy_adjusted_forecast(...)`,把"政策乘子"叠加到基准预测上。

### Q5: 多个需求共享底层数据(R09 = R03/R01),怎么编排?
**A:** 在 `pipelines/predict_all.py` 里按 DAG 顺序跑:R01/R03 先跑 → 各自落库 → R09 从 store 读取 R01/R03 最新预测 → 计算比例。Airflow 部署时直接用 task 依赖。

### Q6: 模型文件太多(20 需求 × 月度版本 × 3 年 = 720 个),磁盘吃不消?
**A:** 备份策略保留 3 个月本地 + 全部归档 OSS。本地按需 clean:
```bash
find logs/outputs -name 'model_*.pkl' -mtime +90 -delete
```

### Q7: 中途想换模型(XGBoost → LightGBM)?
**A:** 改 yaml 配置 `model.type: lightgbm`,重新 train 即可。框架的 `ml_models` 注册表已注册了 11 种模型。

### Q8: 怎么对比新模型 vs 老模型?
**A:** 训练时 `--shadow` 模式,**老模型仍在用,新模型并行预测**,两套预测都进 store(model_id 不同)。一周后对比 MAPE,人工决定切换。

---

## 十、收尾 checklist(项目交付前自检)

- [ ] 所有 20 个需求 `python -m pipelines.train_one --req R0X` 都能跑通
- [ ] 所有 20 个需求 `python -m pipelines.predict_one --req R0X --date YYYY-MM` 都能跑通
- [ ] `pytest tests/smoke/` 5 分钟内绿
- [ ] `monitor.db` 三表 (predictions / metrics_snapshot / alerts) 都有数据
- [ ] `alerts.log` 只有 WARNING 以上记录
- [ ] cron / Airflow / K8s 任意一种部署跑通一个完整月份周期
- [ ] 钉钉/邮件通道收到测试告警
- [ ] 备份脚本可手动执行 + 还原一次
- [ ] 模型回滚脚本可手动执行
- [ ] 文档 README 写明:谁能 run、怎么 run、坏了怎么找谁

---

**关键设计哲学回顾:**

1. **TSF_Frame 提供能力, 业务项目提供编排.** 不要在业务项目里重新造监控/特征/模型,直接用框架。
2. **20 个需求 = 1 套模板 × 20 份配置 + 必要差异化代码.** 不要 20 份完全独立的代码,要复用脚手架。
3. **target_ts 对齐 + settle_actual 异步回填** 是 HPF 月度场景能跑得正确的核心保证。
4. **配置外置 (yaml), 代码内聚 (Python).** 调阈值不应改代码,加新需求只动一个目录。
5. **先跑通,再优化.** 第 1 个需求能跑就提 PR,余 19 个慢慢补。

---

*文档版本: v1.0 / 2026-05*
*维护者: TSF_Frame 团队*
