"""
tsf_frame.monitoring — 时序预测监控包
=====================================

公共 API 分为 6 大类:

1. **契约 / Contracts**: 抽象基类、数据类、注册表、工厂
2. **协调器 / Orchestrators**: 可直接实例化的门面
   (``ModelMonitor``, ``PipelineMonitor``)
3. **性能 / Performance**: ``PerformanceMonitor`` + 内建/可扩展指标
4. **质量 & 漂移 / Quality & Drift**: 原始数据质量 + 3 种漂移检测
5. **规则 & 重训 / Rules & Retraining**: 声明式规则引擎 + 重训决策
6. **持久化 & 告警 & 报表 / I-O**: Store 后端 / Channel / Report

典型用法::

    from tsf_frame.monitoring import (
        ModelMonitor, SQLiteStore, ConsoleChannel, FileChannel,
        DataDriftDetector, RuleEngine,
    )

    store = SQLiteStore('./logs/mon.db')
    mon = ModelMonitor(
        model_id='my-model',
        store=store,
        data_drift=DataDriftDetector(reference=X_ref),
        rule_engine=RuleEngine(rule_ids=['R1_NON_NEGATIVE']),
    )
    mon.alert_manager.add_channel(ConsoleChannel())
    mon.alert_manager.add_channel(FileChannel('./logs/alerts.log'))

    mon.record_prediction(timestamp=..., prediction=..., actual=...,
                          features=...)
    status = mon.check_status()
"""

from __future__ import annotations

# ──────────────────── 契约 / Contracts ─────────────────────
from .interfaces import (
    # 枚举 & 常量
    AlertLevel,
    LEVEL_ORDER,
    DriftType,
    # 数据类
    Alert,
    RuleViolation,
    DriftResult,
    QualityIssue,
    MonitoringStatus,
    StageEvent,
    # 抽象基类
    MetricStore,
    AlertChannel,
    DriftDetector,
    QualityChecker,
    RuleChecker,
    ReportGenerator,
    # 类型别名
    MetricFn,
    # 注册表
    STORE_REGISTRY,
    ALERT_CHANNEL_REGISTRY,
    DRIFT_DETECTOR_REGISTRY,
    QUALITY_CHECKER_REGISTRY,
    RULE_REGISTRY,
    REPORT_REGISTRY,
    METRIC_REGISTRY,
    # 注册装饰器
    register_store,
    register_alert_channel,
    register_drift_detector,
    register_quality_checker,
    register_rule,
    register_report,
    register_metric,
    # 工厂
    create_store,
    create_alert_channel,
    create_drift_detector,
    create_quality_checker,
    create_report,
    get_metric_fn,
    # 工具
    level_ge,
    level_max,
    list_registries,
)

# ─────────────────── 基类 / Base ───────────────────────────
from .base_monitor import BaseMonitor

# ─────────────────── 持久化 / Stores ───────────────────────
from .stores import InMemoryStore, SQLiteStore, JsonlStore

# ─────────────────── 告警 / Alerting ───────────────────────
from .alert_manager import (
    AlertManager,
    ConsoleChannel,
    LoggingChannel,
    FileChannel,
    CallbackChannel,
    StoreChannel,
)

# ─────────────────── 性能 / Performance ────────────────────
from .performance_monitor import (
    PerformanceMonitor,
    MultiHorizonMonitor,
    MultiTargetMonitor,
    mae, mse, rmse, mape, smape, r2,
    picp, miw, winkler,
)

# ─────────────────── 质量 / Quality ────────────────────────
from .data_quality import (
    DataQualityMonitor,
    MissingRateCheck,
    OutlierCheck,
    SchemaCheck,
    FrequencyCheck,
    RangeCheck,
)

# ─────────────────── 漂移 / Drift ──────────────────────────
from .drift_detector import (
    DataDriftDetector,
    ConceptDriftDetector,
    PredictionDriftDetector,
    calc_psi,
    calc_ks,
    calc_js,
)

# ─────────────────── 规则引擎 / Rules ──────────────────────
from .rule_engine import (
    Rule,
    RuleEngine,
    DEFAULT_RULE_IDS,
    rule_non_negative,
    rule_sudden_change,
    rule_out_of_band,
    rule_monotonic_expected,
)

# ─────────────────── 重训 / Retraining ─────────────────────
from .retraining_trigger import (
    RetrainingRule,
    RetrainingTrigger,
    RetrainingDecision,
)

# ─────────────────── 报表 / Reports ────────────────────────
from .reporters import TextReport, PlotReport

# ─────────────────── 协调器 / Orchestrators ────────────────
from .model_monitor import ModelMonitor
from .pipeline_monitor import PipelineMonitor


__all__ = [
    # 枚举/常量
    'AlertLevel', 'LEVEL_ORDER', 'DriftType',
    # 数据类
    'Alert', 'RuleViolation', 'DriftResult', 'QualityIssue',
    'MonitoringStatus', 'StageEvent',
    # ABC
    'MetricStore', 'AlertChannel', 'DriftDetector', 'QualityChecker',
    'RuleChecker', 'ReportGenerator',
    # 类型
    'MetricFn',
    # 注册表
    'STORE_REGISTRY', 'ALERT_CHANNEL_REGISTRY', 'DRIFT_DETECTOR_REGISTRY',
    'QUALITY_CHECKER_REGISTRY', 'RULE_REGISTRY', 'REPORT_REGISTRY',
    'METRIC_REGISTRY',
    # 装饰器
    'register_store', 'register_alert_channel', 'register_drift_detector',
    'register_quality_checker', 'register_rule', 'register_report',
    'register_metric',
    # 工厂
    'create_store', 'create_alert_channel', 'create_drift_detector',
    'create_quality_checker', 'create_report', 'get_metric_fn',
    # 工具
    'level_ge', 'level_max', 'list_registries',
    # 基类
    'BaseMonitor',
    # 持久化
    'InMemoryStore', 'SQLiteStore', 'JsonlStore',
    # 告警
    'AlertManager', 'ConsoleChannel', 'LoggingChannel', 'FileChannel',
    'CallbackChannel', 'StoreChannel',
    # 性能
    'PerformanceMonitor', 'MultiHorizonMonitor', 'MultiTargetMonitor',
    'mae', 'mse', 'rmse', 'mape', 'smape', 'r2',
    'picp', 'miw', 'winkler',
    # 质量
    'DataQualityMonitor', 'MissingRateCheck', 'OutlierCheck',
    'SchemaCheck', 'FrequencyCheck', 'RangeCheck',
    # 漂移
    'DataDriftDetector', 'ConceptDriftDetector', 'PredictionDriftDetector',
    'calc_psi', 'calc_ks', 'calc_js',
    # 规则
    'Rule', 'RuleEngine', 'DEFAULT_RULE_IDS',
    'rule_non_negative', 'rule_sudden_change',
    'rule_out_of_band', 'rule_monotonic_expected',
    # 重训
    'RetrainingRule', 'RetrainingTrigger', 'RetrainingDecision',
    # 报表
    'TextReport', 'PlotReport',
    # 协调器
    'ModelMonitor', 'PipelineMonitor',
]
