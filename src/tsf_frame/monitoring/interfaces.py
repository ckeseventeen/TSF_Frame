"""
监控框架公共契约 / Monitoring framework common contracts
=========================================================

这是整个 ``monitoring`` 包的"骨架":

* 所有可插拔角色 (存储后端, 告警通道, 漂移检测器, 质量检查器, 规则,
  指标函数, 报表) 都以 **抽象基类 (ABC) + 注册表 (registry) + 工厂
  (factory)** 的三段式组织。
* 新的角色实现只需: 继承 ABC → 用装饰器 ``@register_*`` 登记 → 在任何
  地方用 ``create_*`` 或 ``get_*`` 按字符串名取到实例。
* 数据类 (``MonitoringStatus``, ``Alert``, ``RuleViolation`` 等) 统一
  在本文件声明, 避免循环导入。

设计原则 / Design principles:

1. **接口隔离**: 每个 ABC 只暴露最小必须方法, 实现可自由扩展。
2. **注册即可用**: 框架核心不 ``import`` 具体实现, 通过注册表解耦。
3. **无状态契约**: 契约本身不持有状态, 状态在实现内部管理。
4. **字符串级告警级别**: 使用 ``'info' < 'warning' < 'error' < 'critical'``
   的字符串序, 便于 JSON 序列化和配置文件。

扩展示例 / Extension example::

    from tsf_frame.monitoring.interfaces import (
        MetricStore, register_store, create_store,
    )

    @register_store('redis')
    class RedisStore(MetricStore):
        def insert_prediction(self, **kw): ...
        def insert_metrics_snapshot(self, **kw): ...
        def insert_alert(self, **kw): ...
        # ... etc

    store = create_store('redis', host='localhost')
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
)

import numpy as np

__all__ = [
    # 枚举/常量 / Enums & constants
    'AlertLevel',
    'LEVEL_ORDER',
    'DriftType',
    # 数据类 / Dataclasses
    'Alert',
    'RuleViolation',
    'DriftResult',
    'QualityIssue',
    'MonitoringStatus',
    'StageEvent',
    # 抽象角色 / Abstract roles
    'MetricStore',
    'AlertChannel',
    'DriftDetector',
    'QualityChecker',
    'RuleChecker',
    'ReportGenerator',
    # 类型别名 / Type aliases
    'MetricFn',
    # 注册表 / Registries
    'STORE_REGISTRY',
    'ALERT_CHANNEL_REGISTRY',
    'DRIFT_DETECTOR_REGISTRY',
    'QUALITY_CHECKER_REGISTRY',
    'RULE_REGISTRY',
    'REPORT_REGISTRY',
    'METRIC_REGISTRY',
    # 注册装饰器 / Register decorators
    'register_store',
    'register_alert_channel',
    'register_drift_detector',
    'register_quality_checker',
    'register_rule',
    'register_report',
    'register_metric',
    # 工厂 / Factories
    'create_store',
    'create_alert_channel',
    'create_drift_detector',
    'create_quality_checker',
    'create_report',
    'get_metric_fn',
    # 工具 / Utilities
    'level_ge',
    'level_max',
    'list_registries',
]


# ==========================================================================
# 级别 & 常量 / Levels & constants
# ==========================================================================

class AlertLevel:
    """
    告警级别常量 / Alert level string constants.

    使用字符串而非 Enum 的原因:
    * JSON / YAML 配置天然可读
    * 跨进程/跨语言 (例如写入 SQLite 后被仪表盘读取) 无需解析

    顺序约定: ``INFO < WARNING < ERROR < CRITICAL`` (见 LEVEL_ORDER)。
    """

    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'


#: 级别顺序字典 / Level order mapping.
LEVEL_ORDER: Dict[str, int] = {
    AlertLevel.INFO: 0,
    AlertLevel.WARNING: 1,
    AlertLevel.ERROR: 2,
    AlertLevel.CRITICAL: 3,
}


def level_ge(a: str, b: str) -> bool:
    """
    比较告警级别 a >= b / Compare levels.

    未知级别视为 INFO, 不抛异常 (监控本身不应崩)。
    """
    return LEVEL_ORDER.get(a, 0) >= LEVEL_ORDER.get(b, 0)


def level_max(*levels: str) -> str:
    """取多个级别中最严重的 / Return the most severe level."""
    if not levels:
        return AlertLevel.INFO
    return max(levels, key=lambda lv: LEVEL_ORDER.get(lv, 0))


class DriftType:
    """
    漂移类型 / Drift type constants.

    * DATA: 输入特征分布变化 (协变量漂移)
    * CONCEPT: 输入-输出关系变化 (残差分布 / error trend)
    * PREDICTION: 模型输出分布变化 (覆盖率 / 方差扩散)
    """

    DATA = 'data'
    CONCEPT = 'concept'
    PREDICTION = 'prediction'


# ==========================================================================
# 数据类 / Dataclasses
# ==========================================================================

@dataclass
class Alert:
    """
    单条告警 / A single alert record.

    Attributes:
        alert_id: 唯一 ID (通常 "{model_id}_{iso_ts_with_us}")
        model_id: 所属模型标识
        level: AlertLevel.*
        message: 人类可读的告警信息
        timestamp: 告警时刻
        source: 告警来源 (rule_id / metric 名 / 组件名)
        details: 任意结构化上下文 (会被 JSON 序列化后存储)
        acknowledged: 是否已被人工处理
    """

    alert_id: str
    model_id: str
    level: str
    message: str
    timestamp: datetime
    source: str = ''
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class RuleViolation:
    """
    规则违反记录 / A rule violation.

    ``rule_engine`` 中的 RuleChecker 扫描特征/预测, 产出本对象列表。
    """

    rule_id: str
    severity: str                         # AlertLevel.*
    message: str
    value: Optional[float] = None         # 触发规则的量纲值 (如 missing_rate)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftResult:
    """
    漂移检测单次输出 / Output of one drift detection pass.

    Attributes:
        drift_type: DriftType.*
        detected: 是否判定为漂移
        severity: 建议告警级别 (子类可覆盖)
        score: 主要统计量 (如 PSI, KS stat, residual mean shift)
        p_value: 若是假设检验, 报告 p-value
        per_feature: 多维场景下每维得分
        details: 任意附加信息
    """

    drift_type: str
    detected: bool
    severity: str = AlertLevel.WARNING
    score: float = 0.0
    p_value: Optional[float] = None
    per_feature: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityIssue:
    """
    数据质量问题 / A data quality issue.

    Attributes:
        issue_id: 问题代码 (MISSING / OUTLIER / SCHEMA / FREQ / RANGE ...)
        severity: AlertLevel.*
        column: 触发列名 (全表问题留空)
        message: 可读描述
        value: 量化指标 (缺失率等)
        details: 附加上下文
    """

    issue_id: str
    severity: str
    column: str = ''
    message: str = ''
    value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringStatus:
    """
    监控系统整体状态快照 / Snapshot of monitor state.

    ModelMonitor.check_status() 或 PipelineMonitor.check_status()
    都返回本对象, 便于外部统一处理与序列化。
    """

    model_id: str
    timestamp: datetime
    alert_level: str = AlertLevel.INFO
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    data_drift: Optional[DriftResult] = None
    concept_drift: Optional[DriftResult] = None
    prediction_drift: Optional[DriftResult] = None
    quality_issues: List[QualityIssue] = field(default_factory=list)
    rule_violations: List[RuleViolation] = field(default_factory=list)
    needs_retraining: bool = False
    recommendations: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageEvent:
    """
    管道阶段事件 / A pipeline stage event (used by PipelineMonitor).

    Attributes:
        stage: 阶段名 (load / preprocess / feature / train / predict ...)
        started_at: 开始时间
        ended_at: 结束时间 (失败时仍记录)
        duration_ms: 耗时毫秒
        success: 是否成功
        error: 失败堆栈 / 异常信息
        metadata: 附加信息 (样本数/特征数等)
    """

    stage: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==========================================================================
# 抽象角色 / Abstract roles
# ==========================================================================

class MetricStore(ABC):
    """
    持久化抽象层 / Persistence abstraction.

    后端实现: InMemoryStore (测试/默认), SQLiteStore (生产单机),
    可扩展 FileJsonlStore, PostgresStore, RedisStore 等。

    所有 insert_* 必须幂等可重入, query_* 应支持按 ``model_id``
    与时间窗过滤。
    """

    # --- 写 -------------------------------------------------------------
    @abstractmethod
    def insert_prediction(
        self,
        *,
        model_id: str,
        timestamp: datetime,
        target: str,
        y_pred: float,
        y_lower: Optional[float] = None,
        y_upper: Optional[float] = None,
        y_actual: Optional[float] = None,
    ) -> None:
        """记录一次预测 / Persist one prediction."""

    @abstractmethod
    def update_actual(
        self,
        *,
        model_id: str,
        timestamp: datetime,
        target: str,
        y_actual: float,
    ) -> None:
        """回填真实值 / Back-fill the actual value."""

    @abstractmethod
    def insert_metrics_snapshot(
        self,
        *,
        model_id: str,
        timestamp: datetime,
        metrics: Mapping[str, float],
        window: Optional[int] = None,
    ) -> None:
        """写一组聚合指标 / Persist a metric snapshot."""

    @abstractmethod
    def insert_alert(self, alert: Alert) -> None:
        """落库一条告警 / Persist an alert."""

    # --- 读 -------------------------------------------------------------
    @abstractmethod
    def query_predictions(
        self,
        *,
        model_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """查询预测记录 / Query predictions."""

    @abstractmethod
    def query_metrics(
        self,
        *,
        model_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查询指标快照 / Query metric snapshots."""

    @abstractmethod
    def query_alerts(
        self,
        *,
        model_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        min_level: Optional[str] = None,
    ) -> List[Alert]:
        """查询告警 / Query alerts."""

    # --- 维护 -----------------------------------------------------------
    def cleanup_old(self, *, retain_days: int) -> int:
        """删除超过 N 天的记录, 返回删除条数 (子类选填)。"""
        return 0

    def close(self) -> None:
        """关闭连接/释放资源 (子类选填)。"""


class AlertChannel(ABC):
    """
    告警分发通道 / Alert delivery channel.

    已内置: ConsoleChannel (stdout), LoggingChannel (stdlib logging),
    FileChannel (append 行日志), CallbackChannel (包裹任意 callable)。
    """

    @abstractmethod
    def send(self, alert: Alert) -> None:
        """发送告警 / Deliver one alert."""

    def close(self) -> None:
        """释放资源 / Optional cleanup."""


class DriftDetector(ABC):
    """
    漂移检测器 / Drift detector.

    典型用法::

        det.update(new_batch)            # 喂新数据
        result = det.detect()            # 得到 DriftResult
    """

    drift_type: str = DriftType.DATA     # 由子类覆盖

    @abstractmethod
    def update(self, data: np.ndarray) -> None:
        """追加一批观测 / Append a batch."""

    @abstractmethod
    def detect(self) -> DriftResult:
        """执行一次检测 / Run one detection pass."""

    @abstractmethod
    def reset(self) -> None:
        """清空内部状态 / Reset state."""


class QualityChecker(ABC):
    """
    数据质量检查器 / Data quality checker.

    一次 ``check()`` 产出若干 QualityIssue。
    """

    @abstractmethod
    def check(self, data: Any) -> List[QualityIssue]:
        """执行一次质量检查 / Run checks on data batch."""


class RuleChecker(ABC):
    """
    业务/自定义规则检查器 / Rule checker.

    与 QualityChecker 的区别: RuleChecker 面向 (features, prediction)
    组合, QualityChecker 主要面向原始数据。
    """

    @abstractmethod
    def check(
        self,
        *,
        features: Any = None,
        prediction: Any = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> List[RuleViolation]:
        """执行一次规则扫描 / Run all enabled rules."""


class ReportGenerator(ABC):
    """
    监控报表生成器 / Report generator.

    ``generate()`` 返回产物路径 (文本/图片/HTML) 供 UI 或外部系统读取。
    """

    @abstractmethod
    def generate(
        self,
        *,
        model_id: str,
        store: MetricStore,
        out_path: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """生成一份报表, 返回产物路径 / Produce report and return its path."""


# ==========================================================================
# 指标函数 / Metric functions
# ==========================================================================

#: 指标函数签名 / Signature of a metric function.
#:     (y_true, y_pred, **kwargs) -> float
MetricFn = Callable[..., float]


# ==========================================================================
# 注册表 / Registries
# ==========================================================================

STORE_REGISTRY: Dict[str, Type[MetricStore]] = {}
ALERT_CHANNEL_REGISTRY: Dict[str, Type[AlertChannel]] = {}
DRIFT_DETECTOR_REGISTRY: Dict[str, Type[DriftDetector]] = {}
QUALITY_CHECKER_REGISTRY: Dict[str, Type[QualityChecker]] = {}
RULE_REGISTRY: Dict[str, Callable[..., List[RuleViolation]]] = {}
REPORT_REGISTRY: Dict[str, Type[ReportGenerator]] = {}
METRIC_REGISTRY: Dict[str, MetricFn] = {}


def _make_register(registry: Dict[str, Any], kind: str):
    """内部: 生成 ``@register_xxx(name)`` 风格装饰器。"""

    def decorator(name: str):
        def _wrap(obj):
            registry[name] = obj
            return obj
        return _wrap
    decorator.__name__ = f'register_{kind}'
    decorator.__doc__ = (
        f'Register a {kind} implementation under a string name.\n\n'
        f'Example::\n\n'
        f'    @register_{kind}("my_impl")\n'
        f'    class MyImpl(...): ...\n'
    )
    return decorator


register_store = _make_register(STORE_REGISTRY, 'store')
register_alert_channel = _make_register(ALERT_CHANNEL_REGISTRY, 'alert_channel')
register_drift_detector = _make_register(DRIFT_DETECTOR_REGISTRY, 'drift_detector')
register_quality_checker = _make_register(QUALITY_CHECKER_REGISTRY, 'quality_checker')
register_rule = _make_register(RULE_REGISTRY, 'rule')
register_report = _make_register(REPORT_REGISTRY, 'report')
register_metric = _make_register(METRIC_REGISTRY, 'metric')


# ==========================================================================
# 工厂 / Factories
# ==========================================================================

def _factory(registry: Dict[str, Any], kind: str):
    """内部: 统一的 create/get 工厂生成器。"""

    def _build(name: str, *args, **kwargs):
        if name not in registry:
            available = ', '.join(sorted(registry.keys())) or '(empty)'
            raise KeyError(
                f'Unknown {kind} "{name}". Available: {available}'
            )
        impl = registry[name]
        if isinstance(impl, type):
            return impl(*args, **kwargs)
        return impl
    _build.__name__ = f'create_{kind}'
    _build.__doc__ = f'Instantiate a registered {kind} by name.'
    return _build


create_store = _factory(STORE_REGISTRY, 'store')
create_alert_channel = _factory(ALERT_CHANNEL_REGISTRY, 'alert_channel')
create_drift_detector = _factory(DRIFT_DETECTOR_REGISTRY, 'drift_detector')
create_quality_checker = _factory(QUALITY_CHECKER_REGISTRY, 'quality_checker')
create_report = _factory(REPORT_REGISTRY, 'report')


def get_metric_fn(name: str) -> MetricFn:
    """
    根据名字取一个指标函数 / Fetch a metric function by name.

    Raises:
        KeyError: 如果未注册。
    """
    if name not in METRIC_REGISTRY:
        available = ', '.join(sorted(METRIC_REGISTRY.keys())) or '(empty)'
        raise KeyError(f'Unknown metric "{name}". Available: {available}')
    return METRIC_REGISTRY[name]


# ==========================================================================
# 工具 / Utilities
# ==========================================================================

def list_registries() -> Dict[str, List[str]]:
    """
    列出所有注册表当前内容 (用于调试/自省) /
    Return a snapshot of every registry.
    """
    return {
        'stores': sorted(STORE_REGISTRY),
        'alert_channels': sorted(ALERT_CHANNEL_REGISTRY),
        'drift_detectors': sorted(DRIFT_DETECTOR_REGISTRY),
        'quality_checkers': sorted(QUALITY_CHECKER_REGISTRY),
        'rules': sorted(RULE_REGISTRY),
        'reports': sorted(REPORT_REGISTRY),
        'metrics': sorted(METRIC_REGISTRY),
    }


# ==========================================================================
# main — 自省演示 / Introspection demo
# ==========================================================================

def main() -> None:
    """打印当前注册表内容。在 monitoring 的其他模块导入本模块后, 各自
    的子类会通过装饰器填入, 这里正是验证 "插拔" 已生效的最简方式。"""
    try:
        from . import (                           # noqa: F401
            alert_manager, stores, performance_monitor, drift_detector,
            data_quality, rule_engine, reporters,
        )
    except Exception as exc:  # pragma: no cover
        print(f'[warn] 部分模块尚未就绪: {exc}')

    print('=' * 70)
    print(' monitoring.interfaces — Registry snapshot')
    print('=' * 70)
    snap = list_registries()
    for kind, names in snap.items():
        print(f'\n[{kind}]')
        for n in names:
            print(f'  - {n}')
    print()
    print(f'AlertLevel order: {LEVEL_ORDER}')
    print(f'DriftType: DATA={DriftType.DATA!r}, '
          f'CONCEPT={DriftType.CONCEPT!r}, '
          f'PREDICTION={DriftType.PREDICTION!r}')


if __name__ == '__main__':
    main()
