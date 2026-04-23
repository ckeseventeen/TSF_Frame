"""
模型监控协调器 / Model monitor orchestrator
============================================

``ModelMonitor`` 把前面每个子模块组装成一站式监控门面。所有子系统都是
**可选 + 可替换** (依赖注入), 没传的会用合理默认值构造。

子系统全景::

    ModelMonitor(model_id)
    ├── store               : MetricStore         (default: InMemoryStore)
    ├── alert_manager       : AlertManager        (default: auto)
    ├── performance_monitor : PerformanceMonitor  (default: auto)
    ├── data_quality        : DataQualityMonitor  (可选)
    ├── data_drift          : DataDriftDetector   (可选, 传 reference 才启用)
    ├── concept_drift       : ConceptDriftDetector
    ├── prediction_drift    : PredictionDriftDetector (可选)
    ├── rule_engine         : RuleEngine          (可选)
    └── retraining_trigger  : RetrainingTrigger   (default: 默认规则集)

数据流::

    record_prediction(...)
        ├─► store.insert_prediction        (持久化)
        ├─► performance_monitor.update     (指标窗口)
        ├─► data_drift.update              (特征)
        ├─► prediction_drift.update        (y_pred)
        └─► concept_drift.update           (y_true-y_pred 残差, 若有 actual)

    record_data_batch(df)
        └─► data_quality.check             (原始数据质量)

    check_status()
        ├── 聚合 current metrics
        ├── detect 三类 drift
        ├── rule_engine.check 预测规则
        ├── retraining_trigger.check 是否重训
        ├── 合成 alert_level → alert_manager.emit
        └── 写 metrics_snapshot 入 store
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .alert_manager import AlertManager
from .base_monitor import BaseMonitor
from .data_quality import DataQualityMonitor
from .drift_detector import (
    ConceptDriftDetector,
    DataDriftDetector,
    PredictionDriftDetector,
)
from .interfaces import (
    AlertLevel,
    DriftResult,
    LEVEL_ORDER,
    MetricStore,
    MonitoringStatus,
    QualityIssue,
    RuleViolation,
    level_max,
)
from .performance_monitor import PerformanceMonitor
from .retraining_trigger import RetrainingTrigger
from .rule_engine import RuleEngine
from .stores import InMemoryStore

__all__ = ['ModelMonitor']


class ModelMonitor(BaseMonitor):
    """
    模型监控一站式协调器 / Model monitoring facade.

    Args:
        model_id:            模型标识
        store:               MetricStore 实现; 默认 InMemoryStore
        alert_manager:       告警管理器; 默认自建并绑定 store
        performance_monitor: PerformanceMonitor (度量窗口); 默认自建
        data_quality:        DataQualityMonitor; None 表示不做原始数据检查
        data_drift:          DataDriftDetector; None 表示不做特征漂移
        concept_drift:       ConceptDriftDetector; 默认自建
        prediction_drift:    PredictionDriftDetector; None 表示不做预测漂移
        rule_engine:         RuleEngine; None 表示不做业务规则
        retraining_trigger:  RetrainingTrigger; 默认使用 default_rules()
        target_name:         目标列名 (记入 store 与告警 detail)
        cold_start_samples:  积累 < 此数量前跳过漂移检测 (防小样本误报)
    """

    def __init__(
        self,
        model_id: str,
        *,
        store: Optional[MetricStore] = None,
        alert_manager: Optional[AlertManager] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        data_quality: Optional[DataQualityMonitor] = None,
        data_drift: Optional[DataDriftDetector] = None,
        concept_drift: Optional[ConceptDriftDetector] = None,
        prediction_drift: Optional[PredictionDriftDetector] = None,
        rule_engine: Optional[RuleEngine] = None,
        retraining_trigger: Optional[RetrainingTrigger] = None,
        target_name: str = 'y',
        cold_start_samples: int = 20,
        window_size: int = 100,
        history_size: int = 20000,
    ):
        super().__init__(model_id, history_size=history_size)

        self.store: MetricStore = store or InMemoryStore()
        self.alert_manager: AlertManager = (
            alert_manager or AlertManager(model_id, store=self.store)
        )
        self.perf: PerformanceMonitor = performance_monitor or \
            PerformanceMonitor(model_id, window_size=window_size)
        self.data_quality = data_quality
        self.data_drift = data_drift
        self.concept_drift: ConceptDriftDetector = (
            concept_drift or ConceptDriftDetector(window_size=window_size)
        )
        self.prediction_drift = prediction_drift
        self.rule_engine = rule_engine
        self.retraining_trigger: RetrainingTrigger = (
            retraining_trigger or RetrainingTrigger()
        )

        self.target_name = target_name
        self.cold_start_samples = int(cold_start_samples)

        self._n_records: int = 0
        self._last_actual: Optional[float] = None
        self._samples_since_last_train: int = 0
        self._last_train_at: Optional[datetime] = self.created_at

    # ------------------------------------------------------------------
    # BaseMonitor 必须实现的抽象
    # ------------------------------------------------------------------
    def record(self, event: Dict[str, Any]) -> None:
        """通用入口: dispatch 到 record_prediction / record_data_batch。"""
        if 'prediction' in event or 'y_pred' in event:
            self.record_prediction(
                timestamp=event.get('timestamp', datetime.now()),
                prediction=event.get('prediction', event.get('y_pred')),
                actual=event.get('actual', event.get('y_true')),
                features=event.get('features'),
                y_lower=event.get('y_lower'),
                y_upper=event.get('y_upper'),
            )
        elif 'data' in event:
            self.record_data_batch(event['data'])
        else:
            self.history.append({'timestamp': datetime.now(), **event})

    def reset(self) -> None:
        self._n_records = 0
        self._last_actual = None
        self._samples_since_last_train = 0
        self.history.clear()
        self.perf.reset()
        if self.data_drift is not None:
            self.data_drift.reset()
        self.concept_drift.reset()
        if self.prediction_drift is not None:
            self.prediction_drift.reset()
        self.alert_manager.clear()

    # ------------------------------------------------------------------
    # 数据入口
    # ------------------------------------------------------------------
    def record_data_batch(self, data: pd.DataFrame) -> List[QualityIssue]:
        """
        用于处理**原始数据质量** (非预测) 场景 — 把 df 喂给
        DataQualityMonitor, 并把严重问题上报告警。
        """
        if self.data_quality is None:
            return []
        issues = self.data_quality.check(data)
        for iss in issues:
            if iss.severity in (AlertLevel.ERROR, AlertLevel.CRITICAL):
                self.alert_manager.emit(
                    iss.severity, iss.message,
                    source=f'quality.{iss.issue_id}',
                    details={'column': iss.column, 'value': iss.value,
                             **iss.details},
                )
        return issues

    def record_prediction(
        self,
        *,
        timestamp: datetime,
        prediction: float,
        actual: Optional[float] = None,
        features: Optional[Any] = None,
        y_lower: Optional[float] = None,
        y_upper: Optional[float] = None,
    ) -> None:
        """记录一次模型预测 (核心入口)。"""
        self._n_records += 1
        self._samples_since_last_train += 1
        prediction = float(prediction)
        actual_f = None if actual is None else float(actual)

        # 1) 持久化
        try:
            self.store.insert_prediction(
                model_id=self.model_id, timestamp=timestamp,
                target=self.target_name,
                y_pred=prediction, y_lower=y_lower, y_upper=y_upper,
                y_actual=actual_f,
            )
        except Exception:  # pragma: no cover
            pass

        # 2) 性能窗口
        self.perf.update(
            y_pred=prediction, y_true=actual_f,
            y_lower=y_lower, y_upper=y_upper, timestamp=timestamp,
        )

        # 3) 特征漂移
        if self.data_drift is not None and features is not None:
            arr = np.atleast_2d(np.asarray(features, dtype=float))
            self.data_drift.update(arr)

        # 4) 预测漂移
        if self.prediction_drift is not None:
            self.prediction_drift.update(np.array([prediction]))

        # 5) 概念漂移 (需要 actual)
        if actual_f is not None:
            resid = actual_f - prediction
            self.concept_drift.update(np.array([resid]))
            self._last_actual = actual_f

        self.history.append({
            'timestamp': timestamp,
            'y_pred': prediction, 'y_actual': actual_f,
            'y_lower': y_lower, 'y_upper': y_upper,
        })

    # ------------------------------------------------------------------
    # 状态聚合
    # ------------------------------------------------------------------
    def check_status(self) -> MonitoringStatus:
        """
        产生并广播一次完整的监控快照。

        步骤:
        1. 算当前窗口 metrics → baseline 对比
        2. 检测三类漂移 (冷启动期跳过)
        3. 规则引擎
        4. 重训决策
        5. 合成 alert_level, 发送告警 + 写 metrics_snapshot
        """
        now = datetime.now()

        # 1) metrics
        metrics = self.perf.current()

        # 2) drift (冷启动跳过)
        cold = self._n_records < self.cold_start_samples
        data_drift: Optional[DriftResult] = None
        concept_drift: Optional[DriftResult] = None
        prediction_drift: Optional[DriftResult] = None
        if not cold:
            if self.data_drift is not None:
                data_drift = self.data_drift.detect()
            concept_drift = self.concept_drift.detect()
            if self.prediction_drift is not None:
                prediction_drift = self.prediction_drift.detect()

        # 3) rules
        rule_violations: List[RuleViolation] = []
        if self.rule_engine is not None:
            last_pred = (self.history[-1]['y_pred']
                         if self.history else None)
            if last_pred is not None:
                rule_violations = self.rule_engine.check(
                    prediction=[last_pred],
                    context={
                        'target': self.target_name,
                        'last_actual': self._last_actual,
                    },
                )

        # 4) retrain
        hours_since_train = (
            (now - self._last_train_at).total_seconds() / 3600
            if self._last_train_at else 0
        )
        decision = self.retraining_trigger.check({
            'performance': metrics,
            'data_drift': bool(data_drift and data_drift.detected),
            'concept_drift': bool(
                concept_drift and concept_drift.detected),
            'prediction_drift': bool(
                prediction_drift and prediction_drift.detected),
            'samples_since_last_train': self._samples_since_last_train,
            'hours_since_last_train': hours_since_train,
            'now': now,
        })

        # 5) 合成 alert_level + 告警
        levels: List[str] = [AlertLevel.INFO]
        recs: List[str] = []

        for d in (data_drift, concept_drift, prediction_drift):
            if d and d.detected:
                levels.append(d.severity)
                recs.append(f'检测到 {d.drift_type} 漂移 (score={d.score:.3f})')
        for v in rule_violations:
            levels.append(v.severity)
            recs.append(f'[{v.rule_id}] {v.message}')
        if decision.should_retrain:
            levels.append(AlertLevel.CRITICAL)
            recs.append('建议重新训练模型: ' + '; '.join(decision.reasons))

        final_level = level_max(*levels)

        # 写快照
        try:
            self.store.insert_metrics_snapshot(
                model_id=self.model_id, timestamp=now,
                metrics={k: v for k, v in metrics.items()
                         if isinstance(v, (int, float))
                         and not (isinstance(v, float) and np.isnan(v))},
                window=self.perf.window_size,
            )
        except Exception:  # pragma: no cover
            pass

        # 分级告警
        if final_level == AlertLevel.CRITICAL:
            self.alert_manager.critical(
                '模型需要干预', source='model_monitor',
                details={'metrics': metrics, 'recommendations': recs,
                         'triggered_rules': decision.triggered},
            )
        elif final_level == AlertLevel.ERROR:
            self.alert_manager.error(
                '模型性能显著下降', source='model_monitor',
                details={'metrics': metrics, 'recommendations': recs},
            )
        elif final_level == AlertLevel.WARNING:
            self.alert_manager.warning(
                '模型监控告警', source='model_monitor',
                details={'metrics': metrics, 'recommendations': recs},
            )

        return MonitoringStatus(
            model_id=self.model_id,
            timestamp=now,
            alert_level=final_level,
            performance_metrics=metrics,
            data_drift=data_drift,
            concept_drift=concept_drift,
            prediction_drift=prediction_drift,
            rule_violations=rule_violations,
            needs_retraining=decision.should_retrain,
            recommendations=recs,
            extra={
                'n_records': self._n_records,
                'cold_start': cold,
                'retraining_decision': {
                    'triggered': decision.triggered,
                    'reasons': decision.reasons,
                },
            },
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def record_retraining(self, when: Optional[datetime] = None) -> None:
        """通知监控器一次重训已发生, 重置相关计数与冷却。"""
        self._last_train_at = when or datetime.now()
        self._samples_since_last_train = 0
        self.retraining_trigger.record_retraining(self._last_train_at)
        self.alert_manager.info(
            '模型已完成重训', source='model_monitor',
            details={'at': self._last_train_at.isoformat()},
        )

    def set_performance_baseline(self, baseline: Mapping[str, float]) -> None:
        self.perf.set_baseline(baseline)


# ==========================================================================
# main — 全链路演示
# ==========================================================================

def main() -> None:
    """
    端到端演示: 200 条样本, 中途注入特征/概念漂移 + 负预测规则,
    观察 alert_level 从 info → warning → critical 的自动升级。
    """
    from .alert_manager import ConsoleChannel
    from .data_quality import DataQualityMonitor, RangeCheck
    from .stores import InMemoryStore

    print('=' * 70)
    print(' model_monitor — end-to-end pluggable demo')
    print('=' * 70)

    store = InMemoryStore()
    rng = np.random.default_rng(0)
    n = 200

    ref_features = rng.standard_normal((100, 3))
    mon = ModelMonitor(
        model_id='demo_model',
        store=store,
        data_drift=DataDriftDetector(reference=ref_features,
                                     feature_names=['f1', 'f2', 'f3']),
        prediction_drift=PredictionDriftDetector(
            reference=rng.normal(100, 5, 200)),
        rule_engine=RuleEngine(
            rule_ids=['R1_NON_NEGATIVE', 'R2_SUDDEN_CHANGE'],
            params={'R2_SUDDEN_CHANGE': {'max_ratio': 0.3}},
        ),
        data_quality=(DataQualityMonitor()
                      .add(RangeCheck(non_negative=['y']))),
        target_name='y',
        cold_start_samples=30,
        window_size=50,
    )
    mon.alert_manager.add_channel(ConsoleChannel(min_level=AlertLevel.WARNING))
    mon.set_performance_baseline({'mae': 1.0, 'mape': 0.02, 'rmse': 1.5})

    # --- 前 100 条: 正常 ---
    for i in range(100):
        feats = rng.standard_normal(3)
        true_y = float(rng.normal(100, 3))
        pred_y = true_y + float(rng.normal(0, 1))
        mon.record_prediction(
            timestamp=datetime.now(),
            prediction=pred_y, actual=true_y,
            features=feats, y_lower=pred_y - 2, y_upper=pred_y + 2,
        )
    s1 = mon.check_status()
    print(f'\n[前 100 条]  level={s1.alert_level}  '
          f'MAPE={s1.performance_metrics.get("mape", float("nan")):.4f}')

    # --- 后 100 条: 注入 3σ 特征漂移 + 残差放大 + 一次负预测 ---
    for i in range(100, n):
        feats = rng.standard_normal(3) + 2.5       # data drift
        true_y = float(rng.normal(100, 3))
        pred_y = true_y + float(rng.normal(5, 5))  # 残差大
        if i == 180:
            pred_y = -10.0                         # 触发 R1 规则
        mon.record_prediction(
            timestamp=datetime.now(),
            prediction=pred_y, actual=true_y,
            features=feats, y_lower=pred_y - 4, y_upper=pred_y + 4,
        )
    s2 = mon.check_status()
    print(f'\n[后 100 条]  level={s2.alert_level}')
    print(f'   data_drift    = {s2.data_drift.detected if s2.data_drift else None}')
    print(f'   concept_drift = {s2.concept_drift.detected if s2.concept_drift else None}')
    print(f'   pred_drift    = {s2.prediction_drift.detected if s2.prediction_drift else None}')
    print(f'   rule_viol     = {len(s2.rule_violations)}')
    print(f'   needs_retrain = {s2.needs_retraining}')
    print('\n recommendations:')
    for r in s2.recommendations[:10]:
        print(f'   - {r}')

    # 告警分布
    alerts = store.query_alerts(model_id='demo_model')
    dist: Dict[str, int] = {}
    for a in alerts:
        dist[a.level] = dist.get(a.level, 0) + 1
    print(f'\n告警总数 {len(alerts)}, 分布: {dist}')
    print(f'指标快照数  {len(store.query_metrics(model_id="demo_model"))}')


if __name__ == '__main__':
    main()
