"""
模型监控协调器 / Model monitor orchestrator
============================================

``ModelMonitor`` 把前面每个子模块组装成一站式监控门面。所有子系统都是
**可选 + 可替换** (依赖注入), 没传的会用合理默认值构造。

子系统全景::

    ModelMonitor(model_id)
    ├── store               : MetricStore         (default: InMemoryStore)  把数据存起来
    ├── alert_manager       : AlertManager        (default: auto)           把告警发出去
    ├── performance_monitor : PerformanceMonitor  (default: auto)           算MAE/MAPE 等指标
    ├── data_quality        : DataQualityMonitor  (可选)                     检查原始数据缺失/异常
    ├── data_drift          : DataDriftDetector   (可选, 传 reference 才启用)  检测特征分布是否变了
    ├── concept_drift       : ConceptDriftDetector                          检测残差分布是否变了
    ├── prediction_drift    : PredictionDriftDetector (可选)                 检测预测值分布是否变了
    ├── rule_engine         : RuleEngine          (可选)                      跑业务规则（如"预测不能为负"）
    └── retraining_trigger  : RetrainingTrigger   (default: 默认规则集)          判断是否该重训

数据流::
    每来一条预测时
    record_prediction(...)
        ├─► store.insert_prediction        (持久化)  (存进数据库)
        ├─► performance_monitor.update     (指标窗口)   (更新指标)
        ├─► data_drift.update              (特征)         (喂特征)
        ├─► prediction_drift.update        (y_pred)     (喂预测值)
        └─► concept_drift.update           (y_true-y_pred 残差, 若有 actual)    喂残差，如果有真实值)

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

import logging
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
        window_size：        指标的"近期窗口"
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
        cold_start_samples: int = 20,   # 冷启动样本数
        window_size: int = 100,             # PerformanceMonitor 队列容量
        metric_window: int = 12,            # 指标计算窗口 (近 N 条已 settle 的记录)
        history_size: int = 20000,           # 事件留底缓冲
    ):
        super().__init__(model_id, history_size=history_size)

        # ── 子系统 (依赖注入, 全部可替换) ────────────────────────────
        # 持久化后端 (Memory / SQLite / JSONL); 默认内存
        # / Pluggable persistence backend
        self.store: MetricStore = store or InMemoryStore()
        # 告警分发中心; 默认自建并绑定 store
        # / Alert manager (auto-built & bound to store if not provided)
        self.alert_manager: AlertManager = (
            alert_manager or AlertManager(model_id, store=self.store)
        )
        # 性能滑窗监控 (target_ts 对齐, metric_window 默认 12)
        # / Sliding-window performance monitor
        self.perf: PerformanceMonitor = performance_monitor or \
            PerformanceMonitor(
                model_id,
                window_size=window_size,
                metric_window=metric_window,
            )
        # 原始数据质量检查器, None 表示不做该检查
        # / Optional raw-data quality monitor
        self.data_quality = data_quality
        # 数据漂移检测器 (PSI/KS on 特征), None 不检测
        # / Optional data (covariate) drift detector
        self.data_drift = data_drift
        # 概念漂移检测器 (残差分布), 默认自建; 必有
        # / Concept drift detector (residual-based); always created
        self.concept_drift: ConceptDriftDetector = (
            concept_drift or ConceptDriftDetector(window_size=window_size)
        )
        # 预测分布漂移检测器, None 不检测
        # / Optional prediction-output drift detector
        self.prediction_drift = prediction_drift
        # 业务规则引擎 (R1_NON_NEGATIVE 等), None 不跑规则
        # / Optional declarative rule engine
        self.rule_engine = rule_engine
        # 重训触发器, 默认带 5 条通用规则
        # / Retraining trigger; default rules are reasonable for general use
        self.retraining_trigger: RetrainingTrigger = (
            retraining_trigger or RetrainingTrigger()
        )

        # ── 业务上下文 ──────────────────────────────────────────────
        # 目标列名 (落库 + 告警 detail 用), e.g. 'monthly_deposit'
        # / Target column name for storage / alert details
        self.target_name = target_name
        # 冷启动期样本阈值: 累计 < 此数量时跳过漂移检测, 防小样本误报
        # / Skip drift detection until this many samples accumulated
        self.cold_start_samples = int(cold_start_samples)

        # ── 内部累积计数器 ───────────────────────────────────────────
        # 累计收到的预测条数 (用于冷启动判断)
        # / Total predictions ever recorded
        self._n_records: int = 0
        # 最近一次回填到的真值, 供 R2_SUDDEN_CHANGE 等规则使用
        # / Last settled actual (used by sudden-change rule)
        self._last_actual: Optional[float] = None
        # 自上次重训以来累计收到的预测条数, 触发量驱动重训
        # / Predictions since last retrain (volume-based retrain rule)
        self._samples_since_last_train: int = 0
        # 上次重训发生时刻; 默认初始化为创建时刻 (有"时长"概念)
        # / Last retraining timestamp (drives time-based retrain rule)
        self._last_train_at: Optional[datetime] = self.created_at

        # ── 健康度状态 ──────────────────────────────────────────────
        # 持久化是否健康: store.insert_xxx 失败时置 False, check_status 时
        # 上报到 MonitoringStatus.extra 让运维可见. 之前 try/except: pass
        # 会让磁盘满 / 权限错被静默吞掉, 监控系统坏了用户都不知道.
        # / Storage healthy flag; flipped to False on persistent backend errors
        self._storage_healthy: bool = True
        # 最近一次 store 错误信息, 用于诊断
        self._last_storage_error: Optional[str] = None
        # 模块内 logger, 用于上报库级别错误 (区别于 alert_manager 的业务告警)
        self._logger = logging.getLogger(f'tsf_frame.monitoring.{model_id}')

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
        update_drift: bool = True,
    ) -> None:
        """
        记录一次模型预测 / Record one prediction.

        ``timestamp`` 解释为 **target_ts** (预测的目标时点),
        以保证后续 ``settle_actual(target_ts=...)`` 能精确回填。

        多步 horizon 跑批用法 (推荐):
            # 一次跑批输出 12 个 horizon, 但 features 只喂一次到 drift_detector
            for h in range(1, 13):
                target_ts = forecast_time + relativedelta(months=h)
                mon.record_prediction(
                    timestamp=target_ts,
                    prediction=preds[h-1],
                    features=feat_at_forecast if h == 1 else None,
                    update_drift=(h == 1),       # 只在 h=1 喂 prediction_drift
                )

        异步真值到达时, **不要再次调用 record_prediction**, 改用
        ``settle_actual(target_ts=..., y_actual=...)`` 回填。

        Args:
            timestamp:    预测目标时点 (target_ts), 作为对齐键
            prediction:   预测点估计
            actual:       同步可知的真值 (一般为 None, HPF 等异步场景)
            features:     特征向量 (1D) 或矩阵 (2D), 仅在 update_drift 时
                          推入 data_drift; HPF 跑批每月只应推一次
            y_lower/y_upper: 概率预测区间
            update_drift: 是否把本条推入 prediction_drift / data_drift;
                          多步 horizon 跑批时建议**只在 h=1 设 True**, 避免
                          同一组特征 / 同一次跑批被 12 倍放大
        """
        self._n_records += 1
        self._samples_since_last_train += 1
        prediction = float(prediction)
        actual_f = None if actual is None else float(actual)

        # 1) 持久化 — 失败不静默吞,记 logger.error + 翻 storage_healthy=False
        # / Persist with error logging (no more silent except: pass)
        try:
            self.store.insert_prediction(
                model_id=self.model_id, timestamp=timestamp,
                target=self.target_name,
                y_pred=prediction, y_lower=y_lower, y_upper=y_upper,
                y_actual=actual_f,
            )
        except Exception as exc:
            self._storage_healthy = False
            self._last_storage_error = (
                f'insert_prediction failed: {type(exc).__name__}: {exc}'
            )
            self._logger.error(
                'ModelMonitor[%s] store.insert_prediction failed: %s',
                self.model_id, exc, exc_info=True,
            )

        # 2) 性能窗口 (按 target_ts 对齐)
        self.perf.update(
            y_pred=prediction, y_true=actual_f,
            y_lower=y_lower, y_upper=y_upper, timestamp=timestamp,
        )

        # 3) 特征漂移 (仅在 update_drift 时推入, 防 horizon 放大)
        if update_drift and self.data_drift is not None and features is not None:
            arr = np.atleast_2d(np.asarray(features, dtype=float))
            self.data_drift.update(arr)

        # 4) 预测漂移 (同上)
        if update_drift and self.prediction_drift is not None:
            self.prediction_drift.update(np.array([prediction]))

        # 5) 概念漂移 (需要 actual; HPF 异步场景下一般在 settle_actual 中触发)
        if actual_f is not None:
            resid = actual_f - prediction
            self.concept_drift.update(np.array([resid]))
            self._last_actual = actual_f

        self.history.append({
            'timestamp': timestamp,
            'y_pred': prediction, 'y_actual': actual_f,
            'y_lower': y_lower, 'y_upper': y_upper,
        })

    def settle_actual(
        self,
        *,
        target_ts: datetime,
        y_actual: float,
    ) -> bool:
        """
        异步真值到达时回填 / Settle the actual value for a target_ts.

        触发链:
        1. ``perf.fill_actual(target_ts, y_actual)`` — 按 target_ts 对齐回填
        2. ``store.update_actual(...)`` — 持久化真值
        3. 若回填成功且能取到对应 y_pred → 算残差喂 ``concept_drift``
        4. 更新 ``_last_actual`` (供 R2_SUDDEN_CHANGE 等规则用)

        Returns:
            True 若 perf 窗口里找到该 target_ts 并回填; False 已被淘汰。
            (即便返回 False, store 仍会尝试 update_actual, 因 SQLite 历史可能保留更长。)

        典型用法:
            # 4 月底 3 月真值出来了
            mon.settle_actual(target_ts=pd.Timestamp('2026-03-01'),
                              y_actual=178.5)
        """
        y_actual_f = float(y_actual)

        # 1) 取本窗口里的 y_pred, 算残差用 (走公共接口, 不再触私有 _records)
        rec = self.perf.get_record(target_ts)
        y_pred_for_residual: Optional[float] = (
            rec['y_pred'] if rec is not None else None
        )

        # 2) 回填 perf 窗口
        filled = self.perf.fill_actual(target_ts=target_ts, y_true=y_actual_f)

        # 3) 持久化回填 (store 历史可能更长, 即使 perf 已淘汰也尝试更新)
        try:
            self.store.update_actual(
                model_id=self.model_id, timestamp=target_ts,
                target=self.target_name, y_actual=y_actual_f,
            )
        except Exception as exc:
            self._storage_healthy = False
            self._last_storage_error = (
                f'update_actual failed: {type(exc).__name__}: {exc}'
            )
            self._logger.error(
                'ModelMonitor[%s] store.update_actual failed: %s',
                self.model_id, exc, exc_info=True,
            )

        # 4) 残差喂 concept_drift
        if y_pred_for_residual is not None:
            resid = y_actual_f - y_pred_for_residual
            self.concept_drift.update(np.array([resid]))
            self._last_actual = y_actual_f

        # 5) history 留底 (便于审计)
        self.history.append({
            'timestamp': target_ts,
            'event': 'settle_actual',
            'y_actual': y_actual_f,
            'filled_in_perf': filled,
        })
        return filled

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
        # 多步 horizon 场景下, perf 窗口里最新的 target_ts 即"下一步预测",
        # 等价于 h=1 视角 — 这是规则检查 (尤其 R2_SUDDEN_CHANGE) 应该看的。
        # history[-1] 不可靠 (可能是 settle_actual 事件或 h=12 的预测)。
        rule_violations: List[RuleViolation] = []
        if self.rule_engine is not None:
            # 走 PerformanceMonitor 公共接口取最新一条 (h=1 视角); 不触私有字段
            latest_rec = self.perf.get_latest_record()
            last_pred: Optional[float] = (
                latest_rec['y_pred'] if latest_rec is not None else None
            )
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

        # 写快照 (window 字段记录的是"指标计算窗口", 而非"队列容量")
        try:
            self.store.insert_metrics_snapshot(
                model_id=self.model_id, timestamp=now,
                metrics={k: v for k, v in metrics.items()
                         if isinstance(v, (int, float))
                         and not (isinstance(v, float) and np.isnan(v))},
                window=self.perf.metric_window,
            )
        except Exception as exc:
            self._storage_healthy = False
            self._last_storage_error = (
                f'insert_metrics_snapshot failed: {type(exc).__name__}: {exc}'
            )
            self._logger.error(
                'ModelMonitor[%s] store.insert_metrics_snapshot failed: %s',
                self.model_id, exc, exc_info=True,
            )

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
                    'errored_rules': list(decision.errored_rules),
                },
                'storage_healthy': self._storage_healthy,
                'last_storage_error': self._last_storage_error,
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
