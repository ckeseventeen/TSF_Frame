"""
管道监控 / Pipeline stage monitor
==================================

**框架层面的监控** — 跟踪一个时序预测流水线的各个阶段:

    load → preprocess → feature → split → train → predict → postprocess

每个阶段是一段代码, ``PipelineMonitor.stage(name)`` 返回一个上下文
管理器, 自动记录:

* 开始/结束时间
* 耗时 (毫秒)
* 异常 (含堆栈) → 自动升级告警
* 用户附加的 ``metadata`` (样本数/特征数/模型文件大小...)

使用::

    pm = PipelineMonitor(pipeline_id='hpf-daily-train')
    with pm.stage('load', n=144):
        df = load_data()
    with pm.stage('feature', cols=len(df.columns)):
        feat = build_features(df)
    ...

统计信息:
* ``pm.summary()`` — 各阶段耗时均值/P95/失败次数
* ``pm.events``    — 原始 StageEvent 列表
* ``pm.check_status()`` — 汇总为 MonitoringStatus, 便于和 ModelMonitor
                           共享同一 store / alert_manager

与 ModelMonitor 的分工:
* PipelineMonitor — "我的管道跑得怎么样" (framework concern)
* ModelMonitor   — "我的模型还准不准" (model concern)
"""

from __future__ import annotations

import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .alert_manager import AlertManager
from .base_monitor import BaseMonitor
from .interfaces import (
    AlertLevel,
    MetricStore,
    MonitoringStatus,
    StageEvent,
)

__all__ = ['PipelineMonitor']


class PipelineMonitor(BaseMonitor):
    """
    管道阶段监控 / Pipeline-stage monitoring facade.

    Args:
        pipeline_id: 管道标识 (同时作为 BaseMonitor.model_id)
        store:       可选 MetricStore, 指标/告警会写入
        alert_manager: 可选自定义 AlertManager, 默认自建
        slow_threshold_ms: 阶段耗时超过此值即 WARNING
        history_size: 最多保留的 StageEvent 数量
    """

    def __init__(
        self,
        pipeline_id: str,
        *,
        store: Optional[MetricStore] = None,
        alert_manager: Optional[AlertManager] = None,
        slow_threshold_ms: float = 10_000.0,
        history_size: int = 5000,
    ):
        super().__init__(pipeline_id, history_size=history_size)
        # 持久化后端 (stage 耗时和成功率会被写为 metrics_snapshot)
        # / Optional MetricStore for stage metrics
        self.store = store
        # 告警分发中心: 阶段失败 → ERROR, 超时 → WARNING
        # / Alert manager for stage failures / slowdowns
        self.alert_manager = alert_manager or AlertManager(
            pipeline_id, store=store)
        # 阶段耗时阈值 (毫秒), 超过即触发 WARNING 告警
        # / Per-stage duration threshold (ms) above which WARNING fires
        self.slow_threshold_ms = float(slow_threshold_ms)
        # 全部阶段事件的有序列表 (每次 stage() 上下文管理器结束都追加一条)
        # / Append-only log of all stage events ever recorded
        self.events: List[StageEvent] = []

    # ------------------------------------------------------------------
    # 阶段上下文管理器
    # ------------------------------------------------------------------
    @contextmanager
    def stage(self, name: str, **metadata: Any) -> Iterator[Dict[str, Any]]:
        """
        开启一个阶段 / Enter a stage context.

        ``yield`` 出的 dict 可在上下文内写入额外 metadata
        (执行完再一并记入 StageEvent).
        """
        started = datetime.now()
        start_ns = time.perf_counter_ns()
        ctx: Dict[str, Any] = dict(metadata)
        err: Optional[BaseException] = None
        try:
            yield ctx
        except BaseException as exc:  # noqa: BLE001
            err = exc
            raise
        finally:
            ended = datetime.now()
            dur_ms = (time.perf_counter_ns() - start_ns) / 1e6
            event = StageEvent(
                stage=name,
                started_at=started,
                ended_at=ended,
                duration_ms=dur_ms,
                success=(err is None),
                error=(
                    f'{type(err).__name__}: {err}\n'
                    + ''.join(traceback.format_exception(
                        type(err), err, err.__traceback__))
                    if err is not None else None
                ),
                metadata=ctx,
            )
            self._handle_event(event)

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------
    def _handle_event(self, event: StageEvent) -> None:
        self.events.append(event)
        self.history.append({
            'timestamp': event.ended_at,
            'stage': event.stage,
            'duration_ms': event.duration_ms,
            'success': event.success,
            'metadata': event.metadata,
        })

        # 写 store (按 metric 快照落库)
        if self.store is not None:
            try:
                self.store.insert_metrics_snapshot(
                    model_id=self.model_id,
                    timestamp=event.ended_at,
                    metrics={
                        f'stage.{event.stage}.duration_ms': event.duration_ms,
                        f'stage.{event.stage}.success':
                            1.0 if event.success else 0.0,
                    },
                )
            except Exception:  # pragma: no cover
                pass

        # 告警
        if not event.success:
            self.alert_manager.error(
                f'阶段 "{event.stage}" 执行失败',
                source=f'stage.{event.stage}',
                details={'duration_ms': event.duration_ms,
                         'error': event.error,
                         'metadata': event.metadata},
            )
        elif event.duration_ms > self.slow_threshold_ms:
            self.alert_manager.warning(
                f'阶段 "{event.stage}" 耗时 {event.duration_ms:.0f}ms '
                f'超阈值 {self.slow_threshold_ms:.0f}ms',
                source=f'stage.{event.stage}',
                details={'duration_ms': event.duration_ms,
                         'metadata': event.metadata},
            )

    # ------------------------------------------------------------------
    # 必需的 BaseMonitor 接口
    # ------------------------------------------------------------------
    def record(self, event: Dict[str, Any]) -> None:
        """
        外部直接注入事件 / Manual event injection.

        适用于阶段并非由 ``stage()`` 包裹的场景 (如已在别处计时)。
        要求 event 含 {'stage', 'duration_ms', 'success'} 最少三字段。
        """
        stage_event = StageEvent(
            stage=str(event.get('stage', 'unknown')),
            started_at=event.get('started_at', datetime.now()),
            ended_at=event.get('ended_at', datetime.now()),
            duration_ms=float(event.get('duration_ms', 0.0)),
            success=bool(event.get('success', True)),
            error=event.get('error'),
            metadata=dict(event.get('metadata', {})),
        )
        self._handle_event(stage_event)

    def check_status(self) -> MonitoringStatus:
        """聚合为 MonitoringStatus 快照。"""
        summary = self.summary()
        has_failure = any(not e.success for e in self.events)
        has_slow = any(e.duration_ms > self.slow_threshold_ms
                       for e in self.events if e.success)
        level = (AlertLevel.ERROR if has_failure
                 else AlertLevel.WARNING if has_slow
                 else AlertLevel.INFO)
        return MonitoringStatus(
            model_id=self.model_id,
            timestamp=datetime.now(),
            alert_level=level,
            extra={'pipeline_summary': summary,
                   'total_events': len(self.events)},
            recommendations=[
                '检查失败阶段日志' if has_failure else '',
                '考虑优化耗时最长的阶段' if has_slow else '',
            ],
        )

    def reset(self) -> None:
        self.events.clear()
        self.history.clear()
        self.alert_manager.clear()

    # ------------------------------------------------------------------
    # 汇总统计
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Dict[str, Any]]:
        """
        按阶段汇总 / Per-stage statistics.

        Returns:
            {stage: {n, n_failed, mean_ms, p95_ms, max_ms, total_ms}}
        """
        by_stage: Dict[str, List[StageEvent]] = defaultdict(list)
        for e in self.events:
            by_stage[e.stage].append(e)

        out: Dict[str, Dict[str, Any]] = {}
        for stage, evs in by_stage.items():
            durs = np.array([e.duration_ms for e in evs], dtype=float)
            out[stage] = {
                'n': len(evs),
                'n_failed': sum(1 for e in evs if not e.success),
                'mean_ms': float(durs.mean()) if durs.size else 0.0,
                'p95_ms': (float(np.percentile(durs, 95))
                           if durs.size else 0.0),
                'max_ms': float(durs.max()) if durs.size else 0.0,
                'total_ms': float(durs.sum()),
            }
        return out


# ==========================================================================
# main — 演示
# ==========================================================================

def main() -> None:
    """人工跑 5 个阶段, 故意让一个慢、一个失败, 看告警和 summary。"""
    from .alert_manager import ConsoleChannel
    from .stores import InMemoryStore

    print('=' * 70)
    print(' pipeline_monitor — staged demo')
    print('=' * 70)

    store = InMemoryStore()
    pm = PipelineMonitor('demo-pipeline', store=store,
                         slow_threshold_ms=200.0)
    pm.alert_manager.add_channel(ConsoleChannel())

    with pm.stage('load', n_rows=144) as ctx:
        time.sleep(0.05)
        ctx['bytes'] = 1024

    with pm.stage('preprocess') as ctx:
        time.sleep(0.03)

    with pm.stage('feature_engineering') as ctx:
        time.sleep(0.25)     # 故意超阈值
        ctx['features'] = 23

    try:
        with pm.stage('train'):
            time.sleep(0.02)
            raise RuntimeError('模拟训练失败 (OOM)')
    except RuntimeError:
        pass

    with pm.stage('predict'):
        time.sleep(0.04)

    print('\n== summary ==')
    for stg, stat in pm.summary().items():
        print(f'  {stg:<22} n={stat["n"]}  '
              f'mean={stat["mean_ms"]:.1f}ms  '
              f'p95={stat["p95_ms"]:.1f}ms  '
              f'failed={stat["n_failed"]}')

    status = pm.check_status()
    print(f'\nstatus.alert_level = {status.alert_level}')
    print(f'store has {len(store.query_metrics(model_id="demo-pipeline"))} '
          f'metric snapshots')


if __name__ == '__main__':
    main()
