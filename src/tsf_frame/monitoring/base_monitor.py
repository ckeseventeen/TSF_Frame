"""
监控器基类 / Base monitor abstract class
=========================================

``BaseMonitor`` 是所有"协调器"类型监控器 (ModelMonitor /
PipelineMonitor / 自定义业务监控) 的共同父类。它规定了一个
**最小、但足够表达所有编排语义** 的接口:

* ``record(...)``     — 接收一条观测/事件, 更新内部状态
* ``check_status()``  — 当前状态的结构化快照 (MonitoringStatus)
* ``reset()``         — 清空内部状态

子类可以在此基础上新增领域方法 (如 ``record_prediction``,
``record_stage``), 但 ``record/check_status/reset`` 的语义
必须保持。

与 ``interfaces.py`` 的分工:
* ``interfaces.py``: 契约 (ABC 与 dataclass)
* 本模块          : 通用协调器骨架 + 历史/时间戳/模型 ID 等公共字段
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from .interfaces import MonitoringStatus

__all__ = ['BaseMonitor', 'MonitoringStatus']


class BaseMonitor(ABC):
    """
    监控协调器基类 / Base orchestrator for any monitoring facade.

    Args:
        model_id: 本监控实例所属的模型/管道标识
        history_size: ``history`` 双端队列最多保留多少条 (防内存膨胀)

    Attributes:
        model_id: 标识
        created_at: 监控器实例创建时间
        history: 最近若干条 ``record()`` 的回放缓冲 (deque,  O(1) 入队/
            弹出最老)
    """

    def __init__(self, model_id: str, history_size: int = 10_000):
        if not model_id:
            raise ValueError('model_id must be non-empty')
        self.model_id: str = model_id
        self.created_at: datetime = datetime.now()
        self.history: Deque[Dict[str, Any]] = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # 必需接口 / Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def record(self, event: Dict[str, Any]) -> None:
        """
        收到一条事件 / Ingest one event.

        子类自己定义 ``event`` 的结构, 但推荐最少包含
        ``{'timestamp': datetime, ...}``。
        """

    @abstractmethod
    def check_status(self) -> MonitoringStatus:
        """
        生成当前状态快照 / Produce a MonitoringStatus snapshot.

        调用方可按需要序列化 / 输出 / 告警。
        """

    @abstractmethod
    def reset(self) -> None:
        """重置全部内部状态 / Reset all internal state."""

    # ------------------------------------------------------------------
    # 通用辅助 / Shared helpers
    # ------------------------------------------------------------------

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """取最近 n 条事件 / Latest n events for inspection."""
        n = max(0, int(n))
        return list(self.history)[-n:] if n else []

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f'{type(self).__name__}(model_id={self.model_id!r}, '
            f'len(history)={len(self.history)})'
        )


# ==========================================================================
# main — Echo 子类演示 / Minimal subclass demo
# ==========================================================================

def main() -> None:
    """
    Demo: 写一个最小 Echo 子类, 演示 BaseMonitor 的三方法。
    """
    from .interfaces import AlertLevel

    class EchoMonitor(BaseMonitor):
        """只记录事件计数的最小监控 / minimal counter monitor."""

        def record(self, event: Dict[str, Any]) -> None:
            event = {'timestamp': datetime.now(), **event}
            self.history.append(event)

        def check_status(self) -> MonitoringStatus:
            return MonitoringStatus(
                model_id=self.model_id,
                timestamp=datetime.now(),
                alert_level=AlertLevel.INFO,
                extra={'total_events': len(self.history)},
            )

        def reset(self) -> None:
            self.history.clear()

    print('=' * 60)
    print(' base_monitor — EchoMonitor demo')
    print('=' * 60)
    mon = EchoMonitor('demo-model')
    for i in range(5):
        mon.record({'i': i, 'payload': f'event-{i}'})
    print('Recent events:', len(mon.recent(3)))
    status = mon.check_status()
    print(
        f'Status: model_id={status.model_id}  level={status.alert_level}  '
        f'extra={status.extra}'
    )
    mon.reset()
    print(f'After reset: history size = {len(mon.history)}')


if __name__ == '__main__':
    main()
