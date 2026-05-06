"""
告警管理与通道 / Alert manager & channels
==========================================

``AlertManager`` 是告警流水账 + 分发器:

* ``add_channel(ch)``    — 叠加一个 ``AlertChannel`` 实例
* ``emit(level, msg)``   — 产生并广播一条告警
* ``info/warning/error/critical`` — 便捷 shortcut
* ``get_alerts(...)``    — 查询本地缓存的告警

内置四个 ``AlertChannel`` 实现:

=====================  ================================================
ConsoleChannel         直接 print 到 stderr, 纯文本
LoggingChannel         走 stdlib ``logging``, 映射 level
FileChannel            追加到指定文件的 "一行一条" 纯文本日志
CallbackChannel        把任意 ``callable`` 包装成通道 (便于测试/粘合)
=====================  ================================================

典型组装::

    mgr = AlertManager(model_id='m1')
    mgr.add_channel(ConsoleChannel())
    mgr.add_channel(FileChannel('./logs/alerts.log',
                                min_level=AlertLevel.WARNING))
    mgr.warning('MAPE 升高', details={'mape': 0.12})

任何自定义通道只需 ``class X(AlertChannel): def send(alert): ...``
再 ``mgr.add_channel(X())`` 即可——这就是"插拔"。
"""

from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .interfaces import (
    Alert,
    AlertChannel,
    AlertLevel,
    LEVEL_ORDER,
    MetricStore,
    level_ge,
    register_alert_channel,
)

__all__ = [
    'AlertManager',
    'ConsoleChannel',
    'LoggingChannel',
    'FileChannel',
    'CallbackChannel',
    'StoreChannel',
]


# ==========================================================================
# AlertManager
# ==========================================================================

class AlertManager:
    """
    告警产生 + 分发中心 / Alert producer and fan-out.

    * ``emit()`` 生成唯一微秒级 ``alert_id``, 防重复。
    * 同时落本地缓冲 (``_alerts``) 与所有通道; 通道异常不会中断主流程。
    * 可选 ``store``: 如传入, 所有告警同时持久化。

    Args:
        model_id:       模型/管道标识
        store:          可选持久化后端 (MetricStore 任何实现)
        min_level:      丢弃低于此级别的告警 (默认 INFO, 即全部接受)
    """

    def __init__(
        self,
        model_id: str,
        *,
        store: Optional[MetricStore] = None,
        min_level: str = AlertLevel.INFO,
    ):
        # 告警归属的模型/管道标识
        # / Owning model identifier
        self.model_id = model_id
        # 可选持久化后端; 设置后所有 emit 都会同步落库
        # / Optional persistence backend
        self.store = store
        # 全局最低级别过滤; 低于此级别的 emit 直接丢弃
        # / Global minimum level — anything below is dropped at emit
        self.min_level = min_level
        # 已挂载的告警通道列表 (Console / Logging / File / Callback / Store)
        # / Mounted output channels (fan-out targets)
        self._channels: List[AlertChannel] = []
        # 本地告警缓冲 (供 get_alerts 查询; 持久化交给 store 通道)
        # / In-memory buffer for local query
        self._alerts: List[Alert] = []
        # 单实例内的告警序号, 与微秒时间戳一起组成 alert_id 防重
        # / Per-instance counter ensuring unique alert_id under same timestamp
        self._counter = 0
        # 互斥锁, 保护 _counter 自增和 _alerts append
        # / Mutex for thread-safe emit
        self._lock = threading.Lock()

    # ---- 通道管理 -----------------------------------------------------
    def add_channel(self, channel: AlertChannel) -> 'AlertManager':
        """追加一个告警通道 / Append one channel (returns self for chain)."""
        self._channels.append(channel)
        return self

    def add_callback(
        self, fn: Callable[[Alert], None],
        min_level: str = AlertLevel.INFO,
    ) -> 'AlertManager':
        """向后兼容: 把 callable 包成 CallbackChannel 挂上。"""
        return self.add_channel(CallbackChannel(fn, min_level=min_level))

    @property
    def channels(self) -> List[AlertChannel]:
        return list(self._channels)

    # ---- 产生告警 -----------------------------------------------------
    def emit(
        self,
        level: str,
        message: str,
        *,
        source: str = '',
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Alert]:
        """
        产生并广播一条告警 / Produce and broadcast.

        若 ``level`` < ``min_level``, 直接丢弃返回 None。
        """
        if not level_ge(level, self.min_level):
            return None
        ts = timestamp or datetime.now()
        with self._lock:
            self._counter += 1
            alert_id = (f'{self.model_id}_'
                        f'{ts.strftime("%Y%m%dT%H%M%S%f")}_'
                        f'{self._counter}')
        alert = Alert(
            alert_id=alert_id,
            model_id=self.model_id,
            level=level,
            message=message,
            timestamp=ts,
            source=source,
            details=dict(details or {}),
        )
        self._alerts.append(alert)

        # 持久化 (若配置)
        if self.store is not None:
            try:
                self.store.insert_alert(alert)
            except Exception as exc:  # pragma: no cover
                print(f'[AlertManager] store.insert_alert failed: {exc}',
                      file=sys.stderr)

        # 广播通道 (任何一个异常都不影响其他)
        for ch in self._channels:
            try:
                ch.send(alert)
            except Exception as exc:  # pragma: no cover
                print(f'[AlertManager] channel {type(ch).__name__} '
                      f'failed: {exc}', file=sys.stderr)
        return alert

    # ---- 便捷 shortcut ------------------------------------------------
    def info(self, message: str, **kw) -> Optional[Alert]:
        return self.emit(AlertLevel.INFO, message, **kw)

    def warning(self, message: str, **kw) -> Optional[Alert]:
        return self.emit(AlertLevel.WARNING, message, **kw)

    def error(self, message: str, **kw) -> Optional[Alert]:
        return self.emit(AlertLevel.ERROR, message, **kw)

    def critical(self, message: str, **kw) -> Optional[Alert]:
        return self.emit(AlertLevel.CRITICAL, message, **kw)

    # ---- 查询 ---------------------------------------------------------
    def get_alerts(
        self,
        *,
        level: Optional[str] = None,
        min_level: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Alert]:
        """查询本地缓冲告警 (非 store) / Query in-memory buffer."""
        min_rank = LEVEL_ORDER.get(min_level, 0) if min_level else 0
        out: List[Alert] = []
        for a in self._alerts:
            if level is not None and a.level != level:
                continue
            if LEVEL_ORDER.get(a.level, 0) < min_rank:
                continue
            if start is not None and a.timestamp < start:
                continue
            if end is not None and a.timestamp > end:
                continue
            out.append(a)
        return out

    def clear(self) -> None:
        """清空本地缓冲 / Clear in-memory buffer (store untouched)."""
        self._alerts.clear()


# ==========================================================================
# 内置通道 / Built-in channels
# ==========================================================================

class _LevelFilterMixin:
    """所有通道共享的 min_level 过滤逻辑。"""

    min_level: str = AlertLevel.INFO

    def _should_send(self, alert: Alert) -> bool:
        return level_ge(alert.level, self.min_level)


@register_alert_channel('console')
class ConsoleChannel(AlertChannel, _LevelFilterMixin):
    """
    控制台通道 / Print to stderr.

    适合开发/命令行演示。
    """

    def __init__(self, min_level: str = AlertLevel.INFO,
                 stream=None) -> None:
        # 通道独立级别阈值 (与 AlertManager.min_level 取严)
        # / Per-channel min level (stricter than manager's wins)
        self.min_level = min_level
        # 输出流 (默认 stderr, 可替换为任意 file-like)
        # / Output stream — defaults to sys.stderr
        self._stream = stream or sys.stderr

    def send(self, alert: Alert) -> None:
        if not self._should_send(alert):
            return
        ts = alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        prefix = alert.level.upper().ljust(8)
        line = f'[{ts}] {prefix} {alert.model_id}: {alert.message}'
        print(line, file=self._stream)


@register_alert_channel('logging')
class LoggingChannel(AlertChannel, _LevelFilterMixin):
    """
    标准 logging 通道 / stdlib logging adapter.

    level 映射:
        INFO     → logger.info
        WARNING  → logger.warning
        ERROR    → logger.error
        CRITICAL → logger.critical
    """

    _LEVEL_MAP = {
        AlertLevel.INFO: logging.INFO,
        AlertLevel.WARNING: logging.WARNING,
        AlertLevel.ERROR: logging.ERROR,
        AlertLevel.CRITICAL: logging.CRITICAL,
    }

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        *,
        name: str = 'tsf_frame.monitoring',
        min_level: str = AlertLevel.INFO,
    ):
        # 通道级别阈值
        # / Channel-level threshold
        self.min_level = min_level
        # 标准 logging.Logger; 默认按 name 取一个全局 logger
        # / Underlying stdlib logger (use given or look up by name)
        self.logger = logger or logging.getLogger(name)

    def send(self, alert: Alert) -> None:
        if not self._should_send(alert):
            return
        lvl = self._LEVEL_MAP.get(alert.level, logging.INFO)
        self.logger.log(
            lvl, '%s [%s] %s',
            alert.model_id, alert.source or '-', alert.message,
        )


@register_alert_channel('file')
class FileChannel(AlertChannel, _LevelFilterMixin):
    """
    追加式文件通道 / Plain-text file channel.

    每行一条: ``<ts>\\t<level>\\t<model_id>\\t<source>\\t<message>``
    """

    def __init__(
        self,
        path: str,
        *,
        min_level: str = AlertLevel.WARNING,
        encoding: str = 'utf-8',
    ):
        # 输出文件路径; 父目录会自动创建
        # / Output file path; parent dir auto-created
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 通道级别阈值, 默认 WARNING (文件通常只记重要告警)
        # / Channel-level threshold; default WARNING for noise reduction
        self.min_level = min_level
        # 文件编码 (Windows 中文环境下显式 utf-8 防乱码)
        # / Text encoding (utf-8 by default for non-ASCII safety)
        self.encoding = encoding
        # 文件写入锁, 保证多线程下不交错
        # / Mutex for thread-safe file append
        self._lock = threading.Lock()

    def send(self, alert: Alert) -> None:
        if not self._should_send(alert):
            return
        ts = alert.timestamp.isoformat(timespec='seconds')
        line = (f'{ts}\t{alert.level}\t{alert.model_id}\t'
                f'{alert.source or "-"}\t{alert.message}\n')
        with self._lock:
            with self.path.open('a', encoding=self.encoding) as f:
                f.write(line)


@register_alert_channel('callback')
class CallbackChannel(AlertChannel, _LevelFilterMixin):
    """
    回调通道 / Wrap any ``Callable[[Alert], None]`` as a channel.

    用途: 测试注入 / 粘合其他框架 (钉钉/企业微信/Slack)。
    """

    def __init__(
        self,
        fn: Callable[[Alert], None],
        *,
        min_level: str = AlertLevel.INFO,
    ):
        # 通道级别阈值
        # / Channel-level threshold
        self.min_level = min_level
        # 实际执行的回调函数, 接收 Alert 对象
        # / The wrapped callable
        self._fn = fn

    def send(self, alert: Alert) -> None:
        if not self._should_send(alert):
            return
        self._fn(alert)


@register_alert_channel('store')
class StoreChannel(AlertChannel, _LevelFilterMixin):
    """
    把告警写入 MetricStore 的通道 / Persist alert via MetricStore.

    与 AlertManager(store=...) 等价, 但可显式叠加多个 store (例如同时
    写 SQLite 和 JSONL)。
    """

    def __init__(self, store: MetricStore,
                 *, min_level: str = AlertLevel.INFO):
        # 通道级别阈值
        # / Channel-level threshold
        self.min_level = min_level
        # 写入目标 MetricStore (可叠加多个 store, e.g. 同时写 SQLite 和 JSONL)
        # / Target persistence backend
        self.store = store

    def send(self, alert: Alert) -> None:
        if not self._should_send(alert):
            return
        self.store.insert_alert(alert)


# ==========================================================================
# main — 组装演示
# ==========================================================================

def main() -> None:
    """演示 4 个通道一起工作 + AlertManager 级别过滤。"""
    import tempfile

    print('=' * 70)
    print(' alert_manager — channel composition demo')
    print('=' * 70)

    logging.basicConfig(level=logging.INFO,
                        format='[logging] %(levelname)s %(message)s')
    tmp_log = Path(tempfile.mkdtemp(prefix='tsf_alerts_')) / 'alerts.log'

    received: List[str] = []
    mgr = AlertManager(model_id='demo_mgr')
    (mgr
     .add_channel(ConsoleChannel(min_level=AlertLevel.INFO))
     .add_channel(LoggingChannel(min_level=AlertLevel.WARNING))
     .add_channel(FileChannel(str(tmp_log), min_level=AlertLevel.WARNING))
     .add_callback(lambda a: received.append(a.level)))

    mgr.info('启动')                       # Console + Callback
    mgr.warning('MAPE 偏高', details={'mape': 0.13})  # 全通道
    mgr.error('模型性能下降')
    mgr.critical('强烈建议重训',
                 source='retrain_trigger',
                 details={'rule': 'mape_hard'})

    print('\nCallback 收到级别序列:', received)
    print('本地缓冲条数:', len(mgr.get_alerts()))
    print('WARNING+ 条数:', len(mgr.get_alerts(min_level=AlertLevel.WARNING)))
    print('File 写入:', tmp_log.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
