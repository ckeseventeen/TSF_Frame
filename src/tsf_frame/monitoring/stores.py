"""
持久化后端 / Persistence backends
==================================

本模块提供 ``MetricStore`` 的三种开箱即用实现:

====================  ==========================  ==========================
实现                  场景                        依赖
====================  ==========================  ==========================
InMemoryStore         单进程, 测试, 原型          纯内存, 无任何依赖
SQLiteStore           单机落盘, 小规模生产        sqlite3 (stdlib)
JsonlStore            追加式日志, 便于 grep/导出  仅标准库 json
====================  ==========================  ==========================

三者接口完全一致, 可自由替换::

    from tsf_frame.monitoring import create_store
    store = create_store('sqlite', db_path='./logs/mon.db')
    # store = create_store('memory')
    # store = create_store('jsonl', base_dir='./logs/jsonl')

SQLite schema:

* ``predictions`` — 每条预测一条记录, 可后填真实值
* ``metrics_snapshot`` — 每次 check_status 产出的 metric_name → value
* ``alerts`` — 告警流水, 含 level / source / details (JSON)

线程安全:
* SQLiteStore 使用 WAL 模式 + 每次操作开新连接, 可在单进程多线程下
  并发读写。
* InMemoryStore 使用 ``threading.Lock`` 保护列表操作。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .interfaces import (
    Alert,
    AlertLevel,
    LEVEL_ORDER,
    MetricStore,
    register_store,
)

__all__ = ['InMemoryStore', 'SQLiteStore', 'JsonlStore']


def _iso(ts: datetime) -> str:
    """datetime → ISO8601 字符串, 精度到微秒, 便于 SQLite ORDER BY。"""
    return ts.isoformat(timespec='microseconds')


def _parse_iso(s: str) -> datetime:
    """ISO8601 → datetime, 兼容多种精度。"""
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # 兜底: 去掉时区部分重试
        return datetime.fromisoformat(s.split('+')[0].split('Z')[0])


# ==========================================================================
# InMemoryStore — 测试 / 默认
# ==========================================================================

@register_store('memory')
class InMemoryStore(MetricStore):
    """
    纯内存实现 / Pure in-memory store.

    * 进程退出即丢失, 仅用于测试、原型、单次脚本。
    * 接口与 SQLiteStore 完全一致, 切换零成本。
    """

    def __init__(self) -> None:
        # 预测记录列表 (一条 = insert_prediction 一次)
        # / In-memory predictions table
        self._predictions: List[Dict[str, Any]] = []
        # 索引: (model_id, target, timestamp) → 该记录在 _predictions 中的位置列表
        # 用于把 update_actual 从 O(N) 全表扫降到 O(k), k = 同 key 记录数(通常 1).
        # / Index for O(1) update_actual lookup
        self._pred_index: Dict[tuple, List[int]] = {}
        # 指标快照列表 (扁平化为 {model_id, ts, name, value, window})
        # / Flattened metrics snapshot rows
        self._metrics: List[Dict[str, Any]] = []
        # 告警对象列表
        # / Alert records
        self._alerts: List[Alert] = []
        # 互斥锁, 保护多线程并发写
        # / Mutex for thread-safe writes
        self._lock = threading.Lock()

    # ---- writes ----
    def insert_prediction(
        self, *, model_id: str, timestamp: datetime, target: str,
        y_pred: float, y_lower: Optional[float] = None,
        y_upper: Optional[float] = None, y_actual: Optional[float] = None,
    ) -> None:
        with self._lock:
            row = {
                'model_id': model_id, 'timestamp': timestamp,
                'target': target, 'y_pred': float(y_pred),
                'y_lower': None if y_lower is None else float(y_lower),
                'y_upper': None if y_upper is None else float(y_upper),
                'y_actual': None if y_actual is None else float(y_actual),
            }
            idx = len(self._predictions)
            self._predictions.append(row)
            # 维护索引: 同 key 可能多条记录 (e.g. 同 target_ts 不同 horizon),
            # 所以用 list 而非单值
            key = (model_id, target, timestamp)
            self._pred_index.setdefault(key, []).append(idx)

    def update_actual(
        self, *, model_id: str, timestamp: datetime, target: str,
        y_actual: float,
    ) -> None:
        # O(k) 索引查找替代 O(N) 全表扫描; k 通常为 1 或 horizon 数 (≤12)
        # / O(k) indexed lookup vs O(N) scan
        with self._lock:
            indices = self._pred_index.get((model_id, target, timestamp), [])
            for idx in indices:
                self._predictions[idx]['y_actual'] = float(y_actual)

    def insert_metrics_snapshot(
        self, *, model_id: str, timestamp: datetime,
        metrics: Mapping[str, float], window: Optional[int] = None,
    ) -> None:
        with self._lock:
            for name, value in metrics.items():
                self._metrics.append({
                    'model_id': model_id, 'timestamp': timestamp,
                    'name': name, 'value': float(value),
                    'window': window,
                })

    def insert_alert(self, alert: Alert) -> None:
        with self._lock:
            self._alerts.append(alert)

    # ---- reads ----
    def query_predictions(
        self, *, model_id: str, start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                dict(r) for r in self._predictions
                if r['model_id'] == model_id
                and (start is None or r['timestamp'] >= start)
                and (end is None or r['timestamp'] <= end)
            ]

    def query_metrics(
        self, *, model_id: str, start: Optional[datetime] = None,
        end: Optional[datetime] = None, name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                dict(r) for r in self._metrics
                if r['model_id'] == model_id
                and (name is None or r['name'] == name)
                and (start is None or r['timestamp'] >= start)
                and (end is None or r['timestamp'] <= end)
            ]

    def query_alerts(
        self, *, model_id: str, start: Optional[datetime] = None,
        end: Optional[datetime] = None, min_level: Optional[str] = None,
    ) -> List[Alert]:
        min_rank = LEVEL_ORDER.get(min_level, 0) if min_level else 0
        with self._lock:
            return [
                a for a in self._alerts
                if a.model_id == model_id
                and LEVEL_ORDER.get(a.level, 0) >= min_rank
                and (start is None or a.timestamp >= start)
                and (end is None or a.timestamp <= end)
            ]

    def cleanup_old(self, *, retain_days: int) -> int:
        cutoff = datetime.now().timestamp() - retain_days * 86400
        removed = 0
        with self._lock:
            before = (len(self._predictions) + len(self._metrics)
                      + len(self._alerts))
            self._predictions = [
                r for r in self._predictions
                if r['timestamp'].timestamp() >= cutoff
            ]
            self._metrics = [
                r for r in self._metrics
                if r['timestamp'].timestamp() >= cutoff
            ]
            self._alerts = [
                a for a in self._alerts
                if a.timestamp.timestamp() >= cutoff
            ]
            removed = before - (len(self._predictions) + len(self._metrics)
                                + len(self._alerts))
        return removed


# ==========================================================================
# SQLiteStore — 单机生产
# ==========================================================================

@register_store('sqlite')
class SQLiteStore(MetricStore):
    """
    SQLite 持久化 / SQLite-backed store.

    默认启用:
    * ``journal_mode=WAL`` — 读写并发性更好
    * ``synchronous=NORMAL`` — 性能与安全的折中
    * 每个方法自开一个 short-lived 连接, 避免跨线程复用同一连接的坑

    Schema 见模块顶部。
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS predictions (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      model_id    TEXT    NOT NULL,
      ts          TEXT    NOT NULL,
      target      TEXT    NOT NULL,
      y_pred      REAL    NOT NULL,
      y_lower     REAL,
      y_upper     REAL,
      y_actual    REAL,
      created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_pred_model_ts
        ON predictions(model_id, ts);

    CREATE TABLE IF NOT EXISTS metrics_snapshot (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      model_id     TEXT    NOT NULL,
      ts           TEXT    NOT NULL,
      name         TEXT    NOT NULL,
      value        REAL,
      window_size  INTEGER,
      created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_metric_model_ts_name
        ON metrics_snapshot(model_id, ts, name);

    CREATE TABLE IF NOT EXISTS alerts (
      alert_id      TEXT PRIMARY KEY,
      model_id      TEXT NOT NULL,
      ts            TEXT NOT NULL,
      level         TEXT NOT NULL,
      source        TEXT,
      message       TEXT NOT NULL,
      details       TEXT,
      acknowledged  INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_alert_model_ts_level
        ON alerts(model_id, ts, level);
    """

    def __init__(self, db_path: str = './logs/monitor.db') -> None:
        # SQLite 数据库文件绝对路径, 父目录会自动创建
        # / Absolute path to SQLite db file
        self.db_path = os.path.abspath(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # 启动时建表 + 索引 (CREATE IF NOT EXISTS, 幂等)
        self._init_schema()

    # ---- low-level helpers ---------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """开启一个短生命期连接, 由调用方负责 close。"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(self.SCHEMA_SQL)

    # ---- writes --------------------------------------------------------
    def insert_prediction(
        self, *, model_id: str, timestamp: datetime, target: str,
        y_pred: float, y_lower: Optional[float] = None,
        y_upper: Optional[float] = None, y_actual: Optional[float] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO predictions '
                '(model_id, ts, target, y_pred, y_lower, y_upper, y_actual) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)',
                (model_id, _iso(timestamp), target, float(y_pred),
                 None if y_lower is None else float(y_lower),
                 None if y_upper is None else float(y_upper),
                 None if y_actual is None else float(y_actual)),
            )

    def update_actual(
        self, *, model_id: str, timestamp: datetime, target: str,
        y_actual: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                'UPDATE predictions SET y_actual=? '
                'WHERE model_id=? AND ts=? AND target=?',
                (float(y_actual), model_id, _iso(timestamp), target),
            )

    def insert_metrics_snapshot(
        self, *, model_id: str, timestamp: datetime,
        metrics: Mapping[str, float], window: Optional[int] = None,
    ) -> None:
        """
        批量写入指标快照 / Batch insert metric rows.

        显式事务边界: ``with conn:`` 上下文进入即开启 implicit transaction,
        正常退出 commit, 异常时 rollback (sqlite3 标准行为). 这样多个
        ``executemany`` 写入要么全成功要么全回滚, 防止断电/崩溃中途留下
        部分数据.
        / Explicit transaction: with-block commits on success, rolls back on error.
        """
        rows = [
            (model_id, _iso(timestamp), name, float(value), window)
            for name, value in metrics.items()
        ]
        with self._connect() as conn:
            # with conn 块即事务边界; 正常出 commit, 异常自动 rollback
            conn.executemany(
                'INSERT INTO metrics_snapshot '
                '(model_id, ts, name, value, window_size) '
                'VALUES (?, ?, ?, ?, ?)', rows,
            )

    def insert_alert(self, alert: Alert) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT OR IGNORE INTO alerts '
                '(alert_id, model_id, ts, level, source, message, details, '
                ' acknowledged) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (alert.alert_id, alert.model_id, _iso(alert.timestamp),
                 alert.level, alert.source, alert.message,
                 json.dumps(alert.details, ensure_ascii=False, default=str),
                 int(alert.acknowledged)),
            )

    # ---- reads ---------------------------------------------------------
    def _time_where(
        self,
        start: Optional[datetime], end: Optional[datetime],
    ) -> tuple:
        clauses, params = [], []
        if start is not None:
            clauses.append('ts >= ?'); params.append(_iso(start))
        if end is not None:
            clauses.append('ts <= ?'); params.append(_iso(end))
        return clauses, params

    def query_predictions(
        self, *, model_id: str, start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        clauses, params = self._time_where(start, end)
        where = 'model_id=?' + (' AND ' + ' AND '.join(clauses) if clauses
                                else '')
        with self._connect() as conn:
            cur = conn.execute(
                f'SELECT * FROM predictions WHERE {where} ORDER BY ts',
                [model_id, *params],
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    'model_id': r['model_id'],
                    'timestamp': _parse_iso(r['ts']),
                    'target': r['target'],
                    'y_pred': r['y_pred'],
                    'y_lower': r['y_lower'],
                    'y_upper': r['y_upper'],
                    'y_actual': r['y_actual'],
                })
            return rows

    def query_metrics(
        self, *, model_id: str, start: Optional[datetime] = None,
        end: Optional[datetime] = None, name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses, params = self._time_where(start, end)
        if name is not None:
            clauses.append('name=?'); params.append(name)
        where = 'model_id=?' + (' AND ' + ' AND '.join(clauses) if clauses
                                else '')
        with self._connect() as conn:
            cur = conn.execute(
                f'SELECT * FROM metrics_snapshot WHERE {where} '
                f'ORDER BY ts, name',
                [model_id, *params],
            )
            return [
                {
                    'model_id': r['model_id'],
                    'timestamp': _parse_iso(r['ts']),
                    'name': r['name'],
                    'value': r['value'],
                    'window': r['window_size'],
                }
                for r in cur.fetchall()
            ]

    def query_alerts(
        self, *, model_id: str, start: Optional[datetime] = None,
        end: Optional[datetime] = None, min_level: Optional[str] = None,
    ) -> List[Alert]:
        clauses, params = self._time_where(start, end)
        where = 'model_id=?' + (' AND ' + ' AND '.join(clauses) if clauses
                                else '')
        with self._connect() as conn:
            cur = conn.execute(
                f'SELECT * FROM alerts WHERE {where} ORDER BY ts',
                [model_id, *params],
            )
            min_rank = LEVEL_ORDER.get(min_level, 0) if min_level else 0
            out: List[Alert] = []
            for r in cur.fetchall():
                if LEVEL_ORDER.get(r['level'], 0) < min_rank:
                    continue
                try:
                    details = json.loads(r['details']) if r['details'] else {}
                except Exception:
                    details = {}
                out.append(Alert(
                    alert_id=r['alert_id'],
                    model_id=r['model_id'],
                    level=r['level'],
                    message=r['message'],
                    timestamp=_parse_iso(r['ts']),
                    source=r['source'] or '',
                    details=details,
                    acknowledged=bool(r['acknowledged']),
                ))
            return out

    def cleanup_old(self, *, retain_days: int) -> int:
        # 单次取 now, 避免毫秒级两次调用产生不一致;
        # 用 datetime.fromtimestamp 类方法 (不是 instance.fromtimestamp, 那是反模式).
        # / Single now() call; use classmethod fromtimestamp.
        ts = datetime.now().timestamp() - retain_days * 86400
        cutoff = _iso(datetime.fromtimestamp(ts))
        total = 0
        with self._connect() as conn:
            for tbl in ('predictions', 'metrics_snapshot', 'alerts'):
                cur = conn.execute(f'DELETE FROM {tbl} WHERE ts < ?',
                                   (cutoff,))
                total += cur.rowcount or 0
        return total


# ==========================================================================
# JsonlStore — 行 JSON 日志
# ==========================================================================

@register_store('jsonl')
class JsonlStore(MetricStore):
    """
    JSONL 追加式日志 / Append-only JSONL files.

    目录布局::

        base_dir/
          predictions.jsonl
          metrics.jsonl
          alerts.jsonl

    优点: 人类可读 / 便于 ``grep`` / 可无损导入其他系统。
    缺点: 查询需全量扫描, 仅适合 < 百万级总量或离线分析。
    """

    def __init__(self, base_dir: str = './logs/jsonl') -> None:
        # JSONL 文件根目录 (每类一个文件: predictions / metrics / alerts)
        # / Root directory containing the three jsonl files
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # 预测追加日志, 每行一个 JSON 对象
        # / Append-only predictions log
        self._pred_file = self.base_dir / 'predictions.jsonl'
        # 指标快照追加日志
        # / Append-only metrics snapshot log
        self._metric_file = self.base_dir / 'metrics.jsonl'
        # 告警追加日志
        # / Append-only alerts log
        self._alert_file = self.base_dir / 'alerts.jsonl'
        # 写入锁, 保证多线程下行不交错
        # / Mutex for thread-safe append
        self._lock = threading.Lock()

    def _append(self, path: Path, obj: Dict[str, Any]) -> None:
        with self._lock:
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(obj, ensure_ascii=False, default=str))
                f.write('\n')

    def _read_all(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        out = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    # ---- writes ----
    def insert_prediction(
        self, *, model_id, timestamp, target, y_pred,
        y_lower=None, y_upper=None, y_actual=None,
    ) -> None:
        self._append(self._pred_file, {
            'model_id': model_id, 'ts': _iso(timestamp), 'target': target,
            'y_pred': float(y_pred),
            'y_lower': None if y_lower is None else float(y_lower),
            'y_upper': None if y_upper is None else float(y_upper),
            'y_actual': None if y_actual is None else float(y_actual),
        })

    def update_actual(self, *, model_id, timestamp, target, y_actual) -> None:
        # jsonl 不支持 in-place update, 用 update 记录追加
        self._append(self._pred_file, {
            'model_id': model_id, 'ts': _iso(timestamp), 'target': target,
            'y_actual_update': float(y_actual),
        })

    def insert_metrics_snapshot(
        self, *, model_id, timestamp, metrics, window=None,
    ) -> None:
        for name, value in metrics.items():
            self._append(self._metric_file, {
                'model_id': model_id, 'ts': _iso(timestamp),
                'name': name, 'value': float(value), 'window': window,
            })

    def insert_alert(self, alert: Alert) -> None:
        self._append(self._alert_file, {
            'alert_id': alert.alert_id,
            'model_id': alert.model_id,
            'ts': _iso(alert.timestamp),
            'level': alert.level,
            'source': alert.source,
            'message': alert.message,
            'details': alert.details,
            'acknowledged': alert.acknowledged,
        })

    # ---- reads ----
    def _filter_time(self, rows, start, end):
        out = []
        for r in rows:
            ts = _parse_iso(r['ts'])
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue
            r = dict(r)
            r['timestamp'] = ts
            out.append(r)
        return out

    def query_predictions(self, *, model_id, start=None, end=None):
        """
        查询预测 / Query predictions with async actual backfill applied.

        JSONL 是追加式存储, ``update_actual`` 不能就地修改, 而是追加一条
        ``y_actual_update`` 记录. 历史 bug: 之前 ``query_predictions`` 直接
        过滤掉这些 update 记录, 导致回填的真值**永远不可见**.

        修复: 用 last-write-wins 合并:
        1. 顺序读所有行
        2. 原始 prediction 行进入 (model_id, ts, target) → row 字典
        3. 遇到 y_actual_update 行就用 ``y_actual`` 覆盖对应记录
        / Merge update lines into base rows (last-write-wins).
        """
        all_rows = [r for r in self._read_all(self._pred_file)
                    if r.get('model_id') == model_id]

        merged: Dict[tuple, Dict[str, Any]] = {}
        for r in all_rows:
            key = (r.get('model_id'), r.get('ts'), r.get('target'))
            if 'y_actual_update' in r:
                # 回填记录: 仅用 y_actual_update 的值更新已有 prediction 的 y_actual
                if key in merged:
                    merged[key]['y_actual'] = float(r['y_actual_update'])
                # 若对应 prediction 还没出现 (理论上不该发生, 防御性), 忽略
            else:
                # 原始 prediction 行: 后写覆盖前写 (last-write-wins)
                merged[key] = dict(r)

        rows = list(merged.values())
        return self._filter_time(rows, start, end)

    def query_metrics(self, *, model_id, start=None, end=None, name=None):
        rows = [r for r in self._read_all(self._metric_file)
                if r.get('model_id') == model_id
                and (name is None or r.get('name') == name)]
        return self._filter_time(rows, start, end)

    def query_alerts(self, *, model_id, start=None, end=None,
                     min_level=None):
        rows = self._read_all(self._alert_file)
        min_rank = LEVEL_ORDER.get(min_level, 0) if min_level else 0
        out: List[Alert] = []
        for r in rows:
            if r.get('model_id') != model_id:
                continue
            if LEVEL_ORDER.get(r.get('level', ''), 0) < min_rank:
                continue
            ts = _parse_iso(r['ts'])
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue
            out.append(Alert(
                alert_id=r['alert_id'], model_id=r['model_id'],
                level=r['level'], message=r['message'], timestamp=ts,
                source=r.get('source', ''),
                details=r.get('details') or {},
                acknowledged=bool(r.get('acknowledged', False)),
            ))
        return out


# ==========================================================================
# main — 三实现对照演示
# ==========================================================================

def main() -> None:
    """演示 Memory / SQLite / JSONL 三种 store 的一致接口。"""
    import tempfile

    print('=' * 70)
    print(' stores — three backends side by side')
    print('=' * 70)

    tmp = tempfile.mkdtemp(prefix='tsf_stores_')
    backends = {
        'memory': InMemoryStore(),
        'sqlite': SQLiteStore(db_path=os.path.join(tmp, 'mon.db')),
        'jsonl': JsonlStore(base_dir=os.path.join(tmp, 'jsonl')),
    }

    now = datetime.now()
    model_id = 'demo'

    for name, store in backends.items():
        print(f'\n[{name}] inserting 3 predictions + 2 metrics + 1 alert')
        for i in range(3):
            store.insert_prediction(
                model_id=model_id, timestamp=now, target='y',
                y_pred=10.0 + i, y_lower=9.0 + i, y_upper=11.0 + i,
                y_actual=10.5 + i,
            )
        store.insert_metrics_snapshot(
            model_id=model_id, timestamp=now,
            metrics={'mae': 0.5, 'mape': 0.02}, window=30,
        )
        store.insert_alert(Alert(
            alert_id=f'{model_id}_{_iso(now)}',
            model_id=model_id, level=AlertLevel.WARNING,
            message='demo alert', timestamp=now,
            source='demo', details={'x': 1},
        ))

        preds = store.query_predictions(model_id=model_id)
        metrics = store.query_metrics(model_id=model_id)
        alerts = store.query_alerts(model_id=model_id)
        print(f'   predictions={len(preds)}  metrics={len(metrics)}  '
              f'alerts={len(alerts)}')

    print(f'\ntemp dir: {tmp}')
    print('(inspect files manually if you like; they persist)')


if __name__ == '__main__':
    main()
