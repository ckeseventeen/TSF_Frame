"""
性能监控 / Performance monitoring
==================================

关注模型预测质量随时间变化的模块。包含两层:

1. **点预测指标** (PointMetrics): MAE / MSE / RMSE / MAPE / SMAPE / R²,
   通过 ``METRIC_REGISTRY`` 可插拔扩展。
2. **概率预测指标** (ProbabilisticMetrics): 若模型输出 lower/upper,
   计算 *prediction interval coverage (PICP)*, *mean interval width
   (MIW)*, *Winkler 分数*。

``PerformanceMonitor`` 负责:

* 维护滑动窗口 (``window_size``)
* 喂数据时懒累积 (``update(pred, actual, lower, upper)``)
* 查询时集中计算所有已注册指标
* 与 ``baseline`` 对比, 给出 ``relative_change``

可插拔:
* 任何满足 ``(y_true, y_pred, **kw) -> float`` 的函数, 用
  ``@register_metric('my_metric')`` 注册后即可被窗口统计。
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .interfaces import (
    METRIC_REGISTRY,
    MetricFn,
    get_metric_fn,
    register_metric,
)

__all__ = [
    'PerformanceMonitor',
    'MultiHorizonMonitor',
    'MultiTargetMonitor',
    'mae',
    'mse',
    'rmse',
    'mape',
    'smape',
    'r2',
    'picp',
    'miw',
    'winkler',
]


# ==========================================================================
# 内建指标 / Built-in metrics (self-registering)
# ==========================================================================

def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


@register_metric('mae')
def mae(y_true, y_pred, **_) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    if len(y_true) == 0:
        return float('nan')
    return float(np.mean(np.abs(y_true - y_pred)))


@register_metric('mse')
def mse(y_true, y_pred, **_) -> float:
    """Mean Squared Error."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    if len(y_true) == 0:
        return float('nan')
    return float(np.mean((y_true - y_pred) ** 2))


@register_metric('rmse')
def rmse(y_true, y_pred, **_) -> float:
    """Root Mean Squared Error."""
    return math.sqrt(mse(y_true, y_pred))


@register_metric('mape')
def mape(y_true, y_pred, eps: float = 1e-8, **_) -> float:
    """Mean Absolute Percentage Error (小数形式, 0.1 == 10%)."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    if len(y_true) == 0:
        return float('nan')
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


@register_metric('smape')
def smape(y_true, y_pred, eps: float = 1e-8, **_) -> float:
    """Symmetric MAPE, 范围 [0, 2]."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    if len(y_true) == 0:
        return float('nan')
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    den = np.where(den < eps, eps, den)
    return float(np.mean(num / den))


@register_metric('r2')
def r2(y_true, y_pred, **_) -> float:
    """Coefficient of determination."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    if len(y_true) < 2:
        return float('nan')
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


# ----- 概率预测指标 -------------------------------------------------------

@register_metric('picp')
def picp(y_true, y_pred, y_lower=None, y_upper=None, **_) -> float:
    """
    Prediction Interval Coverage Probability.

    区间覆盖率: 落在 [y_lower, y_upper] 内的真实值占比。
    理想值 = 置信水平 (例如 95% → 0.95)。
    """
    if y_lower is None or y_upper is None:
        return float('nan')
    y_true = _arr(y_true)
    lo = _arr(y_lower); hi = _arr(y_upper)
    if len(y_true) == 0:
        return float('nan')
    inside = (y_true >= lo) & (y_true <= hi)
    return float(np.mean(inside))


@register_metric('miw')
def miw(y_true=None, y_pred=None, y_lower=None, y_upper=None, **_) -> float:
    """Mean Interval Width, 区间宽度均值。"""
    if y_lower is None or y_upper is None:
        return float('nan')
    lo = _arr(y_lower); hi = _arr(y_upper)
    if len(lo) == 0:
        return float('nan')
    return float(np.mean(hi - lo))


@register_metric('winkler')
def winkler(y_true, y_pred=None, y_lower=None, y_upper=None,
            alpha: float = 0.05, **_) -> float:
    """
    Winkler 分数 (区间 + 覆盖惩罚), 越小越好。

    针对 (1-alpha) 区间: 落在外时额外叠加 2/alpha * 距离。
    """
    if y_lower is None or y_upper is None:
        return float('nan')
    y = _arr(y_true); lo = _arr(y_lower); hi = _arr(y_upper)
    if len(y) == 0:
        return float('nan')
    width = hi - lo
    below = y < lo
    above = y > hi
    penalty = np.zeros_like(y)
    penalty[below] = (2 / alpha) * (lo[below] - y[below])
    penalty[above] = (2 / alpha) * (y[above] - hi[above])
    return float(np.mean(width + penalty))


# ==========================================================================
# PerformanceMonitor
# ==========================================================================

@dataclass
class _Record:
    """
    单条预测记录 / Single prediction record keyed by target_ts.

    一条记录表示"对某个目标时点 (target_ts) 的一次预测", 真值通过
    ``fill_actual`` 异步回填。
    """
    # 预测的目标时点 (作为对齐键, 不依赖位置索引)
    target_ts: datetime
    # 模型预测点估计 (必填)
    y_pred: float
    # 真值, 异步回填; None 表示尚未到达 / awaiting fill
    y_true: Optional[float] = None
    # 概率预测下界, None 表示该模型不输出区间
    y_lower: Optional[float] = None
    # 概率预测上界, None 表示该模型不输出区间
    y_upper: Optional[float] = None


class PerformanceMonitor:
    """
    滑窗性能监控 / Sliding-window performance monitor.

    **对齐方式**: 以 ``target_ts`` (预测目标时点) 为唯一键, 不依赖位置
    索引。多步预测/异步真值回填场景下不会错位。

    生命周期:
    1. ``update(y_pred, target_ts)`` 在跑批时记录预测 (y_true 留空);
    2. ``fill_actual(target_ts, y_true)`` 在真值到达时回填;
    3. ``current()`` 仅基于已回填的记录计算指标。

    **两层窗口**:
    * ``window_size``   — *存储* 容量, 队列里最多保留多少 target_ts
                          (越大可追溯历史越长, 但内存占用越多)
    * ``metric_window`` — *计算* 窗口, 算指标时只取最近 N 条已回填的记录
                          (HPF 月度场景默认 12, 即"近 12 个月")

    保证 ``metric_window <= window_size``, 否则 clamp 并警告。

    Args:
        model_id:       模型标识
        window_size:    存储队列容量, 默认 100
        metric_window:  指标计算窗口, 默认 12 (近 12 期); 必须 <= window_size
        metrics:        要计算的指标名列表, 默认 [mae, mape, rmse, r2]
        baseline:       基线指标字典, 便于 compare_to_baseline
        probabilistic:  若为 True, 强制计算 picp/miw/winkler (自动追加)
    """

    DEFAULT_METRICS = ['mae', 'mape', 'rmse', 'r2']
    PROB_METRICS = ['picp', 'miw', 'winkler']

    def __init__(
        self,
        model_id: str,
        *,
        window_size: int = 100,
        metric_window: int = 12,
        metrics: Optional[List[str]] = None,
        baseline: Optional[Mapping[str, float]] = None,
        probabilistic: bool = False,
    ):
        # 模型唯一标识, 落库时区分多个模型 / Model identifier for store partitioning
        self.model_id = model_id
        # 队列存储容量上限 (FIFO 淘汰最早的 target_ts)
        # / Storage capacity (FIFO eviction)
        self.window_size = int(window_size)

        # 校验 metric_window: 必须为正, 不超过 window_size
        mw = int(metric_window)
        if mw <= 0:
            raise ValueError(
                f'metric_window 必须 > 0, 实际 {mw}'
            )
        if mw > self.window_size:
            warnings.warn(
                f'metric_window={mw} 超过 window_size={self.window_size}, '
                f'已 clamp 到 {self.window_size}. '
                f'建议 metric_window <= window_size.',
                RuntimeWarning, stacklevel=2,
            )
            mw = self.window_size
        # 计算指标时只取最近 N 条已 settle 的记录; 默认 12 (HPF 月度场景)
        # / Sliding metric window (only the last N settled records contribute)
        self.metric_window: int = mw

        # 基线指标字典 {metric_name: baseline_value}, 用于 compare_to_baseline
        # / Baseline metric values for relative comparison
        self.baseline: Dict[str, float] = dict(baseline or {})
        # 是否启用概率预测指标 (PICP/MIW/Winkler)
        # / Whether to compute probabilistic interval metrics
        self.probabilistic = probabilistic
        # 实际计算的指标名列表; probabilistic=True 时会自动追加 prob 指标
        # / Active metric names; auto-augmented when probabilistic
        self.metric_names: List[str] = list(metrics or self.DEFAULT_METRICS)
        if probabilistic:
            for m in self.PROB_METRICS:
                if m not in self.metric_names:
                    self.metric_names.append(m)

        # 主存储: target_ts → _Record, 插入顺序即 FIFO 淘汰顺序
        # OrderedDict 保证: 同 key 重复 update 不改变位置; popitem(last=False) 淘汰最早
        # / Primary store; OrderedDict guarantees insertion-ordered FIFO eviction
        self._records: 'OrderedDict[datetime, _Record]' = OrderedDict()

        # 每次 snapshot() 调用追加一行的指标历史, 便于趋势分析
        # / History of snapshots produced by snapshot()
        self.snapshots: List[Dict[str, Any]] = []

        # 最近一次 current() 调用中, 哪些 metric 计算抛了异常 ({metric_name: str(exc)}).
        # 此前混在返回 dict 里 (如 'mae__error') 会破坏 Dict[str, float] 类型契约,
        # 现在单独存放. 调用方需要查问题时读它.
        # / Per-metric error details from the latest current() call
        self.last_errors: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 更新 / 查询
    # ------------------------------------------------------------------
    def update(
        self,
        *,
        y_pred: float,
        y_true: Optional[float] = None,
        y_lower: Optional[float] = None,
        y_upper: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        记录一条预测 / Append or update a prediction record.

        ``timestamp`` 解释为 ``target_ts`` (预测的目标时点)。
        若该 ``target_ts`` 已存在, 则**就地更新**字段 (不改变窗口顺序)。
        若不存在且窗口已满, 按 FIFO 淘汰最早 ``target_ts``。

        典型用法:
        - 跑批时: ``update(y_pred=p, target_ts=T+1)``
        - 真值到达: ``fill_actual(target_ts=T+1, y_true=t)`` (推荐)
          或 ``update(y_pred=p, y_true=t, timestamp=T+1)`` (一并提交)
        """
        target_ts = timestamp or datetime.now()
        existing = self._records.get(target_ts)
        if existing is not None:
            # 同 target_ts 再次提交: 就地更新, 不改窗口顺序
            existing.y_pred = float(y_pred)
            if y_true is not None:
                existing.y_true = float(y_true)
            if y_lower is not None:
                existing.y_lower = float(y_lower)
            if y_upper is not None:
                existing.y_upper = float(y_upper)
            return

        # 新 target_ts: 满则 FIFO 淘汰
        if len(self._records) >= self.window_size:
            self._records.popitem(last=False)

        self._records[target_ts] = _Record(
            target_ts=target_ts,
            y_pred=float(y_pred),
            y_true=None if y_true is None else float(y_true),
            y_lower=None if y_lower is None else float(y_lower),
            y_upper=None if y_upper is None else float(y_upper),
        )
    # 解耦预测和验证
    # 假设你在 2026-05-01 预测了 2026-05-06 的销量。到了今天（2026-05-06），真实的销量数据产生了。
    # 行为：你调用 fill_actual(target_ts=2026-05-06, y_true=实际销量)。
    # 逻辑：
    # 它去内存的 _records 字典里，查找 target_ts 为 2026-05-06 的那条记录。
    # 如果找到了，就把 y_true 填进去（rec.y_true = float(y_true)），返回 True。
    # 如果没找到（比如窗口已满，那条记录已经被淘汰了），返回 False。
    def fill_actual(
        self,
        target_ts: Union[datetime, int],
        y_true: float,
    ) -> bool:
        """
        按 ``target_ts`` 回填真实值 / Backfill actual by target_ts.

        Returns:
            True 若找到并填好; False 若该记录已被淘汰出窗口。

        向后兼容: 若传入 ``int``, 视为旧的位置索引接口, 发出
        ``DeprecationWarning`` 并退化为按插入顺序的第 N 条 (不保证正确)。
        """
        if isinstance(target_ts, int):
            warnings.warn(
                'fill_actual(index, y_true) 已废弃; 请用 '
                'fill_actual(target_ts=..., y_true=...). '
                '位置索引在窗口满时会错位.',
                DeprecationWarning, stacklevel=2,
            )
            keys = list(self._records.keys())
            if 0 <= target_ts < len(keys):
                self._records[keys[target_ts]].y_true = float(y_true)
                return True
            return False

        rec = self._records.get(target_ts)
        if rec is None:
            return False
        rec.y_true = float(y_true)
        return True

    def has(self, target_ts: datetime) -> bool:
        """判断某 target_ts 是否还在窗口内。"""
        return target_ts in self._records

    def pending_targets(self) -> List[datetime]:
        """列出已记录但 y_true 还未回填的 target_ts (按时间)。"""
        return [ts for ts, r in self._records.items() if r.y_true is None]

    def get_record(
        self, target_ts: datetime,
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        按 target_ts 取出单条预测记录的只读快照 / Public accessor.

        外部调用者(例如 ModelMonitor)无需触碰 ``_records`` 私有字段,
        通过本接口获得稳定 dict 视图: 字段命名与 ``_Record`` 对齐,
        修改返回值不会影响内部存储。

        Returns:
            ``{'target_ts','y_pred','y_true','y_lower','y_upper'}`` 字典;
            未找到该 target_ts 时返回 ``None``。
        """
        rec = self._records.get(target_ts)
        if rec is None:
            return None
        return {
            'target_ts': rec.target_ts,
            'y_pred': rec.y_pred,
            'y_true': rec.y_true,
            'y_lower': rec.y_lower,
            'y_upper': rec.y_upper,
        }

    def get_latest_record(
        self,
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        取最新一条预测记录 (按插入序最末) / Latest record snapshot.

        与 ``get_record`` 同样返回字段化 dict; 队列为空时返回 ``None``。
        """
        if not self._records:
            return None
        # OrderedDict: next(reversed(...)) 即最新插入的 key
        latest_ts = next(reversed(self._records))
        return self.get_record(latest_ts)

    def iter_records(self):
        """
        惰性遍历所有记录 (按插入序) / Lazy iterator over records.

        返回 ``(target_ts, dict)`` 二元组迭代器, 适合 MonitoringStatus
        填充 / 报表等只读场景。
        """
        for ts, rec in self._records.items():
            yield ts, {
                'target_ts': rec.target_ts,
                'y_pred': rec.y_pred,
                'y_true': rec.y_true,
                'y_lower': rec.y_lower,
                'y_upper': rec.y_upper,
            }

    def n_settled(self) -> int:
        """已回填真值的记录数 / Count of records with y_true filled."""
        return sum(1 for r in self._records.values()
                   if r.y_true is not None)

    def current(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        计算指标 / Compute metrics over the latest ``window`` settled records.

        Args:
            window: 临时覆盖 ``self.metric_window``; None 则用默认。
                    取值会被 clamp 到 [1, window_size]。

        只用 ``y_true is not None`` 的记录, 按 target_ts 插入序取最近 N 条。

        返回值约定 (类型严格 ``Dict[str, float]``):
        - 全部未回填或异常计算失败 → 该 metric 值为 ``NaN``
        - 计算异常细节通过 ``self.last_errors: Dict[str, str]`` 单独保存,
          不再混入返回 dict (此前 ``f'{name}__error'`` 字符串混入是类型 bug)
        - 调用方判断"无数据"用 ``np.isnan(v)``, 判断"算错了"看
          ``self.last_errors``

        / Returns strict Dict[str, float]; errors stored in self.last_errors.
        """
        # 重置错误记录, 每次 current() 都是独立快照
        self.last_errors: Dict[str, str] = {}

        n = int(window) if window is not None else self.metric_window
        n = max(1, min(n, self.window_size))

        # 取最近 n 个 target_ts: 用 reversed + islice 避免 list(...) 创建全量列表.
        # 先 reverse 拿尾部 n 个, 再 reverse 回升序便于按时间序聚合.
        # / Use reversed+islice instead of list()[-n:] to avoid full-list materialization.
        total = len(self._records)
        if n >= total:
            recent_iter = self._records.values()  # 等于全量, 直接用原序
        else:
            tail = list(itertools.islice(reversed(self._records.values()), n))
            tail.reverse()
            recent_iter = tail

        ys_true: List[float] = []
        ys_pred: List[float] = []
        ys_lo: List[Optional[float]] = []
        ys_hi: List[Optional[float]] = []
        for r in recent_iter:
            if r.y_true is None:
                continue
            ys_true.append(r.y_true)
            ys_pred.append(r.y_pred)
            ys_lo.append(r.y_lower)
            ys_hi.append(r.y_upper)

        if not ys_true:
            return {name: float('nan') for name in self.metric_names}

        y_true_f = np.asarray(ys_true, dtype=float)
        y_pred_f = np.asarray(ys_pred, dtype=float)

        lower_arr: Optional[np.ndarray] = None
        upper_arr: Optional[np.ndarray] = None
        if any(v is not None for v in ys_lo):
            lo = np.array([np.nan if v is None else v for v in ys_lo],
                          dtype=float)
            hi = np.array([np.nan if v is None else v for v in ys_hi],
                          dtype=float)
            if not (np.isnan(lo).all() or np.isnan(hi).all()):
                lower_arr, upper_arr = lo, hi

        out: Dict[str, float] = {}
        for name in self.metric_names:
            try:
                fn: MetricFn = get_metric_fn(name)
                out[name] = float(fn(
                    y_true_f, y_pred_f,
                    y_lower=lower_arr, y_upper=upper_arr,
                ))
            except Exception as exc:
                out[name] = float('nan')
                # 错误细节单独存, 不污染返回 dict 的类型
                self.last_errors[name] = str(exc)
        return out

    def set_metric_window(self, n: int) -> None:
        """运行时调整指标窗口大小 (会自动 clamp 到 [1, window_size])。"""
        n = int(n)
        if n <= 0:
            raise ValueError(f'metric_window 必须 > 0, 实际 {n}')
        self.metric_window = min(n, self.window_size)

    def snapshot(self, window: Optional[int] = None) -> Dict[str, Any]:
        """
        记录当前指标并返回 (用于 check_status 时写入 store)。

        Args:
            window: 见 ``current()`` 同名参数。
        """
        used_window = (int(window) if window is not None
                       else self.metric_window)
        used_window = max(1, min(used_window, self.window_size))
        metrics = self.current(window=used_window)
        row = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'metric_window': used_window,
            'n': len(self._records),
            'n_settled': sum(1 for r in self._records.values()
                             if r.y_true is not None),
        }
        self.snapshots.append(row)
        return row

    def compare_to_baseline(
        self, window: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        与基线对比 / Compare current metrics against baseline.

        Args:
            window: 见 ``current()`` 同名参数。

        返回 {metric: relative_change}: 正值表示恶化 (MAE/MAPE 升高) 或
        改善 (R² 升高), 调用方须结合指标语义判断方向。
        """
        cur = self.current(window=window)
        out: Dict[str, float] = {}
        for k, base in self.baseline.items():
            if k in cur and base not in (0, None) and not math.isnan(cur[k]):
                out[k] = (cur[k] - base) / abs(base)
        return out

    def reset(self) -> None:
        """清空窗口与快照 / Clear window and snapshots."""
        self._records.clear()
        self.snapshots.clear()

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def set_baseline(self, baseline: Mapping[str, float]) -> None:
        self.baseline = dict(baseline)


# ==========================================================================
# MultiHorizonMonitor — 单模型多输出监控
# ==========================================================================

class MultiHorizonMonitor:
    """
    多步 horizon 性能监控 / Per-horizon performance tracking.

    适用: **单个模型一次性输出 H 步预测**(典型场景: HPF 月度跑批
    输出未来 12 个月)。内部为每个 horizon 维护一个独立的
    ``PerformanceMonitor``, 实现:

    * 每个 horizon 独立的 MAPE/MAE/RMSE/R² 等指标
    * 同一 ``target_ts`` 的真值能**自动回填所有 horizon**
    * 短期(h=1)/长期(h=H) 表现可独立观察, 暴露"长期模型已坏"等问题
    * 跨 horizon **加权聚合** (默认按各 horizon settled 样本数加权)

    Args:
        model_id:       模型标识 (内部各 PerformanceMonitor 标识为 ``{model_id}_h{h}``)
        horizons:       要追踪的步数列表, 默认 ``[1..12]`` (HPF 月度场景)
        window_size:    每个内部 PerformanceMonitor 的存储队列容量
        metric_window:  每个内部 PerformanceMonitor 的指标计算窗口
        metrics:        指标名列表, None 用 PerformanceMonitor 默认
        baseline:       基线指标字典, 共享给每个 horizon
        probabilistic:  是否启用 PICP/MIW/Winkler

    用法::

        from tsf_frame.monitoring import MultiHorizonMonitor

        mhm = MultiHorizonMonitor(
            model_id='hpf_deposit',
            horizons=range(1, 13),
            window_size=36, metric_window=12,
        )

        # 1) 跑批: 一次记 12 个 horizon
        from dateutil.relativedelta import relativedelta
        forecast_time = pd.Timestamp('2026-04-01')
        target_times = [forecast_time + relativedelta(months=h)
                        for h in range(1, 13)]
        mhm.record_forecast(
            forecast_time=forecast_time,
            predictions=preds,                # shape (12,)
            target_times=target_times,
        )

        # 2) 真值到达: 一次回填所有 horizon
        mhm.settle_actual(
            target_ts=pd.Timestamp('2026-05-01'),
            y_actual=178_500_000.0,
        )

        # 3) 查询: 看每个 horizon 表现
        for h, m in mhm.current().items():
            print(f'h={h:2d}  MAPE={m["mape"]:.2%}')
        # h= 1  MAPE=2.10%
        # h= 6  MAPE=8.30%
        # h=12  MAPE=15.20%   ← 长期已坏

        # 4) 跨 horizon 聚合 (默认按 settled 样本数加权)
        agg = mhm.aggregated()
    """

    def __init__(
        self,
        model_id: str,
        *,
        horizons: Sequence[int] = tuple(range(1, 13)),
        window_size: int = 100,
        metric_window: int = 12,
        metrics: Optional[List[str]] = None,
        baseline: Optional[Mapping[str, float]] = None,
        probabilistic: bool = False,
    ):
        # 顶层模型标识, 内部各 PerformanceMonitor 名为 {model_id}_h{h}
        # / Parent model identifier
        self.model_id = model_id
        # 要追踪的步数列表 (升序去重), 例如 [1, 3, 6, 12]
        # / Sorted unique horizon list
        self.horizons: List[int] = sorted(set(int(h) for h in horizons))
        if not self.horizons:
            raise ValueError('horizons 不能为空')
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f'horizons 必须 > 0, got {self.horizons}')

        # 主存储: horizon → 独立的 PerformanceMonitor, 每个 horizon 各算各的指标
        # / Per-horizon performance monitor map; one independent PM per horizon
        self.per_horizon: Dict[int, PerformanceMonitor] = {
            h: PerformanceMonitor(
                f'{model_id}_h{h}',
                window_size=window_size,
                metric_window=metric_window,
                metrics=metrics,
                baseline=baseline,
                probabilistic=probabilistic,
            )
            for h in self.horizons
        }

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def record_forecast(
        self,
        *,
        forecast_time: datetime,
        predictions: Sequence[float],
        target_times: Optional[Sequence[datetime]] = None,
        y_lower: Optional[Sequence[float]] = None,
        y_upper: Optional[Sequence[float]] = None,
    ) -> None:
        """
        记录一次完整多步预测 / Record one batch of horizon forecasts.

        Args:
            forecast_time: 预测产生时点 (跑批日)
            predictions:   长度 = ``len(self.horizons)`` 的预测序列
            target_times:  各 horizon 对应的 target_ts; None 时按月递推
                           (要求 horizons 的步距以"月"解释)
            y_lower:       各 horizon 的预测下界 (可选, 概率预测时使用)
            y_upper:       各 horizon 的预测上界 (同上)

        Raises:
            ValueError: 长度不匹配
        """
        preds = list(predictions)
        if len(preds) != len(self.horizons):
            raise ValueError(
                f'predictions 长度 {len(preds)} ≠ horizons '
                f'数量 {len(self.horizons)}'
            )

        if target_times is None:
            try:
                from dateutil.relativedelta import relativedelta
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    '默认 target_times 推导需要 dateutil; '
                    '请显式传入 target_times 或安装 python-dateutil'
                ) from exc
            target_times_list = [forecast_time + relativedelta(months=h)
                                 for h in self.horizons]
        else:
            target_times_list = list(target_times)
            if len(target_times_list) != len(self.horizons):
                raise ValueError(
                    f'target_times 长度 {len(target_times_list)} ≠ '
                    f'horizons 数量 {len(self.horizons)}'
                )

        lo_list = list(y_lower) if y_lower is not None else None
        hi_list = list(y_upper) if y_upper is not None else None

        for i, h in enumerate(self.horizons):
            self.per_horizon[h].update(
                y_pred=float(preds[i]),
                timestamp=target_times_list[i],
                y_lower=None if lo_list is None else float(lo_list[i]),
                y_upper=None if hi_list is None else float(hi_list[i]),
            )

    def settle_actual(
        self,
        *,
        target_ts: datetime,
        y_actual: float,
    ) -> Dict[int, bool]:
        """
        某个 ``target_ts`` 的真值到达 → **同时**回填所有 horizon.

        Returns:
            {horizon: filled_or_not} — 每个 horizon 是否找到该 target_ts 并回填
            (False 表示该 horizon 的窗口里已淘汰该记录)
        """
        out: Dict[int, bool] = {}
        for h, pm in self.per_horizon.items():
            out[h] = pm.fill_actual(target_ts=target_ts, y_true=float(y_actual))
        return out

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def current(
        self, window: Optional[int] = None,
    ) -> Dict[int, Dict[str, float]]:
        """各 horizon 的当前指标 / Per-horizon metrics dict."""
        return {h: pm.current(window=window)
                for h, pm in self.per_horizon.items()}

    def aggregated(
        self,
        window: Optional[int] = None,
        weights: Optional[Mapping[int, float]] = None,
    ) -> Dict[str, float]:
        """
        跨 horizon 聚合指标 / Aggregated metrics across horizons.

        Args:
            window:  per-horizon ``current()`` 的 window 覆盖
            weights: ``{horizon: weight}`` 加权;
                     None 时按各 horizon **已 settled 的样本数**加权

        Returns:
            {metric_name: weighted_value}
        """
        per_h = self.current(window=window)
        if not per_h:
            return {}

        # 默认权重: 各 horizon settled 数量 (走 PM 公共接口)
        if weights is None:
            counts = {
                h: pm.n_settled()
                for h, pm in self.per_horizon.items()
            }
            total = sum(counts.values())
            if total > 0:
                weights = {h: c / total for h, c in counts.items()}
            else:
                # 全空, 平均
                weights = {h: 1.0 / len(per_h) for h in per_h}

        # 收集所有出现过的指标名
        all_names: List[str] = []
        for m in per_h.values():
            for n in m:
                if n not in all_names and not n.endswith('__error'):
                    all_names.append(n)

        out: Dict[str, float] = {}
        for name in all_names:
            num, den = 0.0, 0.0
            for h, m in per_h.items():
                v = m.get(name)
                w = float(weights.get(h, 0.0))
                if (v is not None
                        and not (isinstance(v, float) and np.isnan(v))
                        and w > 0):
                    num += v * w
                    den += w
            out[name] = num / den if den > 0 else float('nan')
        return out

    def snapshot(self, window: Optional[int] = None) -> Dict[str, Any]:
        """组合 per-horizon + 聚合的快照, 便于写 store / 报表。"""
        per_h = self.current(window=window)
        agg = self.aggregated(window=window)
        used_window = (int(window) if window is not None
                       else next(iter(self.per_horizon.values())).metric_window)
        return {
            'timestamp': datetime.now(),
            'model_id': self.model_id,
            'horizons': self.horizons,
            'metric_window': used_window,
            'per_horizon': per_h,
            'aggregated': agg,
            'n_settled_per_horizon': {
                h: pm.n_settled()
                for h, pm in self.per_horizon.items()
            },
        }

    def pending_targets(self) -> Dict[int, List[datetime]]:
        """各 horizon 中还未 settle 的 target_ts."""
        return {h: pm.pending_targets()
                for h, pm in self.per_horizon.items()}

    def reset(self) -> None:
        for pm in self.per_horizon.values():
            pm.reset()

    def __len__(self) -> int:
        return sum(len(pm) for pm in self.per_horizon.values())


# ==========================================================================
# MultiTargetMonitor — 单模型多目标监控
# ==========================================================================

class MultiTargetMonitor:
    """
    多目标性能监控 / Per-target performance tracking.

    适用: **单个模型同时预测多个不同的物理量**(典型场景: 一次性输出
    温度 + 湿度 + 气压, 或 HPF 同时预测 deposit + withdrawal + loan_balance)。

    与 ``MultiHorizonMonitor`` 的区别:
    * 后者按"时间步"拆分 (同物理量, 不同 horizon)
    * 本类按"物理量"拆分 (不同物理量, 同时间步)

    每个目标维护独立的 ``PerformanceMonitor``, 便于:
    * 各目标独立的 baseline / 阈值 (温度 MAPE 5% 警戒, 湿度 10% 才警戒)
    * 各目标独立的概率区间开关 (温度有区间, 湿度无)
    * 真值部分到达 (温度先到, 湿度迟到)
    * 各目标独立的 metric 集合

    Args:
        model_id:               模型标识 (内部各 PerformanceMonitor 标识为 ``{model_id}_{target}``)
        targets:                目标名列表, e.g. ``['temperature', 'humidity']``
        window_size:            每个目标的存储队列容量
        metric_window:          每个目标的指标计算窗口
        metrics:                指标名列表 (所有目标共用); None 用 PerformanceMonitor 默认
        baseline_per_target:    ``{target: {metric: baseline_value}}``, 各目标独立基线
        probabilistic_targets:  启用 PICP/MIW/Winkler 的目标列表

    用法::

        from tsf_frame.monitoring import MultiTargetMonitor

        mtm = MultiTargetMonitor(
            model_id='weather',
            targets=['temperature', 'humidity', 'pressure'],
            window_size=200, metric_window=24,
            baseline_per_target={
                'temperature': {'mape': 0.03},   # 温度基线 3%
                'humidity':    {'mape': 0.08},   # 湿度基线 8%
                'pressure':    {'mape': 0.005},  # 气压非常稳定
            },
            probabilistic_targets=['temperature'],   # 只温度有区间
        )

        # 1) 一次记录所有目标
        mtm.record_prediction(
            timestamp=now,
            predictions={'temperature': 25.3, 'humidity': 60.0, 'pressure': 1013.2},
            y_lowers={'temperature': 24.0},        # 仅温度有区间
            y_uppers={'temperature': 26.5},
        )

        # 2) 真值部分到达 (湿度传感器还没回数)
        mtm.settle_actuals(
            target_ts=now,
            actuals={'temperature': 25.1, 'pressure': 1013.5},
        )
        # 后续湿度真值到了, 单独 settle:
        mtm.settle_actual(
            target_ts=now, target='humidity', y_actual=58.2,
        )

        # 3) 各目标查 MAPE
        for tgt, m in mtm.current().items():
            print(f'{tgt:12s}  MAPE={m["mape"]:.2%}')
        # temperature   MAPE=0.79%
        # humidity      MAPE=3.00%
        # pressure      MAPE=0.03%
    """

    def __init__(
        self,
        model_id: str,
        *,
        targets: Sequence[str],
        window_size: int = 100,
        metric_window: int = 12,
        metrics: Optional[List[str]] = None,
        baseline_per_target: Optional[Mapping[str, Mapping[str, float]]] = None,
        probabilistic_targets: Optional[Sequence[str]] = None,
    ):
        # 顶层模型标识, 内部各 PerformanceMonitor 名为 {model_id}_{target}
        # / Parent model identifier
        self.model_id = model_id
        # 目标名列表 (保留顺序), e.g. ['temperature', 'humidity', 'pressure']
        # / Ordered list of target names
        self.targets: List[str] = [str(t) for t in targets]
        if not self.targets:
            raise ValueError('targets 不能为空')
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(f'targets 包含重复: {self.targets}')

        # 严格校验配置 — 早抛错胜过静默错误
        # / Validate config — fail fast over silent error
        baselines = dict(baseline_per_target or {})
        prob_set = set(probabilistic_targets or [])
        unknown_prob = prob_set - set(self.targets)
        if unknown_prob:
            raise ValueError(
                f'probabilistic_targets 中有未声明的目标: {unknown_prob}; '
                f'已声明 {self.targets}'
            )
        unknown_base = set(baselines) - set(self.targets)
        if unknown_base:
            raise ValueError(
                f'baseline_per_target 中有未声明的目标: {unknown_base}; '
                f'已声明 {self.targets}'
            )

        # 主存储: target_name → 独立的 PerformanceMonitor
        # 每个目标独立的 baseline / 概率开关 / 指标窗口
        # / Per-target performance monitor map; each target has independent config
        self.per_target: Dict[str, PerformanceMonitor] = {
            t: PerformanceMonitor(
                f'{model_id}_{t}',
                window_size=window_size,
                metric_window=metric_window,
                metrics=metrics,
                baseline=baselines.get(t),
                probabilistic=(t in prob_set),
            )
            for t in self.targets
        }

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def record_prediction(
        self,
        *,
        timestamp: datetime,
        predictions: Mapping[str, float],
        actuals: Optional[Mapping[str, float]] = None,
        y_lowers: Optional[Mapping[str, float]] = None,
        y_uppers: Optional[Mapping[str, float]] = None,
    ) -> None:
        """
        一次记录多个目标 / Record a multi-target prediction in one shot.

        Args:
            timestamp:    本次预测的目标时点 (target_ts)
            predictions:  ``{target_name: y_pred}``;
                          支持只覆盖部分目标 (其他目标本时刻不记)
            actuals:      ``{target_name: y_true}`` 同步可知的真值, 可选
            y_lowers:     各目标的预测下界, 可选
            y_uppers:     各目标的预测上界, 可选

        Raises:
            ValueError: predictions 中出现未声明的 target_name
        """
        actuals = dict(actuals or {})
        y_lowers = dict(y_lowers or {})
        y_uppers = dict(y_uppers or {})

        unknown = set(predictions) - set(self.targets)
        if unknown:
            raise ValueError(
                f'predictions 中有未声明的目标: {unknown}; '
                f'已声明 {self.targets}'
            )

        for t, y_p in predictions.items():
            self.per_target[t].update(
                y_pred=float(y_p),
                y_true=(None if t not in actuals
                        else float(actuals[t])),
                y_lower=(None if t not in y_lowers
                         else float(y_lowers[t])),
                y_upper=(None if t not in y_uppers
                         else float(y_uppers[t])),
                timestamp=timestamp,
            )

    def settle_actual(
        self,
        *,
        target_ts: datetime,
        target: str,
        y_actual: float,
    ) -> bool:
        """
        回填指定目标的真值 / Backfill actual for one target.

        Returns:
            True 若找到并回填; False 若该目标该 target_ts 已被淘汰出窗口

        Raises:
            ValueError: target 未声明
        """
        if target not in self.per_target:
            raise ValueError(
                f'未知目标 {target!r}; 已声明 {self.targets}'
            )
        return self.per_target[target].fill_actual(
            target_ts=target_ts, y_true=float(y_actual),
        )

    def settle_actuals(
        self,
        *,
        target_ts: datetime,
        actuals: Mapping[str, float],
    ) -> Dict[str, bool]:
        """
        一次回填多个目标 / Backfill multiple targets at one target_ts.

        Returns:
            ``{target: filled_or_not}`` — 每个目标是否成功回填
        """
        out: Dict[str, bool] = {}
        for t, v in actuals.items():
            out[t] = self.settle_actual(
                target_ts=target_ts, target=t, y_actual=float(v),
            )
        return out

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def current(
        self, window: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """各目标当前指标 / Per-target metrics dict."""
        return {t: pm.current(window=window)
                for t, pm in self.per_target.items()}

    def compare_to_baseline(
        self, window: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """各目标的 baseline 相对偏差 / Per-target relative_change vs baseline."""
        return {t: pm.compare_to_baseline(window=window)
                for t, pm in self.per_target.items()}

    def snapshot(self, window: Optional[int] = None) -> Dict[str, Any]:
        per_t = self.current(window=window)
        used_window = (int(window) if window is not None
                       else next(iter(self.per_target.values())).metric_window)
        return {
            'timestamp': datetime.now(),
            'model_id': self.model_id,
            'targets': self.targets,
            'metric_window': used_window,
            'per_target': per_t,
            'n_settled_per_target': {
                t: pm.n_settled()
                for t, pm in self.per_target.items()
            },
        }

    def pending_targets(self) -> Dict[str, List[datetime]]:
        """各目标中还未 settle 的 target_ts."""
        return {t: pm.pending_targets()
                for t, pm in self.per_target.items()}

    def reset(self) -> None:
        for pm in self.per_target.values():
            pm.reset()

    def __len__(self) -> int:
        return sum(len(pm) for pm in self.per_target.values())


# ==========================================================================
# main — 演示
# ==========================================================================

def main() -> None:
    """200 样本滚动窗口 + 概率区间指标 + 自定义指标扩展。"""
    print('=' * 70)
    print(' performance_monitor — window + baseline demo')
    print('=' * 70)

    # 自定义指标: 残差最大值 (加入注册表后自动可用)
    @register_metric('max_abs_err')
    def max_abs_err(y_true, y_pred, **_):
        return float(np.max(np.abs(_arr(y_true) - _arr(y_pred))))

    rng = np.random.default_rng(0)
    mon = PerformanceMonitor(
        model_id='demo_perf',
        window_size=100,
        metrics=['mae', 'mape', 'rmse', 'max_abs_err'],
        baseline={'mae': 0.6, 'mape': 0.05},
        probabilistic=True,
    )
    for i in range(200):
        true_y = float(rng.normal(100, 5))
        pred_y = true_y + float(rng.normal(0, 1 if i < 100 else 3))
        mon.update(y_pred=pred_y, y_true=true_y,
                   y_lower=pred_y - 2, y_upper=pred_y + 2)

    cur = mon.current()
    print('\n当前窗口指标:')
    for k, v in cur.items():
        print(f'  {k:<15} {v:.4f}')
    print('\n与基线对比 (相对变化, 正值=指标值上升):')
    for k, v in mon.compare_to_baseline().items():
        print(f'  {k:<15} {v*100:+.1f}%')
    print(f'\nwindow 中样本数: {len(mon)}')
    print('可用指标注册表:', sorted(METRIC_REGISTRY))


if __name__ == '__main__':
    main()
