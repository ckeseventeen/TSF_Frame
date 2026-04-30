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

import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from .interfaces import (
    METRIC_REGISTRY,
    MetricFn,
    get_metric_fn,
    register_metric,
)

__all__ = [
    'PerformanceMonitor',
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
    """单条预测记录 / Single prediction record keyed by target_ts."""
    target_ts: datetime
    y_pred: float
    y_true: Optional[float] = None
    y_lower: Optional[float] = None
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
        self.model_id = model_id
        self.window_size = int(window_size)

        # 校验 metric_window
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
        self.metric_window: int = mw

        self.baseline: Dict[str, float] = dict(baseline or {})
        self.probabilistic = probabilistic
        self.metric_names: List[str] = list(metrics or self.DEFAULT_METRICS)
        if probabilistic:
            for m in self.PROB_METRICS:
                if m not in self.metric_names:
                    self.metric_names.append(m)

        # target_ts -> _Record, 插入顺序 = FIFO 淘汰顺序
        # OrderedDict 保证: 同 key 重复 update 不改变顺序; popitem(last=False) FIFO 淘汰
        self._records: 'OrderedDict[datetime, _Record]' = OrderedDict()

        self.snapshots: List[Dict[str, Any]] = []

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

    def current(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        计算指标 / Compute metrics over the latest ``window`` settled records.

        Args:
            window: 临时覆盖 ``self.metric_window``; None 则用默认。
                    取值会被 clamp 到 [1, window_size]。

        只用 ``y_true is not None`` 的记录, 按 target_ts 插入序取最近 N 条。
        若全部未回填, 返回 NaN dict。
        """
        n = int(window) if window is not None else self.metric_window
        n = max(1, min(n, self.window_size))

        # 取最近 n 个 target_ts (按插入顺序的尾部)
        recent_keys = list(self._records.keys())[-n:]

        ys_true: List[float] = []
        ys_pred: List[float] = []
        ys_lo: List[Optional[float]] = []
        ys_hi: List[Optional[float]] = []
        for k in recent_keys:
            r = self._records[k]
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
                out[name] = fn(
                    y_true_f, y_pred_f,
                    y_lower=lower_arr, y_upper=upper_arr,
                )
            except Exception as exc:
                out[name] = float('nan')
                out[f'{name}__error'] = str(exc)  # type: ignore[assignment]
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
