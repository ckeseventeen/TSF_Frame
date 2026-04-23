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
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

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

class PerformanceMonitor:
    """
    滑窗性能监控 / Sliding-window performance monitor.

    Args:
        model_id:       模型标识
        window_size:    保留多少条 (pred, actual, lower, upper) 做滚动指标
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
        metrics: Optional[List[str]] = None,
        baseline: Optional[Mapping[str, float]] = None,
        probabilistic: bool = False,
    ):
        self.model_id = model_id
        self.window_size = int(window_size)
        self.baseline: Dict[str, float] = dict(baseline or {})
        self.probabilistic = probabilistic
        self.metric_names: List[str] = list(metrics or self.DEFAULT_METRICS)
        if probabilistic:
            for m in self.PROB_METRICS:
                if m not in self.metric_names:
                    self.metric_names.append(m)

        self._y_true: Deque[float] = deque(maxlen=self.window_size)
        self._y_pred: Deque[float] = deque(maxlen=self.window_size)
        self._y_lower: Deque[Optional[float]] = deque(maxlen=self.window_size)
        self._y_upper: Deque[Optional[float]] = deque(maxlen=self.window_size)
        self._timestamps: Deque[datetime] = deque(maxlen=self.window_size)

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
        追加一条观测 / Append a point.

        ``y_true`` 可为 None (未回填), 调用 ``fill_actual()`` 后补。
        """
        self._y_pred.append(float(y_pred))
        self._y_true.append(float('nan') if y_true is None else float(y_true))
        self._y_lower.append(None if y_lower is None else float(y_lower))
        self._y_upper.append(None if y_upper is None else float(y_upper))
        self._timestamps.append(timestamp or datetime.now())

    def fill_actual(self, index: int, y_true: float) -> None:
        """回填第 index 条 (从窗口起点计数) 的真实值。"""
        if 0 <= index < len(self._y_true):
            # deque 不支持 index 写, 转 list 再写回
            buf = list(self._y_true)
            buf[index] = float(y_true)
            self._y_true = deque(buf, maxlen=self.window_size)

    def current(self) -> Dict[str, float]:
        """
        计算当前窗口全部已注册指标 / Compute all registered metrics now.

        ``y_true`` 中若有 NaN 会被同位置 pair 一同剔除再算。
        """
        y_true = np.array(self._y_true, dtype=float)
        y_pred = np.array(self._y_pred, dtype=float)
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not mask.any():
            return {name: float('nan') for name in self.metric_names}

        y_true_f = y_true[mask]
        y_pred_f = y_pred[mask]

        lower_arr, upper_arr = None, None
        if any(v is not None for v in self._y_lower):
            lo = np.array([np.nan if v is None else v
                           for v in self._y_lower], dtype=float)
            hi = np.array([np.nan if v is None else v
                           for v in self._y_upper], dtype=float)
            lo_f = lo[mask]; hi_f = hi[mask]
            if not (np.isnan(lo_f).all() or np.isnan(hi_f).all()):
                lower_arr, upper_arr = lo_f, hi_f

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

    def snapshot(self) -> Dict[str, Any]:
        """记录当前指标并返回 (用于 check_status 时写入 store)。"""
        metrics = self.current()
        row = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'n': len(self._y_pred),
        }
        self.snapshots.append(row)
        return row

    def compare_to_baseline(self) -> Dict[str, float]:
        """
        与基线对比 / Compare current metrics against baseline.

        返回 {metric: relative_change}: 正值表示恶化 (MAE/MAPE 升高) 或
        改善 (R² 升高), 调用方须结合指标语义判断方向。
        """
        cur = self.current()
        out: Dict[str, float] = {}
        for k, base in self.baseline.items():
            if k in cur and base not in (0, None) and not math.isnan(cur[k]):
                out[k] = (cur[k] - base) / abs(base)
        return out

    def reset(self) -> None:
        """清空窗口与快照 / Clear window and snapshots."""
        self._y_true.clear()
        self._y_pred.clear()
        self._y_lower.clear()
        self._y_upper.clear()
        self._timestamps.clear()
        self.snapshots.clear()

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._y_pred)

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
