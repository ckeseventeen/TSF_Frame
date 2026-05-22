"""
MultiHorizonMonitor lifecycle tests / 多 horizon 监控完整生命周期测试.

覆盖:
  - 构造时按 horizons 创建独立 PerformanceMonitor
  - record_forecast 把一次 N 步预测分发到各 horizon
  - settle_actual 在所有 horizon 同时回填一个 target_ts
  - current / aggregated 计算
  - pending_targets / snapshot / reset
  - 默认 target_times 推导(月度递推)
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tsf_frame.monitoring.performance_monitor import MultiHorizonMonitor


# ────────────────────────────────────────────────────────────────────
# 构造
# ────────────────────────────────────────────────────────────────────

def test_construction_creates_one_pm_per_horizon():
    mhm = MultiHorizonMonitor(
        model_id='m', horizons=[1, 3, 6, 12],
        window_size=24, metric_window=6,
    )
    assert mhm.horizons == [1, 3, 6, 12]
    assert set(mhm.per_horizon.keys()) == {1, 3, 6, 12}
    # 各 horizon 各自独立的 PM
    for h, pm in mhm.per_horizon.items():
        assert pm.model_id == f'm_h{h}'
        assert pm.window_size == 24
        assert pm.metric_window == 6


def test_construction_dedup_and_sort_horizons():
    mhm = MultiHorizonMonitor(
        model_id='m', horizons=[12, 3, 12, 1, 6],
    )
    assert mhm.horizons == [1, 3, 6, 12]


def test_construction_rejects_empty_and_non_positive():
    with pytest.raises(ValueError, match='不能为空'):
        MultiHorizonMonitor(model_id='m', horizons=[])
    with pytest.raises(ValueError, match='> 0'):
        MultiHorizonMonitor(model_id='m', horizons=[1, 0, 3])


# ────────────────────────────────────────────────────────────────────
# record_forecast / settle_actual
# ────────────────────────────────────────────────────────────────────

def test_record_forecast_dispatches_to_each_horizon():
    mhm = MultiHorizonMonitor(model_id='m', horizons=[1, 2, 3])
    forecast_time = pd.Timestamp('2026-01-01')
    target_times = [pd.Timestamp(f'2026-{m:02d}-01')
                    for m in (2, 3, 4)]   # h=1,2,3 → 2 月,3 月,4 月
    mhm.record_forecast(
        forecast_time=forecast_time,
        predictions=[10.0, 20.0, 30.0],
        target_times=target_times,
    )
    # 每个 horizon 收到对应的预测
    assert mhm.per_horizon[1].get_record(target_times[0])['y_pred'] == 10.0
    assert mhm.per_horizon[2].get_record(target_times[1])['y_pred'] == 20.0
    assert mhm.per_horizon[3].get_record(target_times[2])['y_pred'] == 30.0


def test_settle_actual_fills_all_horizons():
    mhm = MultiHorizonMonitor(model_id='m', horizons=[1, 2])
    t1 = pd.Timestamp('2026-02-01')
    t2 = pd.Timestamp('2026-03-01')
    mhm.record_forecast(
        forecast_time=pd.Timestamp('2026-01-01'),
        predictions=[10.0, 20.0],
        target_times=[t1, t2],
    )
    # t1 的真值同时影响 h=1 (作为 h=1 预测的目标)
    # h=2 的窗口里没有 t1, 应该返回 False
    result = mhm.settle_actual(target_ts=t1, y_actual=11.0)
    assert result[1] is True
    assert result[2] is False
    # h=1 的记录里 y_true 已填
    assert mhm.per_horizon[1].get_record(t1)['y_true'] == 11.0


def test_record_forecast_validates_lengths():
    mhm = MultiHorizonMonitor(model_id='m', horizons=[1, 2, 3])
    with pytest.raises(ValueError, match='长度'):
        mhm.record_forecast(
            forecast_time=pd.Timestamp('2026-01-01'),
            predictions=[10.0, 20.0],   # 只有 2 个, 但 horizons 是 3 个
        )


# ────────────────────────────────────────────────────────────────────
# current / aggregated
# ────────────────────────────────────────────────────────────────────

def test_current_returns_per_horizon_metrics():
    mhm = MultiHorizonMonitor(model_id='m', horizons=[1, 2], metric_window=12)
    # 模拟 5 期预测 + 真值
    base = pd.Timestamp('2026-01-01')
    for k in range(5):
        ft = base + pd.DateOffset(months=k)
        t1 = ft + pd.DateOffset(months=1)
        t2 = ft + pd.DateOffset(months=2)
        mhm.record_forecast(
            forecast_time=ft,
            predictions=[100.0 + k, 100.0 + k * 2],
            target_times=[t1, t2],
        )
        # 真值大约比预测低 1
        mhm.settle_actual(target_ts=t1, y_actual=99.0 + k)
    metrics = mhm.current()
    assert set(metrics.keys()) == {1, 2}
    # h=1 有 settled 真值, mae 应 ≈ 1
    assert metrics[1]['mae'] == pytest.approx(1.0, abs=0.1)


def test_aggregated_with_equal_settled_samples_uses_count_weights():
    """
    h=1 和 h=12 不会因为窗口重叠互相影响 (12 月跨度足够大).
    settle h=1 → 只填 h=1 的 _records, 不波及 h=12 (h=12 的目标时点更远).
    """
    mhm = MultiHorizonMonitor(model_id='m', horizons=[1, 12])
    base = pd.Timestamp('2026-01-01')
    for k in range(3):
        ft = base + pd.DateOffset(months=k)
        t1 = ft + pd.DateOffset(months=1)
        t12 = ft + pd.DateOffset(months=12)
        mhm.record_forecast(
            forecast_time=ft, predictions=[10.0, 20.0],
            target_times=[t1, t12],
        )
        mhm.settle_actual(target_ts=t1, y_actual=10.0)   # 只波及 h=1 (t1 不在 h=12 窗口内)
    agg = mhm.aggregated()
    # h=1 有 3 个 settled, h=12 有 0 个 → h=1 权重 = 1.0
    # MAE = 0 (h=1 预测完美), 加权后 = 0
    assert agg['mae'] == pytest.approx(0.0)


def test_settle_propagates_when_target_ts_overlaps():
    """
    重要边界: settle_actual(target_ts) 会同时回填**所有** horizon 中包含该 target_ts
    的记录. 这是设计预期行为(同一时点的真值对所有 horizon 都生效), 但会让短期/长期
    horizon 的窗口产生意外联动 — 调用方需要注意 target_ts 重叠时的统计含义.
    """
    mhm = MultiHorizonMonitor(model_id='m', horizons=[1, 2])
    base = pd.Timestamp('2026-01-01')
    # 故意构造: 第 0 期 h=2 写 t=2026-03, 第 1 期 h=1 也写 t=2026-03 → 重叠
    mhm.record_forecast(
        forecast_time=base, predictions=[10.0, 20.0],
        target_times=[base + pd.DateOffset(months=1), base + pd.DateOffset(months=2)],
    )
    mhm.record_forecast(
        forecast_time=base + pd.DateOffset(months=1), predictions=[10.0, 20.0],
        target_times=[base + pd.DateOffset(months=2), base + pd.DateOffset(months=3)],
    )
    # settle 2026-03 → 影响 h=1 (来自第 1 期) 和 h=2 (来自第 0 期)
    result = mhm.settle_actual(target_ts=base + pd.DateOffset(months=2), y_actual=10.0)
    assert result[1] is True
    assert result[2] is True


# ────────────────────────────────────────────────────────────────────
# pending_targets / snapshot / reset
# ────────────────────────────────────────────────────────────────────

def test_pending_targets_lists_unsettled():
    mhm = MultiHorizonMonitor(model_id='m', horizons=[1, 2])
    base = pd.Timestamp('2026-01-01')
    mhm.record_forecast(
        forecast_time=base,
        predictions=[10.0, 20.0],
        target_times=[base + pd.DateOffset(months=1),
                      base + pd.DateOffset(months=2)],
    )
    pending = mhm.pending_targets()
    assert len(pending[1]) == 1
    assert len(pending[2]) == 1


def test_snapshot_returns_expected_keys():
    mhm = MultiHorizonMonitor(model_id='m', horizons=[1, 2])
    mhm.record_forecast(
        forecast_time=pd.Timestamp('2026-01-01'),
        predictions=[10.0, 20.0],
        target_times=[pd.Timestamp('2026-02-01'), pd.Timestamp('2026-03-01')],
    )
    snap = mhm.snapshot()
    assert 'timestamp' in snap
    assert snap['model_id'] == 'm'
    assert snap['horizons'] == [1, 2]
    assert 'per_horizon' in snap and 'aggregated' in snap
    assert 'n_settled_per_horizon' in snap


def test_reset_clears_all_horizons():
    mhm = MultiHorizonMonitor(model_id='m', horizons=[1, 2])
    mhm.record_forecast(
        forecast_time=pd.Timestamp('2026-01-01'),
        predictions=[10.0, 20.0],
        target_times=[pd.Timestamp('2026-02-01'), pd.Timestamp('2026-03-01')],
    )
    assert len(mhm) == 2
    mhm.reset()
    assert len(mhm) == 0


def test_default_target_times_use_monthly_offset():
    """target_times=None 时, 走 dateutil.relativedelta 按月递推."""
    mhm = MultiHorizonMonitor(model_id='m', horizons=[1, 3, 6])
    ft = pd.Timestamp('2026-01-15')
    mhm.record_forecast(
        forecast_time=ft, predictions=[1.0, 2.0, 3.0],
    )
    # h=1 应该是 2026-02-15, h=3 是 2026-04-15, h=6 是 2026-07-15
    assert mhm.per_horizon[1].get_record(pd.Timestamp('2026-02-15')) is not None
    assert mhm.per_horizon[3].get_record(pd.Timestamp('2026-04-15')) is not None
    assert mhm.per_horizon[6].get_record(pd.Timestamp('2026-07-15')) is not None
