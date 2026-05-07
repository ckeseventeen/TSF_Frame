"""
监控模块核心契约测试 / Monitoring core contract tests.

锁定本次重构的关键不变式, 防止回归:

1. **PerformanceMonitor 用 target_ts 对齐, 不靠位置索引**
   — 窗口满 + 真值乱序回填仍正确.
2. **metric_window 与 window_size 解耦**
   — 默认 12, 可超过 window_size 时自动 clamp.
3. **ModelMonitor.settle_actual 异步真值闭环**
   — perf 回填 + concept_drift 残差 + last_actual 更新.
4. **SQLiteStore / InMemoryStore 读写一致**
   — insert → query roundtrip 数据无损.
5. **RuleEngine 内建规则触发**
   — R1_NON_NEGATIVE 在负预测时产生 violation.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest

from tsf_frame.monitoring import (
    AlertLevel,
    InMemoryStore,
    ModelMonitor,
    MultiHorizonMonitor,
    MultiTargetMonitor,
    PerformanceMonitor,
    RuleEngine,
    SQLiteStore,
)


# ──────────────────────────────────────────────────────────────────────
# 1. PerformanceMonitor 对齐 (本次重构的核心 bug)
# ──────────────────────────────────────────────────────────────────────

def test_perf_target_ts_alignment_after_window_full():
    """
    窗口满后真值乱序到达, 必须按 target_ts 精确回填,
    被淘汰的 target_ts 应返回 False 而非错位填充.
    """
    pm = PerformanceMonitor('test', window_size=5, metric_window=5)
    ts = [datetime(2026, m, 1) for m in range(1, 13)]

    # 12 个月预测, 窗口 5 → 只保留最后 5 个 (8月~12月)
    for i, t in enumerate(ts):
        pm.update(y_pred=100.0 + i, timestamp=t)

    assert len(pm) == 5, '窗口满后应只保留 maxlen 条'

    # 已淘汰 (3月) 的 target_ts → 拒绝回填
    assert pm.fill_actual(target_ts=ts[2], y_true=999.0) is False
    # 仍在窗口 (9月) → 正常回填
    assert pm.fill_actual(target_ts=ts[8], y_true=108.5) is True

    # 关键: 9月的真值应该和 9月的预测 (i=8 → pred=108) 配对, 残差 = 0.5
    metrics = pm.current()
    assert metrics['mae'] == pytest.approx(0.5), \
        f'9月真值108.5 vs 预测108 → MAE=0.5; got {metrics["mae"]}'


def test_perf_metric_window_default_is_12():
    """metric_window 默认为 12."""
    pm = PerformanceMonitor('test', window_size=100)
    assert pm.metric_window == 12


def test_perf_metric_window_clamps_when_too_large():
    """metric_window > window_size 时自动 clamp + 警告."""
    with pytest.warns(RuntimeWarning, match='超过 window_size'):
        pm = PerformanceMonitor('test', window_size=10, metric_window=99)
    assert pm.metric_window == 10


def test_perf_metric_window_zero_raises():
    """metric_window <= 0 抛 ValueError."""
    with pytest.raises(ValueError, match='必须 > 0'):
        PerformanceMonitor('test', metric_window=0)


def test_perf_current_window_override():
    """current(window=N) 临时覆盖默认 metric_window."""
    pm = PerformanceMonitor('test', window_size=20, metric_window=10)
    base_ts = datetime(2026, 1, 1)
    # 前 10 个误差 0, 后 10 个误差 5
    for i in range(20):
        err = 0.0 if i < 10 else 5.0
        pm.update(
            y_pred=100.0 + err, y_true=100.0,
            timestamp=base_ts + timedelta(days=i),
        )
    # 默认 window=10 → 只看后 10 个 → MAE=5
    assert pm.current()['mae'] == pytest.approx(5.0)
    # 覆盖 window=20 → 看全部 → MAE=2.5
    assert pm.current(window=20)['mae'] == pytest.approx(2.5)


# ──────────────────────────────────────────────────────────────────────
# 2. ModelMonitor 异步真值闭环
# ──────────────────────────────────────────────────────────────────────

def test_model_monitor_settle_actual_roundtrip():
    """
    record_prediction(actual=None) → settle_actual(target_ts, y_actual)
    应正确触发 perf 回填 + store update + last_actual 更新.
    """
    store = InMemoryStore()
    mon = ModelMonitor(
        model_id='m1',
        store=store,
        window_size=12,
        metric_window=6,
        cold_start_samples=1,  # 加速测试
    )
    target_ts = datetime(2026, 4, 1)
    mon.record_prediction(timestamp=target_ts, prediction=100.0)
    # 此时 perf 里 y_true 是 None
    assert mon.perf._records[target_ts].y_true is None
    assert mon._last_actual is None

    # 真值到达
    ok = mon.settle_actual(target_ts=target_ts, y_actual=105.0)
    assert ok is True
    assert mon.perf._records[target_ts].y_true == 105.0
    assert mon._last_actual == 105.0


def test_model_monitor_settle_actual_after_evicted():
    """target_ts 已被淘汰出 perf 窗口时, settle_actual 仍应尝试 store update."""
    store = InMemoryStore()
    mon = ModelMonitor(
        model_id='m2', store=store,
        window_size=2, metric_window=2, cold_start_samples=1,
    )
    base = datetime(2026, 1, 1)
    # 推 5 个, 窗口只剩 2 个 (3月, 4月)
    for h in range(1, 6):
        mon.record_prediction(
            timestamp=base + timedelta(days=h), prediction=100.0 + h,
        )
    # 1月已被淘汰
    old_ts = base + timedelta(days=1)
    ok = mon.settle_actual(target_ts=old_ts, y_actual=999.0)
    assert ok is False, '已淘汰的 target_ts 应返回 False'


# ──────────────────────────────────────────────────────────────────────
# 3. Stores 一致性
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('store_factory', [
    lambda: InMemoryStore(),
    lambda: SQLiteStore(
        db_path=os.path.join(tempfile.mkdtemp(prefix='tsf_test_'), 'mon.db')),
])
def test_store_prediction_roundtrip(store_factory):
    """两种 store 的 insert_prediction → query_predictions 数据一致."""
    store = store_factory()
    ts = datetime(2026, 4, 1)
    store.insert_prediction(
        model_id='mx', timestamp=ts, target='y',
        y_pred=100.5, y_lower=99.0, y_upper=102.0,
    )
    rows = store.query_predictions(model_id='mx')
    assert len(rows) == 1
    r = rows[0]
    assert r['y_pred'] == pytest.approx(100.5)
    assert r['y_lower'] == pytest.approx(99.0)
    assert r['y_upper'] == pytest.approx(102.0)
    assert r['y_actual'] is None


@pytest.mark.parametrize('store_factory', [
    lambda: InMemoryStore(),
    lambda: SQLiteStore(
        db_path=os.path.join(tempfile.mkdtemp(prefix='tsf_test_'), 'mon.db')),
])
def test_store_update_actual(store_factory):
    """update_actual 应能修改对应记录的 y_actual."""
    store = store_factory()
    ts = datetime(2026, 4, 1)
    store.insert_prediction(
        model_id='mx', timestamp=ts, target='y', y_pred=100.0,
    )
    store.update_actual(
        model_id='mx', timestamp=ts, target='y', y_actual=98.5,
    )
    rows = store.query_predictions(model_id='mx')
    assert rows[0]['y_actual'] == pytest.approx(98.5)


# ──────────────────────────────────────────────────────────────────────
# 4. RuleEngine 内建规则
# ──────────────────────────────────────────────────────────────────────

def test_rule_r1_non_negative_fires_on_negative_pred():
    eng = RuleEngine(rule_ids=['R1_NON_NEGATIVE'])
    viol = eng.check(prediction=[-5.0, 10.0, 20.0], context={'target': 'y'})
    assert len(viol) == 1
    assert viol[0].rule_id == 'R1_NON_NEGATIVE'
    assert viol[0].severity == AlertLevel.CRITICAL


def test_rule_r1_non_negative_silent_on_positive_pred():
    eng = RuleEngine(rule_ids=['R1_NON_NEGATIVE'])
    viol = eng.check(prediction=[10.0, 20.0, 30.0], context={'target': 'y'})
    assert viol == []


def test_rule_r2_sudden_change_fires():
    eng = RuleEngine(
        rule_ids=['R2_SUDDEN_CHANGE'],
        params={'R2_SUDDEN_CHANGE': {'max_ratio': 0.3}},
    )
    # 上月 100, 这月预测 200 → 100% 变化, 超 30%
    viol = eng.check(
        prediction=[200.0],
        context={'target': 'y', 'last_actual': 100.0},
    )
    assert len(viol) == 1
    assert viol[0].rule_id == 'R2_SUDDEN_CHANGE'


# ──────────────────────────────────────────────────────────────────────
# 5. MultiHorizonMonitor — 单模型多输出监控
# ──────────────────────────────────────────────────────────────────────

def test_multi_horizon_record_forecast_dispatches_to_each_pm():
    """record_forecast 应把每个 horizon 的预测送到对应 PerformanceMonitor."""
    mhm = MultiHorizonMonitor('m', horizons=[1, 3, 6, 12])
    forecast_time = datetime(2026, 4, 1)
    target_times = [datetime(2026, 5, 1), datetime(2026, 7, 1),
                    datetime(2026, 10, 1), datetime(2027, 4, 1)]
    preds = [100.0, 105.0, 110.0, 120.0]
    mhm.record_forecast(
        forecast_time=forecast_time,
        predictions=preds,
        target_times=target_times,
    )
    # 每个 horizon 各应记到 1 条
    for h in [1, 3, 6, 12]:
        assert len(mhm.per_horizon[h]) == 1
    # h=1 那条的 target_ts 应是 5 月
    rec = mhm.per_horizon[1]._records[target_times[0]]
    assert rec.y_pred == 100.0


def test_multi_horizon_settle_actual_hits_all_matching_horizons():
    """同一 target_ts 的真值, 应回填所有 horizon 中匹配的记录."""
    mhm = MultiHorizonMonitor('m', horizons=[1, 3])
    # 两次跑批: 4 月跑批 → h=1 是 5 月; 2 月跑批 → h=3 也是 5 月
    mhm.per_horizon[1].update(
        y_pred=100.0, timestamp=datetime(2026, 5, 1))
    mhm.per_horizon[3].update(
        y_pred=98.0, timestamp=datetime(2026, 5, 1))

    # 5 月真值到 → 两个 horizon 都应 settle
    result = mhm.settle_actual(
        target_ts=datetime(2026, 5, 1), y_actual=102.0)
    assert result == {1: True, 3: True}
    assert mhm.per_horizon[1]._records[datetime(2026, 5, 1)].y_true == 102.0
    assert mhm.per_horizon[3]._records[datetime(2026, 5, 1)].y_true == 102.0


def test_multi_horizon_per_horizon_metrics_are_independent():
    """各 horizon 的 MAPE 应独立计算 — h=1 准, h=12 不准."""
    mhm = MultiHorizonMonitor(
        'm', horizons=[1, 12], window_size=5, metric_window=5)
    base = datetime(2026, 1, 1)
    # 5 个目标月, 每个目标月有 h=1 和 h=12 两条记录(同一 target_ts)
    for i in range(5):
        target_ts = base + timedelta(days=i)
        # h=1 误差 1%, h=12 误差 10%
        mhm.per_horizon[1].update(
            y_pred=100.0, y_true=101.0, timestamp=target_ts)
        mhm.per_horizon[12].update(
            y_pred=100.0, y_true=110.0, timestamp=target_ts)
    metrics = mhm.current()
    # h=1 MAPE ≈ 1/101 ≈ 0.0099
    assert metrics[1]['mape'] == pytest.approx(1 / 101, rel=0.01)
    # h=12 MAPE ≈ 10/110 ≈ 0.0909
    assert metrics[12]['mape'] == pytest.approx(10 / 110, rel=0.01)


def test_multi_horizon_aggregated_weights_by_settled_count():
    """聚合时默认按各 horizon settled 样本数加权."""
    mhm = MultiHorizonMonitor('m', horizons=[1, 12])
    base = datetime(2026, 1, 1)
    # h=1 有 3 条已 settled (MAPE=0)
    for i in range(3):
        mhm.per_horizon[1].update(
            y_pred=100.0, y_true=100.0,
            timestamp=base + timedelta(days=i),
        )
    # h=12 只有 1 条已 settled (MAPE=0.5)
    mhm.per_horizon[12].update(
        y_pred=100.0, y_true=200.0, timestamp=base + timedelta(days=10),
    )
    # 聚合: 0*3/4 + 0.5*1/4 = 0.125
    agg = mhm.aggregated()
    assert agg['mape'] == pytest.approx(0.125, rel=0.01)


def test_multi_horizon_record_forecast_length_mismatch_raises():
    mhm = MultiHorizonMonitor('m', horizons=[1, 3, 6])
    with pytest.raises(ValueError, match='predictions 长度'):
        mhm.record_forecast(
            forecast_time=datetime(2026, 4, 1),
            predictions=[100.0, 110.0],  # 长度 2 ≠ horizons 数量 3
            target_times=[datetime(2026, 5, 1)] * 2,
        )


def test_multi_horizon_snapshot_contains_per_horizon_and_aggregated():
    mhm = MultiHorizonMonitor('m', horizons=[1, 6])
    base = datetime(2026, 1, 1)
    for h in [1, 6]:
        mhm.per_horizon[h].update(
            y_pred=100.0, y_true=100.0, timestamp=base)
    snap = mhm.snapshot()
    assert snap['model_id'] == 'm'
    assert snap['horizons'] == [1, 6]
    assert set(snap['per_horizon'].keys()) == {1, 6}
    assert 'mape' in snap['aggregated']
    assert snap['n_settled_per_horizon'] == {1: 1, 6: 1}


# ──────────────────────────────────────────────────────────────────────
# 6. MultiTargetMonitor — 单模型多目标(温度+湿度+...)
# ──────────────────────────────────────────────────────────────────────

def test_multi_target_record_dispatches_to_each_target():
    """record_prediction 把每个 target 送到对应 PerformanceMonitor."""
    mtm = MultiTargetMonitor(
        'weather', targets=['temperature', 'humidity', 'pressure'],
    )
    ts = datetime(2026, 4, 1)
    mtm.record_prediction(
        timestamp=ts,
        predictions={'temperature': 25.3, 'humidity': 60.0, 'pressure': 1013.2},
    )
    assert len(mtm.per_target['temperature']) == 1
    assert len(mtm.per_target['humidity']) == 1
    assert len(mtm.per_target['pressure']) == 1
    assert mtm.per_target['temperature']._records[ts].y_pred == 25.3


def test_multi_target_partial_predictions_only_record_provided():
    """只覆盖部分目标时, 其他目标本时刻不应有记录."""
    mtm = MultiTargetMonitor('w', targets=['temperature', 'humidity'])
    ts = datetime(2026, 4, 1)
    mtm.record_prediction(
        timestamp=ts, predictions={'temperature': 25.0},
    )
    assert len(mtm.per_target['temperature']) == 1
    assert len(mtm.per_target['humidity']) == 0


def test_multi_target_unknown_target_raises():
    mtm = MultiTargetMonitor('w', targets=['temperature'])
    with pytest.raises(ValueError, match='未声明的目标'):
        mtm.record_prediction(
            timestamp=datetime(2026, 4, 1),
            predictions={'humidity': 60.0},
        )


def test_multi_target_duplicate_targets_raises():
    with pytest.raises(ValueError, match='包含重复'):
        MultiTargetMonitor('w', targets=['x', 'y', 'x'])


def test_multi_target_settle_actuals_partial_arrival():
    """真值部分到达 (湿度晚到), 后续单独 settle 仍正确."""
    mtm = MultiTargetMonitor('w', targets=['temperature', 'humidity'])
    ts = datetime(2026, 4, 1)
    mtm.record_prediction(
        timestamp=ts,
        predictions={'temperature': 25.0, 'humidity': 60.0},
    )
    # 第一次只到温度
    r1 = mtm.settle_actuals(target_ts=ts, actuals={'temperature': 25.5})
    assert r1 == {'temperature': True}
    assert mtm.per_target['temperature']._records[ts].y_true == 25.5
    assert mtm.per_target['humidity']._records[ts].y_true is None
    # 后续湿度到了
    ok = mtm.settle_actual(target_ts=ts, target='humidity', y_actual=58.0)
    assert ok is True
    assert mtm.per_target['humidity']._records[ts].y_true == 58.0


def test_multi_target_per_target_baseline_independent():
    """每个目标的 baseline 应独立, 互不影响."""
    mtm = MultiTargetMonitor(
        'w', targets=['temperature', 'humidity'],
        baseline_per_target={
            'temperature': {'mape': 0.03},
            'humidity':    {'mape': 0.10},
        },
    )
    assert mtm.per_target['temperature'].baseline == {'mape': 0.03}
    assert mtm.per_target['humidity'].baseline == {'mape': 0.10}


def test_multi_target_probabilistic_only_for_specified():
    """只有 probabilistic_targets 列表里的目标才启用 PICP/MIW/Winkler."""
    mtm = MultiTargetMonitor(
        'w', targets=['temperature', 'humidity'],
        probabilistic_targets=['temperature'],
    )
    assert 'picp' in mtm.per_target['temperature'].metric_names
    assert 'picp' not in mtm.per_target['humidity'].metric_names


def test_multi_target_metrics_independent_after_settling():
    """各目标的 MAPE 应基于各自的真值/预测对独立计算."""
    mtm = MultiTargetMonitor('w', targets=['temperature', 'humidity'])
    base = datetime(2026, 1, 1)
    for i in range(5):
        ts = base + timedelta(days=i)
        mtm.record_prediction(
            timestamp=ts,
            predictions={'temperature': 25.0, 'humidity': 60.0},
            actuals={'temperature': 26.0, 'humidity': 90.0},
        )
    metrics = mtm.current()
    # 温度: |25-26|/26 = 0.0385
    assert metrics['temperature']['mape'] == pytest.approx(1 / 26, rel=0.01)
    # 湿度: |60-90|/90 = 0.333
    assert metrics['humidity']['mape'] == pytest.approx(30 / 90, rel=0.01)


def test_multi_target_snapshot_structure():
    mtm = MultiTargetMonitor(
        'w', targets=['temperature', 'humidity'],
    )
    ts = datetime(2026, 4, 1)
    mtm.record_prediction(
        timestamp=ts,
        predictions={'temperature': 25.0, 'humidity': 60.0},
        actuals={'temperature': 25.0},   # 仅温度有真值
    )
    snap = mtm.snapshot()
    assert snap['model_id'] == 'w'
    assert snap['targets'] == ['temperature', 'humidity']
    assert set(snap['per_target'].keys()) == {'temperature', 'humidity'}
    assert snap['n_settled_per_target'] == {'temperature': 1, 'humidity': 0}
