"""
Edge case tests / 边界条件测试.

覆盖框架对"非主流"输入的处理:
  - empty DataFrame
  - 单行 DataFrame
  - 整列 NaN
  - dtype 边界(全 int / 含日期列 / 含 object 列)

针对 HPFAdapter / FeatureEngineer / MixedFeatureHandler 三处的行为兜底.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.features.engineering import (
    LagFeatureEngineer, RollingFeatureEngineer, CompositeFeatureEngineer,
    create_feature_engineer,
)
from tsf_frame.features.mixed_feature_handler import MixedFeatureHandler


# ────────────────────────────────────────────────────────────────────
# HPFAdapter — 边界输入
# ────────────────────────────────────────────────────────────────────

def _make_adapter() -> HPFAdapter:
    return HPFAdapter({
        'target_columns': ['monthly_deposit'],
        'normalization': 'zscore',
        'handle_outliers': False,
    })


def test_adapter_preprocess_empty_dataframe():
    """空 DataFrame: preprocess 应能跑通(不抛异常), 但 _scalers 应为空 dict."""
    adapter = _make_adapter()
    df = pd.DataFrame({'monthly_deposit': [], 'date': []}).set_index('date')
    df['monthly_deposit'] = df['monthly_deposit'].astype(float)
    processed, meta = adapter.preprocess(df, fit=True)
    assert len(processed) == 0
    # 空数据的 scaler: std 为 nan, mean 为 nan(pandas 行为), 但不应崩
    assert 'monthly_deposit' in meta['scalers']


def test_adapter_preprocess_single_row():
    """单行: pandas .std() on N=1 returns NaN; framework should not crash."""
    adapter = _make_adapter()
    df = pd.DataFrame({
        'monthly_deposit': [100.0],
    }, index=pd.date_range('2026-01-01', periods=1, freq='MS'))
    processed, meta = adapter.preprocess(df, fit=True)
    # 单行 std=NaN, (x - mean) / (NaN + eps) = NaN — 框架不应崩, 输出可以是 NaN
    assert len(processed) == 1
    assert 'monthly_deposit' in meta['scalers']


def test_adapter_validate_data_rejects_negative_in_non_negative_col():
    adapter = _make_adapter()
    df = pd.DataFrame({
        'monthly_deposit': [100.0, -50.0, 200.0],
    }, index=pd.date_range('2026-01-01', periods=3, freq='MS'))
    ok, msg = adapter.validate_data(df)
    assert ok is False
    assert 'monthly_deposit' in msg or '负' in msg or 'negative' in msg.lower()


def test_adapter_preprocess_fit_false_without_prior_fit_raises():
    adapter = _make_adapter()
    df = pd.DataFrame({
        'monthly_deposit': [100.0, 110.0, 120.0],
    }, index=pd.date_range('2026-01-01', periods=3, freq='MS'))
    with pytest.raises(RuntimeError, match='fit'):
        adapter.preprocess(df, fit=False)


# ────────────────────────────────────────────────────────────────────
# FeatureEngineer — NaN-only / 单行 / 空 df
# ────────────────────────────────────────────────────────────────────

def test_lag_engineer_handles_single_row():
    """单行 + lags=[1,2]: 全部 lag 列应为 NaN(没有历史)."""
    eng = LagFeatureEngineer(config={'target_cols': ['y'], 'lags': [1, 2]})
    df = pd.DataFrame({'y': [10.0]}, index=pd.date_range('2026-01-01', periods=1, freq='MS'))
    eng.fit(df)
    out = eng.transform(df)
    assert out['y_lag_1'].isna().all()
    assert out['y_lag_2'].isna().all()


def test_rolling_engineer_handles_nan_only_column():
    """target_col 全 NaN 时, rolling 应保持 NaN, 不抛异常."""
    eng = RollingFeatureEngineer(config={
        'target_cols': ['y'], 'windows': [3], 'stats': ['mean'],
    })
    df = pd.DataFrame({
        'y': [np.nan, np.nan, np.nan, np.nan, np.nan],
    }, index=pd.date_range('2026-01-01', periods=5, freq='MS'))
    eng.fit(df)
    out = eng.transform(df)
    assert 'y_roll_3_mean' in out.columns
    # 全 NaN 输入 → 全 NaN 输出
    assert out['y_roll_3_mean'].isna().all()


def test_composite_engineer_with_empty_df():
    """空 df: composite 应能跑通空管道(无 ValueError), 输出仍为空."""
    eng = CompositeFeatureEngineer([
        LagFeatureEngineer(config={'target_cols': ['y'], 'lags': [1]}),
    ])
    df = pd.DataFrame({'y': pd.Series([], dtype=float)},
                      index=pd.DatetimeIndex([], name='date'))
    eng.fit(df)
    out = eng.transform(df)
    assert len(out) == 0
    assert 'y_lag_1' in out.columns


def test_create_feature_engineer_factory_returns_composite():
    """工厂函数应返回 CompositeFeatureEngineer."""
    eng = create_feature_engineer(['lag', 'rolling'], config={
        'lag_config': {'target_cols': ['y'], 'lags': [1]},
        'rolling_config': {'target_cols': ['y'], 'windows': [3], 'stats': ['mean']},
    })
    assert isinstance(eng, CompositeFeatureEngineer)
    assert len(eng.engineers) == 2


# ────────────────────────────────────────────────────────────────────
# MixedFeatureHandler — 边界 + 推理协议 API
# ────────────────────────────────────────────────────────────────────

def test_mfh_create_single_input_rejects_short_df():
    """少于 seq_len 行时, create_single_input 应抛 ValueError."""
    mfh = MixedFeatureHandler(
        time_varying_cols=['x'], static_cols=[], target_col='x', seq_len=12,
    )
    df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
    mfh.fit(df)
    with pytest.raises(ValueError, match='12'):   # 错误消息含"至少 12 行"或"requires at least 12"
        mfh.create_single_input(df)


def test_mfh_required_source_columns_dedups_target():
    """target_col 同时在 time_varying_cols 时, required_source_columns 不重复."""
    mfh = MixedFeatureHandler(
        time_varying_cols=['y', 'x', 'z'],
        static_cols=['region'],
        target_col='y',
        seq_len=12,
    )
    cols = mfh.required_source_columns
    # 期望: y (target 在前), x, z, region — 无重复
    assert cols == ['y', 'x', 'z', 'region']
    assert len(cols) == len(set(cols))


def test_mfh_min_required_rows_without_engineer_equals_seq_len():
    mfh = MixedFeatureHandler(
        time_varying_cols=['x'], static_cols=[], target_col='x', seq_len=24,
    )
    assert mfh.min_required_rows() == 24


def test_mfh_min_required_rows_with_composite_engineer():
    """min_required_rows = seq_len + max(lag/rolling/diff) - 1."""
    eng = CompositeFeatureEngineer([
        LagFeatureEngineer(config={'target_cols': ['x'], 'lags': [1, 6]}),
        RollingFeatureEngineer(config={'target_cols': ['x'], 'windows': [12, 24], 'stats': ['mean']}),
    ])
    mfh = MixedFeatureHandler(
        time_varying_cols=['x'], static_cols=[], target_col='x', seq_len=12,
    )
    # max lookback = max(1, 6, 12, 24) = 24; min_rows = 12 + 24 - 1 = 35
    assert mfh.min_required_rows(feature_engineer=eng) == 35


def test_mfh_fit_with_only_static_columns():
    """全是静态列(time_varying 单 target)时 fit 正常."""
    mfh = MixedFeatureHandler(
        time_varying_cols=['y'], static_cols=['gender', 'region'],
        target_col='y', seq_len=3,
    )
    df = pd.DataFrame({
        'y': [10.0, 20.0, 30.0],
        'gender': ['M', 'M', 'M'],
        'region': ['A', 'A', 'A'],
    })
    mfh.fit(df)
    assert mfh._is_fitted
    # static_values 取第一行的静态列
    assert mfh.static_values is not None
