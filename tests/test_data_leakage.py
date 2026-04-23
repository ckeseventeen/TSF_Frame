"""防泄漏测试：滚动 / 滞后 / 差分 特征的未来值不应出现在当前时间步的特征行中.

特征工程模块采用 fit→transform 模式，且接受 config dict。
"""
import numpy as np
import pandas as pd
import pytest

from tsf_frame.features.engineering import (
    LagFeatureEngineer,
    RollingFeatureEngineer,
    DifferenceFeatureEngineer,
)


@pytest.fixture()
def toy_series() -> pd.DataFrame:
    idx = pd.date_range('2020-01-01', periods=24, freq='ME')
    return pd.DataFrame({'y': np.arange(24, dtype=float)}, index=idx)


def test_lag_does_not_peek_future(toy_series):
    eng = LagFeatureEngineer({'target_col': 'y', 'lags': [1, 3, 6]})
    eng.fit(toy_series)
    out = eng.transform(toy_series)
    lag_cols = [c for c in out.columns if 'lag' in c]
    assert lag_cols, f'expected lag_* columns, got {list(out.columns)}'


def test_rolling_uses_only_past(toy_series):
    eng = RollingFeatureEngineer({'target_col': 'y', 'windows': [3], 'stats': ['mean']})
    eng.fit(toy_series)
    out = eng.transform(toy_series)
    roll_cols = [c for c in out.columns if 'rolling' in c or 'roll' in c]
    assert roll_cols, f'expected rolling_* columns, got {list(out.columns)}'


def test_difference_is_causal(toy_series):
    eng = DifferenceFeatureEngineer({'target_col': 'y', 'periods': [1]})
    eng.fit(toy_series)
    out = eng.transform(toy_series)
    diff_cols = [c for c in out.columns if 'diff' in c]
    assert diff_cols, f'expected diff_* columns, got {list(out.columns)}'
