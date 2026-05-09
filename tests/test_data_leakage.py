"""
防泄漏测试 / Data-leakage tests.

锁定: 滚动 / 滞后 / 差分 特征**不应**包含未来值,t 时刻的特征只能由
[0, t] 区间内的原始值构造.

历史教训: 早期版本只断言"lag_X 列存在"(查列名), 没验证数值,
所以即使 transform 写错也能过测试. 本版本**全部用真实数值断言**.
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
    """y = [0, 1, 2, ..., 23] 单调递增, 异常值容易被 lag/rolling 暴露."""
    idx = pd.date_range('2020-01-01', periods=24, freq='ME')
    return pd.DataFrame({'y': np.arange(24, dtype=float)}, index=idx)


# ──────────────────────────────────────────────────────────────────────
# Lag — t 行 lag_k 列必须等于 t-k 行的原值, 且前 k 行为 NaN
# ──────────────────────────────────────────────────────────────────────

def test_lag_1_equals_prev_row(toy_series):
    """LagFeatureEngineer: lag_1 列在 t 行的值必须等于原 y 在 t-1 行的值."""
    eng = LagFeatureEngineer({'target_cols': ['y'], 'lags': [1, 3, 6]})
    eng.fit(toy_series)
    out = eng.transform(toy_series)

    # t=0 行(第 1 行)的 y_lag_1 应是 NaN (没有 t-1)
    assert pd.isna(out['y_lag_1'].iloc[0]), 't=0 的 lag_1 应该是 NaN'

    # t=5 行的 y_lag_1 应等于 t=4 行的 y 原值 (= 4.0)
    assert out['y_lag_1'].iloc[5] == 4.0, \
        f't=5 的 lag_1 应是 4.0, 实际 {out["y_lag_1"].iloc[5]}'

    # t=10 行的 y_lag_3 应等于 t=7 行的 y 原值 (= 7.0)
    assert out['y_lag_3'].iloc[10] == 7.0, \
        f't=10 的 lag_3 应是 7.0, 实际 {out["y_lag_3"].iloc[10]}'

    # t=20 行的 y_lag_6 应等于 t=14 行的 y 原值 (= 14.0)
    assert out['y_lag_6'].iloc[20] == 14.0


def test_lag_does_not_contain_future_values(toy_series):
    """t 行的 lag_k 不能包含 t 之后的任何值."""
    eng = LagFeatureEngineer({'target_cols': ['y'], 'lags': [1, 2, 3]})
    out = eng.fit_transform(toy_series)

    # 对每一行 t 和每个 lag k, 检查 lag_k[t] <= y[t-k] (单调递增序列下严格相等)
    for t in range(3, 24):
        for k in [1, 2, 3]:
            actual = out[f'y_lag_{k}'].iloc[t]
            expected = toy_series['y'].iloc[t - k]
            assert actual == expected, (
                f't={t}, lag={k}: 期望 {expected}, 得到 {actual} '
                f'(若得到 t 之后的值即是数据泄漏)'
            )


# ──────────────────────────────────────────────────────────────────────
# Rolling — t 行 rolling_w_mean 必须只用 [t-w+1, t] 范围内的原值
# ──────────────────────────────────────────────────────────────────────

def test_rolling_mean_only_uses_past_and_current(toy_series):
    """t 行 rolling 窗口的均值必须等于 [t-w+1, t] 切片的均值."""
    eng = RollingFeatureEngineer({
        'target_cols': ['y'], 'windows': [3], 'stats': ['mean', 'max'],
    })
    out = eng.fit_transform(toy_series)

    # 前 w-1 行 = 前 2 行应该是 NaN (样本不足)
    assert pd.isna(out['y_roll_3_mean'].iloc[0])
    assert pd.isna(out['y_roll_3_mean'].iloc[1])

    # t=2 行: rolling_3_mean = mean(y[0..2]) = mean(0, 1, 2) = 1.0
    assert out['y_roll_3_mean'].iloc[2] == pytest.approx(1.0)
    # t=2 行: rolling_3_max = max(0, 1, 2) = 2.0 (不是 3 — 不能看 future)
    assert out['y_roll_3_max'].iloc[2] == 2.0

    # t=10 行: rolling_3_mean = mean(8, 9, 10) = 9.0
    assert out['y_roll_3_mean'].iloc[10] == pytest.approx(9.0)
    # t=10 行: rolling_3_max = max(8, 9, 10) = 10.0 (不是 11 — 不能看 future)
    assert out['y_roll_3_max'].iloc[10] == 10.0


def test_rolling_max_does_not_peek_future(toy_series):
    """关键防泄漏: 单调递增序列下, rolling_max[t] 必须 <= y[t]."""
    eng = RollingFeatureEngineer({
        'target_cols': ['y'], 'windows': [5], 'stats': ['max'],
    })
    out = eng.fit_transform(toy_series)

    for t in range(5, 24):
        rolling_max = out['y_roll_5_max'].iloc[t]
        current_y = toy_series['y'].iloc[t]
        assert rolling_max <= current_y, (
            f't={t}: rolling_max={rolling_max} > y[t]={current_y}, '
            f'说明 rolling 看到了 t 之后的值 (数据泄漏!)'
        )


# ──────────────────────────────────────────────────────────────────────
# Difference — t 行 diff_k 必须等于 y[t] - y[t-k]
# ──────────────────────────────────────────────────────────────────────

def test_diff_1_equals_y_minus_prev(toy_series):
    """diff_1 在 t 行必须等于 y[t] - y[t-1] (这里始终 = 1.0, 单调递增 1)."""
    eng = DifferenceFeatureEngineer({
        'target_cols': ['y'], 'periods': [1, 7],
    })
    out = eng.fit_transform(toy_series)

    # t=0 行 diff_1 = NaN
    assert pd.isna(out['y_diff_1'].iloc[0])

    # t=1..23 行 diff_1 都应该 = 1.0 (步长 1 序列)
    for t in range(1, 24):
        assert out['y_diff_1'].iloc[t] == 1.0, \
            f't={t}: diff_1 应是 1.0, 实际 {out["y_diff_1"].iloc[t]}'

    # t=10 行 diff_7 = y[10] - y[3] = 10 - 3 = 7.0
    assert out['y_diff_7'].iloc[10] == 7.0


def test_diff_does_not_use_future(toy_series):
    """diff_k[t] 不能包含 t+k 时刻的值."""
    eng = DifferenceFeatureEngineer({'target_cols': ['y'], 'periods': [3]})
    out = eng.fit_transform(toy_series)

    for t in range(3, 24):
        actual = out['y_diff_3'].iloc[t]
        expected = toy_series['y'].iloc[t] - toy_series['y'].iloc[t - 3]
        assert actual == expected, (
            f't={t}: diff_3 应是 {expected}, 实际 {actual}'
        )


# ──────────────────────────────────────────────────────────────────────
# fit→transform 分离 — 测试集 transform 不能用测试集统计
# ──────────────────────────────────────────────────────────────────────

def test_fit_transform_separation_for_lag(toy_series):
    """
    LagFeatureEngineer 用 fit 学 target_cols, transform 应用. 测试集 transform
    不应该重新学习; 否则训练/测试列名可能不一致.
    """
    train = toy_series.iloc[:18]
    test = toy_series.iloc[18:]

    eng = LagFeatureEngineer({'target_cols': ['y'], 'lags': [1, 3]})
    eng.fit(train)
    train_out = eng.transform(train)
    test_out = eng.transform(test)

    # 列结构必须一致
    assert list(train_out.columns) == list(test_out.columns), \
        '训练集与测试集 transform 后列结构不一致'
    # 测试集第一行的 lag_1 不应该用训练集最后一行的值 (因为 transform 单独跑)
    # 实际行为: shift 会让 test 的第一行 lag_1 = NaN (因为测试集独立)
    # 这是符合"transform 不引入跨集统计"的设计
    assert pd.isna(test_out['y_lag_1'].iloc[0])
