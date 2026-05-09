"""MetricsCalculator 基础契约测试."""
import numpy as np
import pytest

from tsf_frame.utils.metrics import MetricsCalculator


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = MetricsCalculator.calculate_all(y, y)
    assert out['MAE'] == pytest.approx(0.0)
    assert out['RMSE'] == pytest.approx(0.0)
    assert out['MAPE'] == pytest.approx(0.0)
    assert out['R2'] == pytest.approx(1.0)


def test_constant_prediction_has_positive_error():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, fill_value=y_true.mean())
    out = MetricsCalculator.calculate_all(y_true, y_pred)
    assert out['MAE'] > 0
    assert out['RMSE'] > 0


def test_mape_returns_decimal_not_percentage():
    """
    锁定 MAPE 量纲为**小数** (0.05 = 5%), 与 monitoring 模块统一.
    历史上本模块返回 5.0(百分比), 已统一为小数 0.05.
    """
    y_true = np.array([100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([105.0, 105.0, 105.0, 105.0])  # 一律高估 5%
    mape = MetricsCalculator.mape(y_true, y_pred)
    assert mape == pytest.approx(0.05), (
        f'MAPE 应返回小数 0.05 (5%), 实际 {mape}; '
        f'若得到 5.0 说明量纲又退回百分比形式了'
    )


def test_smape_returns_decimal_not_percentage():
    """同 MAPE — SMAPE 也返回小数."""
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([100.0, 100.0])
    assert MetricsCalculator.smape(y_true, y_pred) == pytest.approx(0.0)
    # 一致性差异: |100-200|/((100+200)/2) = 100/150 = 0.667
    y_pred2 = np.array([200.0, 200.0])
    assert MetricsCalculator.smape(y_true, y_pred2) == pytest.approx(2 / 3, rel=1e-3)
