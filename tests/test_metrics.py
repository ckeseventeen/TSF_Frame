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
