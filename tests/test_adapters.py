"""HPFAdapter 冒烟测试 —— 确认 src-layout 下业务层可被 configs/HPFConfig 正确驱动."""
import numpy as np
import pandas as pd
import pytest

from configs.hpf import HPFConfig
from tsf_frame.business.hpf_adapter import HPFAdapter


@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    rng = pd.date_range('2020-01-31', periods=36, freq='ME')
    rs = np.random.RandomState(0)
    return pd.DataFrame({
        'date': rng,
        'monthly_deposit': np.linspace(1.0e8, 3.0e8, 36) + rs.normal(0, 1e6, 36),
        'monthly_withdrawal': np.linspace(5.0e7, 1.5e8, 36),
        'loan_balance': np.linspace(1.0e9, 2.5e9, 36),
        'depositor_count': np.linspace(5e5, 7e5, 36),
    })


@pytest.fixture()
def adapter() -> HPFAdapter:
    cfg = HPFConfig()
    return HPFAdapter(cfg.to_adapter_config())


def test_preprocess_returns_dataframe_and_metadata(adapter, tiny_df):
    processed, meta = adapter.preprocess(tiny_df)
    assert isinstance(processed, pd.DataFrame)
    assert len(processed) == len(tiny_df)
    assert isinstance(meta, dict)


def test_business_metrics_perfect_prediction(adapter, tiny_df):
    y_true = tiny_df[['monthly_deposit']].reset_index(drop=True)
    y_pred = y_true.copy()
    metrics = adapter.get_business_metrics(y_true, y_pred)
    assert isinstance(metrics, dict)
    # 至少应该返回一个 mape 维度
    assert any('mape' in k.lower() for k in metrics.keys())
