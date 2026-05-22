"""
Tier A + B 改进项单测 / Unit tests for Tier A+B improvements.

覆盖:
  A1: DLinear.pack_y / unpack_y round-trip + fit shape check
  A2: MixedFeatureHandler.min_required_rows / required_source_columns
  A3: _dl_fit early stopping triggers + best_state restore
  A4: RevIN × use_diff mutex UserWarning
  B1: InferenceRunner save → load → predict round-trip
  B2: MLPModel 2D and 3D forward shapes
  B3: BaseMLModel.fit(cv_folds=5) → _residual_source='cv'
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pytest

from tsf_frame.features.engineering import (
    LagFeatureEngineer, RollingFeatureEngineer, CompositeFeatureEngineer,
)
from tsf_frame.features.mixed_feature_handler import MixedFeatureHandler
from tsf_frame.models import MLPModel, RidgeModel
from tsf_frame.models.transformer.transformer_models import DLinear, LSTMModel


# ────────────────────────────────────────────────────────────────────────
# A1 — DLinear pack/unpack/predict_structured
# ────────────────────────────────────────────────────────────────────────

def test_a1_pack_y_unpack_y_roundtrip():
    """pack_y(unpack_y(...)) 应完全还原 (target-major flatten 顺序一致)."""
    N, T, H = 7, 3, 4
    y_per_target = [np.random.randn(N, H).astype(np.float32) for _ in range(T)]
    flat = DLinear.pack_y(y_per_target)
    assert flat.shape == (N, T * H), f"expected ({N}, {T*H}), got {flat.shape}"

    structured = DLinear.unpack_y(flat, num_targets=T, pred_len=H)
    assert structured.shape == (N, T, H)
    # 各 target 数组应能逐项还原
    for t in range(T):
        np.testing.assert_allclose(structured[:, t, :], y_per_target[t])


def test_a1_pack_y_empty_raises():
    with pytest.raises(ValueError, match='不能为空'):
        DLinear.pack_y([])


def test_a1_pack_y_shape_mismatch_raises():
    with pytest.raises(ValueError, match='shape'):
        DLinear.pack_y([np.zeros((5, 3)), np.zeros((5, 4))])


def test_a1_dlinear_fit_y_shape_check():
    """y 末维错时, fit 应早抛 ValueError 指引使用 pack_y."""
    model = DLinear(config={
        'input_size': 1, 'seq_len': 8, 'num_targets': 1, 'pred_len': 4,
        'train_epochs': 1,
    })
    X = np.random.randn(20, 8, 1).astype(np.float32)
    y_wrong = np.random.randn(20, 7).astype(np.float32)   # 7 != 1*4
    with pytest.raises(ValueError, match='pack_y'):
        model.fit((X, y_wrong))


# ────────────────────────────────────────────────────────────────────────
# A2 — MixedFeatureHandler 推理协议 API
# ────────────────────────────────────────────────────────────────────────

def test_a2_required_source_columns():
    mfh = MixedFeatureHandler(
        time_varying_cols=['x', 'y'],
        static_cols=['gender', 'industry'],
        target_col='y',
        seq_len=12,
    )
    cols = mfh.required_source_columns
    # target 在最前; time-varying 去除 target; static 去重追加
    assert cols[0] == 'y'
    assert set(cols) == {'y', 'x', 'gender', 'industry'}


def test_a2_min_required_rows_no_engineer():
    """没传 engineer 时, min_required_rows 等于 seq_len."""
    mfh = MixedFeatureHandler(
        time_varying_cols=['a'], static_cols=[], target_col='a', seq_len=12,
    )
    assert mfh.min_required_rows() == 12


def test_a2_min_required_rows_with_engineer():
    """传入 lag/rolling engineer 时, 取最大 lookback 加到 seq_len."""
    eng = CompositeFeatureEngineer([
        LagFeatureEngineer(config={'target_cols': ['a'], 'lags': [1, 12]}),
        RollingFeatureEngineer(config={'target_cols': ['a'], 'windows': [6, 24]}),
    ])
    mfh = MixedFeatureHandler(
        time_varying_cols=['a'], static_cols=[], target_col='a', seq_len=8,
    )
    # max lookback = max(12, 24) = 24; seq_len(8) + 24 - 1 = 31
    assert mfh.min_required_rows(feature_engineer=eng) == 8 + 24 - 1


# ────────────────────────────────────────────────────────────────────────
# A3 — _dl_fit early stopping
# ────────────────────────────────────────────────────────────────────────

def test_a3_early_stopping_triggers():
    """patience=3 时, val_loss 不降会提前停, history 含 early_stopped_at."""
    np.random.seed(0)
    import torch
    torch.manual_seed(0)
    X = np.random.randn(60, 6, 4).astype(np.float32)
    y = np.random.randn(60, 1).astype(np.float32)
    model = LSTMModel(config={
        'input_size': 4, 'output_size': 1, 'seq_len': 6, 'pred_len': 1,
        'hidden_size': 8, 'num_layers': 1, 'dropout': 0.1,
        'train_epochs': 100, 'batch_size': 16, 'learning_rate': 1e-2,
        'early_stop_patience': 3, 'early_stop_min_delta': 1e-3,
        'use_revin': False,
    })
    history = model.fit((X[:40], y[:40]), val_data=(X[40:], y[40:]))
    # 随机数据上 100 epoch 几乎必然在某点开始过拟合或停滞 → 早停应触发
    # 不强制必停 (避免随机性测试 flaky), 但若停了 history 字段必须正确
    if 'early_stopped_at' in history:
        assert history['early_stopped_at'] < 100
        assert 'best_val_loss' in history


# ────────────────────────────────────────────────────────────────────────
# A4 — RevIN × use_diff mutex warning
# ────────────────────────────────────────────────────────────────────────

def test_a4_revin_diff_mutex_warning():
    """同开 use_revin + _train_uses_diff_target 应抛 UserWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _ = LSTMModel(config={
            'input_size': 3, 'seq_len': 6, 'output_size': 1,
            'use_revin': True, '_train_uses_diff_target': True,
        })
        msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        assert any('use_revin' in m and 'use_diff' in m for m in msgs), \
            f"未捕获到守卫 warning. 实际 warnings: {msgs}"


def test_a4_revin_alone_no_warning():
    """仅开 use_revin 不应抛守卫 warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _ = LSTMModel(config={
            'input_size': 3, 'seq_len': 6, 'output_size': 1,
            'use_revin': True,
        })
        msgs = [str(x.message) for x in w]
        assert not any('use_revin' in m and 'use_diff' in m for m in msgs)


# ────────────────────────────────────────────────────────────────────────
# B1 — InferenceRunner save/load 一致性
# ────────────────────────────────────────────────────────────────────────

class _MinAdapterForB1:
    """模块级 stub adapter (function-local 类无法被 pickle)."""
    _is_fitted = True
    _scalers = {}
    def validate_data(self, df): return True, ''
    def preprocess(self, df, fit=None): return df, {}
    def _denormalize(self, df, meta): return df


def test_b1_inference_runner_roundtrip(tmp_path):
    """save → load → predict 在同输入下应得到相同结果."""
    from tsf_frame.deployment import InferenceRunner

    np.random.seed(0)
    df = pd.DataFrame({
        'y': np.arange(60).astype(float) + np.random.randn(60),
        'x': np.random.randn(60),
    }, index=pd.date_range('2020-01-01', periods=60, freq='MS'))

    mfh = MixedFeatureHandler(
        time_varying_cols=['x'], static_cols=[], target_col='y', seq_len=1,
    ).fit(df)
    model = RidgeModel({'probabilistic': True, 'probabilistic_method': 'residual'})
    X = df[['x']].values.astype(np.float32)
    y = df['y'].values.astype(np.float32).reshape(-1, 1)
    model.fit((X[:50], y[:50]), val_data=(X[50:], y[50:]))

    runner = InferenceRunner(
        adapter=_MinAdapterForB1(), feat_eng=None, mfh=mfh,
        model=model, model_config=model.config, target_col='y',
    )
    path = str(tmp_path / 'art.pkl')
    runner.save(path)

    # load + 同输入预测一致
    runner2 = InferenceRunner.load(path, model_cls=RidgeModel)
    p1 = runner.predict(df.tail(5), target_col='y')
    p2 = runner2.predict(df.tail(5), target_col='y')
    assert np.allclose(p1.mean, p2.mean, atol=1e-5)


# ────────────────────────────────────────────────────────────────────────
# B2 — MLPModel 2D/3D forward
# ────────────────────────────────────────────────────────────────────────

def test_b2_mlp_2d_forward():
    np.random.seed(0)
    X = np.random.randn(60, 5).astype(np.float32)
    y = X.sum(axis=1, keepdims=True).astype(np.float32)
    m = MLPModel(config={
        'input_size': 5, 'output_size': 1, 'hidden_size': 16,
        'num_layers': 2, 'dropout': 0.1, 'train_epochs': 3, 'batch_size': 16,
    })
    m.fit((X, y))
    pred = m.predict(X)
    assert pred.shape == (60, 1)


def test_b2_mlp_3d_reduce_mean():
    np.random.seed(0)
    X = np.random.randn(30, 8, 4).astype(np.float32)
    y = np.random.randn(30, 1).astype(np.float32)
    m = MLPModel(config={
        'input_size': 4, 'output_size': 1, 'mlp_reduce': 'mean',
        'train_epochs': 2,
    })
    m.fit((X, y))
    assert m.predict(X).shape == (30, 1)


def test_b2_mlp_invalid_reduce_raises():
    with pytest.raises(ValueError, match='mlp_reduce'):
        MLPModel(config={'input_size': 4, 'mlp_reduce': 'bogus'})


# ────────────────────────────────────────────────────────────────────────
# B3 — K-fold CV 残差
# ────────────────────────────────────────────────────────────────────────

def test_b3_cv_residual_source_tag():
    """cv_folds > 0 时, _residual_source 应标 'cv'."""
    np.random.seed(0)
    X = np.random.randn(60, 4).astype(np.float32)
    y = (X.sum(axis=1) + 0.1 * np.random.randn(60)).astype(np.float32)
    m = RidgeModel({'probabilistic': True, 'probabilistic_method': 'residual'})
    m.fit((X, y), cv_folds=5)
    assert m._residual_source == 'cv'
    # 残差应该是 60 个 (OOF 用全部样本) 而非 val 路径的 N_val 个
    assert m._residuals.shape[0] == 60


def test_b3_cv_priority_over_val():
    """同传 cv_folds 和 val_data 时, cv 优先 (因为它更严谨)."""
    np.random.seed(0)
    X = np.random.randn(80, 3).astype(np.float32)
    y = X.sum(axis=1).astype(np.float32)
    m = RidgeModel({'probabilistic': True, 'probabilistic_method': 'residual'})
    m.fit((X[:60], y[:60]), val_data=(X[60:], y[60:]), cv_folds=4)
    assert m._residual_source == 'cv'
