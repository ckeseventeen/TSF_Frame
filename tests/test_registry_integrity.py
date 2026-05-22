"""
Registry integrity tests / 模型注册表完整性测试.

验证 MODEL_REGISTRY 和 DL_MODEL_REGISTRY 包含所有已实装模型, 且每个模型可以
用最小化 config 实例化(catches 注册漏接 / __init__ 签名漂移).
"""

from __future__ import annotations

import pytest

from tsf_frame.models.classical.ml_models import MODEL_REGISTRY, get_ml_model
from tsf_frame.models.transformer.transformer_models import (
    DL_MODEL_REGISTRY, get_dl_model,
)
# 触发 MLPModel 自注册到 DL_MODEL_REGISTRY['mlp']
from tsf_frame.models import MLPModel  # noqa: F401


# 期望注册的名字 — 如果新增模型, 这两个 set 也要同步更新
EXPECTED_ML_MODELS = {
    'linear_regression', 'ridge', 'lasso',
    'random_forest', 'gradient_boosting', 'decision_tree',
    'xgboost', 'lightgbm', 'catboost',
    'svr', 'knn',
}
EXPECTED_DL_MODELS = {
    'lstm', 'transformer', 'autoformer', 'itransformer', 'timesnet', 'dlinear',
    'mlp',   # 来自 tabular/mlp_model.py 的延迟注册
}


def test_ml_registry_count_and_names():
    assert set(MODEL_REGISTRY.keys()) == EXPECTED_ML_MODELS, (
        f"MODEL_REGISTRY 内容与期望不符. 实际: {set(MODEL_REGISTRY.keys())}, "
        f"期望: {EXPECTED_ML_MODELS}"
    )
    assert len(MODEL_REGISTRY) == 11


def test_dl_registry_count_and_names():
    assert set(DL_MODEL_REGISTRY.keys()) == EXPECTED_DL_MODELS, (
        f"DL_MODEL_REGISTRY 内容与期望不符. 实际: {set(DL_MODEL_REGISTRY.keys())}, "
        f"期望: {EXPECTED_DL_MODELS}"
    )
    assert len(DL_MODEL_REGISTRY) == 7   # 6 时序 + MLP


@pytest.mark.parametrize('name', sorted(EXPECTED_ML_MODELS))
def test_ml_model_can_instantiate(name):
    """每个 ML 模型可用最小 config 通过工厂函数实例化."""
    model = get_ml_model(name, {'random_seed': 42})
    assert model is not None
    assert model.model_name == name


# 各 DL 模型最小可用 config — input_size 必须 ≥ 1, seq_len ≥ 8 (TimesNet 要求)
_DL_MINIMAL_CONFIG = {
    'input_size': 3,
    'output_size': 1,
    'seq_len': 12,            # 满足 TimesNet >= 8 + Autoformer kernel_size guard
    'pred_len': 1,
    'd_model': 16,
    'nhead': 4,                # d_model % nhead == 0
    'num_layers': 1,
    'hidden_size': 16,
    'dim_feedforward': 32,
    'dropout': 0.1,
    'train_epochs': 1,
    'batch_size': 8,
    'device': 'cpu',
}


@pytest.mark.parametrize('name', sorted(EXPECTED_DL_MODELS))
def test_dl_model_can_instantiate(name):
    """每个 DL 模型可用最小 config 实例化(覆盖 _init_revin / optimizer 创建路径)."""
    cfg = dict(_DL_MINIMAL_CONFIG)
    if name == 'dlinear':
        # DLinear 多目标参数: num_targets ≤ input_size; 这里跑单目标
        cfg['num_targets'] = 1
    model = get_dl_model(name, cfg)
    assert model is not None
    # MLP 默认 use_revin=False, 其他默认 True
    if name == 'mlp':
        assert model.revin is None
    else:
        assert model.revin is not None, f"{name}: revin 默认应该开启"


def test_factory_rejects_unknown_name():
    with pytest.raises(ValueError, match='not found'):
        get_ml_model('nonexistent_model', {})
    with pytest.raises(ValueError, match='not found'):
        get_dl_model('nonexistent_model', {})
