"""
经典机器学习模型模块 / Classical ML models module

包含 11 个基于 sklearn / xgboost / lightgbm / catboost 的时序预测模型,
通过统一的 BaseMLModel 接口与主框架交互。
Contains 11 classical ML models (sklearn/xgboost/lightgbm/catboost)
exposed via a unified BaseMLModel interface.

对外接口 / Public API:
    get_ml_model(name, config) -> BaseMLModel
    MODEL_REGISTRY: {name: class} 模型注册表 / Name-to-class registry
    BaseMLModel: 统一基类,自定义模型需继承 / Base class for custom models
"""

from .ml_models import get_ml_model, MODEL_REGISTRY, BaseMLModel

__all__ = ['get_ml_model', 'MODEL_REGISTRY', 'BaseMLModel']
