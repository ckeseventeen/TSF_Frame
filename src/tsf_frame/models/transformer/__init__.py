"""
深度学习时序模型模块 / Deep-learning time-series models module

包含 LSTM + Transformer 系列(Transformer / Informer / Autoformer /
iTransformer / TimesNet),继承 BaseModel 提供统一的 fit / predict /
predict_probabilistic(MC Dropout) 接口。
Contains LSTM + Transformer families (Transformer/Informer/Autoformer/
iTransformer/TimesNet), all derived from BaseModel with unified
fit/predict/MC-Dropout interfaces.

对外接口 / Public API:
    get_dl_model(name, config) -> BaseModel
    DL_MODEL_REGISTRY: {name: class} 深度学习模型注册表 / Registry map
"""

from .transformer_models import get_dl_model, DL_MODEL_REGISTRY

__all__ = ['get_dl_model', 'DL_MODEL_REGISTRY']
