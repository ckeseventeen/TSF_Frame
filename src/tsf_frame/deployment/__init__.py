"""
tsf_frame.deployment — 生产部署工具 / Production deployment utilities.

提供端到端的 save / load / predict wrapper, 把训练时的 5 个状态
(adapter / feature_engineer / handler / model / config) 打包成单一 artifact,
推理侧一行 ``InferenceRunner.load(path).predict(latest_df)`` 即可出预测.
"""

from .inference_runner import InferenceRunner

__all__ = ['InferenceRunner']
