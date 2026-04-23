"""
基础模型模块 / Base model module

定义时序预测框架中所有模型的抽象基类和概率预测容器。
Defines the abstract base class for all models in the time-series
forecasting framework, along with a probabilistic prediction container.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn


class ProbabilisticPrediction:
    """
    概率预测结果容器 / Probabilistic prediction result container

    封装均值预测及其不确定性信息（置信区间、标准差、采样分布）。
    Wraps point predictions together with uncertainty information
    (confidence intervals, standard deviation, sample distribution).

    Attributes:
        mean: 均值预测 / Point prediction (mean).
        lower: 置信下界 / Lower bound of the confidence interval.
        upper: 置信上界 / Upper bound of the confidence interval.
        std: 标准差 / Standard deviation of predictions.
        samples: MC采样结果 / Raw Monte-Carlo samples, shape (S, N, output_size).
    """

    def __init__(self, mean: np.ndarray,
                 lower: Optional[np.ndarray] = None,
                 upper: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None,
                 samples: Optional[np.ndarray] = None):
        self.mean = mean
        self.lower = lower
        self.upper = upper
        self.std = std
        self.samples = samples

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典 / Convert to dictionary

        仅包含非空字段，便于序列化和日志记录。
        Only includes non-None fields for easy serialisation and logging.
        """
        result = {'mean': self.mean}
        if self.lower is not None:
            result['lower'] = self.lower
        if self.upper is not None:
            result['upper'] = self.upper
        if self.std is not None:
            result['std'] = self.std
        if self.samples is not None:
            result['samples'] = self.samples
        return result


class BaseModel(ABC, nn.Module):
    """
    基础模型抽象类 / Base model abstract class

    定义所有时序预测模型的统一接口，同时集成概率预测能力。
    继承自 ABC（强制子类实现核心方法）和 nn.Module（兼容 PyTorch 训练流程）。
    Defines the unified interface for all time-series forecasting models and
    integrates probabilistic prediction capabilities.  Inherits from ABC
    (enforcing subclass implementation) and nn.Module (PyTorch compatibility).

    子类必须实现 / Subclasses must implement:
        - forward(): 前向传播 / Forward pass.
        - fit(): 训练 / Training loop.
        - predict(): 推理 / Inference.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化基础模型 / Initialize base model.

        Args:
            config: 模型配置字典 / Model configuration dictionary. 支持如下键:
                - model_name (str):        模型名称标识,默认 'base_model'
                                           Model identifier, default 'base_model'.
                - device (str):            运行设备,'cpu' 或 'cuda'
                                           Device, 'cpu' or 'cuda'.
                - probabilistic (bool):    是否启用概率预测,默认 False
                                           Enable probabilistic prediction.
                - probabilistic_method:    概率方法,'residual'(残差法)/'mc_dropout'(MC Dropout)/
                                           'quantile'(分位数回归),默认 'residual'
                                           Probabilistic method.
                - quantiles (List[float]): 分位数列表,仅分位数回归使用
                                           Quantile list, used by quantile regression.
                - num_samples (int):       MC Dropout 采样次数,默认 100
                                           MC Dropout sample count.
                - confidence_level (float): 置信水平,默认 0.95(对应 95% 置信区间)
                                            Confidence level.
        """
        super().__init__()
        self.config = config
        self.model_name = config.get('model_name', 'base_model')
        self.device = config.get('device', 'cpu')

        # --- 概率预测相关配置 / Probabilistic prediction settings ---
        self.probabilistic = config.get('probabilistic', False)
        self.probabilistic_method = config.get('probabilistic_method', 'residual')
        self.quantiles = config.get('quantiles', [0.025, 0.5, 0.975])
        self.num_samples = config.get('num_samples', 100)
        self.confidence_level = config.get('confidence_level', 0.95)

        # 训练阶段缓存的残差，用于残差法置信区间 / Cached residuals for residual-based CI
        # 结构: np.ndarray, shape (N,) 或 (N, output_size)
        # Populated by _fit_residuals() during training when probabilistic='residual'.
        self._residuals = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播 / Forward pass. 子类必须实现 / Must be implemented by subclasses."""
        pass

    @abstractmethod
    def fit(self, train_data: Any, val_data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        训练模型 / Train the model.

        Returns:
            训练历史字典，至少含 'train_loss' 键。
            Training history dict with at least a 'train_loss' key.
        """
        pass

    @abstractmethod
    def predict(self, test_data: Any, **kwargs) -> np.ndarray:
        """
        推理/预测 / Run inference.

        Returns:
            预测结果数组 / Prediction array, shape (N, output_size).
        """
        pass

    def predict_probabilistic(self, test_data: Any, **kwargs) -> ProbabilisticPrediction:
        """
        概率预测入口 / Entry point for probabilistic prediction.

        优先调用子类的 _predict_probabilistic()；若未实现则退化为点预测。
        Delegates to _predict_probabilistic() if the subclass provides one;
        otherwise falls back to a plain point prediction.
        """
        if hasattr(self, '_predict_probabilistic'):
            return self._predict_probabilistic(test_data, **kwargs)
        mean = self.predict(test_data, **kwargs)
        return ProbabilisticPrediction(mean=mean)

    def _fit_residuals(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        计算并缓存训练残差 / Compute and cache training residuals.

        残差 = 真实值 - 预测值，后续用于构建经验置信区间。
        Residuals = y_true - y_pred, later used to build empirical CIs.
        """
        self._residuals = y_true - y_pred

    def _get_residual_interval(self, y_pred: np.ndarray) -> tuple:
        """
        基于残差经验分布计算置信区间 / Compute confidence interval
        from the empirical residual distribution.

        算法 / Algorithm:
          1. 由 confidence_level（如 0.95）算出 alpha = 0.05
          2. 取残差的 alpha/2 和 1-alpha/2 分位数
          3. 将分位数加到点预测上得到 [lower, upper]
          Step 1: Derive alpha from confidence_level (e.g. 0.95 -> 0.05).
          Step 2: Get the alpha/2 and 1-alpha/2 percentiles of residuals.
          Step 3: Add these percentiles to the point prediction.

        Returns:
            (lower, upper) 置信区间数组 / Confidence interval arrays.
        """
        if self._residuals is None:
            return y_pred, y_pred

        # 显著性水平 / Significance level: e.g. 0.05 for 95% CI
        alpha = 1 - self.confidence_level
        # 双侧分位数百分比 / Two-tailed percentile bounds
        lower_percentile = (alpha / 2) * 100       # e.g. 2.5
        upper_percentile = (1 - alpha / 2) * 100   # e.g. 97.5

        # 从训练残差分布取分位数 / Get quantiles from the training residual distribution
        residual_lower = np.percentile(self._residuals, lower_percentile)
        residual_upper = np.percentile(self._residuals, upper_percentile)

        # 残差为负 -> 模型高估 -> lower 向下偏移；反之 upper 向上偏移
        # Negative residual = model over-predicted -> lower shifts down
        lower = y_pred + residual_lower
        upper = y_pred + residual_upper

        return lower, upper
    
    def save_model(self, save_path: str):
        """
        保存模型权重和配置到文件 / Save model weights and config to file.

        使用 torch.save 序列化为字典,包含 state_dict 和 config 两部分,
        便于与 load_model 配对恢复。注意: _residuals 等运行期缓存不会被保存。

        Args:
            save_path: 目标文件路径 / Target file path (e.g. 'checkpoint.pt').
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, save_path)

    def load_model(self, load_path: str):
        """
        从文件加载模型权重和配置 / Load model weights and config from file.

        会将权重张量映射到 self.device,并在加载后显式调用 self.to(self.device),
        确保模型自身的 submodule/buffer 归属与权重一致,避免跨设备推理报错。
        self.config 会被 checkpoint 中的 config 增量更新(update 而非覆盖)。

        Loads weights via map_location=self.device and synchronises the whole
        module to self.device afterwards, so submodules and buffers match the
        loaded tensors. self.config is updated (not replaced) in place.

        Args:
            load_path: 模型文件路径 / Path to the saved checkpoint file.
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config.update(checkpoint['config'])
        # 同步整个模块到 self.device,防止 submodule/buffer 滞留在 CPU
        # Keep submodules and buffers on the target device.
        self.to(self.device)

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型摘要信息 / Get model summary information.

        Returns:
            包含模型名、配置、参数量的字典。
            Dict with model name, config, and total parameter count.
        """
        return {
            'model_name': self.model_name,
            'config': self.config,
            'num_parameters': sum(p.numel() for p in self.parameters())
        }
