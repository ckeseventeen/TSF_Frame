"""
传统机器学习模型模块 / Classical machine learning model module

封装 11 种 sklearn 兼容的回归模型，统一为 BaseMLModel 接口。
支持概率预测（残差法置信区间）、多输出自动包装、特征重要性提取。
Wraps 11 sklearn-compatible regression models under a unified BaseMLModel
interface. Supports probabilistic prediction (residual-based CI),
automatic multi-output wrapping, and feature importance extraction.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

from ..base_model import BaseModel, ProbabilisticPrediction


class BaseMLModel(BaseModel):
    """
    传统 ML 模型的公共基类 / Common base class for classical ML models

    继承自 BaseModel(nn.Module)，但内部使用 sklearn 模型。
    提供统一的 fit/predict/save/load 接口和特征重要性提取。
    Inherits from BaseModel (nn.Module) but uses sklearn models internally.
    Provides unified fit/predict/save/load interface and feature importance.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 sklearn 模型基类 / Initialize sklearn model base class.

        字段说明 / Fields:
            model:         底层 sklearn 估计器,由 _build_model() 构造,
                           多输出时自动包装为 MultiOutputRegressor。
                           Underlying sklearn estimator; wrapped by
                           MultiOutputRegressor when y has multiple outputs.
            scaler:        预留的数据标准化器(当前未使用,由外层特征工程处理)。
                           Placeholder scaler (unused; handled externally).
            feature_names: 特征列名,fit 时自动生成为 feature_0..feature_N-1。
                           Feature names, auto-generated at fit time.
            target_names:  目标列名,fit 时自动生成为 target_0..target_M-1。
                           Target names, auto-generated at fit time.
        """
        super().__init__(config)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_names = None

    def forward(self, x, **kwargs):
        """
        PyTorch 风格前向传播 / PyTorch-style forward pass.

        仅用于满足 nn.Module 接口,实际推理委托给 sklearn 的 predict。
        自动处理 numpy 与 torch.Tensor 两种输入。
        Only to satisfy the nn.Module interface; delegates to predict().
        """
        if isinstance(x, np.ndarray):
            return self.predict(x, **kwargs)
        return self.predict(x.cpu().numpy(), **kwargs)

    @abstractmethod
    def _build_model(self) -> Any:
        """
        构造底层 sklearn 模型实例 / Build the underlying sklearn estimator.

        子类必须重写,读取 self.config 并返回未训练的模型对象。
        Subclasses must override to return an untrained sklearn estimator.
        """
        pass

    def fit(self, train_data: Tuple[np.ndarray, np.ndarray],
            val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, **kwargs) -> Dict[str, Any]:
        """
        训练模型 / Train the model.

        流程 / Workflow:
          1. 将一维 y 提升为二维 (N, 1),以统一单输出/多输出代码路径
             Reshape 1-D y to (N, 1) to unify single/multi-output paths.
          2. 自动生成 feature_names / target_names
          3. 调用 _build_model() 构造 sklearn 估计器
          4. 若 y 为多输出(shape[1] > 1),自动用 MultiOutputRegressor 包装
             Auto-wrap with MultiOutputRegressor for multi-output targets.
          5. 若 val_data 提供,用训练好的模型在验证集上计算 MSE
          6. 若开启 probabilistic='residual',用训练集预测残差缓存到
             self._residuals,供后续 _predict_probabilistic 构建置信区间。
             When probabilistic='residual' is enabled, cache training
             residuals into self._residuals for later CI construction.

        Args:
            train_data: (X_train, y_train) 元组,X 形状 (N, D),y 形状 (N,) 或 (N, M)
                        Training tuple; y can be 1-D or 2-D.
            val_data:   可选 (X_val, y_val) 元组 / Optional validation tuple.

        Returns:
            训练历史字典,含 'train_loss';若有 val_data 也含 'val_loss'。
            Training history dict with 'train_loss' (and 'val_loss' if applicable).
        """
        X_train, y_train = train_data

        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        self.target_names = [f'target_{i}' for i in range(y_train.shape[1])]

        self.model = self._build_model()

        if y_train.shape[1] == 1:
            # 单输出: 用 ravel() 拍平成一维,sklearn 单输出模型期望 1-D y
            # Single output: sklearn expects 1-D y.
            self.model.fit(X_train, y_train.ravel(), **kwargs)
        else:
            # 多输出: 包一层 MultiOutputRegressor,为每个输出维度独立训练一个克隆模型
            # Multi-output: wrap with MultiOutputRegressor (one model per target).
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(self.model)
            self.model.fit(X_train, y_train, **kwargs)

        history = {'train_loss': []}
        if val_data is not None:
            X_val, y_val = val_data
            y_pred = self.predict(X_val)
            from tsf_frame.utils.metrics import MetricsCalculator
            val_mse = MetricsCalculator.mse(y_val, y_pred)
            history['val_loss'] = [val_mse]

        # 残差法概率预测: 缓存训练集残差的经验分布,后续由
        # BaseModel._get_residual_interval 取分位数构造置信区间
        # Residual-based probabilistic prediction: cache training residual
        # distribution for later quantile-based CI construction.
        if self.probabilistic and self.probabilistic_method == 'residual':
            y_train_pred = self.predict(X_train)
            self._fit_residuals(y_train.flatten(), y_train_pred.flatten())

        return history

    def _predict_probabilistic(self, test_data: Any, **kwargs) -> 'ProbabilisticPrediction':
        """
        概率预测 / Probabilistic prediction.

        当前仅支持残差法(residual): 先做点预测得到 mean,再叠加训练残差分位数
        得到 lower/upper。若未启用残差法或残差缓存为空,则降级为点预测。
        残差区间的具体计算见 [BaseModel._get_residual_interval]。
        Currently supports residual method only: point prediction + residual
        quantiles. Falls back to point prediction when residuals are unavailable.

        Args:
            test_data: 测试特征或 (X_test, _) 元组 / Test features or tuple.

        Returns:
            ProbabilisticPrediction 对象,含 mean/lower/upper。
        """
        mean = self.predict(test_data, **kwargs)

        if self.probabilistic_method == 'residual' and self._residuals is not None:
            lower, upper = self._get_residual_interval(mean)
            return ProbabilisticPrediction(mean=mean, lower=lower, upper=upper)

        return ProbabilisticPrediction(mean=mean)

    def predict(self, test_data: Any, **kwargs) -> np.ndarray:
        """
        点预测 / Point prediction.

        兼容两种输入形式:
          - np.ndarray: 直接作为 X_test
          - tuple: 取第一个元素作为 X_test(便于 (X, y) 元组场景复用)
        一维输出会被 reshape 为 (N, 1),保持形状一致性。
        Accepts raw array or (X, y) tuple. 1-D predictions are reshaped to (N, 1).

        Args:
            test_data: 测试特征或 (X_test, _) 元组 / Test features or tuple.

        Returns:
            预测数组,形状 (N, output_size) / Prediction array (N, M).
        """
        if isinstance(test_data, tuple):
            X_test, _ = test_data
        else:
            X_test = test_data

        predictions = self.model.predict(X_test)

        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions

    def save_model(self, save_path: str):
        """
        持久化模型到 pickle 文件 / Pickle the model to disk.

        保存底层 sklearn model、config、feature_names、target_names。
        若 save_path 不含目录(仅文件名),跳过 makedirs 以避免空字符串异常。
        Saves sklearn model, config, feature_names, and target_names. Skips
        makedirs when save_path has no directory component.

        Args:
            save_path: 目标文件路径 / Target pickle file path.
        """
        # B10 修复: dirname 为空时不要调用 makedirs('')(Windows 下会抛异常)
        # B10 fix: skip makedirs when dirname is empty (raises on Windows).
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'feature_names': self.feature_names,
                'target_names': self.target_names
            }, f)

    def load_model(self, load_path: str):
        """
        从 pickle 文件加载模型 / Load model from pickle file.

        恢复底层 sklearn model 与元数据;config 是 update 合并(不会清空原字段)。
        Restores sklearn model and metadata; config is merged via dict.update().

        Args:
            load_path: 源文件路径 / Source pickle file path.
        """
        with open(load_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.model = checkpoint['model']
        self.config.update(checkpoint['config'])
        self.feature_names = checkpoint.get('feature_names')
        self.target_names = checkpoint.get('target_names')

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        获取特征重要性 DataFrame / Get feature importance as DataFrame.

        多输出模型 (MultiOutputRegressor) 会被自动解包,取第 0 个子估计器的
        重要性作为代表(假设各输出子模型结构一致)。
        Tree-based 模型从 feature_importances_ 读取;线性模型从 |coef_| 读取;
        若模型不支持任一种,返回 None。
        For multi-output models, unwraps MultiOutputRegressor and uses the first
        sub-estimator. Reads feature_importances_ for tree models, |coef_| for
        linear models; returns None if neither is available.

        Returns:
            按重要性降序排列的 DataFrame,列为 ['feature', 'importance'];
            不支持的模型返回 None。
            DataFrame sorted by importance (desc), or None if unsupported.
        """
        # 解包 MultiOutputRegressor,访问底层单输出估计器
        # Unwrap MultiOutputRegressor to access the underlying single-output estimator
        model = self.model
        if hasattr(model, 'estimators_') and isinstance(model.estimators_, list):
            model = model.estimators_[0]

        if hasattr(model, 'feature_importances_'):
            # 树模型(RF/GBM/XGBoost/LightGBM/CatBoost/DT) 直接提供
            # Tree-based models expose feature_importances_ directly.
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 线性模型(Linear/Ridge/Lasso) 用系数绝对值作为重要性代理
            # Linear models: use |coef_| as importance proxy.
            importances = np.abs(model.coef_).flatten()
        else:
            # SVR(非线性核)/KNN 等不暴露重要性 / SVR (non-linear)/KNN: unavailable
            return None

        feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importances))]
        return pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)


class LinearRegressionModel(BaseMLModel):
    """线性回归 / Linear Regression"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'linear_regression'
    
    def _build_model(self) -> Any:
        return LinearRegression(
            fit_intercept=self.config.get('fit_intercept', True)
        )


class RidgeModel(BaseMLModel):
    """岭回归 / Ridge Regression (L2 regularization)"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'ridge'
    
    def _build_model(self) -> Any:
        return Ridge(
            alpha=self.config.get('alpha', 1.0),
            fit_intercept=self.config.get('fit_intercept', True)
        )


class LassoModel(BaseMLModel):
    """Lasso 回归 / Lasso Regression (L1 regularization)"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'lasso'
    
    def _build_model(self) -> Any:
        return Lasso(
            alpha=self.config.get('alpha', 1.0),
            fit_intercept=self.config.get('fit_intercept', True)
        )


class RandomForestModel(BaseMLModel):
    """随机森林回归 / Random Forest Regression"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'random_forest'
    
    def _build_model(self) -> Any:
        return RandomForestRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', None),
            min_samples_split=self.config.get('min_samples_split', 2),
            min_samples_leaf=self.config.get('min_samples_leaf', 1),
            random_state=self.config.get('random_seed', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )


class GradientBoostingModel(BaseMLModel):
    """梯度提升回归 / Gradient Boosting Regression"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'gradient_boosting'
    
    def _build_model(self) -> Any:
        return GradientBoostingRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            max_depth=self.config.get('max_depth', 3),
            random_state=self.config.get('random_seed', 42)
        )


class XGBoostModel(BaseMLModel):
    """XGBoost 回归 / XGBoost Regression"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if xgb is None:
            raise ImportError("XGBoost is not installed. Please install it with 'pip install xgboost'")
        self.model_name = 'xgboost'
    
    def _build_model(self) -> Any:
        return xgb.XGBRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            max_depth=self.config.get('max_depth', 6),
            subsample=self.config.get('subsample', 1.0),
            colsample_bytree=self.config.get('colsample_bytree', 1.0),
            random_state=self.config.get('random_seed', 42),
            n_jobs=self.config.get('n_jobs', -1),
            verbosity=self.config.get('verbosity', 0)
        )


class LightGBMModel(BaseMLModel):
    """LightGBM 回归 / LightGBM Regression"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if lgb is None:
            raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'")
        self.model_name = 'lightgbm'
    
    def _build_model(self) -> Any:
        return lgb.LGBMRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            max_depth=self.config.get('max_depth', -1),
            num_leaves=self.config.get('num_leaves', 31),
            subsample=self.config.get('subsample', 1.0),
            colsample_bytree=self.config.get('colsample_bytree', 1.0),
            random_state=self.config.get('random_seed', 42),
            n_jobs=self.config.get('n_jobs', -1),
            verbosity=self.config.get('verbosity', -1)
        )


class CatBoostModel(BaseMLModel):
    """CatBoost 回归 / CatBoost Regression"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if CatBoostRegressor is None:
            raise ImportError("CatBoost is not installed. Please install it with 'pip install catboost'")
        self.model_name = 'catboost'
    
    def _build_model(self) -> Any:
        return CatBoostRegressor(
            iterations=self.config.get('iterations', 1000),
            learning_rate=self.config.get('learning_rate', 0.03),
            depth=self.config.get('depth', 6),
            random_seed=self.config.get('random_seed', 42),
            verbose=self.config.get('verbose', False)
        )


class SVRModel(BaseMLModel):
    """支持向量回归 / Support Vector Regression (SVR)"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'svr'
    
    def _build_model(self) -> Any:
        return SVR(
            kernel=self.config.get('kernel', 'rbf'),
            C=self.config.get('C', 1.0),
            epsilon=self.config.get('epsilon', 0.1)
        )


class KNNModel(BaseMLModel):
    """K近邻回归 / K-Nearest Neighbors Regression"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'knn'
    
    def _build_model(self) -> Any:
        return KNeighborsRegressor(
            n_neighbors=self.config.get('n_neighbors', 5),
            weights=self.config.get('weights', 'uniform'),
            n_jobs=self.config.get('n_jobs', -1)
        )


class DecisionTreeModel(BaseMLModel):
    """决策树回归 / Decision Tree Regression"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'decision_tree'
    
    def _build_model(self) -> Any:
        return DecisionTreeRegressor(
            max_depth=self.config.get('max_depth', None),
            min_samples_split=self.config.get('min_samples_split', 2),
            min_samples_leaf=self.config.get('min_samples_leaf', 1),
            random_state=self.config.get('random_seed', 42)
        )


# 模型注册表：名称 -> 类映射 / Model registry: name -> class mapping
MODEL_REGISTRY = {
    'linear_regression': LinearRegressionModel,
    'ridge': RidgeModel,
    'lasso': LassoModel,
    'random_forest': RandomForestModel,
    'gradient_boosting': GradientBoostingModel,
    'xgboost': XGBoostModel,
    'lightgbm': LightGBMModel,
    'catboost': CatBoostModel,
    'svr': SVRModel,
    'knn': KNNModel,
    'decision_tree': DecisionTreeModel
}


def get_ml_model(model_name: str, config: Dict[str, Any]) -> BaseMLModel:
    """工厂函数：按名称创建 ML 模型实例 / Factory: create ML model instance by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](config)
