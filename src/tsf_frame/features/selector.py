"""
特征选择与降维模块 / Feature selection and dimensionality reduction module

提供多种特征选择器和降维器，以及自动化特征工程流水线。
Provides multiple feature selectors, dimensionality reducers, and an
automated feature engineering pipeline.

选择器 / Selectors:
  - KBestSelector       : 基于 F 回归的 K-Best 选择 / F-regression K-Best
  - RFESelector         : 递归特征消除 / Recursive Feature Elimination
  - ModelBasedSelector  : 基于模型重要性的选择 / Model importance-based
  - LassoSelector       : 基于 Lasso 系数的选择 / Lasso coefficient-based
  - VarianceSelector    : 方差阈值过滤 / Variance threshold filtering
  - CorrelationSelector : 高相关性去冗余 / High-correlation redundancy removal

降维器 / Reducers:
  - PCAReducer : 主成分分析降维 / PCA dimensionality reduction
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class BaseFeatureSelector(ABC):
    """
    特征选择器抽象基类 / Abstract base class for feature selectors.

    统一接口: fit(学习重要性/系数) → transform(按 selected_features 裁列)。
    Unified interface: fit → transform.

    字段 / Fields:
        selected_features:  fit 后保存的选中列名列表 / List of selected column names.
        feature_importances: 各特征的重要性/得分字典 {col: score} / Importance dict.
    """
    def __init__(self):
        self.selected_features = None
        self.feature_importances = None

    @abstractmethod
    def fit(self, X, y=None):
        """学习选择准则(得分/系数/排名等) / Fit selection criterion. Returns self."""
        pass

    @abstractmethod
    def transform(self, X):
        """按 selected_features 裁列,返回子集 DataFrame / Subset X by selected features."""
        pass

    def fit_transform(self, X, y=None):
        """fit + transform 一步完成 / Convenience fit + transform."""
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self):
        """
        返回选中的特征名列表 / Return list of selected feature names.

        Returns:
            List[str]: 已选列名 / Column names that passed the selection.
        Raises:
            ValueError: 未调用 fit / Selector not fitted.
        """
        if self.selected_features is None:
            raise ValueError("Selector has not been fitted yet. Call 'fit' first.")
        return self.selected_features

    def get_feature_importances(self):
        """
        返回特征重要性字典 / Return feature importance dict.

        Returns:
            Dict[str, float]: {feature_name: importance_score} 映射。
                具体语义取决于选择器:
                  - KBest    : F 统计量值 / F-statistic
                  - RFE      : 排名(数字越小越重要) / Ranking (smaller = more important)
                  - ModelBased / Lasso : 重要性或 |系数| / importance or |coef|
                  - Variance : 方差值 / variance
                  - Correlation : 固定为 1.0(保留列) / constant 1.0 for retained cols
        Raises:
            ValueError: 未调用 fit / Selector not fitted.
        """
        if self.feature_importances is None:
            raise ValueError("Selector has not been fitted yet. Call 'fit' first.")
        return self.feature_importances


class BaseFeatureReducer(ABC):
    """特征降维器抽象基类 / Abstract base class for feature reducers"""
    def __init__(self):
        self.reducer = None
    
    @abstractmethod
    def fit(self, X):
        pass
    
    @abstractmethod
    def transform(self, X):
        pass
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class KBestSelector(BaseFeatureSelector):
    """基于 F 回归统计量选择 Top-K 特征 / Select top-K features by F-regression score"""
    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self.selector = None
    
    def fit(self, X, y):
        self.selector = SelectKBest(score_func=f_regression, k=self.k)
        self.selector.fit(X, y)
        mask = self.selector.get_support()
        self.selected_features = X.columns[mask].tolist()
        scores = self.selector.scores_
        self.feature_importances = dict(zip(X.columns, scores))
        return self
    
    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("Selector has not been fitted yet. Call 'fit' first.")
        return X[self.selected_features]


class RFESelector(BaseFeatureSelector):
    """递归特征消除选择器 / Recursive Feature Elimination (RFE) selector"""
    def __init__(self, n_features_to_select=10, estimator=None):
        super().__init__()
        self.n_features_to_select = n_features_to_select
        self.estimator = estimator or RandomForestRegressor(n_estimators=50, random_state=42)
        self.selector = None
    
    def fit(self, X, y):
        self.selector = RFE(self.estimator, n_features_to_select=self.n_features_to_select)
        self.selector.fit(X, y)
        mask = self.selector.get_support()
        self.selected_features = X.columns[mask].tolist()
        self.feature_importances = dict(zip(X.columns, self.selector.ranking_))
        return self
    
    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("Selector has not been fitted yet. Call 'fit' first.")
        return X[self.selected_features]


class ModelBasedSelector(BaseFeatureSelector):
    """
    基于树模型特征重要性的选择器 / Tree model feature-importance based selector.

    Args:
        threshold: 重要性阈值,可选 / Importance threshold.
            - 'median'(默认): 使用重要性的中位数作为阈值,保留前 50%
            - float        : 指定绝对阈值,仅保留 importance >= threshold 的特征
            'median' (default) keeps top 50%; a float sets an absolute cutoff.
        estimator: 基础树模型 / Base tree estimator (需有 feature_importances_ 属性)。
            默认 RandomForestRegressor(n_estimators=100, random_state=42)。
    """
    def __init__(self, threshold='median', estimator=None):
        super().__init__()
        self.threshold = threshold
        self.estimator = estimator or RandomForestRegressor(n_estimators=100, random_state=42)
    
    def fit(self, X, y):
        self.estimator.fit(X, y)
        importances = self.estimator.feature_importances_
        self.feature_importances = dict(zip(X.columns, importances))
        
        if self.threshold == 'median':
            thresh = np.median(importances)
        else:
            thresh = self.threshold
        
        mask = importances >= thresh
        self.selected_features = X.columns[mask].tolist()
        return self
    
    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("Selector has not been fitted yet. Call 'fit' first.")
        return X[self.selected_features]


class LassoSelector(BaseFeatureSelector):
    """基于 Lasso 回归系数的特征选择器 / Lasso coefficient-based selector (zero coef = dropped)"""
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.model = None
    
    def fit(self, X, y):
        self.model = Lasso(alpha=self.alpha, random_state=42)
        self.model.fit(X, y)
        coef = np.abs(self.model.coef_)
        self.feature_importances = dict(zip(X.columns, coef))
        mask = coef > 0
        self.selected_features = X.columns[mask].tolist()
        return self
    
    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("Selector has not been fitted yet. Call 'fit' first.")
        return X[self.selected_features]


class VarianceSelector(BaseFeatureSelector):
    """方差阈值选择器 / Variance threshold selector — 删除方差过低的特征 / Drops low-variance features"""
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold
    
    def fit(self, X, y=None):
        variances = X.var()
        mask = variances > self.threshold
        self.selected_features = X.columns[mask].tolist()
        self.feature_importances = variances.to_dict()
        return self
    
    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("Selector has not been fitted yet. Call 'fit' first.")
        return X[self.selected_features]


class CorrelationSelector(BaseFeatureSelector):
    """高相关性去冗余选择器 / Correlation-based redundancy removal — 删除相关性超过阈值的特征 / Drops features exceeding correlation threshold"""
    def __init__(self, threshold=0.95):
        super().__init__()
        self.threshold = threshold
    
    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        self.selected_features = [col for col in X.columns if col not in to_drop]
        self.feature_importances = {col: 1.0 for col in self.selected_features}
        return self
    
    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("Selector has not been fitted yet. Call 'fit' first.")
        return X[self.selected_features]


class PCAReducer(BaseFeatureReducer):
    """PCA 主成分分析降维器 / PCA dimensionality reducer"""
    def __init__(self, n_components=0.95, use_scaler=True):
        super().__init__()
        self.n_components = n_components
        self.use_scaler = use_scaler
        self.scaler = None
    
    def fit(self, X):
        if self.use_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        self.reducer = PCA(n_components=self.n_components)
        self.reducer.fit(X_scaled)
        return self
    
    def transform(self, X):
        if self.reducer is None:
            raise ValueError("Reducer has not been fitted yet. Call 'fit' first.")
        
        X_processed = X.copy()
        if self.use_scaler and self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
        
        X_reduced = self.reducer.transform(X_processed)
        
        columns = [f'component_{i}' for i in range(X_reduced.shape[1])]
        return pd.DataFrame(X_reduced, index=X.index, columns=columns)
    
    def get_explained_variance(self):
        if self.reducer is None:
            raise ValueError("Reducer has not been fitted yet.")
        return self.reducer.explained_variance_ratio_


class AutoFeatureEngineer:
    """自动化特征工程流水线 / Automated feature engineering pipeline — 串联工程器、选择器和降维器 / Chains engineers, selectors, and reducers"""
    def __init__(self):
        self.feature_engineers = []
        self.selectors = []
        self.reducers = []
        self.pipeline = []
    
    def add_feature_engineer(self, engineer):
        self.feature_engineers.append(engineer)
        return self
    
    def add_selector(self, selector):
        self.selectors.append(selector)
        return self
    
    def add_reducer(self, reducer):
        self.reducers.append(reducer)
        return self
    
    def fit(self, X, y=None):
        """
        训练流水线 / Fit the pipeline end-to-end.

        Lag / rolling 等特征工程器会在序列开头引入 NaN, 因此在
        engineer 阶段结束后会**自动 dropna 并同步 y 的索引**, 否则
        sklearn 的 selectors 会报 "Input contains NaN"。
        """
        current_X = X.copy()
        current_y = y.copy() if y is not None else None

        for engineer in self.feature_engineers:
            current_X = engineer.fit_transform(current_X)

        # 关键: 特征工程后 dropna 并同步 y, 避免下游 selector 拿到 NaN
        if current_X.isna().any().any():
            valid_mask = ~current_X.isna().any(axis=1)
            current_X = current_X[valid_mask]
            if current_y is not None:
                current_y = current_y.loc[current_X.index]

        for selector in self.selectors:
            current_X = selector.fit_transform(current_X, current_y)

        for reducer in self.reducers:
            current_X = reducer.fit_transform(current_X)

        self.pipeline = self.feature_engineers + self.selectors + self.reducers
        return self

    def transform(self, X):
        current_X = X.copy()
        # 应用所有 engineers
        for engineer in self.feature_engineers:
            current_X = engineer.transform(current_X)
        # engineer 后同样 dropna (与 fit 行为一致)
        if current_X.isna().any().any():
            current_X = current_X.dropna()
        # 应用所有 selectors / reducers
        for step in (self.selectors + self.reducers):
            current_X = step.transform(current_X)
        return current_X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


SELECTOR_REGISTRY = {
    'kbest': KBestSelector,
    'rfe': RFESelector,
    'model_based': ModelBasedSelector,
    'lasso': LassoSelector,
    'variance': VarianceSelector,
    'correlation': CorrelationSelector
}

REDUCER_REGISTRY = {
    'pca': PCAReducer
}


def get_selector(selector_name, config=None):
    if selector_name not in SELECTOR_REGISTRY:
        raise ValueError(f"Selector '{selector_name}' not found. Available: {list(SELECTOR_REGISTRY.keys())}")
    return SELECTOR_REGISTRY[selector_name](**(config or {}))


def get_reducer(reducer_name, config=None):
    if reducer_name not in REDUCER_REGISTRY:
        raise ValueError(f"Reducer '{reducer_name}' not found. Available: {list(REDUCER_REGISTRY.keys())}")
    return REDUCER_REGISTRY[reducer_name](**(config or {}))
