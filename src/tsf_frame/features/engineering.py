"""
特征工程模块 / Feature engineering module

提供多种时序特征提取器，以及通过组合模式批量构建特征流水线的工具。
Provides various time-series feature extractors and a composite pipeline
builder for batch feature engineering.

包含的特征工程器 / Included feature engineers:
  - TimeFeatureEngineer    : 时间特征（小时/天/周/月等）/ Calendar features
  - LagFeatureEngineer     : 滞后特征 / Lag features
  - RollingFeatureEngineer : 滚动窗口统计特征 / Rolling window statistics
  - ExpandingFeatureEngineer : 扩展窗口统计特征 / Expanding window statistics
  - DifferenceFeatureEngineer : 差分特征 / Difference features
  - CompositeFeatureEngineer  : 组合流水线 / Composite pipeline
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseFeatureEngineer(ABC):
    """
    特征工程器抽象基类 / Abstract base class for feature engineers.

    统一接口: fit(学习参数) → transform(产出新特征) → get_feature_names(查名)。
    Unified interface: fit → transform → get_feature_names.

    ⚠ NaN 提醒: 子类涉及 shift/rolling/diff 操作会在序列**开头**引入 NaN,
    调用方在进入模型训练前需决定处理策略(dropna/fillna/保留)。
    NaN warning: shift/rolling/diff introduce NaN at the start of the series;
    caller is responsible for handling strategy (dropna/fillna/keep).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_names = []

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseFeatureEngineer':
        """学习必要的统计参数 / Fit any required statistics. Returns self."""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        在 data 基础上追加新特征列,返回新 DataFrame / Append features.

        Returns:
            添加了新特征列的 DataFrame(原列保留) / DataFrame with new columns.
        """
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """fit 后立即 transform 的便捷方法 / Convenience fit + transform."""
        self.fit(data)
        return self.transform(data)

    def get_feature_names(self) -> List[str]:
        """
        返回本工程器最近一次 transform 产出的新特征名列表 / Names of newly added features.
        """
        return self.feature_names


class TimeFeatureEngineer(BaseFeatureEngineer):
    """时间/日历特征提取器 / Calendar feature extractor (hour, day, weekday, month, etc.)"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.time_col = config.get('time_col', 'timestamp') if config else 'timestamp'
        self.features = config.get('features', ['hour', 'day', 'weekday', 'month', 'quarter', 'year', 'is_weekend']) if config else ['hour', 'day', 'weekday', 'month', 'quarter', 'year', 'is_weekend']
    
    def fit(self, data: pd.DataFrame) -> 'TimeFeatureEngineer':
        if self.time_col not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"Time column '{self.time_col}' not found in data")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        if isinstance(data.index, pd.DatetimeIndex):
            idx = pd.Series(data.index, index=data.index)
        else:
            idx = pd.to_datetime(data[self.time_col])
        
        new_features = {}

        if 'hour' in self.features:
            new_features['hour'] = idx.dt.hour  # <--- 注意加了 .dt
        if 'day' in self.features:
            new_features['day'] = idx.dt.day
        if 'weekday' in self.features:
            new_features['weekday'] = idx.dt.weekday
        if 'dayofyear' in self.features:
            new_features['dayofyear'] = idx.dt.dayofyear
        if 'month' in self.features:
            new_features['month'] = idx.dt.month
        if 'quarter' in self.features:
            new_features['quarter'] = idx.dt.quarter
        if 'year' in self.features:
            new_features['year'] = idx.dt.year
        if 'is_weekend' in self.features:
            # Series 比较运算返回布尔序列，astype(int) 转 0/1
            new_features['is_weekend'] = (idx.dt.weekday >= 5).astype(int)
        if 'is_month_start' in self.features:
            new_features['is_month_start'] = idx.dt.is_month_start.astype(int)
        if 'is_month_end' in self.features:
            new_features['is_month_end'] = idx.dt.is_month_end.astype(int)
        
        for name, values in new_features.items():
            data[name] = values
        
        self.feature_names = list(new_features.keys())
        return data

class LagFeatureEngineer(BaseFeatureEngineer):
    """
    滞后特征提取器 / Lag feature extractor.

    对每个目标列生成 t-k 时刻的历史值 (k 取自 lags 列表)。
    Generates historical values at t-k for each target column.

    ⚠ NaN 提醒: 前 max(lags) 行会因没有足够历史而为 NaN。
    NaN warning: The first max(lags) rows will contain NaN.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.target_cols = config.get('target_cols', []) if config else []
        self.lags = config.get('lags', [1, 2, 3, 7, 14, 30]) if config else [1, 2, 3, 7, 14, 30]

    def fit(self, data: pd.DataFrame) -> 'LagFeatureEngineer':
        # 未显式指定目标列时,自动选所有数值列 / Auto-pick numeric columns if not specified
        if not self.target_cols:
            self.target_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        对每个目标列 × 每个 lag,生成新列 `{col}_lag_{k}`。
        Returns:
            追加滞后列的 DataFrame,列名格式 '{col}_lag_{k}'。
            前 max(lags) 行含 NaN。
        """
        data = data.copy()
        new_features = []

        for col in self.target_cols:
            if col in data.columns:
                for lag in self.lags:
                    feature_name = f'{col}_lag_{lag}'
                    # shift(k) 生成 t-k 的值: 例 lag=1 即用 t-1 的值填在 t 行
                    # shift(k): place value from t-k at row t
                    data[feature_name] = data[col].shift(lag)
                    new_features.append(feature_name)

        self.feature_names = new_features
        return data


class RollingFeatureEngineer(BaseFeatureEngineer):
    """
    滚动窗口统计特征 / Rolling window statistics.

    对每个目标列 × 每个窗口大小 × 每种统计量,生成特征列。
    支持 mean/std/min/max/median 五种统计量。
    Generates rolling stats (mean/std/min/max/median) per (col, window).

    ⚠ NaN 提醒: 每个窗口 w 的前 w-1 行因样本不足而为 NaN。
    多个 rolling 特征串联后 NaN 行数会叠加,建议统一 dropna。
    NaN warning: first w-1 rows per window are NaN; stacks with other engineers.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.target_cols = config.get('target_cols', []) if config else []
        self.windows = config.get('windows', [7, 14, 30]) if config else [7, 14, 30]
        self.stats = config.get('stats', ['mean', 'std', 'min', 'max']) if config else ['mean', 'std', 'min', 'max']
    
    def fit(self, data: pd.DataFrame) -> 'RollingFeatureEngineer':
        if not self.target_cols:
            self.target_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        new_features = []
        
        for col in self.target_cols:
            if col in data.columns:
                for window in self.windows:
                    rolling = data[col].rolling(window=window)
                    for stat in self.stats:
                        feature_name = f'{col}_roll_{window}_{stat}'
                        if stat == 'mean':
                            data[feature_name] = rolling.mean()
                        elif stat == 'std':
                            data[feature_name] = rolling.std()
                        elif stat == 'min':
                            data[feature_name] = rolling.min()
                        elif stat == 'max':
                            data[feature_name] = rolling.max()
                        elif stat == 'median':
                            data[feature_name] = rolling.median()
                        new_features.append(feature_name)
        
        self.feature_names = new_features
        return data


class ExpandingFeatureEngineer(BaseFeatureEngineer):
    """扩展窗口统计特征 / Expanding window statistics — 从序列起始累积计算 / Cumulative from start"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.target_cols = config.get('target_cols', []) if config else []
        self.stats = config.get('stats', ['mean', 'std', 'min', 'max']) if config else ['mean', 'std', 'min', 'max']
    
    def fit(self, data: pd.DataFrame) -> 'ExpandingFeatureEngineer':
        if not self.target_cols:
            self.target_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        new_features = []
        
        for col in self.target_cols:
            if col in data.columns:
                expanding = data[col].expanding()
                for stat in self.stats:
                    feature_name = f'{col}_expanding_{stat}'
                    if stat == 'mean':
                        data[feature_name] = expanding.mean()
                    elif stat == 'std':
                        data[feature_name] = expanding.std()
                    elif stat == 'min':
                        data[feature_name] = expanding.min()
                    elif stat == 'max':
                        data[feature_name] = expanding.max()
                    new_features.append(feature_name)
        
        self.feature_names = new_features
        return data


class DifferenceFeatureEngineer(BaseFeatureEngineer):
    """
    差分特征提取器 / Difference feature extractor.

    计算 x(t) - x(t-period) 作为新特征,period 取自 periods 列表。
    常用于把非平稳序列转为平稳序列(如去趋势)。
    Computes x(t) - x(t-period) per column. Useful for stationarising.

    ⚠ NaN 提醒: 前 max(periods) 行会产生 NaN,需调用方处理。
    **强烈建议**: 在 diff 特征后、窗口切分前,统一 dropna 或 fillna,
    否则时间泄漏的 NaN 可能被模型捕捉到(等价于"未来信息"标记)。
    NaN warning: first max(periods) rows are NaN. Strongly recommended to
    dropna or fillna before window slicing to avoid leakage.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.target_cols = config.get('target_cols', []) if config else []
        self.periods = config.get('periods', [1, 7, 30]) if config else [1, 7, 30]
    
    def fit(self, data: pd.DataFrame) -> 'DifferenceFeatureEngineer':
        if not self.target_cols:
            self.target_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        new_features = []
        
        for col in self.target_cols:
            if col in data.columns:
                for period in self.periods:
                    feature_name = f'{col}_diff_{period}'
                    data[feature_name] = data[col].diff(period)
                    new_features.append(feature_name)
        
        self.feature_names = new_features
        return data


class CompositeFeatureEngineer(BaseFeatureEngineer):
    """组合特征工程器 / Composite feature engineer — 串联多个工程器形成流水线 / Chains multiple engineers into a pipeline"""
    def __init__(self, engineers: List[BaseFeatureEngineer]):
        super().__init__()
        self.engineers = engineers
    
    def fit(self, data: pd.DataFrame) -> 'CompositeFeatureEngineer':
        for engineer in self.engineers:
            engineer.fit(data)
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        all_features = []
        
        for engineer in self.engineers:
            result = engineer.transform(result)
            all_features.extend(engineer.get_feature_names())
        
        self.feature_names = all_features
        return result


def create_feature_engineer(feature_types: List[str], config: Optional[Dict[str, Any]] = None) -> CompositeFeatureEngineer:
    """
    工厂函数：按名称列表创建组合特征工程器 / Factory: create composite engineer from type name list

    Args:
        feature_types: 特征类型名称列表，可选 'time','lag','rolling','expanding','difference'
        config: 各类型对应的配置字典 / Config dict keyed by type (e.g. 'time_config', 'lag_config')
    """
    config = config or {}
    engineers = []
    
    feature_configs = {
        'time': config.get('time_config', {}),
        'lag': config.get('lag_config', {}),
        'rolling': config.get('rolling_config', {}),
        'expanding': config.get('expanding_config', {}),
        'difference': config.get('difference_config', {})
    }
    
    for feature_type in feature_types:
        if feature_type == 'time':
            engineers.append(TimeFeatureEngineer(feature_configs['time']))
        elif feature_type == 'lag':
            engineers.append(LagFeatureEngineer(feature_configs['lag']))
        elif feature_type == 'rolling':
            engineers.append(RollingFeatureEngineer(feature_configs['rolling']))
        elif feature_type == 'expanding':
            engineers.append(ExpandingFeatureEngineer(feature_configs['expanding']))
        elif feature_type == 'difference':
            engineers.append(DifferenceFeatureEngineer(feature_configs['difference']))
    
    return CompositeFeatureEngineer(engineers)


if __name__ == '__main__':


    # 1. 构造一个包含 100 天的简单测试数据
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'value': np.sin(np.linspace(0, 10, 100)) * 100 + np.random.normal(0, 10, 100)  # 带有正弦周期的随机数据
    })
    df.set_index('timestamp', inplace=True)

    # 2. 定义你想提取的特征逻辑
    config = {
        'time_config': {'features': ['weekday', 'is_weekend']},
        'lag_config': {'target_cols': ['value'], 'lags': [1, 3]},
        'rolling_config': {'target_cols': ['value'], 'windows': [5], 'stats': ['mean', 'std']}
    }

    # 3. 生成流水线并执行
    engineer = create_feature_engineer(feature_types=['time', 'lag', 'rolling'], config=config)
    df_features = engineer.fit_transform(df)

    # 观察结果，注意前几天因为 rolling 和 lag 会产生 NaN
    print(df_features.head(10))