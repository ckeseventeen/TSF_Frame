"""
公开数据集加载模块 / Public dataset loading module

提供 darts 时序数据集和 yfinance 金融数据集的加载器。
当依赖库未安装时，自动回退到合成数据并发出警告。
Provides loaders for darts time-series datasets and yfinance financial datasets.
Falls back to synthetic data with a warning when dependencies are missing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
# 注: 不再用 `warnings.filterwarnings('ignore')` 模块级抑制 — 那会污染所有
# 下游代码的警告输出, 严重影响 debug. 如需局部抑制, 请用 with 上下文:
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')
#         ...


class BasePublicDataset(ABC):
    """
    公开数据集抽象基类 / Abstract base class for public datasets.

    字段 / Fields:
        config:   构造参数字典 / Config dict passed at construction.
        data:     加载后的 DataFrame(load 前为 None) / Loaded DataFrame.
        metadata: 数据集元信息(名称/形状/列名等) / Dataset metadata.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data = None
        self.metadata = {}

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        加载数据并填充 self.data 与 self.metadata / Load and populate.

        Returns:
            加载后的 DataFrame / Loaded DataFrame.
        """
        pass

    def get_data(self) -> pd.DataFrame:
        """
        获取数据,首次调用会触发 load / Get data, triggers load on first call.
        """
        if self.data is None:
            self.load()
        return self.data

    def get_metadata(self) -> Dict[str, Any]:
        """返回数据集元信息字典 / Return metadata dict."""
        return self.metadata

    def split_train_test(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序切分训练/测试集 / Chronological train/test split.

        ⚠ 重要: 本方法**按行顺序**切分,不做任何排序。调用方需保证
        self.data 的索引已按时间升序排列,否则会造成时间泄漏
        (训练集含有未来数据、测试集含有过去数据)。
        Warning: split is positional; caller must ensure data is sorted
        chronologically, otherwise time leakage will occur.

        Args:
            test_size: 测试集占比(0~1) / Fraction of data for test set.

        Returns:
            (train_df, test_df) 按位置切分后的两个 DataFrame。
        """
        if self.data is None:
            self.load()
        split_idx = int(len(self.data) * (1 - test_size))
        return self.data.iloc[:split_idx], self.data.iloc[split_idx:]


class DartsDataset(BasePublicDataset):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 通过 self.config 读取, super() 已把 None 兜底为 {}
        # / Read via self.config (super() coerced None to {})
        self.dataset_name = self.config.get('dataset_name', 'air_passengers')
    
    def load(self) -> pd.DataFrame:
        try:
            from darts.datasets import (
                AirPassengersDataset,
                AusBeerDataset,
                AustralianElectricityDataset,
                ETTh1Dataset,
                ETTh2Dataset,
                ETTm1Dataset,
                ETTm2Dataset,
                ExchangeRateDataset,
                SolarEnergyDataset,
                TrafficDataset,
                WeatherDataset,
                ElectricityDataset,
                M4Dataset
            )
            
            dataset_map = {
                'air_passengers': AirPassengersDataset,
                'aus_beer': AusBeerDataset,
                'australian_electricity': AustralianElectricityDataset,
                'etth1': ETTh1Dataset,
                'etth2': ETTh2Dataset,
                'ettm1': ETTm1Dataset,
                'ettm2': ETTm2Dataset,
                'exchange_rate': ExchangeRateDataset,
                'solar_energy': SolarEnergyDataset,
                'traffic': TrafficDataset,
                'weather': WeatherDataset,
                'electricity': ElectricityDataset,
                'm4': M4Dataset
            }
            
            if self.dataset_name not in dataset_map:
                raise ValueError(f"Dataset '{self.dataset_name}' not available. Available: {list(dataset_map.keys())}")
            
            dataset_class = dataset_map[self.dataset_name]
            timeseries = dataset_class().load()
            
            self.data = timeseries.pd_dataframe()
            self.data = self.data.reset_index()
            if 'time' in self.data.columns:
                self.data.set_index('time', inplace=True)
            
            self.metadata = {
                'dataset_name': self.dataset_name,
                'shape': self.data.shape,
                'columns': list(self.data.columns.tolist())
            }
            
            return self.data
            
        except ImportError:
            import logging
            # B12 修复: 之前用 %s 但格式化参数拼在字符串外;改用 f-string 避免 logging
            # 格式化的双参数签名(level, msg, *args) 引发参数错配的歧义。
            # B12 fix: switch from %s lazy formatting to f-string for clarity.
            logging.getLogger(__name__).warning(
                f"darts 库未安装,无法加载数据集 '{self.dataset_name}',"
                f"已自动生成合成替代数据。请安装: pip install u8darts"
            )
            return self._generate_fallback_data()

    def _generate_fallback_data(self) -> pd.DataFrame:
        from datetime import datetime, timedelta
        
        start_date = datetime(2000, 1, 1)
        end_date = datetime(2020, 1, 1)
        dates = []
        current_date = start_date
        while current_date < end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 用局部 RNG 实例, 不污染全局 np.random 状态
        # / Local RNG to avoid polluting global state
        rng = np.random.default_rng(42)
        values = rng.standard_normal(len(dates)).cumsum() + 100
        
        self.data = pd.DataFrame({'value': values}, index=dates)
        
        self.metadata = {
            'dataset_name': f'{self.dataset_name}_fallback',
            'shape': self.data.shape,
            'columns': list(self.data.columns.tolist())
        }
        
        return self.data


class SyntheticDataset(BasePublicDataset):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 通过 self.config 读取, 防 config=None 时 .get() 炸
        # / Read via self.config to safely handle config=None
        self.n_samples = self.config.get('n_samples', 1000)
        self.seasonality_period = self.config.get('seasonality_period', 365)
        self.trend_strength = self.config.get('trend_strength', 0.1)
        self.noise_level = self.config.get('noise_level', 0.1)
        # 随机种子: 保证多次调用 load() 结果一致 (便于回归测试 / 复现实验)
        # / RNG seed for reproducibility across load() calls
        self.random_seed = int(self.config.get('random_seed', 42))

    def load(self) -> pd.DataFrame:
        from datetime import datetime, timedelta

        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(self.n_samples)]

        t = np.arange(self.n_samples)
        trend = self.trend_strength * t
        seasonality = np.sin(2 * np.pi * t / self.seasonality_period)
        # 用独立 RNG 实例, 不污染全局 np.random 状态
        # / Use a local Generator to avoid mutating the global RNG state
        rng = np.random.default_rng(self.random_seed)
        noise = rng.normal(0, self.noise_level, self.n_samples)
        
        values = 100 + trend + 10 * seasonality + noise
        
        self.data = pd.DataFrame({'target': values}, index=dates)
        
        self.metadata = {
            'dataset_name': 'synthetic',
            'n_samples': self.n_samples,
            'shape': self.data.shape,
            'columns': list(self.data.columns.tolist())
        }
        
        return self.data


class FinancialDataset(BasePublicDataset):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 通过 self.config 读取, 防 config=None 时 .get() 炸
        self.ticker = self.config.get('ticker', 'AAPL')
        self.start_date = self.config.get('start_date', '2010-01-01')
        self.end_date = self.config.get('end_date', '2023-01-01')
    
    def load(self) -> pd.DataFrame:
        try:
            import yfinance as yf
            
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            
            if data.empty:
                return self._generate_fallback_financial_data()
            
            self.data = data
            self.metadata = {
                'dataset_name': f'financial_{self.ticker}',
                'ticker': self.ticker,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'shape': self.data.shape,
                'columns': list(self.data.columns.tolist())
            }
            
            return self.data
            
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "yfinance 库未安装，无法下载金融数据，已自动生成合成替代数据。"
                "请安装: pip install yfinance"
            )
            return self._generate_fallback_financial_data()
    
    def _generate_fallback_financial_data(self) -> pd.DataFrame:
        from datetime import datetime, timedelta
        
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 用局部 RNG, 防全局污染 / Local RNG
        rng = np.random.default_rng(42)
        n_days = len(dates)
        base_price = 100
        returns = rng.normal(0.001, 0.02, n_days)
        prices = base_price * (1 + returns).cumprod()

        self.data = pd.DataFrame({
            'Open': prices * (1 - rng.uniform(0, 0.02, n_days)),
            'High': prices * (1 + rng.uniform(0, 0.03, n_days)),
            'Low': prices * (1 - rng.uniform(0, 0.03, n_days)),
            'Close': prices,
            'Adj Close': prices,
            'Volume': rng.integers(100000, 1000000, n_days),
        }, index=dates)
        
        self.metadata = {
            'dataset_name': f'financial_{self.ticker}_fallback',
            'ticker': self.ticker,
            'shape': self.data.shape,
            'columns': list(self.data.columns.tolist())
        }
        
        return self.data


class EnergyDataset(BasePublicDataset):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 通过 self.config 读取, 防 config=None 时 .get() 炸
        self.data_type = self.config.get('data_type', 'electricity')
        self.location = self.config.get('location', 'US')
    
    def load(self) -> pd.DataFrame:
        from datetime import datetime, timedelta
        
        start_date = datetime(2020, 1, 1)
        n_hours = 8760
        dates = [start_date + timedelta(hours=i) for i in range(n_hours)]
        
        hour_of_day = np.array([d.hour for d in dates])
        day_of_week = np.array([d.weekday() for d in dates])
        month_of_year = np.array([d.month for d in dates])
        
        base_load = 100
        daily_pattern = 30 * np.sin(2 * np.pi * hour_of_day / 24)
        weekly_pattern = 20 * np.sin(2 * np.pi * day_of_week / 7)
        seasonal_pattern = 15 * np.sin(2 * np.pi * month_of_year / 12)
        # 用局部 RNG, 防全局污染 / Local RNG
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 5, n_hours)

        load = base_load + daily_pattern + weekly_pattern + seasonal_pattern + noise
        load = np.maximum(load, 20)

        temperature = (20 + 10 * np.sin(2 * np.pi * np.arange(n_hours) / 8760)
                       + rng.normal(0, 2, n_hours))
        
        self.data = pd.DataFrame({'load': load, 'temperature': temperature}, index=dates)
        
        self.metadata = {
            'dataset_name': f'energy_{self.data_type}',
            'data_type': self.data_type,
            'location': self.location,
            'shape': self.data.shape,
            'columns': list(self.data.columns.tolist())
        }
        
        return self.data


DATASET_REGISTRY = {
    'darts': DartsDataset,
    'synthetic': SyntheticDataset,
    'financial': FinancialDataset,
    'energy': EnergyDataset
}


def get_dataset(dataset_type: str, config: Optional[Dict[str, Any]] = None) -> BasePublicDataset:
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Dataset type '{dataset_type}' not found. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_type](config)


def list_available_datasets() -> Dict[str, List[str]]:
    darts_datasets = [
        'air_passengers', 'aus_beer', 'australian_electricity',
        'etth1', 'etth2', 'ettm1', 'ettm2',
        'exchange_rate', 'solar_energy', 'traffic',
        'weather', 'electricity', 'm4'
    ]
    
    return {
        'darts': darts_datasets,
        'synthetic': ['synthetic'],
        'financial': ['financial'],
        'energy': ['energy']
    }
