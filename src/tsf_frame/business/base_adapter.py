from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


class BaseBusinessAdapter(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.business_type = config.get('business_type', 'general')
        self.target_columns = config.get('target_columns', ['target'])
        self.feature_columns = config.get('feature_columns', None)
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def postprocess(self, predictions: np.ndarray, metadata: Dict[str, Any], **kwargs) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        pass
    
    @abstractmethod
    def get_business_metrics(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
        pass
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns:
            return data[self.feature_columns]
        return data
    
    def extract_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[self.target_columns]
    
    def get_adapter_info(self) -> Dict[str, Any]:
        return {
            'business_type': self.business_type,
            'target_columns': self.target_columns,
            'feature_columns': self.feature_columns
        }
