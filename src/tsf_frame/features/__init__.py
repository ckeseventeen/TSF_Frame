"""特征工程子包 / Feature engineering subpackage.

合并了原 `features/mixed_feature_handler.py` 与原 `data/feature_*.py`：
    mixed_feature_handler   混合（时序+静态）特征 + 序列化
    engineering             时间/滞后/滚动/扩展/差分 特征
    selector                KBest/RFE/Lasso/相关性/方差/PCA 等选择与降维
"""

from .mixed_feature_handler import (
    MixedFeatureHandler,
    AdvancedMixedFeatureHandler,
    prepare_mixed_features,
)
from .engineering import (
    BaseFeatureEngineer,
    TimeFeatureEngineer,
    LagFeatureEngineer,
    RollingFeatureEngineer,
    ExpandingFeatureEngineer,
    DifferenceFeatureEngineer,
    CompositeFeatureEngineer,
    create_feature_engineer,
)
from .selector import (
    BaseFeatureSelector,
    KBestSelector,
    RFESelector,
    ModelBasedSelector,
    LassoSelector,
    VarianceSelector,
    CorrelationSelector,
    BaseFeatureReducer,
    PCAReducer,
    AutoFeatureEngineer,
    get_selector,
    get_reducer,
)

__all__ = [
    # mixed
    'MixedFeatureHandler',
    'AdvancedMixedFeatureHandler',
    'prepare_mixed_features',
    # engineering
    'BaseFeatureEngineer',
    'TimeFeatureEngineer',
    'LagFeatureEngineer',
    'RollingFeatureEngineer',
    'ExpandingFeatureEngineer',
    'DifferenceFeatureEngineer',
    'CompositeFeatureEngineer',
    'create_feature_engineer',
    # selector
    'BaseFeatureSelector',
    'KBestSelector',
    'RFESelector',
    'ModelBasedSelector',
    'LassoSelector',
    'VarianceSelector',
    'CorrelationSelector',
    'BaseFeatureReducer',
    'PCAReducer',
    'AutoFeatureEngineer',
    'get_selector',
    'get_reducer',
]
