"""
住房公积金（HPF）业务预测配置

设计思路:
  - 月度数据量通常有限（5~20年 = 60~240条），优先使用树模型
  - 季节性显著（月度周期、年度周期），需要充分的滞后/滚动特征
  - 业务要求可解释性和置信区间，推荐 probabilistic=True + residual 方法
  - 对于政策分析场景，Linear/Ridge 可解释性更好
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .hpf_monitoring_config import HPFMonitoringConfig


@dataclass
class HPFDataConfig:
    """公积金数据配置"""
    # 目标预测列，可按需扩展
    target_columns: List[str] = field(default_factory=lambda: ['monthly_deposit'])

    # 协变量列（None 表示使用全部数值列）
    feature_columns: Optional[List[str]] = None

    # 数据频率（月度）
    freq: str = 'M'

    # 训练/验证/测试比例
    test_size: float = 0.15
    val_size: float = 0.10


@dataclass
class HPFFeatureConfig:
    """公积金特征工程配置"""

    # 时间特征（月度数据的关键特征）
    time_features: List[str] = field(default_factory=lambda: [
        'month', 'quarter', 'year', 'is_month_start', 'is_month_end'
    ])

    # 滞后特征: 1=环比, 3=季比, 6=半年比, 12=年同比
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])

    # 滚动统计窗口（季度/半年/年）
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12])
    rolling_stats: List[str] = field(default_factory=lambda: ['mean', 'std'])

    # 差分特征（环比变化、年同比变化）
    diff_periods: List[int] = field(default_factory=lambda: [1, 12])


@dataclass
class HPFModelConfig:
    """公积金模型配置"""

    # 推荐首选：月度小数据集树模型效果好、训练快、可解释
    model_name: str = 'xgboost'

    # 回看 24 个月（捕捉 2 个年度周期）
    seq_len: int = 24
    # 预测步数（1=下月, 3=季度预测, 12=年度预测）
    pred_len: int = 1
    output_size: int = 1

    # 概率预测：月度数据推荐残差法，稳定性优于 mc_dropout
    probabilistic: bool = True
    probabilistic_method: str = 'residual'
    confidence_level: float = 0.95

    # XGBoost/LightGBM 推荐参数（适合月度中小数据集）
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_seed: int = 42

    # LSTM 参数（数据量充足时可切换）
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    train_epochs: int = 200
    batch_size: int = 16


@dataclass
class HPFConfig:
    """
    住房公积金业务预测总配置

    使用示例:
        config = HPFConfig()
        config.data.target_columns = ['monthly_deposit', 'monthly_withdrawal']
        config.model.pred_len = 3  # 预测未来 3 个月

        adapter_cfg = config.to_adapter_config()
        model_cfg   = config.to_model_config()
        feature_cfg = config.to_feature_config()
    """
    project_name: str = 'HPF_Forecast'

    data: HPFDataConfig = field(default_factory=HPFDataConfig)
    features: HPFFeatureConfig = field(default_factory=HPFFeatureConfig)
    model: HPFModelConfig = field(default_factory=HPFModelConfig)
    monitoring: HPFMonitoringConfig = field(default_factory=HPFMonitoringConfig)

    # 预处理参数
    handle_outliers: bool = True
    outlier_method: str = 'iqr'     # 月度数据推荐 IQR 法
    normalization: str = 'zscore'   # z-score 对树模型无影响，对 DL 模型有益

    def to_adapter_config(self) -> Dict[str, Any]:
        return {
            'business_type': 'hpf',
            'target_columns': self.data.target_columns,
            'feature_columns': self.data.feature_columns,
            'handle_outliers': self.handle_outliers,
            'outlier_method': self.outlier_method,
            'normalization': self.normalization,
        }

    def to_model_config(self) -> Dict[str, Any]:
        m = self.model
        cfg = {
            'model_name': m.model_name,
            'seq_len': m.seq_len,
            'pred_len': m.pred_len,
            'output_size': m.output_size,
            'probabilistic': m.probabilistic,
            'probabilistic_method': m.probabilistic_method,
            'confidence_level': m.confidence_level,
            'random_seed': m.random_seed,
        }
        # 树模型参数
        cfg.update({
            'n_estimators': m.n_estimators,
            'max_depth': m.max_depth,
            'learning_rate': m.learning_rate,
            'subsample': m.subsample,
            'colsample_bytree': m.colsample_bytree,
        })
        # 深度学习参数（切换为 lstm/transformer 时生效）
        # 注意: input_size 不在此处设置. DL 模型的 input_size 是**特征工程后的特征
        # 维度**, 在配置阶段还不知道(取决于 lag/rolling/time 特征生成结果).
        # caller 在调 get_dl_model 前必须基于 X_train.shape[-1] 显式设置:
        #   dl_cfg = cfg.to_adapter_config()
        #   dl_cfg['input_size'] = X_train.shape[-1]
        #   model = get_dl_model('lstm', dl_cfg)
        # 历史: 之前误设为 len(target_columns), 在 DL 路径会让 LSTM input_size=1
        # / input_size omitted; caller MUST set after feature engineering
        cfg.update({
            'hidden_size': m.hidden_size,
            'num_layers': m.num_layers,
            'dropout': m.dropout,
            'train_epochs': m.train_epochs,
            'batch_size': m.batch_size,
        })
        return cfg

    def to_feature_config(self) -> Dict[str, Any]:
        f = self.features
        return {
            'time_config': {'features': f.time_features},
            'lag_config': {
                'target_cols': self.data.target_columns,
                'lags': f.lags,
            },
            'rolling_config': {
                'target_cols': self.data.target_columns,
                'windows': f.rolling_windows,
                'stats': f.rolling_stats,
            },
            'difference_config': {
                'target_cols': self.data.target_columns,
                'periods': f.diff_periods,
            },
        }
