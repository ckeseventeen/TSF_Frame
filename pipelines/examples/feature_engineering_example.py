import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
for _p in (_HERE.parent, *_HERE.parents):
    if (_p / 'configs').is_dir() and (_p / 'src').is_dir():
        for _q in (_p, _p / 'src'):
            if str(_q) not in sys.path:
                sys.path.insert(0, str(_q))
        break

from configs.base_config import BaseConfig
from tsf_frame.utils.logger import get_logger
from tsf_frame.features.engineering import create_feature_engineer
from tsf_frame.features.selector import (
    AutoFeatureEngineer,
    KBestSelector,
    LassoSelector,
    VarianceSelector,
    CorrelationSelector,
    PCAReducer
)


def generate_sample_data(hours=1000):
    start_date = datetime.now() - timedelta(hours=hours)
    dates = [start_date + timedelta(hours=i) for i in range(hours)]
    
    np.random.seed(42)
    base_load = 100
    hour_of_day = np.array([d.hour for d in dates])
    
    daily_pattern = 20 * np.sin(2 * np.pi * hour_of_day / 24)
    noise = np.random.normal(0, 5, hours)
    
    power_load = base_load + daily_pattern + noise
    power_load = np.maximum(power_load, 20)
    
    temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(hours) / 8760) + np.random.normal(0, 2, hours)
    humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(hours) / 8760 + np.pi) + np.random.normal(0, 5, hours)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'power_load': power_load,
        'temperature': temperature,
        'humidity': humidity,
        'feature1': np.random.normal(0, 1, hours),
        'feature2': np.random.normal(0, 1, hours),
        'feature3': np.random.normal(0, 1, hours),
        'noise_feature': np.random.normal(0, 10, hours),
        'constant_feature': 5.0
    })
    data.set_index('timestamp', inplace=True)
    
    return data


def main():
    config = BaseConfig()
    logger = get_logger('feature_engineering_example', log_dir=config.log_dir)
    
    logger.info('='*60)
    logger.info('通用特征工程功能示例')
    logger.info('='*60)
    
    logger.info('\n1. 生成示例数据...')
    data = generate_sample_data()
    logger.info(f'原始数据形状: {data.shape}')
    logger.info(f'原始数据列: {list(data.columns)}')
    
    target_col = 'power_load'
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    logger.info(f'\n特征数据形状: {X.shape}')
    logger.info(f'目标变量形状: {y.shape}')
    
    logger.info('\n2. 步骤1: 生成基础特征...')
    feature_config = {
        'time_config': {'features': ['hour', 'weekday', 'is_weekend']},
        'lag_config': {'target_cols': ['temperature', 'humidity'], 'lags': [1, 2, 3, 6, 12]},
        'rolling_config': {
            'target_cols': ['temperature', 'humidity'],
            'windows': [6, 12],
            'stats': ['mean', 'std']
        }
    }
    
    feature_engineer = create_feature_engineer(
        feature_types=['time', 'lag', 'rolling'],
        config=feature_config
    )
    
    X_with_features = feature_engineer.fit_transform(X)
    X_with_features = X_with_features.dropna()
    y_aligned = y.loc[X_with_features.index]
    
    logger.info(f'生成特征后数据形状: {X_with_features.shape}')
    logger.info(f'生成的特征: {feature_engineer.get_feature_names()}')
    
    logger.info('\n3. 步骤2: 方差选择 (去除低方差特征)...')
    variance_selector = VarianceSelector(threshold=0.01)
    X_after_variance = variance_selector.fit_transform(X_with_features)
    logger.info(f'方差选择后数据形状: {X_after_variance.shape}')
    logger.info(f'移除的特征: {[f for f in X_with_features.columns if f not in X_after_variance.columns]}')

    logger.info('\n4. 步骤3: 相关性选择 (去除高相关性特征)...')
    corr_selector = CorrelationSelector(threshold=0.9)
    X_after_corr = corr_selector.fit_transform(X_after_variance)
    logger.info(f'相关性选择后数据形状: {X_after_corr.shape}')
    logger.info(f'移除的特征: {[f for f in X_after_variance.columns if f not in X_after_corr.columns]}')

    logger.info('\n5. 步骤4: KBest特征选择 (F值)...')
    kbest_selector = KBestSelector(k=15)
    X_kbest = kbest_selector.fit_transform(X_after_corr, y_aligned)
    logger.info(f'KBest选择后数据形状: {X_kbest.shape}')
    logger.info(f'选择的特征: {kbest_selector.get_selected_features()}')
    
    if kbest_selector.feature_importances:
        logger.info('\n特征重要性 (Top 10):')
        sorted_importance = sorted(kbest_selector.feature_importances.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
        for feat, imp in sorted_importance:
            logger.info(f'  {feat}: {imp:.4f}')
    
    logger.info('\n6. 步骤5: Lasso特征选择...')
    lasso_selector = LassoSelector(alpha=0.01)
    X_lasso = lasso_selector.fit_transform(X_after_corr, y_aligned)
    logger.info(f'Lasso选择后数据形状: {X_lasso.shape}')
    logger.info(f'选择的特征: {lasso_selector.get_selected_features()}')
    
    if lasso_selector.feature_importances:
        logger.info('\nLasso特征重要性 (Top 10):')
        sorted_importance = sorted(lasso_selector.feature_importances.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
        for feat, imp in sorted_importance:
            if imp > 0:
                logger.info(f'  {feat}: {imp:.4f}')
    
    logger.info('\n7. 步骤6: PCA降维...')
    pca_reducer = PCAReducer(n_components=0.95, use_scaler=True)
    X_pca = pca_reducer.fit_transform(X_kbest)
    logger.info(f'PCA降维后数据形状: {X_pca.shape}')
    
    if hasattr(pca_reducer, 'get_explained_variance'):
        explained_var = pca_reducer.get_explained_variance()
        logger.info(f'累计解释方差: {np.sum(explained_var):.4f}')
        logger.info(f'各主成分解释方差: {[f"{v:.4f}" for v in explained_var[:5]]}')
    
    logger.info('\n8. 使用AutoFeatureEngineer自动化流水线...')
    auto_fe = AutoFeatureEngineer()
    
    auto_fe.add_feature_engineer(feature_engineer)
    auto_fe.add_selector(VarianceSelector(threshold=0.01))
    auto_fe.add_selector(CorrelationSelector(threshold=0.9))
    auto_fe.add_selector(KBestSelector(k=15))
    
    X_final = auto_fe.fit_transform(X, y)
    logger.info(f'自动化流水线后数据形状: {X_final.shape}')
    
    logger.info('\n' + '='*60)
    logger.info('通用特征工程功能示例完成!')
    logger.info('='*60)
    logger.info(f'\n日志保存到: {config.log_dir}')


if __name__ == '__main__':
    main()
