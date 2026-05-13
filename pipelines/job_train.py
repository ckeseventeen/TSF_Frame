import os
import pickle
from tsf_frame.features.engineering import create_feature_engineer
from tsf_frame.models.classical.ml_models import get_ml_model
from tsf_frame.utils.logger import get_logger

logger = get_logger('job_train')

def run_full_retrain(df_raw, config, model_path, adapter):
    """
    全量重训任务：处理特征、训练模型、保存状态
    """
    logger.info(f"Starting full retrain. Data shape: {df_raw.shape}")
    
    # 1. 业务预处理 (Fit scaler)
    processed_df, meta = adapter.preprocess(df_raw, fit=True)
    target_col = config.data.target_columns[0]
    
    # 2. 特征工程
    engineer = create_feature_engineer(
        ['time', 'lag', 'rolling', 'difference'], 
        config.to_feature_config()
    )
    df_feat = engineer.fit_transform(processed_df).dropna()
    
    # 3. 准备训练集
    X = df_feat.drop(columns=[target_col]).values
    y = df_feat[target_col].values
    
    # 4. 模型训练
    model = get_ml_model(config.model.model_name, config.to_model_config())
    model.fit((X, y))
    
    # 5. 持久化 (模型 + 适配器元数据)
    model.save_model(model_path)
    
    meta_path = model_path.replace('.pkl', '_meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
        
    logger.info(f"Retrain finished. Model saved to {model_path}")
    return meta
