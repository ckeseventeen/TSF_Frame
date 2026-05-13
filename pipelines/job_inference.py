import pandas as pd
import numpy as np
import pickle
from tsf_frame.features.engineering import create_feature_engineer
from tsf_frame.models.classical.ml_models import get_ml_model
from tsf_frame.utils.logger import get_logger

logger = get_logger('job_inference')

def run_future_forecast(recent_df, config, model_path, adapter, meta=None):
    """
    极速外推预测：滚动预测未来 N 个月
    """
    target_col = config.data.target_columns[0]
    pred_len = config.model.pred_len
    
    # 1. 如果没有传入 meta，尝试从磁盘加载
    if meta is None:
        meta_path = model_path.replace('.pkl', '_meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

    # 2. 业务预处理 (只 Transform)
    processed_df, _ = adapter.preprocess(recent_df, fit=False)
    
    # 3. 初始化特征工程
    engineer = create_feature_engineer(
        ['time', 'lag', 'rolling', 'difference'], 
        config.to_feature_config()
    )
    
    # 4. 加载模型
    model = get_ml_model(config.model.model_name, config.to_model_config())
    model.load_model(model_path)

    # 5. 滚动外推核心逻辑
    current_buffer = processed_df.copy()
    future_preds = []
    
    logger.info(f"Starting autoregressive forecast for {pred_len} steps")
    
    for i in range(pred_len):
        # A. 更新特征
        df_feat = engineer.fit_transform(current_buffer)
        X_last = df_feat.tail(1).drop(columns=[target_col]).values
        
        # B. 预测一步 (归一化空间)
        y_next_norm = model.predict(X_last)[0][0]
        
        # C. 构造新行并推入 buffer
        next_date = current_buffer.index[-1] + pd.DateOffset(months=1)
        new_row = pd.DataFrame({target_col: [y_next_norm]}, index=[next_date])
        
        # 填充协变量 (简单策略：延续最后一期)
        for col in config.data.feature_columns:
            if col in current_buffer.columns:
                new_row[col] = current_buffer[col].iloc[-1]
                
        current_buffer = pd.concat([current_buffer, new_row])
        future_preds.append(y_next_norm)

    # 6. 后处理 (反归一化 + 非负裁剪)
    future_dates = pd.date_range(
        start=recent_df.index[-1] + pd.DateOffset(months=1), 
        periods=pred_len, 
        freq='MS'
    )
    
    final_raw_preds = np.array(future_preds).reshape(-1, 1)
    final_df = adapter.postprocess(final_raw_preds, meta)
    final_df.index = future_dates
    
    # 7. 计算衍生指标 (同比增长、环比增长)
    # 此处省略具体 pct_change 计算，建议在 adapter 或专用的 business_logic 模块中处理
    
    return final_df
