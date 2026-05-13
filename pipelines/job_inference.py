"""
生产原子任务：极速外推预测

职责：
  - 加载已训练模型 + 归一化元数据 + 差分标记
  - 自回归滚动预测未来 N 个月 (每步预测 1 个月, 回填特征后继续)
  - 树模型自动执行差分累加还原 (anchor + cumsum), 突破训练值域限制
  - 反归一化 + 非负裁剪, 输出业务可读的真实量纲预测
"""

import pandas as pd
import numpy as np
import pickle
from tsf_frame.features.engineering import create_feature_engineer
from tsf_frame.models.classical.ml_models import get_ml_model
from tsf_frame.utils.logger import get_logger

logger = get_logger('job_inference')


def _load_diff_flag(model_path: str) -> dict:
    """加载训练时保存的差分标记"""
    diff_flag_path = model_path.replace('.pkl', '_diff_flag.pkl')
    try:
        with open(diff_flag_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.warning(f"No diff_flag file found at {diff_flag_path}, assuming no diff")
        return {'use_diff': False, 'last_train_value': 0.0, 'feature_cols': []}


def run_future_forecast(recent_df, config, model_path, adapter, meta=None):
    """
    极速外推预测：滚动预测未来 N 个月

    Args:
        recent_df:   最新的历史全量数据 (DatetimeIndex)
        config:      HPFConfig 实例
        model_path:  模型文件路径
        adapter:     HPFAdapter 实例
        meta:        预处理元数据 (可选, 为 None 时自动从磁盘加载)

    Returns:
        final_df: 反归一化后的预测 DataFrame, 索引为未来月份日期
    """
    target_col = config.data.target_columns[0]
    pred_len = config.model.pred_len

    # 1. 加载元数据
    if meta is None:
        meta_path = model_path.replace('.pkl', '_meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

    # 2. 加载差分标记
    diff_flag = _load_diff_flag(model_path)
    use_diff = diff_flag['use_diff']

    # 3. 业务预处理 (只 Transform, 复用训练时学到的 scaler)
    #    🔴 若 adapter 是新实例 (典型场景: 月度跑批只推理, 不重训),
    #    _is_fitted=False, 必须先从 meta 把训练期 scaler 灌回去, 否则
    #    fit=False 会 raise RuntimeError.
    if not adapter._is_fitted and meta.get('scalers'):
        adapter._scalers = dict(meta['scalers'])
        adapter._is_fitted = True
        logger.info(f"Restored {len(meta['scalers'])} scalers from meta to adapter")
    processed_df, _ = adapter.preprocess(recent_df, fit=False)

    # 4. 初始化特征工程
    engineer = create_feature_engineer(
        ['time', 'lag', 'rolling', 'difference'],
        config.to_feature_config()
    )

    # 5. 加载模型
    model = get_ml_model(config.model.model_name, config.to_model_config())
    model.load_model(model_path)

    # 6. 滚动外推核心逻辑
    current_buffer = processed_df.copy()
    future_preds_norm = []  # 归一化空间的预测值

    # 差分模式: 需要一个"锚点" (最后已知的水平值) 用于累加还原
    if use_diff:
        # 用当前 buffer 末尾的归一化值作为锚点
        df_init = engineer.fit_transform(current_buffer).dropna()
        anchor = df_init[target_col].iloc[-1]
        logger.info(f"DiffTransform enabled. Anchor = {anchor:.4f}")
    else:
        anchor = None

    logger.info(f"Starting autoregressive forecast for {pred_len} steps (diff={use_diff})")

    for i in range(pred_len):
        # A. 重建特征
        df_feat = engineer.fit_transform(current_buffer)
        feature_cols = [c for c in df_feat.columns if c != target_col]
        X_last = df_feat[feature_cols].tail(1).values

        # B. 预测一步
        raw_pred = model.predict(X_last)[0][0]

        # C. 差分还原 vs 直接使用
        if use_diff:
            # raw_pred 是差分值 (delta), 需要累加到 anchor
            anchor = anchor + raw_pred
            y_next_norm = anchor
        else:
            y_next_norm = raw_pred

        # D. 构造新行并推入 buffer (用于下一步的 lag/rolling 特征更新)
        next_date = current_buffer.index[-1] + pd.DateOffset(months=1)
        new_row = pd.DataFrame({target_col: [y_next_norm]}, index=[next_date])

        # 填充协变量 (简单策略：延续最后一期)
        if config.data.feature_columns:
            for col in config.data.feature_columns:
                if col in current_buffer.columns:
                    new_row[col] = current_buffer[col].iloc[-1]

        current_buffer = pd.concat([current_buffer, new_row])
        future_preds_norm.append(y_next_norm)

    # 7. 后处理 (反归一化 + 非负裁剪)
    future_dates = pd.date_range(
        start=recent_df.index[-1] + pd.DateOffset(months=1),
        periods=pred_len,
        freq='MS'
    )

    final_raw_preds = np.array(future_preds_norm).reshape(-1, 1)
    final_df = adapter.postprocess(final_raw_preds, meta)
    final_df.index = future_dates

    logger.info(f"Forecast complete. Shape: {final_df.shape}, "
                f"Range: [{final_df.values.min():.2f}, {final_df.values.max():.2f}]")

    return final_df
