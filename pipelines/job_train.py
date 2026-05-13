"""
生产原子任务：全量特征工程与模型重训

职责：
  - 对原始数据做业务预处理 (adapter.preprocess, fit=True 学习 scaler)
  - 特征工程 (时间/滞后/滚动/差分)
  - train/val 分割 (val 用于 OOS 残差置信区间, 避免过拟合低估区间)
  - 树模型自动启用差分训练 (DiffTransform), 解决外推值域锁死问题
  - 持久化模型 (.pkl) + 归一化元数据 (_meta.pkl) + 差分标记 (_diff_flag)
"""

import os
import pickle
import numpy as np
from tsf_frame.features.engineering import create_feature_engineer
from tsf_frame.models.classical.ml_models import get_ml_model
from tsf_frame.utils.logger import get_logger
from tsf_frame.utils.target_transforms import DiffTransform

logger = get_logger('job_train')

# 树模型集合: 这些模型无法外推超出训练值域, 必须做差分
TREE_MODELS = {'xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'catboost', 'decision_tree'}


def run_full_retrain(df_raw, config, model_path, adapter):
    """
    全量重训任务：处理特征、训练模型、保存状态

    Args:
        df_raw:      原始 DataFrame (DatetimeIndex)
        config:      HPFConfig 实例
        model_path:  模型保存路径 (如 logs/models/REQ_01_best.pkl)
        adapter:     HPFAdapter 实例

    Returns:
        meta: 预处理元数据 (含 scalers), 供 job_inference 使用
    """
    logger.info(f"Starting full retrain. Data shape: {df_raw.shape}")

    # 1. 业务预处理 (Fit scaler — 学习归一化参数)
    processed_df, meta = adapter.preprocess(df_raw, fit=True)
    target_col = config.data.target_columns[0]

    # 2. 特征工程
    engineer = create_feature_engineer(
        ['time', 'lag', 'rolling', 'difference'],
        config.to_feature_config()
    )
    df_feat = engineer.fit_transform(processed_df).dropna()

    # 3. train / val 分割
    #    val 用于 OOS 残差 → 概率区间更可靠 (尤其对 XGBoost 等易过拟合模型)
    feature_cols = [c for c in df_feat.columns if c != target_col]
    X = df_feat[feature_cols].values
    y = df_feat[target_col].values

    n_val = max(6, int(len(X) * config.data.val_size))
    n_train = len(X) - n_val
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    logger.info(f"Train: {n_train} samples, Val: {n_val} samples, Features: {len(feature_cols)}")

    # 4. 差分决策: 树模型自动启用 DiffTransform
    model_name = config.model.model_name
    use_diff = model_name in TREE_MODELS
    diff_transform = None

    model = get_ml_model(model_name, config.to_model_config())

    if use_diff:
        logger.info(f"Tree model '{model_name}' detected — enabling DiffTransform")
        diff_transform = DiffTransform()
        y_train_diff, X_train_fit = diff_transform.transform(y_train, X_train)
        # val 也需要做差分 (用于 OOS 残差)
        y_val_diff = np.diff(y_val, prepend=y_train[-1])
        model.fit(
            train_data=(X_train_fit, y_train_diff),
            val_data=(X_val, y_val_diff),
        )
    else:
        model.fit(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
        )

    # 5. 持久化
    # 5a. 模型权重
    model.save_model(model_path)

    # 5b. 归一化元数据 (反归一化必需)
    meta_path = model_path.replace('.pkl', '_meta.pkl')
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    # 5c. 差分标记 + 最后训练水平值 (推理时累加还原需要)
    diff_flag_path = model_path.replace('.pkl', '_diff_flag.pkl')
    with open(diff_flag_path, 'wb') as f:
        pickle.dump({
            'use_diff': use_diff,
            'last_train_value': float(y[-1]),  # 整个特征集最后一个值作为 anchor
            'feature_cols': feature_cols,
        }, f)

    logger.info(f"Retrain finished. Model → {model_path}")
    return meta
