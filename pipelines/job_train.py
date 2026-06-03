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
from tsf_frame.utils.metrics import MetricsCalculator

logger = get_logger('job_train')

# 树模型集合: 这些模型无法外推超出训练值域, 必须做差分
TREE_MODELS = {'xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'catboost', 'decision_tree'}


def _train_and_eval(model_name, X_train, y_train, X_val, y_val, model_cfg):
    """
    训练单个候选模型并在**验证集 level 空间**评估, 供自动选模公平横比.

    树模型走 DiffTransform (学 Δ), 评估时用 one-step-ahead 把 Δ 预测累加回
    level (anchor=真实前值); 非树模型直接预测 level. 两者最终都在同一 level
    量纲上算指标, 保证可比.

    Returns:
        (model, use_diff, metrics_dict)
    """
    use_diff = model_name in TREE_MODELS
    model = get_ml_model(model_name, model_cfg)

    if use_diff:
        diff_transform = DiffTransform()
        y_train_diff, X_train_fit = diff_transform.transform(y_train, X_train)
        y_val_diff = np.diff(y_val, prepend=y_train[-1])
        model.fit(train_data=(X_train_fit, y_train_diff), val_data=(X_val, y_val_diff))
        pred_val_diff = np.asarray(model.predict(X_val)).ravel()
        # one-step-ahead 还原 level: 每个 val 点用其真实前值做 anchor
        y_val_prev = np.concatenate([[y_train[-1]], y_val[:-1]])
        pred_val_level = y_val_prev + pred_val_diff
    else:
        model.fit(train_data=(X_train, y_train), val_data=(X_val, y_val))
        pred_val_level = np.asarray(model.predict(X_val)).ravel()

    metrics = MetricsCalculator.calculate_all(y_val, pred_val_level)
    return model, use_diff, metrics


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

    # 4. 候选模型自动选优
    #    配了 config.model.candidate_models 就横比择优; 为空则退回单模型 (向后兼容).
    #    树模型自动差分, 非树模型不差分 (见 _train_and_eval).
    candidates = list(getattr(config.model, 'candidate_models', None) or [])
    if not candidates:
        candidates = [config.model.model_name]
    model_cfg = config.to_model_config()

    if len(candidates) > 1:
        logger.info(f"Auto model selection over {len(candidates)} candidates: {candidates}")

    results = []  # [(name, model, use_diff, metrics), ...]
    for name in candidates:
        try:
            mdl, ud, metrics = _train_and_eval(name, X_train, y_train, X_val, y_val, model_cfg)
        except Exception as exc:  # 未安装/不支持的候选跳过, 不让整体失败
            logger.warning(f"  candidate '{name}' failed, skipped: {exc}")
            continue
        logger.info(
            f"  [{name}] val MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
            f"MAPE={metrics['MAPE']:.2%}  (diff={ud})"
        )
        results.append((name, mdl, ud, metrics))

    if not results:
        raise RuntimeError(f"All candidate models failed to train: {candidates}")

    # 按验证集 MAE 选最优 (level 量纲, 公平)
    model_name, model, use_diff, best_metrics = min(results, key=lambda r: r[3]['MAE'])
    if len(results) > 1:
        logger.info(
            f"✓ Selected '{model_name}' (val MAE={best_metrics['MAE']:.4f}, "
            f"MAPE={best_metrics['MAPE']:.2%}) out of {len(results)} trained candidates"
        )
    elif model_name in TREE_MODELS:
        logger.info(f"Tree model '{model_name}' — DiffTransform enabled")

    # 5. 持久化
    # 5a. 模型权重
    model.save_model(model_path)

    # 5b. 归一化元数据 (反归一化必需)
    meta_path = model_path.replace('.pkl', '_meta.pkl')
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    # 5c. 差分标记 + anchor + 选中的模型名 (推理端据此用对的模型类加载)
    diff_flag_path = model_path.replace('.pkl', '_diff_flag.pkl')
    with open(diff_flag_path, 'wb') as f:
        pickle.dump({
            'use_diff': use_diff,
            'last_train_value': float(y[-1]),  # 整个特征集最后一个值作为 anchor
            'feature_cols': feature_cols,
            'model_name': model_name,          # 自动选模选中者, job_inference 用它实例化
        }, f)

    logger.info(f"Retrain finished. Model '{model_name}' → {model_path}")
    return meta




