"""
InferenceRunner 端到端演示 / End-to-end inference demo
========================================================

模拟生产部署链路:
  1. 训练阶段: 用 12 年 HPF 模拟数据训练 Ridge 模型 + 概率残差区间
  2. 保存阶段: InferenceRunner.save() 把 5 个状态打包成单一 pickle
  3. 重载阶段: 模拟"新进程"载入(实际上同进程, 但 InferenceRunner.load() 重建)
  4. 推理阶段: 拿"刚到的新月份"原始数据, 一行 .predict(latest_df) 出预测

验证:
  - load 后 adapter._is_fitted 自动为 True (不会误重新 fit 新数据 → 数据泄露)
  - predict 端到端 (validate → preprocess(fit=False) → feat_eng.transform
    → mfh.create_single_input → model.predict_probabilistic → 反归一化)
  - 同输入下 load-后预测 与 原 process 内 predict 结果一致 (允许 1e-6 浮点误差)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, *_HERE.parents):
    if (_p / 'configs').is_dir() and (_p / 'src').is_dir():
        for _q in (_p, _p / 'src'):
            if str(_q) not in sys.path:
                sys.path.insert(0, str(_q))
        break

import numpy as np
import pandas as pd

from configs.hpf.hpf_config import HPFConfig
from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.features.engineering import create_feature_engineer
from tsf_frame.features.mixed_feature_handler import MixedFeatureHandler
from tsf_frame.models.classical.ml_models import RidgeModel
from tsf_frame.deployment import InferenceRunner
from tsf_frame.utils.logger import get_logger

# 复用 run_hpf_forecast 的合成数据生成 (动态导入避免 pipelines/ 不是包导入路径)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'run_hpf_forecast', str(_HERE.parent / 'run_hpf_forecast.py'),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
generate_hpf_data = _mod.generate_hpf_data


def main() -> None:
    project_root = next(p for p in _HERE.parents if (p / 'src').is_dir())
    logger = get_logger('inference_demo', log_dir=str(project_root / 'logs' / 'runs'))
    logger.info('=' * 65)
    logger.info('  InferenceRunner 端到端演示')
    logger.info('=' * 65)

    # ── 1. 训练数据 (12 年 144 个月) ──────────────────────────────────────
    raw = generate_hpf_data(years=12, start_year=2012)
    target_col = 'monthly_deposit'
    logger.info(f'[1/5] 模拟数据: {raw.shape}, {raw.index[0].date()} ~ {raw.index[-1].date()}')

    # 切分: 前 130 个月训练, 末 14 个月保留做"未来新数据"演示
    train_raw = raw.iloc[:130].copy()
    future_raw = raw.iloc[-14:].copy()   # 14 = seq_len(12) + 推理时 lag/rolling buffer

    # ── 2. 训练流程 ────────────────────────────────────────────────────────
    logger.info('[2/5] 训练 Ridge + 残差概率区间')
    cfg = HPFConfig()
    adapter = HPFAdapter(cfg.to_adapter_config())
    processed, _ = adapter.preprocess(train_raw, fit=True)

    feat_eng = create_feature_engineer(
        feature_types=['time', 'lag', 'rolling'],
        config={
            'time_config':    {'features': ['month', 'quarter']},
            'lag_config':     {'target_cols': [target_col], 'lags': [1, 3, 6, 12]},
            'rolling_config': {'target_cols': [target_col], 'windows': [3, 12], 'stats': ['mean']},
        },
    )
    df_feat = feat_eng.fit_transform(processed).dropna()
    feature_cols = [c for c in df_feat.columns if c != target_col]

    # MixedFeatureHandler 走 2D 路径 (Ridge 是 sklearn 模型, 不要 3D), 仅用其
    # required_source_columns / min_required_rows / create_single_input 协议侧能力.
    # 为了让 ML 路径也走 InferenceRunner, 这里手工构造一个 thin handler.
    seq_len = 1
    mfh = MixedFeatureHandler(
        time_varying_cols=feature_cols,
        static_cols=[],
        target_col=target_col,
        seq_len=seq_len,
        pred_len=1,
    ).fit(df_feat)

    X_train_2d = df_feat[feature_cols].values.astype(np.float32)
    y_train_2d = df_feat[target_col].values.astype(np.float32).reshape(-1, 1)
    # 训练/验证切分 (验证残差用于 OOS CI)
    split = int(len(X_train_2d) * 0.85)
    model = RidgeModel(config={
        'probabilistic': True,
        'probabilistic_method': 'residual',
        'random_seed': 42,
    })
    model.fit(
        (X_train_2d[:split], y_train_2d[:split]),
        val_data=(X_train_2d[split:], y_train_2d[split:]),
    )
    model.feature_names = feature_cols
    logger.info(f'  训练样本 {split}, 验证样本 {len(X_train_2d) - split}, '
                f'残差源={model._residual_source}')

    # ── 3. 保存 artifact ──────────────────────────────────────────────────
    artifact_path = project_root / 'logs' / 'models' / 'inference_demo_ridge.pkl'
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    runner = InferenceRunner(
        adapter=adapter, feat_eng=feat_eng, mfh=mfh, model=model,
        model_config=model.config, target_col=target_col,
    )
    runner.save(str(artifact_path))
    logger.info(f'[3/5] 保存 artifact → {artifact_path}')

    # ── 4. 模拟新进程: load + 预测 ────────────────────────────────────────
    runner2 = InferenceRunner.load(str(artifact_path), model_cls=RidgeModel)
    logger.info(f'[4/5] load 完成, adapter._is_fitted={runner2.adapter._is_fitted}')
    assert runner2.adapter._is_fitted is True, 'load 后必须强制 _is_fitted=True'

    # ── 5. 推理新月份(原始 raw df, 不预处理) ───────────────────────────────
    # 这里 future_raw 是 14 个月, mfh.min_required_rows(feat_eng) 应该 ≤ 14
    min_rows = mfh.min_required_rows(feature_engineer=feat_eng)
    logger.info(f'[5/5] min_required_rows = {min_rows}, future_raw 有 {len(future_raw)} 行')

    prob = runner2.predict(future_raw, target_col=target_col)

    # 真值最后一个月 (推理目标的 ground truth 在原始 future_raw 末行)
    actual_last = float(future_raw[target_col].iloc[-1])
    logger.info(f'  预测 mean: {prob.mean[0]:.2f} (raw 量纲, 单位亿元)')
    if prob.lower is not None and prob.upper is not None:
        logger.info(f'  95% 区间:  [{prob.lower[0]:.2f}, {prob.upper[0]:.2f}]')
    logger.info(f'  对比真值:  {actual_last:.2f}')

    # ── 一致性检查: load-前后预测一致 ──
    prob_orig = runner.predict(future_raw, target_col=target_col)
    diff = abs(prob.mean[0] - prob_orig.mean[0])
    logger.info(f'  load 前后预测差: {diff:.2e} (应近 0)')
    assert diff < 1e-5, f'load 前后预测不一致! diff={diff}'

    logger.info('=' * 65)
    logger.info(f'  InferenceRunner 演示完成, artifact: {artifact_path}')
    logger.info('=' * 65)


if __name__ == '__main__':
    main()
