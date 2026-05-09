"""通用单模型训练入口 / Generic single-model training entry.

使用示例:
    # 经典 ML 模型, darts 内置数据集
    python pipelines/train_model.py --model xgboost --dataset air_passengers

    # darts 类型 + 默认数据集
    python pipelines/train_model.py --model ridge --dataset darts

    # 合成数据 (无需外部依赖)
    python pipelines/train_model.py --model lightgbm --dataset synthetic

    # 深度学习模型 (需要 torch)
    python pipelines/train_model.py --model lstm --dataset air_passengers --epochs 20

该脚本是一个**最小可运行骨架**, 作为业务团队二次开发的起点:
  - ``--model``: 任意已注册的 ML 或 DL 模型 (从注册表动态读取)
  - ``--dataset``: 公开数据集类型 (darts/synthetic/financial/energy) 或
                   darts 内置数据集名 (air_passengers/etth1/...);
                   会自动判断走哪条路径
  - 滑窗组装 (X seq → y next), 训练 → 测试 → 输出 metrics
  - 模型权重保存到 ``--output-dir`` (默认 logs/models/)

真实生产场景下, 请把 `--config <path>` 加进来 (读取 configs/xxx),
并把数据来源替换为 business/<name>_adapter.py 中的 preprocess 接口.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

# 让脚本无论是否执行过 `pip install -e .` 都能导入 configs.* 与 tsf_frame.*
# 同时锚定项目根, 让日志/模型权重始终落在 <root>/logs/ 下而不是 CWD.
# / Project root anchor; CWD-independent
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent  # 兜底
for _p in (_HERE.parent, *_HERE.parents):
    if (_p / 'configs').is_dir() and (_p / 'src').is_dir():
        _PROJECT_ROOT = _p
        for _q in (_p, _p / 'src'):
            if str(_q) not in sys.path:
                sys.path.insert(0, str(_q))
        break

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsf_frame.utils.logger import get_logger  # noqa: E402
from tsf_frame.utils.metrics import MetricsCalculator  # noqa: E402
from tsf_frame.models.classical.ml_models import (  # noqa: E402
    MODEL_REGISTRY as ML_MODEL_REGISTRY,
    get_ml_model,
)
from tsf_frame.models.transformer.transformer_models import (  # noqa: E402
    DL_MODEL_REGISTRY,
    get_dl_model,
)
from tsf_frame.data.datasets.public_datasets import (  # noqa: E402
    DATASET_REGISTRY,
    get_dataset,
)


# 模型清单从注册表动态生成 — 加新模型不用维护双份
# / Model lists derived from registries (no double maintenance)
CLASSICAL_MODELS = set(ML_MODEL_REGISTRY.keys())
DL_MODELS = set(DL_MODEL_REGISTRY.keys())
ALL_MODELS = sorted(CLASSICAL_MODELS | DL_MODELS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='TSF_Frame generic trainer')
    p.add_argument('--model', required=True,
                   choices=ALL_MODELS,
                   help='Model name (registered in ML/DL registries)')
    p.add_argument('--dataset', required=True,
                   help='Dataset type (darts/synthetic/financial/energy) '
                        'OR a darts dataset_name (air_passengers/etth1/...)')
    p.add_argument('--seq-len', type=int, default=12,
                   help='Sliding-window length used to build (X, y) pairs')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=20,
                   help='DL-only: number of training epochs')
    p.add_argument('--batch-size', type=int, default=32,
                   help='DL-only: training batch size')
    p.add_argument('--device', default='cpu',
                   help='DL-only: torch device, e.g. cpu / cuda / cuda:0')
    p.add_argument(
        '--output-dir',
        default=str(_PROJECT_ROOT / 'logs' / 'models'),
        help='Where to save model checkpoint (default: <root>/logs/models)',
    )
    return p.parse_args()


def _load_dataframe(dataset_arg: str) -> pd.DataFrame:
    """
    智能加载: dataset_arg 命中 DATASET_REGISTRY 时按"类型"加载,
    否则当作 darts 内置数据集名加载.

    / Smart loader: treat as dataset type if registered, else as darts name.
    """
    if dataset_arg in DATASET_REGISTRY:
        ds = get_dataset(dataset_arg)
    else:
        # 默认走 darts (内置常用时序数据集)
        ds = get_dataset('darts', {'dataset_name': dataset_arg})
    return ds.load()


def _to_sliding_window(
    series: np.ndarray, seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    把一维序列切成滑窗 (X seq, y next): X 形状 (N, seq_len), y 形状 (N,).

    / Slice 1D series into (X, y) where X is shape (N, seq_len), y is (N,).
    """
    if len(series) <= seq_len:
        raise ValueError(
            f'series length {len(series)} <= seq_len {seq_len}; '
            f'reduce --seq-len or use a longer dataset'
        )
    X = np.lib.stride_tricks.sliding_window_view(series, seq_len)[:-1]
    y = series[seq_len:]
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def main() -> int:
    args = parse_args()
    logger = get_logger(
        'train_model',
        log_dir=str(_PROJECT_ROOT / 'logs' / 'runs'),
    )
    logger.info('Trainer started: model=%s  dataset=%s  seq_len=%d',
                args.model, args.dataset, args.seq_len)

    # ── 1. 加载数据集 ──────────────────────────────────────────────
    try:
        df = _load_dataframe(args.dataset)
    except Exception as exc:
        logger.error('Failed to load dataset %s: %s', args.dataset, exc)
        return 1
    if df is None or len(df) == 0:
        logger.error('Dataset %s returned empty DataFrame', args.dataset)
        return 1

    target = df.columns[-1]   # 约定: 最后一列为目标 / Convention: last column = target
    series = df[target].to_numpy(dtype=float)
    logger.info('Loaded %d rows; target column = %r', len(df), target)

    # ── 2. 滑窗 + train/test 切分 ───────────────────────────────────
    X_full, y_full = _to_sliding_window(series, seq_len=args.seq_len)
    split = int(len(X_full) * (1 - args.test_size))
    X_train, X_test = X_full[:split], X_full[split:]
    y_train, y_test = y_full[:split], y_full[split:]
    logger.info('Train: %d  Test: %d  feature_dim: %d',
                len(X_train), len(X_test), X_train.shape[1])

    # ── 3. 训练 + 预测 ──────────────────────────────────────────────
    save_method = 'save_model'
    if args.model in CLASSICAL_MODELS:
        # ML 路径: X 保持二维 (N, seq_len) 即可
        # / ML path: X stays 2D (N, seq_len)
        model_config = {'random_seed': 42}
        model = get_ml_model(args.model, model_config)
        model.fit(
            train_data=(X_train, y_train),
            val_data=(X_test, y_test),
        )
        y_pred = model.predict(X_test)
    else:
        # DL 路径: X 需 reshape 为 (N, seq_len, 1) 三维; y → (N, 1) 单输出
        # / DL path: X reshape to (N, seq_len, 1); y to (N, 1)
        dl_config = {
            'input_size': 1,
            'output_size': 1,
            'seq_len': args.seq_len,
            'train_epochs': args.epochs,
            'batch_size': args.batch_size,
            'device': args.device,
            'learning_rate': 0.001,
        }
        model = get_dl_model(args.model, dl_config)
        X_train_3d = X_train.reshape(*X_train.shape, 1)
        X_test_3d = X_test.reshape(*X_test.shape, 1)
        y_train_2d = y_train.reshape(-1, 1)
        y_test_2d = y_test.reshape(-1, 1)
        model.fit(
            train_data=(X_train_3d, y_train_2d),
            val_data=(X_test_3d, y_test_2d),
        )
        y_pred = model.predict(X_test_3d)

    # ── 4. 评估指标 ────────────────────────────────────────────────
    metrics = MetricsCalculator.calculate_all(
        y_test, np.asarray(y_pred).ravel(),
    )
    logger.info('=' * 50)
    logger.info('Test metrics:')
    for k, v in metrics.items():
        logger.info('  %-6s = %.4f', k, v)
    logger.info('=' * 50)

    # ── 5. 保存模型 ────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_ext = '.pkl' if args.model in CLASSICAL_MODELS else '.pt'
    ckpt = Path(args.output_dir) / f'{args.model}_{args.dataset}{ckpt_ext}'
    if hasattr(model, save_method):
        try:
            getattr(model, save_method)(str(ckpt))
            logger.info('Saved checkpoint to %s', ckpt)
        except Exception as exc:  # noqa: BLE001
            logger.warning('Save failed (%s); continuing.', exc)
    return 0


if __name__ == '__main__':
    sys.exit(main())
