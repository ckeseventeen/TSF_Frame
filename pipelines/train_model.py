"""通用单模型训练入口 / Generic single-model training entry.

使用示例:
    # 经典 ML 模型
    python pipelines/train_model.py --model xgboost --dataset airpassengers
    # 深度学习模型
    python pipelines/train_model.py --model lstm --dataset sunspots --epochs 20

该脚本是一个最小可运行骨架, 作为业务团队二次开发的起点:
  - 读取 `--model` 指定的模型名, 使用 models/__init__.py 里的工厂函数
  - 读取 `--dataset` 指定的公开数据集, 使用 tsf_frame.data.datasets.public_datasets.get_dataset
  - 划分训练/测试, 输出 metrics 到 stdout, 模型权重保存到 `experiments/checkpoints/`

真实生产场景下, 请把 `--config <path>` 支持加进来（读取 configs/xxx），
并把数据来源替换为 business/<name>_adapter.py 中的 preprocess 接口。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 让脚本无论是否执行过 `pip install -e .` 都能导入 configs.* 与 tsf_frame.*
_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, *_HERE.parents):
    if (_p / 'configs').is_dir() and (_p / 'src').is_dir():
        for _q in (_p, _p / 'src'):
            if str(_q) not in sys.path:
                sys.path.insert(0, str(_q))
        break

import numpy as np  # noqa: E402

from tsf_frame.utils.logger import get_logger  # noqa: E402
from tsf_frame.utils.metrics import MetricsCalculator  # noqa: E402


CLASSICAL_MODELS = {
    'linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting',
    'xgboost', 'lightgbm', 'catboost', 'svr', 'knn', 'decision_tree',
}
DL_MODELS = {'lstm', 'transformer', 'informer', 'moirai'}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='TSF_Frame generic trainer')
    p.add_argument('--model', required=True, help=f'one of {sorted(CLASSICAL_MODELS | DL_MODELS)}')
    p.add_argument('--dataset', required=True, help='public dataset name, e.g. airpassengers')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=10, help='DL-only: number of training epochs')
    p.add_argument('--output-dir', default='experiments/checkpoints')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger('train_model', log_dir='logs')
    logger.info('Trainer started: model=%s dataset=%s', args.model, args.dataset)

    from tsf_frame.data.datasets.public_datasets import get_dataset

    df = get_dataset(args.dataset)
    if df is None or len(df) == 0:
        logger.error('Dataset %s unavailable; aborting.', args.dataset)
        return 1

    target = df.columns[-1]  # 约定：最后一列为目标
    split = int(len(df) * (1 - args.test_size))
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    X_train = np.arange(len(train_df)).reshape(-1, 1).astype(float)
    y_train = train_df[target].to_numpy(dtype=float)
    X_test = np.arange(len(train_df), len(df)).reshape(-1, 1).astype(float)
    y_test = test_df[target].to_numpy(dtype=float)

    if args.model in CLASSICAL_MODELS:
        from tsf_frame.models.classical.ml_models import get_ml_model
        model = get_ml_model(args.model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif args.model in DL_MODELS:
        from tsf_frame.models.transformer.transformer_models import get_dl_model
        model = get_dl_model(args.model, input_dim=X_train.shape[1])
        model.fit(X_train, y_train, epochs=args.epochs)
        y_pred = model.predict(X_test)
    else:
        logger.error('Unknown model: %s', args.model)
        return 2

    metrics = MetricsCalculator.compute_all(y_test, np.asarray(y_pred).ravel())
    for k, v in metrics.items():
        logger.info('  %s = %.4f', k, v)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = Path(args.output_dir) / f'{args.model}_{args.dataset}.pkl'
    if hasattr(model, 'save'):
        try:
            model.save(str(ckpt))
            logger.info('Saved checkpoint to %s', ckpt)
        except Exception as exc:  # noqa: BLE001
            logger.warning('Save failed (%s); continuing.', exc)
    return 0


if __name__ == '__main__':
    sys.exit(main())
