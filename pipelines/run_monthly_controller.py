"""
月度跑批主控 —— 遍历 TASKS, 每个任务: 读数据 → 监控比对 → (可选)重训 → 外推预测 → 落库.

数据源切换 (CSV 兜底 / Hive 真接入) 通过 DataSourceConfig 控制:
  - 默认从环境变量初始化 (TSF_HPF_SOURCE / TSF_HPF_SINK 等)
  - 也可在本文件 main() 里显式构造并 set_default_config(...) 覆盖

运行:
    # CSV 模式 (本地调试)
    TSF_HPF_SOURCE=csv python -m pipelines.run_monthly_controller

    # Hive + MySQL 生产模式
    export TSF_HPF_SOURCE=hive TSF_HPF_HIVE_HOST=hive-prod-01 \\
           TSF_HPF_SINK=mysql TSF_HPF_MYSQL_URL=mysql+pymysql://u:p@h:3306/hpf
    python -m pipelines.run_monthly_controller
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / 'src'))
sys.path.insert(0, str(_HERE.parents[1]))

from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.monitoring import ModelMonitor, SQLiteStore
from tsf_frame.utils.logger import get_logger

from configs.hpf.task_registry import TASKS
from configs.hpf.data_source_config import DataSourceConfig
from pipelines.production_loader import (
    fetch_data, fetch_last_prediction, save_predictions,
    set_default_config, get_default_config,
)
from pipelines.job_train import run_full_retrain
from pipelines.job_inference import run_future_forecast


def main(data_source_config: Optional[DataSourceConfig] = None) -> None:
    logger = get_logger('monthly_batch', log_dir='logs/runs')
    logger.info("Starting monthly batch process...")

    # 1. 数据源配置 —— 显式传入优先, 否则按环境变量初始化
    if data_source_config is not None:
        set_default_config(data_source_config)
    ds_cfg = get_default_config()
    logger.info(f"DataSource: {ds_cfg.to_dict()}")

    # 2. 初始化监控存储
    store = SQLiteStore('logs/monitor/hpf_monitor.db')

    for task in TASKS:
        task_id = task['task_id']
        task_name = task['task_name']
        logger.info(f"Processing task: {task_name} ({task_id})")

        try:
            # 3a. 加载配置与适配器
            config = task['config_builder']()
            adapter = HPFAdapter(config.to_adapter_config())
            monitor = ModelMonitor(model_id=task_id, store=store)
            target_col = config.data.target_columns[0]

            # 3b. 读取**训练**历史数据 (拉全量, 一般 5 年)
            #     task dict 可显式指定 csv_path 覆盖默认推导
            #     训练/监控比对走 sql_path; 推理走 sql_infer_path (短窗口, 见 3c)
            # / Fetch training data (full history); inference uses sql_infer_path below.
            df_train = fetch_data(
                sql_path=task['sql_path'],
                data_path=task.get('csv_path'),
                config=ds_cfg,
            )
            if df_train.empty:
                logger.warning(f"No training data fetched for {task_id}, skipping...")
                continue

            # ── Defensive: 训练 SQL 拉到的行数够不够 ──
            # 公式: seq_len + max(lags ∪ rolling_windows ∪ diff_periods) - 1 + pred_len
            # 不够则训练/推理会因 lag/rolling 大量 NaN 失败. 早抛错.
            # / Defensive check on training SQL rows; aligns with MixedFeatureHandler.min_required_rows.
            lookback_pool = list(getattr(config.features, 'lags', []) or [0])
            lookback_pool += list(getattr(config.features, 'rolling_windows', []) or [0])
            lookback_pool += list(getattr(config.features, 'diff_periods', []) or [0])
            max_lb = max(lookback_pool) if lookback_pool else 0
            required_train_rows = config.model.seq_len + max(0, max_lb - 1) + config.model.pred_len
            if len(df_train) < required_train_rows:
                logger.error(
                    f"{task_id}: 训练 SQL 拉到 {len(df_train)} 行 < 要求 {required_train_rows} "
                    f"(seq_len={config.model.seq_len}, max_lookback={max_lb}, "
                    f"pred_len={config.model.pred_len}). 跳过此任务."
                )
                continue

            # 确保日期索引
            if 'YCRQ' in df_train.columns:
                df_train['YCRQ'] = pd.to_datetime(df_train['YCRQ'])
                df_train.set_index('YCRQ', inplace=True)
            df_train.sort_index(inplace=True)

            # 4. 监控体检：比对上月预测值与本月真实值
            #    target_ts = 本月最后一行的索引时间 (即上月预测的 horizon=1 目标月)
            #    监控比对用训练 df 的尾部即可 (训练 SQL 一定覆盖最新月份)
            last_pred = fetch_last_prediction(task_id, target_col, config=ds_cfg)
            if last_pred is not None:
                actual_val = float(df_train[target_col].iloc[-1])
                target_ts = df_train.index[-1] if hasattr(df_train.index[-1], 'to_pydatetime') else datetime.now()
                monitor.record_prediction(
                    timestamp=target_ts,
                    prediction=float(last_pred),
                    actual=actual_val,
                )

            status = monitor.check_status()
            logger.info(f"Health Status: {status.alert_level}")

            # 5. 持续训练决策 (CT) —— 不存在 OR warning/critical 时重训
            #    AlertLevel 用小写字符串常量 (见 monitoring.interfaces.AlertLevel)
            model_path = f"logs/models/{task_id}_best.pkl"
            need_retrain = (
                not os.path.exists(model_path)
                or status.alert_level in ('warning', 'critical')
                or status.needs_retraining
            )
            if need_retrain:
                reason = (
                    'model file missing' if not os.path.exists(model_path)
                    else f"alert={status.alert_level}, recommendations={status.recommendations}"
                )
                logger.warning(f"Triggering retraining for {task_id}. Reason: {reason}")
                meta = run_full_retrain(df_train, config, model_path, adapter)
            else:
                logger.info("Model is healthy. Skipping retraining.")
                meta = None  # inference 会自动尝试加载

            # 3c. 读取**推理**短窗口数据 (典型 1-2 年, 避免每月 Hive 全量扫描)
            #     若 task 未指定 sql_infer_path → 退化为 sql_path (向后兼容旧 task)
            # / Fetch inference window data (short SQL); fallback to sql_path if not specified.
            sql_infer_path = task.get('sql_infer_path')
            if sql_infer_path and sql_infer_path != task['sql_path']:
                df_infer = fetch_data(
                    sql_path=sql_infer_path,
                    data_path=task.get('csv_infer_path'),
                    config=ds_cfg,
                )
                if 'YCRQ' in df_infer.columns:
                    df_infer['YCRQ'] = pd.to_datetime(df_infer['YCRQ'])
                    df_infer.set_index('YCRQ', inplace=True)
                df_infer.sort_index(inplace=True)
                # Defensive: 推理 SQL 拉到的行数够不够算 lag/rolling
                # / pred_len 不参与推理侧 (那是输出长度), 只检查特征工程的回看
                required_infer_rows = config.model.seq_len + max(0, max_lb - 1)
                if len(df_infer) < required_infer_rows:
                    logger.error(
                        f"{task_id}: 推理 SQL 拉到 {len(df_infer)} 行 < 要求 {required_infer_rows} "
                        f"(seq_len + max_lookback - 1). 跳过推理."
                    )
                    continue
                logger.info(
                    f"推理用短窗口 SQL: {len(df_infer)} 行 (训练 SQL 同时拉了 {len(df_train)} 行)"
                )
            else:
                df_infer = df_train
                logger.info(f"未配置 sql_infer_path, 推理沿用训练数据 ({len(df_infer)} 行)")

            # 6. 执行外推预测 (pred_len 由 config 决定, REQ_01 默认 60 个月)
            forecast_df = run_future_forecast(df_infer, config, model_path, adapter, meta)

            # 7. 结果持久化 (CSV 或 MySQL) —— 用 task_id 而非中文 task_name 做目录/表名
            location = save_predictions(task_id, forecast_df, config=ds_cfg)
            logger.info(f"Task {task_name} ({task_id}) finished successfully. Output → {location}")

        except Exception as e:
            logger.error(f"Task {task_name} failed: {str(e)}", exc_info=True)
            continue


if __name__ == "__main__":
    main()
