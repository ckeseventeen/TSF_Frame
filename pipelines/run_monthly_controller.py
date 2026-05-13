import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / 'src'))
sys.path.insert(0, str(_HERE.parents[1]))

from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.monitoring import ModelMonitor, SQLiteStore
from tsf_frame.utils.logger import get_logger

from configs.hpf.task_registry import TASKS
from pipelines.production_loader import fetch_hive_data, fetch_last_prediction, save_to_mysql
from pipelines.job_train import run_full_retrain
from pipelines.job_inference import run_future_forecast

def main():
    logger = get_logger('monthly_batch', log_dir='logs/runs')
    logger.info("Starting monthly batch process...")

    # 初始化监控存储
    store = SQLiteStore('logs/monitor/hpf_monitor.db')

    for task in TASKS:
        task_id = task['task_id']
        task_name = task['task_name']
        logger.info(f"Processing task: {task_name} ({task_id})")

        try:
            # 1. 加载配置与适配器
            config = task['config_builder']()
            adapter = HPFAdapter(config.to_adapter_config())
            monitor = ModelMonitor(model_id=task_id, store=store)
            target_col = config.data.target_columns[0]

            # 2. 读取 Hive 历史全量数据
            df_raw = fetch_hive_data(task['sql_path'])
            if df_raw.empty:
                logger.warning(f"No data fetched for {task_id}, skipping...")
                continue
                
            # 确保日期索引
            if 'YCRQ' in df_raw.columns:
                df_raw['YCRQ'] = pd.to_datetime(df_raw['YCRQ'])
                df_raw.set_index('YCRQ', inplace=True)
            df_raw.sort_index(inplace=True)

            # 3. 监控体检：比对上月预测值与本月真实值
            last_pred = fetch_last_prediction(task_id, target_col)
            if last_pred is not None:
                actual_val = df_raw[target_col].iloc[-1]
                # 记录到监控库
                monitor.record_prediction(
                    timestamp=datetime.now(),
                    prediction=last_pred,
                    actual=actual_val,
                    target_col=target_col
                )
            
            status = monitor.check_status()
            logger.info(f"Health Status: {status.alert_level}")

            # 4. 持续训练决策 (CT)
            model_path = f"logs/models/{task_id}_best.pkl"
            
            # 触发条件：模型不存在 OR 监控报警 (WARNING/CRITICAL)
            if not os.path.exists(model_path) or status.alert_level in ['WARNING', 'CRITICAL']:
                logger.warning(f"Triggering retraining for {task_id}. Reason: {status.message}")
                meta = run_full_retrain(df_raw, config, model_path, adapter)
            else:
                logger.info(f"Model is healthy. Skipping retraining.")
                meta = None # inference 会自动尝试加载

            # 5. 执行外推预测 (60个月)
            forecast_df = run_future_forecast(df_raw, config, model_path, adapter, meta)
            
            # 6. 结果持久化与推送
            save_to_mysql(task_name, forecast_df)
            logger.info(f"Task {task_name} finished successfully.")

        except Exception as e:
            logger.error(f"Task {task_name} failed: {str(e)}", exc_info=True)
            continue

if __name__ == "__main__":
    main()
