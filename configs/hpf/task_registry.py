"""
HPF 业务任务注册总线 / HPF business task registry.

约定 / Conventions:
  - 每个任务 (REQ_XX) 必须有 sql_path (训练 / 监控比对 用, 拉全量历史)
  - **可选** sql_infer_path: 月度跑批推理用的短窗口 SQL.
    未指定时退化为 sql_path (即训练/推理同一份 SQL, 旧行为).
    指定时, run_monthly_controller 用 sql_infer_path 拉数据喂 run_future_forecast,
    显著减少 Hive 全量扫描的 I/O 浪费.
  - config_builder 返回 HPFConfig 实例

为什么训练/推理分两份 SQL?
  - 训练拉 60 个月: 滑窗/flatten 后产足够多的样本, 模型更鲁棒
  - 推理拉 24 个月: 只需要"够算 lag/rolling 特征 + 喂模型 base buffer"的近期数据,
    每月跑批避免全量扫描
  - 两份 SQL 各自是合法可独立执行的 SQL, DBA 能直接 copy 到 HiveCLI 验证,
    不引入模板渲染语法
"""

from configs.hpf.hpf_config import HPFConfig


def get_collection_prediction_config() -> HPFConfig:
    """归集额预测专属配置"""
    cfg = HPFConfig()

    # 1. 核心目标设置
    cfg.data.target_columns = ['YDGJJE']

    # 2. 特征列定义 (对应文档协变量)
    cfg.data.feature_columns = [
        'GJZHSL', 'GDPZZL', 'JMSRSP', 'JCBLBH', 'CSRKQLQ', 'XZJY'
    ]

    # 3. 特征工程细节
    cfg.features.lags = [1, 2, 3, 6, 12]
    cfg.features.rolling_windows = [3, 6, 12]

    # 4. 模型与预测步数 (未来 60 个月)
    cfg.model.model_name = 'xgboost'   # 单模型兜底 (candidate_models 为空时用)
    # 自动选模: 每月重训时横比这些候选, 按验证集 MAE 择优落盘
    cfg.model.candidate_models = ['xgboost', 'random_forest', 'ridge']
    cfg.model.pred_len = 60

    # 5. 监控报警阈值 (10%)
    cfg.monitoring.performance_alert_threshold = 0.10

    return cfg


# 20 个任务集中注册总线
# task dict 字段:
#   task_id          : 唯一标识 (REQ_XX), 用于落库表名 / 日志命名空间 / 模型路径
#   task_name        : 业务可读名 (展示用)
#   sql_path         : 训练/监控比对 SQL (拉全量, 一般 5 年)
#   sql_infer_path   : (可选) 推理 SQL (拉近期窗口, 一般 1-2 年), 未指定退化为 sql_path
#   config_builder   : 返回 HPFConfig 的 callable
TASKS = [
    {
        "task_id": "REQ_01",
        "task_name": "归集额预测",
        "sql_path":       "configs/hpf/sql_templates/req_01_collection_amount.sql",        # 训练拉 60 个月
        "sql_infer_path": "configs/hpf/sql_templates/req_01_collection_amount_infer.sql",  # 推理拉 24 个月
        "config_builder": get_collection_prediction_config,
    },
    # ... 后续 19 个任务可在此扩展; 不写 sql_infer_path 则训练/推理共用 sql_path
]
