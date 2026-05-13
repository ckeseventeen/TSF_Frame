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
    cfg.model.model_name = 'xgboost'
    cfg.model.pred_len = 60 
    
    # 5. 监控报警阈值 (10%)
    cfg.monitoring.performance_alert_threshold = 0.10
    
    return cfg

# 20个任务集中注册总线
TASKS = [
    {
        "task_id": "REQ_01",
        "task_name": "归集额预测",
        "sql_path": "configs/hpf/sql_templates/req_01_collection_amount.sql",
        "config_builder": get_collection_prediction_config
    },
    # ... 后续 19 个任务可在此扩展
]
