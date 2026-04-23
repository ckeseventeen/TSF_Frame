"""TSF_Frame — 通用时序预测框架主包 / main package.

子包:
    business/     业务防腐层（BaseBusinessAdapter + HPFAdapter）
    data/         数据加载与切分（datasets 子包）
    features/     特征工程（mixed / engineering / selector）
    models/       模型层（classical / transformer / moirai）
    monitoring/   监控与告警（性能/漂移/业务规则/SQLite/报表）
    visualization/ 预测与监控图表
    utils/        通用工具（logger / metrics）
"""

__version__ = '0.2.0'
__all__ = ['__version__']
