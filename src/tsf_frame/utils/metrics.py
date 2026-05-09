"""
评估指标模块 / Evaluation metrics module

提供时序预测常用的评估指标计算: MAE / MSE / RMSE / MAPE / SMAPE / R²。

✅ 量纲约定 (全项目统一, 与 ``tsf_frame.monitoring.performance_monitor`` 一致):
    所有比例类指标 (MAPE / SMAPE) 都返回 **小数形式**:
        0.05 表示 5%, 0.10 表示 10%

    与之配套的:
        - ``HPFMonitoringConfig.mape_warning = 0.10``  (10%)
        - ``HPFMonitoringConfig.mape_critical = 0.15`` (15%)
        - ``RetrainingTrigger`` 默认规则 ``mape > 0.15``

    展示时 (CLI 输出/报表/告警文案) 请用 ``f'{mape:.2%}'`` 转成百分比文本.

历史: 本模块原先返回百分比 (× 100), 与监控模块不一致, 已在 2026-Q2 统一为小数.
新业务建议直接用监控模块的注册指标函数 (可被 PerformanceMonitor 自动消费):

    from tsf_frame.monitoring import mae, mape, get_metric_fn
"""

import numpy as np
from typing import Dict, Any, List


class MetricsCalculator:
    """
    指标计算器 / Metrics calculator

    提供静态方法计算各种回归/预测评估指标。
    Provides static methods to compute various regression/forecasting evaluation metrics.
    """
    @staticmethod
    def mae(y_true, y_pred):
        """计算平均绝对误差 / Calculate Mean Absolute Error (MAE)"""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mse(y_true, y_pred):
        """计算均方误差 / Calculate Mean Squared Error (MSE)"""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true, y_pred):
        """计算均方根误差 / Calculate Root Mean Squared Error (RMSE)"""
        return np.sqrt(MetricsCalculator.mse(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred, eps: float = 1e-8):
        """
        计算平均绝对百分比误差 / MAPE.

        ⚠ 返回 **小数形式** (0.05 = 5%), 与 monitoring 模块统一.
        展示时请用 ``f'{value:.2%}'`` 转百分比.

        近零值保护: 当 |y_true| < eps 时, 用 eps 作分母防止天文值
        (而非简单丢弃, 与 ``monitoring.mape`` 行为一致).
        / Near-zero protection: clamp denominator at eps (consistent with monitoring.mape).
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size == 0:
            return float('nan')
        denom = np.where(np.abs(y_true) < eps, eps, y_true)
        return float(np.mean(np.abs((y_true - y_pred) / denom)))

    @staticmethod
    def smape(y_true, y_pred):
        """
        计算对称平均绝对百分比误差 / SMAPE.

        ⚠ 返回 **小数形式** (0.05 = 5%), 与 monitoring 模块统一.
        Returns decimal; use ``:.2%`` for display.

        相比 MAPE 对真实值和预测值的尺度更对称.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0  # 过滤分母为零 / Filter out zero denominators
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]))

    @staticmethod
    def r2(y_true, y_pred):
        """
        计算R2决定系数 / Calculate R-squared (coefficient of determination)

        衡量模型对数据方差的解释能力，1.0为完美拟合。
        Measures how well the model explains variance in data; 1.0 indicates a perfect fit.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)      # 残差平方和 / Residual sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 总平方和 / Total sum of squares
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def calculate_all(y_true, y_pred):
        """
        计算所有指标并返回字典 / Calculate all metrics and return as a dictionary

        Returns:
            包含 MAE, MSE, RMSE, MAPE, SMAPE, R2 的字典
            Dictionary containing MAE, MSE, RMSE, MAPE, SMAPE, R2
        """
        return {
            'MAE': MetricsCalculator.mae(y_true, y_pred),
            'MSE': MetricsCalculator.mse(y_true, y_pred),
            'RMSE': MetricsCalculator.rmse(y_true, y_pred),
            'MAPE': MetricsCalculator.mape(y_true, y_pred),
            'SMAPE': MetricsCalculator.smape(y_true, y_pred),
            'R2': MetricsCalculator.r2(y_true, y_pred)
        }
