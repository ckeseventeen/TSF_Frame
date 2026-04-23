"""
评估指标模块 / Evaluation metrics module

提供时序预测常用的评估指标计算，包括MAE、MSE、RMSE、MAPE、SMAPE和R2。
Provides common evaluation metrics for time series forecasting, including MAE, MSE, RMSE, MAPE, SMAPE, and R2.
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
    def mape(y_true, y_pred):
        """
        计算平均绝对百分比误差 / Calculate Mean Absolute Percentage Error (MAPE)

        自动过滤真实值为零的样本以避免除零错误。
        Automatically filters out samples where y_true is zero to avoid division by zero.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0  # 过滤零值 / Filter out zero values
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def smape(y_true, y_pred):
        """
        计算对称平均绝对百分比误差 / Calculate Symmetric Mean Absolute Percentage Error (SMAPE)

        相比MAPE，对真实值和预测值的尺度更对称。
        More symmetric with respect to the scale of y_true and y_pred compared to MAPE.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0  # 过滤分母为零的情况 / Filter out zero denominators
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

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
