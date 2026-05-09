"""
住房公积金（HPF）业务监控配置

阈值的设计思路:
  - 月度数据噪声大于日度数据，MAPE 阈值放宽到 10%/15%
  - 冷启动期(<6 月)只做业务规则校验，跳过漂移检测
  - 滑窗以"月"为单位，12 个月覆盖一个完整年度周期
  - SQLite 作为默认持久化后端，无外部依赖
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


# 项目根目录: configs/hpf/hpf_monitoring_config.py → parents[2] → <root>
# 不依赖 CWD, 让所有 example/cron/IDE 启动方式都能落到同一份 logs 下
# / Project root anchor; CWD-independent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _root_path(*parts: str) -> str:
    """构造基于项目根的绝对路径 / Build abs path from project root."""
    return str(_PROJECT_ROOT.joinpath(*parts))


@dataclass
class HPFMonitoringConfig:
    """HPF 业务监控总配置"""

    # ── 监控窗口（月度适配）───────────────────────────────────────────────
    # 滑动窗口长度（月）；用于计算近期业务指标
    window_months: int = 12
    # 冷启动阈值；累计样本 < 此数量时仅做规则校验
    cold_start_months: int = 6
    # 基线参考期（月）；用于 compare_to_baseline
    baseline_months: int = 24

    # ── 业务规则（R1-R5）───────────────────────────────────────────────
    # R1 非负列：出现负值即 CRITICAL
    non_negative_cols: List[str] = field(default_factory=lambda: [
        'monthly_deposit', 'monthly_withdrawal', 'loan_balance',
        'depositor_count', 'loan_count', 'loan_issued',
    ])
    # R2 月度突变比例：|pred - last_actual| / |last_actual| 超过此值即告警
    sudden_change_ratio: float = 0.30
    # R3 特征缺失率阈值
    missing_rate_threshold: float = 0.10
    # R5 异常回声：预测值相对近期历史 sigma 的倍数，超过即告警
    outlier_sigma_ratio: float = 3.0

    # ── 业务指标告警阈值 ─────────────────────────────────────────────────
    # MAPE 阈值 (小数形式, 0.10 = 10%, 与 monitoring.mape / MetricsCalculator.mape 统一)
    # 展示时用 f'{value:.2%}' 转百分比
    # MAPE thresholds (decimal; 0.10 == 10%)
    mape_warning: float = 0.10
    mape_critical: float = 0.15
    # 方向准确率下限 (小数, 0.60 = 60%)
    direction_accuracy_floor: float = 0.60
    # YoY_MAE / mean(y_true) 阈值 (小数)
    yoy_mae_ratio_warning: float = 0.20

    # ── 漂移阈值 ─────────────────────────────────────────────────────────
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    ks_pvalue: float = 0.05

    # ── 持久化 ───────────────────────────────────────────────────────────
    # 统一目录布局: <project_root>/logs/{runs,monitor,reports/hpf,outputs}
    # 默认值用 project_root 锚定的绝对路径, 防止从 IDE 直接 Run 文件时
    # CWD = 文件所在目录, 导致日志写到 pipelines/examples/logs 这种"飞地".
    # 用户 yaml 覆盖时仍可用相对/绝对路径自由控制.
    # / Defaults are absolute (project-root anchored) to avoid CWD drift.

    # SQLite 数据库路径
    sqlite_path: str = field(default_factory=lambda: _root_path(
        'logs', 'monitor', 'hpf_monitor.db'))
    # 历史保留月数；cleanup_old 按此裁剪
    retain_months: int = 36

    # ── 报表 ─────────────────────────────────────────────────────────────
    report_dir: str = field(default_factory=lambda: _root_path(
        'logs', 'reports', 'hpf'))
    # 是否在 WARNING+ 时自动生成报表 PNG
    report_on_alert: bool = True
    report_dpi: int = 120

    # ── 告警通道 ─────────────────────────────────────────────────────────
    # 控制台 + 主日志由 get_logger 输出
    log_name: str = 'hpf_monitor'
    log_dir: str = field(default_factory=lambda: _root_path(
        'logs', 'runs'))
    # 专用告警文件（仅 WARNING 以上）
    alert_log_file: str = field(default_factory=lambda: _root_path(
        'logs', 'monitor', 'hpf_alerts.log'))

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（便于日志和调试）"""
        return {
            'window_months': self.window_months,
            'cold_start_months': self.cold_start_months,
            'baseline_months': self.baseline_months,
            'non_negative_cols': self.non_negative_cols,
            'sudden_change_ratio': self.sudden_change_ratio,
            'missing_rate_threshold': self.missing_rate_threshold,
            'mape_warning': self.mape_warning,
            'mape_critical': self.mape_critical,
            'direction_accuracy_floor': self.direction_accuracy_floor,
            'yoy_mae_ratio_warning': self.yoy_mae_ratio_warning,
            'psi_warning': self.psi_warning,
            'psi_critical': self.psi_critical,
            'ks_pvalue': self.ks_pvalue,
            'sqlite_path': self.sqlite_path,
            'retain_months': self.retain_months,
            'report_dir': self.report_dir,
            'report_on_alert': self.report_on_alert,
            'log_name': self.log_name,
            'log_dir': self.log_dir,
            'alert_log_file': self.alert_log_file,
        }
