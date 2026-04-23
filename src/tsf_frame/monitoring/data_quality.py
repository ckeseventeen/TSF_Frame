"""
数据质量监控 / Data quality monitoring
=======================================

面向时序预测管道入口的**原始数据质量**检查, 区别于业务层
``rule_engine``: 这里关心的是 **数据本身是否可用**, 而非业务含义。

五类核心检查:

======================  =========================================
MissingRateCheck        每列缺失率, 超阈值告警
OutlierCheck            Z-score / IQR 异常值统计
SchemaCheck             必须列是否齐、dtype 是否匹配
FrequencyCheck          时间索引频率是否连续 (时序专用)
RangeCheck              值域约束 (min/max, 非负/单调递增等)
======================  =========================================

每个检查实现 ``QualityChecker`` 接口, 通过 ``@register_quality_checker``
加入注册表; ``DataQualityMonitor`` 是一个**组合器**, 按配置挑选并串联
多个检查器。

与 ``PerformanceMonitor`` / ``DriftDetector`` 的分工:
* Quality   — 单批数据是否"干净"
* Drift     — 两批数据分布是否一致
* Perf      — 预测是否准
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .interfaces import (
    AlertLevel,
    QualityChecker,
    QualityIssue,
    register_quality_checker,
)

__all__ = [
    'MissingRateCheck',
    'OutlierCheck',
    'SchemaCheck',
    'FrequencyCheck',
    'RangeCheck',
    'DataQualityMonitor',
]


# ==========================================================================
# 基础检查实现
# ==========================================================================

@register_quality_checker('missing_rate')
class MissingRateCheck(QualityChecker):
    """
    缺失率检查 / Per-column missing rate.

    Args:
        threshold: 警戒线 (0-1). 超过即 WARNING, 超过 2x 升为 ERROR。
        columns: 只检查这些列; None 表示全部。
    """

    def __init__(
        self, threshold: float = 0.1,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.columns = list(columns) if columns else None

    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        issues: List[QualityIssue] = []
        if not isinstance(data, pd.DataFrame):
            return issues
        cols = self.columns or list(data.columns)
        for col in cols:
            if col not in data.columns:
                continue
            rate = float(data[col].isna().mean())
            if rate > self.threshold * 2:
                severity = AlertLevel.ERROR
            elif rate > self.threshold:
                severity = AlertLevel.WARNING
            else:
                continue
            issues.append(QualityIssue(
                issue_id='MISSING',
                severity=severity,
                column=col,
                message=f'列 "{col}" 缺失率 {rate:.1%} 超阈值 '
                        f'{self.threshold:.1%}',
                value=rate,
                details={'threshold': self.threshold},
            ))
        return issues


@register_quality_checker('outlier')
class OutlierCheck(QualityChecker):
    """
    异常值比例检查 / Outlier ratio.

    方法: Z-score (|z| > z_threshold) 或 IQR (1.5 * IQR 外)。
    Args:
        method: 'zscore' 或 'iqr'
        z_threshold: Z 阈值 (method='zscore')
        ratio_threshold: 异常值占比超过即 WARNING
    """

    def __init__(
        self, method: str = 'zscore', z_threshold: float = 3.0,
        ratio_threshold: float = 0.05,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        if method not in ('zscore', 'iqr'):
            raise ValueError(f"method must be 'zscore'|'iqr', got {method!r}")
        self.method = method
        self.z_threshold = float(z_threshold)
        self.ratio_threshold = float(ratio_threshold)
        self.columns = list(columns) if columns else None

    def _outlier_mask(self, x: np.ndarray) -> np.ndarray:
        x = x[~np.isnan(x)]
        if len(x) < 3:
            return np.zeros_like(x, dtype=bool)
        if self.method == 'zscore':
            std = x.std(ddof=0)
            if std == 0:
                return np.zeros_like(x, dtype=bool)
            return np.abs((x - x.mean()) / std) > self.z_threshold
        else:  # iqr
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return (x < lo) | (x > hi)

    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        issues: List[QualityIssue] = []
        if not isinstance(data, pd.DataFrame):
            return issues
        cols = self.columns or [c for c in data.columns
                                if pd.api.types.is_numeric_dtype(data[c])]
        for col in cols:
            if col not in data.columns:
                continue
            arr = data[col].to_numpy(dtype=float, na_value=np.nan)
            arr = arr[~np.isnan(arr)]
            if len(arr) < 3:
                continue
            mask = self._outlier_mask(arr)
            ratio = float(mask.mean())
            if ratio > self.ratio_threshold:
                issues.append(QualityIssue(
                    issue_id='OUTLIER',
                    severity=AlertLevel.WARNING,
                    column=col,
                    message=f'列 "{col}" 异常值占比 {ratio:.1%} 超阈值 '
                            f'{self.ratio_threshold:.1%} ({self.method})',
                    value=ratio,
                    details={'method': self.method,
                             'threshold': self.ratio_threshold},
                ))
        return issues


@register_quality_checker('schema')
class SchemaCheck(QualityChecker):
    """
    Schema 检查 / Required columns & dtypes.

    Args:
        required: 必须存在的列名列表
        dtypes:   {col: pandas_dtype_str} 期望 dtype (可选)
    """

    def __init__(
        self, required: Sequence[str],
        dtypes: Optional[Mapping[str, str]] = None,
    ):
        self.required = list(required)
        self.dtypes = dict(dtypes or {})

    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        issues: List[QualityIssue] = []
        if not isinstance(data, pd.DataFrame):
            issues.append(QualityIssue(
                issue_id='SCHEMA', severity=AlertLevel.ERROR,
                message=f'期望 DataFrame, 实际 {type(data).__name__}',
            ))
            return issues

        for col in self.required:
            if col not in data.columns:
                issues.append(QualityIssue(
                    issue_id='SCHEMA_MISSING_COL',
                    severity=AlertLevel.CRITICAL,
                    column=col,
                    message=f'缺少必须列 "{col}"',
                ))

        for col, expected in self.dtypes.items():
            if col not in data.columns:
                continue
            actual = str(data[col].dtype)
            if not actual.startswith(expected):
                issues.append(QualityIssue(
                    issue_id='SCHEMA_DTYPE',
                    severity=AlertLevel.WARNING,
                    column=col,
                    message=f'列 "{col}" dtype={actual} 与期望 '
                            f'{expected!r} 不符',
                    details={'expected': expected, 'actual': actual},
                ))
        return issues


@register_quality_checker('frequency')
class FrequencyCheck(QualityChecker):
    """
    时间索引频率连续性 / Time index frequency continuity.

    Args:
        expected_freq: pandas offset 字符串 (e.g. 'D', 'M', 'MS', 'H')
        tolerance:    允许的缺失期数 (默认 0, 严格连续)
    """

    def __init__(self, expected_freq: str, tolerance: int = 0):
        self.expected_freq = expected_freq
        self.tolerance = int(tolerance)

    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        issues: List[QualityIssue] = []
        if not isinstance(data, pd.DataFrame):
            return issues
        idx = data.index
        if not isinstance(idx, pd.DatetimeIndex):
            issues.append(QualityIssue(
                issue_id='FREQ_NOT_DATETIME',
                severity=AlertLevel.WARNING,
                message='数据索引不是 DatetimeIndex, 跳过频率检查',
            ))
            return issues
        if len(idx) < 2:
            return issues

        try:
            full = pd.date_range(idx.min(), idx.max(),
                                 freq=self.expected_freq)
        except Exception as exc:
            issues.append(QualityIssue(
                issue_id='FREQ_INVALID',
                severity=AlertLevel.ERROR,
                message=f'无法按频率 {self.expected_freq} 生成参考序列: '
                        f'{exc}',
            ))
            return issues

        missing = full.difference(idx)
        if len(missing) > self.tolerance:
            issues.append(QualityIssue(
                issue_id='FREQ_DISCONTINUOUS',
                severity=AlertLevel.ERROR,
                message=f'时间索引不连续: 缺失 {len(missing)} 期 '
                        f'(容忍 {self.tolerance})',
                value=float(len(missing)),
                details={
                    'expected_freq': self.expected_freq,
                    'missing_examples': [str(t) for t in missing[:5]],
                },
            ))
        return issues


@register_quality_checker('range')
class RangeCheck(QualityChecker):
    """
    值域约束 / Value range constraints.

    Args:
        constraints: {col: (min, max)} — None 表示无下/上界
        non_negative: 简写, 列表中的列强制 min=0
        monotonic_increasing: 列表中的列必须单调不减
    """

    def __init__(
        self,
        constraints: Optional[Mapping[str, tuple]] = None,
        non_negative: Optional[Sequence[str]] = None,
        monotonic_increasing: Optional[Sequence[str]] = None,
    ):
        self.constraints: Dict[str, tuple] = dict(constraints or {})
        for col in (non_negative or []):
            prev = self.constraints.get(col, (None, None))
            self.constraints[col] = (0, prev[1])
        self.mono_inc = list(monotonic_increasing or [])

    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        issues: List[QualityIssue] = []
        if not isinstance(data, pd.DataFrame):
            return issues

        for col, (lo, hi) in self.constraints.items():
            if col not in data.columns:
                continue
            s = data[col].dropna()
            if lo is not None:
                bad = (s < lo).sum()
                if bad > 0:
                    issues.append(QualityIssue(
                        issue_id='RANGE_BELOW_MIN',
                        severity=AlertLevel.CRITICAL
                        if lo == 0 else AlertLevel.ERROR,
                        column=col,
                        message=f'列 "{col}" 有 {bad} 条低于 {lo}',
                        value=float(bad),
                    ))
            if hi is not None:
                bad = (s > hi).sum()
                if bad > 0:
                    issues.append(QualityIssue(
                        issue_id='RANGE_ABOVE_MAX',
                        severity=AlertLevel.ERROR,
                        column=col,
                        message=f'列 "{col}" 有 {bad} 条高于 {hi}',
                        value=float(bad),
                    ))

        for col in self.mono_inc:
            if col not in data.columns:
                continue
            s = data[col].dropna().to_numpy()
            if len(s) >= 2 and not np.all(np.diff(s) >= 0):
                drops = int(np.sum(np.diff(s) < 0))
                issues.append(QualityIssue(
                    issue_id='RANGE_NOT_MONOTONIC',
                    severity=AlertLevel.WARNING,
                    column=col,
                    message=f'列 "{col}" 非单调不减, 出现 {drops} 次下降',
                    value=float(drops),
                ))
        return issues


# ==========================================================================
# 组合器 / DataQualityMonitor
# ==========================================================================

@dataclass
class DataQualityMonitor:
    """
    数据质量检查组合器 / Compose multiple QualityChecker instances.

    Attributes:
        checkers: 实际执行的检查器列表; 顺序即扫描顺序
        max_severity: 所有 issue 里最严重的 severity, 调用方可据此决定
                      是否阻断管道
    """

    checkers: List[QualityChecker] = field(default_factory=list)

    def add(self, checker: QualityChecker) -> 'DataQualityMonitor':
        self.checkers.append(checker)
        return self

    def check(self, data: Any) -> List[QualityIssue]:
        """并行串跑所有 checker, 合并 issue 列表。"""
        issues: List[QualityIssue] = []
        for ch in self.checkers:
            try:
                issues.extend(ch.check(data))
            except Exception as exc:  # 检查器本身崩溃也要捕获
                issues.append(QualityIssue(
                    issue_id='CHECKER_ERROR',
                    severity=AlertLevel.ERROR,
                    message=(f'{type(ch).__name__} 抛异常: {exc}'),
                ))
        return issues


# ==========================================================================
# main — 5 种异常人工注入演示
# ==========================================================================

def main() -> None:
    """构造一个"不干净"的 DataFrame, 观察各 checker 的报告。"""
    print('=' * 70)
    print(' data_quality — 5 checkers on a messy DataFrame')
    print('=' * 70)

    idx = pd.date_range('2024-01-01', periods=12, freq='MS')
    # 故意丢掉一个月制造频率不连续
    idx = idx.delete(6)
    df = pd.DataFrame({
        'deposit':     [100, 110, 120, np.nan, np.nan, 130, 140,
                        150, 160, 170, 180],
        'withdrawal':  [20, 25, 22, 23, 24, -1, 500, 27, 28, 29, 30],
        'count':       [10, 11, 12, 13, 14, 15, 14, 17, 18, 19, 20],
    }, index=idx)

    dqm = (DataQualityMonitor()
           .add(SchemaCheck(required=['deposit', 'withdrawal', 'count']))
           .add(MissingRateCheck(threshold=0.1,
                                 columns=['deposit', 'withdrawal']))
           .add(OutlierCheck(method='iqr', ratio_threshold=0.05))
           .add(FrequencyCheck(expected_freq='MS'))
           .add(RangeCheck(non_negative=['deposit', 'withdrawal'],
                           monotonic_increasing=['count'])))

    issues = dqm.check(df)
    print(f'\n共发现 {len(issues)} 个问题:\n')
    for iss in issues:
        col = f' [{iss.column}]' if iss.column else ''
        v = f' value={iss.value:.3f}' if iss.value is not None else ''
        print(f'  {iss.severity.upper():<8} {iss.issue_id:<22}{col}{v}')
        print(f'           {iss.message}')


if __name__ == '__main__':
    main()
