"""
漂移检测 / Drift detection
===========================

三类漂移 (三种 ``DriftDetector`` 实现) 针对不同信号:

=======================  ====================================================
DataDriftDetector        输入特征分布变化 (协变量漂移)
                          统计量: PSI / KS / JS
ConceptDriftDetector     输入 → 输出关系变化 (残差分布)
                          方法: 残差均值漂移 / 方差扩大 / 滚动误差趋势
PredictionDriftDetector  模型输出本身分布变化 (上下游同漂移时区分用)
                          统计量: PSI on y_pred
=======================  ====================================================

每个检测器:
1. 持有参考数据 (reference / baseline)
2. 通过 ``update(batch)`` 累积新窗口
3. ``detect()`` 返回 ``DriftResult`` (detected/score/p_value/...)
4. ``reset()`` 清空非参考状态

实现细节:
* PSI: 按参考分布分箱 (默认 10 分位数箱), 计算 Σ(pN-pR)·ln(pN/pR)
* KS:  使用 scipy.stats.ks_2samp (可降级到 numpy 实现)
* JS:  0.5·KL(P||M) + 0.5·KL(Q||M), M = 0.5(P+Q)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .interfaces import (
    AlertLevel,
    DriftDetector,
    DriftResult,
    DriftType,
    register_drift_detector,
)

try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_stats = None
    _HAS_SCIPY = False


__all__ = [
    'DataDriftDetector',
    'ConceptDriftDetector',
    'PredictionDriftDetector',
    'calc_psi',
    'calc_ks',
    'calc_js',
]


# ==========================================================================
# 统计量 / Statistics helpers
# ==========================================================================

def calc_psi(
    reference: np.ndarray, current: np.ndarray, bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index.

    约定解读:
    * < 0.1  稳定
    * 0.1-0.25 轻度漂移 (WARNING)
    * > 0.25 显著漂移 (CRITICAL)
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    # 用 reference 分位数做切点, 保证稳定性
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_ratio = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_ratio = cur_counts / max(cur_counts.sum(), 1) + eps
    return float(np.sum((cur_ratio - ref_ratio)
                        * np.log(cur_ratio / ref_ratio)))


def calc_ks(reference: np.ndarray, current: np.ndarray) -> tuple:
    """
    双样本 KS 检验, 返回 (statistic, p_value).

    scipy 可用时直接用 ks_2samp; 否则手工算统计量 (p 置为 nan)。
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0, 1.0
    if _HAS_SCIPY:
        res = _scipy_stats.ks_2samp(ref, cur)
        return float(res.statistic), float(res.pvalue)
    # 手动 KS stat
    combined = np.concatenate([ref, cur])
    grid = np.sort(np.unique(combined))
    cdf_ref = np.searchsorted(np.sort(ref), grid, side='right') / len(ref)
    cdf_cur = np.searchsorted(np.sort(cur), grid, side='right') / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur))), float('nan')


def calc_js(
    reference: np.ndarray, current: np.ndarray, bins: int = 30,
    eps: float = 1e-12,
) -> float:
    """Jensen-Shannon divergence (以 2 为底, 范围 [0,1])."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    lo = min(ref.min(), cur.min())
    hi = max(ref.max(), cur.max())
    if hi == lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(ref, bins=edges, density=True)
    q, _ = np.histogram(cur, bins=edges, density=True)
    p = p / (p.sum() + eps) + eps
    q = q / (q.sum() + eps) + eps
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


# ==========================================================================
# DataDriftDetector
# ==========================================================================

@register_drift_detector('data')
class DataDriftDetector(DriftDetector):
    """
    数据 (协变量) 漂移检测 / Covariate drift.

    支持 1D 或 2D (多特征). 对 2D 逐列计算 PSI/KS, 报告最严重列。

    Args:
        reference:   参考窗口数据 (1D 或 2D)
        window_size: 新样本滑动窗口大小
        psi_warn:    WARNING PSI 阈值
        psi_crit:    CRITICAL PSI 阈值
        ks_alpha:    KS p-value 拒绝阈值
        feature_names: 2D 时的特征名列表 (便于报告)
    """

    drift_type = DriftType.DATA

    def __init__(
        self,
        reference: Optional[np.ndarray] = None,
        *,
        window_size: int = 100,
        psi_warn: float = 0.1,
        psi_crit: float = 0.25,
        ks_alpha: float = 0.05,
        feature_names: Optional[List[str]] = None,
    ):
        self.reference = (None if reference is None
                          else np.atleast_2d(np.asarray(reference)))
        self.window_size = int(window_size)
        self.psi_warn = float(psi_warn)
        self.psi_crit = float(psi_crit)
        self.ks_alpha = float(ks_alpha)
        self.feature_names = list(feature_names) if feature_names else None
        self._buf: Deque[np.ndarray] = deque(maxlen=self.window_size)

    def set_reference(self, reference: np.ndarray) -> None:
        self.reference = np.atleast_2d(np.asarray(reference))

    def update(self, data: np.ndarray) -> None:
        arr = np.atleast_2d(np.asarray(data, dtype=float))
        for row in arr:
            self._buf.append(row)

    def detect(self) -> DriftResult:
        if self.reference is None or len(self._buf) < 2:
            return DriftResult(drift_type=self.drift_type, detected=False)
        cur = np.asarray(self._buf)
        ref = self.reference
        if ref.shape[1] != cur.shape[1]:
            return DriftResult(
                drift_type=self.drift_type, detected=False,
                details={'error': f'dim mismatch ref={ref.shape[1]} '
                                  f'cur={cur.shape[1]}'})

        per_feature: Dict[str, float] = {}
        psi_vals: List[float] = []
        ks_vals: List[float] = []
        ks_ps: List[float] = []
        names = (self.feature_names
                 or [f'f{i}' for i in range(ref.shape[1])])

        for i, name in enumerate(names):
            psi = calc_psi(ref[:, i], cur[:, i])
            ks_stat, ks_p = calc_ks(ref[:, i], cur[:, i])
            per_feature[f'{name}__psi'] = psi
            per_feature[f'{name}__ks'] = ks_stat
            psi_vals.append(psi); ks_vals.append(ks_stat); ks_ps.append(ks_p)

        max_psi = max(psi_vals) if psi_vals else 0.0
        min_p = min(ks_ps) if ks_ps else 1.0

        if max_psi > self.psi_crit:
            severity = AlertLevel.CRITICAL
            detected = True
        elif max_psi > self.psi_warn or (min_p < self.ks_alpha):
            severity = AlertLevel.WARNING
            detected = True
        else:
            severity = AlertLevel.INFO
            detected = False

        return DriftResult(
            drift_type=self.drift_type,
            detected=detected,
            severity=severity,
            score=max_psi,
            p_value=min_p,
            per_feature=per_feature,
            details={'max_psi_feature': names[int(np.argmax(psi_vals))]
                     if psi_vals else None,
                     'window_size': len(self._buf),
                     'reference_size': len(ref)},
        )

    def reset(self) -> None:
        self._buf.clear()


# ==========================================================================
# ConceptDriftDetector
# ==========================================================================

@register_drift_detector('concept')
class ConceptDriftDetector(DriftDetector):
    """
    概念漂移检测 / Concept drift via residuals.

    以预测残差为信号, 同时监测:
    1. 残差均值是否偏离 0 (模型系统性偏差)
    2. 残差方差是否扩大
    3. 残差绝对值滚动均值是否持续上升 (误差趋势)

    Args:
        window_size:     滑窗大小
        mean_shift_std:  均值偏离被判定显著的标准差倍数 (3.0 → 3σ)
        var_ratio_warn:  current_var / reference_var 超过即 WARNING
    """

    drift_type = DriftType.CONCEPT

    def __init__(
        self,
        *,
        window_size: int = 100,
        mean_shift_std: float = 3.0,
        var_ratio_warn: float = 2.0,
        var_ratio_crit: float = 4.0,
    ):
        self.window_size = int(window_size)
        self.mean_shift_std = float(mean_shift_std)
        self.var_ratio_warn = float(var_ratio_warn)
        self.var_ratio_crit = float(var_ratio_crit)
        self._ref_residuals: Optional[np.ndarray] = None
        self._buf: Deque[float] = deque(maxlen=self.window_size)

    def set_reference(self, residuals: np.ndarray) -> None:
        """把已知 baseline 时期的残差设为参考。"""
        arr = np.asarray(residuals, dtype=float).ravel()
        self._ref_residuals = arr[~np.isnan(arr)]

    def update(self, data: np.ndarray) -> None:
        """``data`` 为残差 (y_true - y_pred) 的 1D 数组。"""
        arr = np.atleast_1d(np.asarray(data, dtype=float)).ravel()
        for x in arr:
            if not np.isnan(x):
                self._buf.append(float(x))

    def detect(self) -> DriftResult:
        if len(self._buf) < 5:
            return DriftResult(drift_type=self.drift_type, detected=False)
        cur = np.array(self._buf)
        cur_mean = float(cur.mean())
        cur_std = float(cur.std(ddof=0))

        severity = AlertLevel.INFO
        detected = False
        details: Dict[str, Any] = {
            'current_mean': cur_mean,
            'current_std': cur_std,
            'n': len(cur),
        }

        # 与参考对比
        if self._ref_residuals is not None and len(self._ref_residuals) >= 2:
            ref_mean = float(self._ref_residuals.mean())
            ref_std = max(float(self._ref_residuals.std(ddof=0)), 1e-9)
            z = abs(cur_mean - ref_mean) / ref_std
            var_ratio = (cur_std ** 2) / max(ref_std ** 2, 1e-12)
            details.update({'ref_mean': ref_mean, 'ref_std': ref_std,
                            'mean_z': z, 'var_ratio': var_ratio})
            if var_ratio > self.var_ratio_crit or z > self.mean_shift_std * 1.5:
                severity = AlertLevel.CRITICAL
                detected = True
            elif var_ratio > self.var_ratio_warn or z > self.mean_shift_std:
                severity = AlertLevel.WARNING
                detected = True
            score = max(z, var_ratio - 1.0)
        else:
            # 无参考: 只能看偏差
            if abs(cur_mean) > self.mean_shift_std * cur_std / max(
                    np.sqrt(len(cur)), 1):
                severity = AlertLevel.WARNING
                detected = True
            score = abs(cur_mean)

        # 趋势 (前后半段绝对残差均值对比)
        half = len(cur) // 2
        if half >= 3:
            early = np.mean(np.abs(cur[:half]))
            late = np.mean(np.abs(cur[half:]))
            details['trend_ratio'] = float(late / max(early, 1e-9))
            if details['trend_ratio'] > 2.0 and severity == AlertLevel.INFO:
                severity = AlertLevel.WARNING
                detected = True

        return DriftResult(
            drift_type=self.drift_type,
            detected=detected,
            severity=severity,
            score=float(score),
            details=details,
        )

    def reset(self) -> None:
        self._buf.clear()


# ==========================================================================
# PredictionDriftDetector
# ==========================================================================

@register_drift_detector('prediction')
class PredictionDriftDetector(DriftDetector):
    """
    预测输出分布漂移 / Prediction output drift.

    与 DataDriftDetector 对偶, 监测 y_pred 分布是否随时间变化。
    典型用法:
    * 参考 = 验证集预测
    * 当前 = 近期线上预测
    """

    drift_type = DriftType.PREDICTION

    def __init__(
        self,
        reference: Optional[np.ndarray] = None,
        *,
        window_size: int = 100,
        psi_warn: float = 0.1,
        psi_crit: float = 0.25,
    ):
        self.reference = (None if reference is None
                          else np.asarray(reference, dtype=float).ravel())
        self.window_size = int(window_size)
        self.psi_warn = float(psi_warn)
        self.psi_crit = float(psi_crit)
        self._buf: Deque[float] = deque(maxlen=self.window_size)

    def set_reference(self, reference: np.ndarray) -> None:
        self.reference = np.asarray(reference, dtype=float).ravel()

    def update(self, data: np.ndarray) -> None:
        arr = np.atleast_1d(np.asarray(data, dtype=float)).ravel()
        for v in arr:
            if not np.isnan(v):
                self._buf.append(float(v))

    def detect(self) -> DriftResult:
        if self.reference is None or len(self._buf) < 2:
            return DriftResult(drift_type=self.drift_type, detected=False)
        cur = np.array(self._buf)
        psi = calc_psi(self.reference, cur)
        js = calc_js(self.reference, cur)
        ks_stat, ks_p = calc_ks(self.reference, cur)
        if psi > self.psi_crit:
            severity, detected = AlertLevel.CRITICAL, True
        elif psi > self.psi_warn:
            severity, detected = AlertLevel.WARNING, True
        else:
            severity, detected = AlertLevel.INFO, False
        return DriftResult(
            drift_type=self.drift_type,
            detected=detected,
            severity=severity,
            score=psi,
            p_value=ks_p,
            details={'js': js, 'ks_stat': ks_stat,
                     'n_current': len(cur),
                     'n_reference': len(self.reference)},
        )

    def reset(self) -> None:
        self._buf.clear()


# ==========================================================================
# main — 三类漂移演示
# ==========================================================================

def main() -> None:
    """人工制造三类漂移, 分别触发一个 detector。"""
    print('=' * 70)
    print(' drift_detector — data / concept / prediction demos')
    print('=' * 70)

    rng = np.random.default_rng(42)

    # 1) Data drift: 参考 N(0,1), 当前 N(0.8,1.2)
    ref = rng.standard_normal((200, 3))
    cur = rng.standard_normal((200, 3)) * 1.2 + 0.8
    det = DataDriftDetector(reference=ref,
                            feature_names=['f1', 'f2', 'f3'])
    det.update(cur)
    r = det.detect()
    print(f'\n[data]       detected={r.detected} severity={r.severity} '
          f'PSI={r.score:.3f}  ks_p={r.p_value:.4f}')
    for k, v in list(r.per_feature.items())[:4]:
        print(f'             {k} = {v:.3f}')

    # 2) Concept drift: 残差从 N(0,1) 渐变为 N(1.5,1.5)
    cd = ConceptDriftDetector(window_size=80)
    cd.set_reference(rng.standard_normal(200))
    residuals = rng.standard_normal(80) * 1.5 + 1.5
    cd.update(residuals)
    r = cd.detect()
    print(f'\n[concept]    detected={r.detected} severity={r.severity} '
          f'score={r.score:.3f}')
    print(f'             details={ {k:round(float(v),3) if isinstance(v,(int,float)) else v for k,v in r.details.items()} }')

    # 3) Prediction drift: 预测分布从 [0,1] 偏到 [0.5, 1.5]
    pd_det = PredictionDriftDetector(reference=rng.uniform(0, 1, 500))
    pd_det.update(rng.uniform(0.5, 1.5, 100))
    r = pd_det.detect()
    print(f'\n[prediction] detected={r.detected} severity={r.severity} '
          f'PSI={r.score:.3f}  ks_p={r.p_value:.4f}')
    print(f'             js={r.details.get("js"):.3f}')

    print(f'\nscipy 可用: {_HAS_SCIPY}')


if __name__ == '__main__':
    main()
