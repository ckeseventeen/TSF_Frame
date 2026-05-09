"""
声明式规则引擎 / Declarative rule engine
=========================================

用于表达 **业务层 / 领域层** 的硬约束, 这些约束 *不是* "数据干不干净"
(那是 data_quality) 也不是 "分布变没变" (那是 drift_detector),
而是 *"这样的预测结果在业务上根本不合理"* — 例如:

* 负预测 (存款不可能为负)
* 月环比跳动 > 30% (政策变动之外不应该)
* 与历史同期差距过大
* 自定义任意 callable

两种注册方式:

1. **函数式** (轻量): ``@register_rule('name')`` 装饰一个满足签名

       fn(*, prediction, features=None, last_actual=None,
          context=None, **kw) -> List[RuleViolation]

2. **类式**: 继承 ``Rule`` 基类, 更方便封装状态 (如历史记忆)。

``RuleEngine`` 聚合若干规则, 支持按名字启用/禁用, 统一调用。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .interfaces import (
    AlertLevel,
    RULE_REGISTRY,
    RuleChecker,
    RuleViolation,
    register_rule,
)

__all__ = [
    'Rule',
    'RuleEngine',
    # 内置规则函数
    'rule_non_negative',
    'rule_sudden_change',
    'rule_out_of_band',
    'rule_monotonic_expected',
    'DEFAULT_RULE_IDS',
]


# ==========================================================================
# Rule 基类 (供需要状态的规则继承)
# ==========================================================================

@dataclass
class Rule:
    """
    规则基类 (类式规则) / Base rule as callable class.

    子类可覆盖 ``__call__`` 或 ``evaluate`` 以实现复杂逻辑。
    """

    # 规则唯一标识, 用于注册表 / 告警 source / 配置覆盖
    # / Unique rule id used in registry and alert.source
    rule_id: str
    # 默认告警级别 (具体规则可在 evaluate 时按情况升降)
    # / Default severity used when rule fires
    severity: str = AlertLevel.WARNING

    def __call__(self, **kw) -> List[RuleViolation]:
        return self.evaluate(**kw)

    def evaluate(self, **kw) -> List[RuleViolation]:  # pragma: no cover
        raise NotImplementedError


# ==========================================================================
# 内置规则函数
# ==========================================================================

@register_rule('R1_NON_NEGATIVE')
def rule_non_negative(
    *, prediction, target: str = 'y',
    severity: str = AlertLevel.CRITICAL, **_,
) -> List[RuleViolation]:
    """预测必须 >= 0 / Prediction must be non-negative."""
    arr = np.atleast_1d(np.asarray(prediction, dtype=float))
    if arr.size == 0:
        return []
    neg_count = int(np.sum(arr < 0))
    if neg_count == 0:
        return []
    return [RuleViolation(
        rule_id='R1_NON_NEGATIVE',
        severity=severity,
        message=(f'目标 "{target}" 预测出现 {neg_count} 个负值 '
                 f'(min={float(arr.min()):.3f})'),
        value=float(arr.min()),
        details={'target': target, 'neg_count': neg_count},
    )]


@register_rule('R2_SUDDEN_CHANGE')
def rule_sudden_change(
    *, prediction, last_actual: Optional[float] = None,
    max_ratio: float = 0.3, target: str = 'y',
    severity: str = AlertLevel.WARNING,
    min_baseline: float = 1e-6,
    skip_near_zero: bool = True, **_,
) -> List[RuleViolation]:
    """
    相比上一个实际值的环比变化 > max_ratio 即告警。

    Args:
        prediction:    本期预测序列 (取 ``arr[0]`` 与 last_actual 比)
        last_actual:   上一期真值; ``None`` 时跳过本规则
        max_ratio:     环比变化比率阈值 (默认 0.3)
        target:        目标名 (写进 message / details)
        severity:      告警级别 (默认 WARNING)
        min_baseline:  ``|last_actual| < min_baseline`` 视为"接近 0",
                       此时除法会数值放大噪声 → 默认跳过, 避免
                       对零基线的"环比"误报 (如刚启动期、节假日断点).
                       仍想检测? 设 ``skip_near_zero=False``, 此时
                       ``|last_actual|`` 会被 clamp 到 ``min_baseline``,
                       规则照常触发但分母不会爆.
        skip_near_zero: ``True`` (默认) 时基线近 0 → 跳过本规则;
                       ``False`` 时基线近 0 → 用 ``min_baseline`` clamp
                       后照常算 (谁负责定义业务"小到可忽略"由调用方传入).

    (对月度数据常见业务约束)
    """
    if last_actual is None:
        return []
    abs_base = abs(float(last_actual))
    # 近 0 基线: 默认跳过 (避免误报); 显式关闭跳过则 clamp 到 min_baseline.
    if abs_base < min_baseline:
        if skip_near_zero:
            return []
        abs_base = min_baseline
    arr = np.atleast_1d(np.asarray(prediction, dtype=float))
    if arr.size == 0:
        return []
    first = float(arr[0])
    ratio = abs(first - last_actual) / abs_base
    if ratio <= max_ratio:
        return []
    return [RuleViolation(
        rule_id='R2_SUDDEN_CHANGE',
        severity=severity,
        message=(f'目标 "{target}" 环比变化 {ratio:.1%} '
                 f'超过阈值 {max_ratio:.0%} '
                 f'(pred={first:.2f}, last={last_actual:.2f})'),
        value=ratio,
        details={'target': target, 'max_ratio': max_ratio,
                 'pred': first, 'last_actual': last_actual,
                 'min_baseline': min_baseline,
                 'baseline_clamped': abs(float(last_actual)) < min_baseline},
    )]


@register_rule('R3_OUT_OF_BAND')
def rule_out_of_band(
    *, prediction, lo: Optional[float] = None,
    hi: Optional[float] = None, target: str = 'y',
    severity: str = AlertLevel.ERROR, **_,
) -> List[RuleViolation]:
    """预测超出业务可接受值域 (若提供了 lo / hi)。"""
    if lo is None and hi is None:
        return []
    arr = np.atleast_1d(np.asarray(prediction, dtype=float))
    viol: List[RuleViolation] = []
    if lo is not None and np.any(arr < lo):
        viol.append(RuleViolation(
            rule_id='R3_OUT_OF_BAND',
            severity=severity,
            message=f'目标 "{target}" 有预测 < {lo}',
            value=float(arr.min()),
            details={'target': target, 'bound': 'lo', 'value': lo},
        ))
    if hi is not None and np.any(arr > hi):
        viol.append(RuleViolation(
            rule_id='R3_OUT_OF_BAND',
            severity=severity,
            message=f'目标 "{target}" 有预测 > {hi}',
            value=float(arr.max()),
            details={'target': target, 'bound': 'hi', 'value': hi},
        ))
    return viol


@register_rule('R4_MONOTONIC_EXPECTED')
def rule_monotonic_expected(
    *, prediction, direction: str = 'increasing',
    target: str = 'y', severity: str = AlertLevel.WARNING, **_,
) -> List[RuleViolation]:
    """
    预测序列应单调 (业务强约束场景, 如累计余额).

    direction: 'increasing' 或 'decreasing'。
    """
    arr = np.atleast_1d(np.asarray(prediction, dtype=float))
    if len(arr) < 2:
        return []
    diff = np.diff(arr)
    bad = (diff < 0).sum() if direction == 'increasing' else (diff > 0).sum()
    if bad == 0:
        return []
    return [RuleViolation(
        rule_id='R4_MONOTONIC_EXPECTED',
        severity=severity,
        message=(f'目标 "{target}" 预期 {direction} 但出现 '
                 f'{int(bad)} 次反向变化'),
        value=float(bad),
        details={'target': target, 'direction': direction},
    )]


#: 默认启用的内建规则 (RuleEngine 默认使用)
DEFAULT_RULE_IDS: List[str] = [
    'R1_NON_NEGATIVE',
    'R2_SUDDEN_CHANGE',
]


# ==========================================================================
# RuleEngine
# ==========================================================================

class RuleEngine(RuleChecker):
    """
    规则聚合器 / Compose registered rules.

    Args:
        rule_ids: 启用的规则 ID 列表; None → DEFAULT_RULE_IDS
        params:   {rule_id: {kw}} 每条规则的固定参数 (例如 R2 的
                  max_ratio, R3 的 lo/hi)
        extra_rules: 临时添加的 (id → callable) 映射, 无需注册即可使用
    """

    def __init__(
        self,
        rule_ids: Optional[Sequence[str]] = None,
        *,
        params: Optional[Mapping[str, Mapping[str, Any]]] = None,
        extra_rules: Optional[Mapping[str, Callable]] = None,
    ):
        # 启用的规则 ID 列表 (顺序即检查顺序); None 时使用 DEFAULT_RULE_IDS
        # / Active rule IDs (order = check order)
        self.rule_ids = list(rule_ids) if rule_ids is not None \
            else list(DEFAULT_RULE_IDS)
        # 每条规则的固定参数 {rule_id: {kw}}, 例如 R2 的 max_ratio, R3 的 lo/hi
        # / Per-rule kwargs forwarded to the rule callable
        self.params: Dict[str, Dict[str, Any]] = {
            k: dict(v) for k, v in (params or {}).items()
        }
        # 临时规则映射 {rule_id: callable}, 不进全局注册表 (本实例独享)
        # / Instance-local extra rules (not in the global registry)
        self._extra: Dict[str, Callable] = dict(extra_rules or {})

    # ---- 规则管理 -----------------------------------------------------
    def enable(self, rule_id: str) -> None:
        if rule_id not in self.rule_ids:
            self.rule_ids.append(rule_id)

    def disable(self, rule_id: str) -> None:
        self.rule_ids = [r for r in self.rule_ids if r != rule_id]

    def set_params(self, rule_id: str, **kw) -> None:
        self.params.setdefault(rule_id, {}).update(kw)

    def add_rule(self, rule_id: str, fn: Callable) -> None:
        """临时添加一个规则 (不进全局注册表)。"""
        self._extra[rule_id] = fn
        if rule_id not in self.rule_ids:
            self.rule_ids.append(rule_id)

    def available(self) -> List[str]:
        """当前可用的规则 (合并全局注册 + 本实例 extra)。"""
        return sorted(set(list(RULE_REGISTRY.keys()) + list(self._extra)))

    # ---- 主调度 -------------------------------------------------------
    def check(
        self,
        *,
        features: Any = None,
        prediction: Any = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> List[RuleViolation]:
        context = dict(context or {})
        all_viol: List[RuleViolation] = []
        for rid in self.rule_ids:
            fn = self._extra.get(rid) or RULE_REGISTRY.get(rid)
            if fn is None:
                all_viol.append(RuleViolation(
                    rule_id=rid,
                    severity=AlertLevel.WARNING,
                    message=f'规则 "{rid}" 未注册, 已跳过',
                ))
                continue
            # context 里的键允许直接展开到规则函数 kwargs (如 'last_actual'),
            # 但**不能覆盖** prediction / features / context 这三个保留参数,
            # 否则 caller 传 context={'prediction': ...} 会篡改第 1 参数(安全 bug).
            # / Don't let context override the reserved kwargs.
            _RESERVED = ('prediction', 'features', 'context')
            context_safe = {k: v for k, v in (context or {}).items()
                            if k not in _RESERVED}
            kw: Dict[str, Any] = {
                'prediction': prediction,
                'features': features,
                'context': context,
                **context_safe,
                **self.params.get(rid, {}),
            }
            # 只传函数能接收的参数
            sig = inspect.signature(fn)
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD
                       for p in sig.parameters.values()):
                kw = {k: v for k, v in kw.items() if k in sig.parameters}
            try:
                viol = fn(**kw) or []
                all_viol.extend(viol)
            except Exception as exc:
                all_viol.append(RuleViolation(
                    rule_id=rid,
                    severity=AlertLevel.ERROR,
                    message=f'规则 "{rid}" 执行异常: {exc}',
                ))
        return all_viol


# ==========================================================================
# main — 注入 5 种情形演示
# ==========================================================================

def main() -> None:
    """演示 4 个内建规则 + 1 个临时自定义规则。"""
    print('=' * 70)
    print(' rule_engine — declarative rules demo')
    print('=' * 70)

    # 自定义临时规则: 预测方差过大
    def high_variance(*, prediction, max_std=10, **_):
        arr = np.atleast_1d(np.asarray(prediction, dtype=float))
        if len(arr) < 2:
            return []
        s = float(arr.std())
        if s > max_std:
            return [RuleViolation(
                rule_id='R_VARIANCE',
                severity=AlertLevel.WARNING,
                message=f'预测标准差 {s:.2f} 超过 {max_std}',
                value=s,
            )]
        return []

    engine = RuleEngine(
        rule_ids=['R1_NON_NEGATIVE', 'R2_SUDDEN_CHANGE',
                  'R3_OUT_OF_BAND', 'R4_MONOTONIC_EXPECTED'],
        params={
            'R2_SUDDEN_CHANGE': {'max_ratio': 0.3, 'target': 'deposit'},
            'R3_OUT_OF_BAND':   {'lo': 0, 'hi': 1000, 'target': 'deposit'},
            'R4_MONOTONIC_EXPECTED': {'direction': 'increasing',
                                      'target': 'cum_balance'},
        },
    )
    engine.add_rule('R_VARIANCE', high_variance)
    engine.set_params('R_VARIANCE', max_std=5.0)

    scenarios = {
        '1. 正常': dict(prediction=[100, 105, 110, 115]),
        '2. 负值': dict(prediction=[-20, 10, 12]),
        '3. 突变': dict(prediction=[200], context={'last_actual': 100}),
        '4. 超上界': dict(prediction=[1500, 1200]),
        '5. 非单调': dict(prediction=[100, 95, 98, 97]),
        '6. 高方差': dict(prediction=[1, 50, 100, 8]),
    }
    for name, kw in scenarios.items():
        ctx = kw.pop('context', {})
        viol = engine.check(prediction=kw['prediction'], context=ctx)
        print(f'\n{name}: {len(viol)} 条违规')
        for v in viol:
            print(f'   {v.severity.upper():<8} {v.rule_id}: {v.message}')

    print(f'\n当前启用的规则: {engine.rule_ids}')
    print(f'全局注册表中可用: {sorted(RULE_REGISTRY)}')


if __name__ == '__main__':
    main()
