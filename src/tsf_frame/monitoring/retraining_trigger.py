"""
重训触发器 / Retraining trigger
================================

决定 *"模型此刻是否应当重训"* 的规则引擎。它接收每次
``check_status()`` 的汇总信号 (性能指标, 漂移标志, 时间, 样本量)
并返回一个决策对象:

    {
        'should_retrain': bool,
        'reasons':        List[str],
        'triggered':      List[str],    # 命中的规则 ID
        'cooldown_until': Optional[datetime],
    }

设计:
* 规则以 ``RetrainingRule`` dataclass 描述, 谓词是一个 callable;
* 内置 4 类规则: performance / drift / time / volume, 全部可选;
* 支持"冷却期" (cooldown) — 一次重训后短期内不再触发, 防止抖动;
* ``record_retraining()`` 显式记录一次重训, 推进内部时钟。

与 AlertManager 的分工:
* AlertManager 负责 *"现在告警"*
* RetrainingTrigger 负责 *"是否应当走重训流程"*, 这是更贵的动作
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional

__all__ = [
    'RetrainingRule',
    'RetrainingTrigger',
    'RetrainingDecision',
]


# ==========================================================================
# 数据类
# ==========================================================================

@dataclass
class RetrainingRule:
    """
    单条重训规则 / A single retraining rule.

    Attributes:
        rule_id: 规则标识
        kind:    'performance' | 'drift' | 'time' | 'volume' | 'custom'
        predicate: callable(context) -> bool
        reason:  命中时展示的说明
        enabled: 是否启用
    """

    rule_id: str
    kind: str
    predicate: Callable[[Dict[str, Any]], bool]
    reason: str = ''
    enabled: bool = True


@dataclass
class RetrainingDecision:
    """重训决策 / Decision bundle."""

    should_retrain: bool
    reasons: List[str] = field(default_factory=list)
    triggered: List[str] = field(default_factory=list)
    cooldown_until: Optional[datetime] = None
    checked_at: datetime = field(default_factory=datetime.now)


# ==========================================================================
# Trigger
# ==========================================================================

class RetrainingTrigger:
    """
    重训触发决策器 / Retraining decision engine.

    Args:
        rules:           初始规则列表; 空则默认调用 ``default_rules()``
        cooldown_hours:  一次重训后多少小时内强制抑制; 0 = 无冷却

    Context schema (每次 ``check()`` 传入):
        performance:    Mapping[str, float]
        data_drift:     bool
        concept_drift:  bool
        prediction_drift: bool
        samples_since_last_train: int
        hours_since_last_train:   float
        now:                      datetime
    """

    def __init__(
        self,
        rules: Optional[List[RetrainingRule]] = None,
        *,
        cooldown_hours: float = 0.0,
    ):
        self.rules: List[RetrainingRule] = list(
            rules if rules is not None else self.default_rules()
        )
        self.cooldown_hours = float(cooldown_hours)
        self.last_retraining: Optional[datetime] = None

    # ---- 规则管理 -----------------------------------------------------
    def add(self, rule: RetrainingRule) -> None:
        self.rules.append(rule)

    def remove(self, rule_id: str) -> None:
        self.rules = [r for r in self.rules if r.rule_id != rule_id]

    def enable(self, rule_id: str, enabled: bool = True) -> None:
        for r in self.rules:
            if r.rule_id == rule_id:
                r.enabled = enabled

    def list_rules(self) -> List[str]:
        return [r.rule_id for r in self.rules]

    # ---- 决策 ---------------------------------------------------------
    def check(self, context: Mapping[str, Any]) -> RetrainingDecision:
        ctx = dict(context)
        ctx.setdefault('now', datetime.now())

        cooldown_until = None
        if self.last_retraining and self.cooldown_hours > 0:
            cooldown_until = self.last_retraining + timedelta(
                hours=self.cooldown_hours)
            if ctx['now'] < cooldown_until:
                return RetrainingDecision(
                    should_retrain=False,
                    reasons=[f'处于冷却期内, 直到 '
                             f'{cooldown_until.isoformat(timespec="seconds")}'],
                    cooldown_until=cooldown_until,
                    checked_at=ctx['now'],
                )

        triggered, reasons = [], []
        for r in self.rules:
            if not r.enabled:
                continue
            try:
                if bool(r.predicate(ctx)):
                    triggered.append(r.rule_id)
                    reasons.append(r.reason or r.rule_id)
            except Exception as exc:
                reasons.append(f'规则 {r.rule_id} 执行异常: {exc}')

        return RetrainingDecision(
            should_retrain=bool(triggered),
            reasons=reasons,
            triggered=triggered,
            cooldown_until=cooldown_until,
            checked_at=ctx['now'],
        )

    def record_retraining(self, when: Optional[datetime] = None) -> None:
        """通知触发器一次重训已经发生, 推进冷却时钟。"""
        self.last_retraining = when or datetime.now()

    def reset(self) -> None:
        self.last_retraining = None

    # ---- 默认规则集 ---------------------------------------------------
    @staticmethod
    def default_rules() -> List[RetrainingRule]:
        """一组合理的通用默认规则, 使用者可按需裁剪。"""
        return [
            RetrainingRule(
                rule_id='mape_hard',
                kind='performance',
                predicate=lambda c: (
                    c.get('performance', {}).get('mape', 0) > 0.15),
                reason='MAPE > 15%',
            ),
            RetrainingRule(
                rule_id='concept_drift',
                kind='drift',
                predicate=lambda c: bool(c.get('concept_drift')),
                reason='检测到概念漂移',
            ),
            RetrainingRule(
                rule_id='data_and_perf',
                kind='drift',
                predicate=lambda c: (
                    bool(c.get('data_drift'))
                    and c.get('performance', {}).get('mape', 0) > 0.10),
                reason='数据漂移叠加性能下降',
            ),
            RetrainingRule(
                rule_id='volume_based',
                kind='volume',
                predicate=lambda c: (
                    int(c.get('samples_since_last_train', 0)) >= 1000),
                reason='累计 ≥ 1000 个新样本',
            ),
            RetrainingRule(
                rule_id='time_based',
                kind='time',
                predicate=lambda c: (
                    float(c.get('hours_since_last_train', 0)) >= 24 * 30),
                reason='距上次训练超过 30 天',
            ),
        ]


# ==========================================================================
# main — 演示
# ==========================================================================

def main() -> None:
    print('=' * 70)
    print(' retraining_trigger — decision demo')
    print('=' * 70)

    trig = RetrainingTrigger(cooldown_hours=1.0)

    print('\n规则列表:', trig.list_rules())

    # 场景 1: 一切正常
    d1 = trig.check({'performance': {'mape': 0.05},
                     'data_drift': False, 'concept_drift': False})
    print(f'\n[1] 正常           should_retrain={d1.should_retrain}  '
          f'triggered={d1.triggered}')

    # 场景 2: MAPE 超 hard 阈值
    d2 = trig.check({'performance': {'mape': 0.20},
                     'data_drift': True, 'concept_drift': False})
    print(f'[2] MAPE 20%+漂移  should_retrain={d2.should_retrain}  '
          f'triggered={d2.triggered}')
    for r in d2.reasons:
        print(f'      - {r}')

    # 记录一次重训 → 接下来处于冷却
    trig.record_retraining()
    d3 = trig.check({'performance': {'mape': 0.30},
                     'data_drift': True, 'concept_drift': True})
    print(f'\n[3] 冷却期内       should_retrain={d3.should_retrain}  '
          f'until={d3.cooldown_until}')

    # 自定义一条规则
    trig.add(RetrainingRule(
        rule_id='custom_outlier',
        kind='custom',
        predicate=lambda c: c.get('performance', {}).get('max_abs_err', 0) > 50,
        reason='单点误差超 50',
    ))
    trig.last_retraining = None  # 跳过冷却看效果
    d4 = trig.check({'performance': {'mape': 0.02, 'max_abs_err': 60}})
    print(f'\n[4] 自定义规则     should_retrain={d4.should_retrain}  '
          f'triggered={d4.triggered}')


if __name__ == '__main__':
    main()
