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

import logging
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
        Rule ID	类型	触发条件	说明
mape_hard	Performance	MAPE > 15%	性能硬阈值：误差太大必须重训
concept_drift	Drift	concept_drift 为 True	概念漂移：数据分布规律变了
data_and_perf	Drift	数据漂移 且 MAPE > 10%	组合条件：数据变了 + 性能降了
volume_based	Volume	新样本 ≥ 1000 个	数据量驱动：积累了足够多新数据
time_based	Time	距上次训练 ≥ 30 天	时间驱动：定期刷新模型
    """

    # 规则唯一 ID, 写入 RetrainingDecision.triggered 列表
    # / Unique rule id for tracking
    rule_id: str
    # 规则分类标签, 用于过滤/统计 (如只看 drift 类)
    # / Rule category for grouping/filtering
    kind: str
    # 谓词函数: 接收 context dict 返回 bool; True 表示触发重训
    # / Predicate function — context → bool (True = trigger)
    predicate: Callable[[Dict[str, Any]], bool]
    # 命中后展示给用户的可读原因
    # / Human-readable reason shown when fired
    reason: str = ''
    # 是否启用 (False 时 check 跳过此规则)
    # / Whether the rule is currently active
    enabled: bool = True


@dataclass
class RetrainingDecision:
    """重训决策 / Decision bundle."""

    # 是否建议重训 (任一规则**业务命中**即 True; 谓词执行异常不会让此值变 True)
    # / Whether retraining is recommended (only business hits, never errors)
    should_retrain: bool
    # 命中规则的人类可读原因列表
    # / Human-readable reasons (one per fired rule)
    reasons: List[str] = field(default_factory=list)
    # 命中规则的 rule_id 列表 (谓词返回 True)
    # / Fired rule ids (predicate returned True)
    triggered: List[str] = field(default_factory=list)
    # 谓词执行抛异常的规则 ID 列表 (与 triggered 互斥, 不计入 should_retrain).
    # 用途: 让运维区分 "规则真命中, 该重训" vs "规则代码出错, 别误判".
    # / Rule ids whose predicate raised. NOT counted as a trigger;
    #   surfaced separately so ops can tell business hits from code errors.
    errored_rules: List[str] = field(default_factory=list)
    # 谓词异常详情 {rule_id: 'ExcType: msg'}, 便于排查
    # / Per-rule predicate error details for debugging
    rule_errors: Dict[str, str] = field(default_factory=dict)
    # 冷却期结束时刻; 在此时刻之前 should_retrain 强制为 False
    # / End of cooldown window (if any)
    cooldown_until: Optional[datetime] = None
    # 决策产生时刻
    # / Decision timestamp
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
        logger: Optional[logging.Logger] = None,
    ):
        # 规则列表; None 时使用 default_rules() 的 5 条通用规则
        # / Active rules; defaults to 5 generic rules
        self.rules: List[RetrainingRule] = list(
            rules if rules is not None else self.default_rules()
        )
        # 一次重训后的强制冷却期 (小时); 0 = 无冷却
        # 防止短时间内反复触发抖动 / Cooldown to prevent thrashing
        self.cooldown_hours = float(cooldown_hours)
        # 最近一次记录到的重训时刻 (record_retraining 设置); 用于计算冷却剩余时长
        # / Timestamp of last recorded retraining
        self.last_retraining: Optional[datetime] = None
        # 谓词异常的日志器; 默认走 'tsf_frame.monitoring.retraining' 命名空间
        # / Logger for predicate-execution errors (separate from business hits)
        self._logger = logger or logging.getLogger(
            'tsf_frame.monitoring.retraining'
        )

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

        # triggered: 业务命中 (谓词返回 True) — 才会让 should_retrain=True
        # errored_rules: 谓词执行抛异常 — 单独存放, 不计入触发, 由运维侧排查
        triggered: List[str] = []
        reasons: List[str] = []
        errored_rules: List[str] = []
        rule_errors: Dict[str, str] = {}
        for r in self.rules:
            if not r.enabled:
                continue
            try:
                hit = bool(r.predicate(ctx))
            except Exception as exc:
                # 谓词代码错误 ≠ 业务命中. 记录 + warning 日志, 不进 triggered.
                err_msg = f'{type(exc).__name__}: {exc}'
                errored_rules.append(r.rule_id)
                rule_errors[r.rule_id] = err_msg
                reasons.append(f'规则 {r.rule_id} 执行异常: {err_msg}')
                self._logger.warning(
                    'RetrainingTrigger rule %r predicate raised: %s',
                    r.rule_id, exc, exc_info=True,
                )
                continue
            if hit:
                triggered.append(r.rule_id)
                reasons.append(r.reason or r.rule_id)

        return RetrainingDecision(
            # 关键: 仅由业务命中决定, 谓词异常不会误触发重训
            should_retrain=bool(triggered),
            reasons=reasons,
            triggered=triggered,
            errored_rules=errored_rules,
            rule_errors=rule_errors,
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
