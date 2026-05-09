"""
HPF (住房公积金) 端到端监控演示 — 使用通用 ModelMonitor
========================================================

本示例展示:

1. 如何用通用的 ``tsf_frame.monitoring.ModelMonitor`` 组合出一套
   面向 HPF 业务的监控栈 (不再依赖已废弃的 HPFMonitor)。
2. 如何通过 ``@register_rule`` 注入业务规则 (非负 / 月环比突变 /
   异常回声), 这些规则 **不会** 污染通用 monitoring 包。
3. 如何把 ``HPFMonitoringConfig`` 的阈值映射到新架构的各组件。
4. 如何同时启用:
       - 性能监控 (PerformanceMonitor + baseline)
       - 数据漂移 (DataDriftDetector, 训练特征作参考)
       - 概念漂移 (ConceptDriftDetector, 残差驱动)
       - 预测漂移 (PredictionDriftDetector, 验证集预测作参考)
       - 原始数据质量 (DataQualityMonitor, 频率/非负)
       - 业务规则 (RuleEngine + 新注册的 R_HPF_* 规则)
       - 重训触发器 (RetrainingTrigger 默认规则)
       - 告警通道 (Console + Logging + File, 仅 WARNING+ 进文件)
       - SQLite 持久化
       - 静态 PNG 报表

运行::

    python pipelines/examples/hpf_monitoring_example.py
"""

from __future__ import annotations

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ------ sys.path 引导 + 项目根锚定 ------
# 项目根 = 包含 configs/ 和 src/ 的最近祖先目录, 不依赖 CWD.
# 这样无论从命令行还是 IDE 直接 Run 此文件, 日志都落在 <root>/logs/ 下,
# 不会出现 pipelines/examples/logs 这种 CWD 飞地.
# / Project root anchor; CWD-independent so logs always land in <root>/logs/
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent  # 兜底
for _p in (_HERE.parent, *_HERE.parents):
    if (_p / 'configs').is_dir() and (_p / 'src').is_dir():
        _PROJECT_ROOT = _p
        for _q in (_p, _p / 'src'):
            if str(_q) not in sys.path:
                sys.path.insert(0, str(_q))
        break

from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from configs.hpf import HPFConfig
from tsf_frame.business.hpf_adapter import HPFAdapter
from tsf_frame.models.classical.ml_models import get_ml_model
from tsf_frame.utils.logger import get_logger

# 通用监控栈
from tsf_frame.monitoring import (
    AlertLevel,
    ConsoleChannel, LoggingChannel, FileChannel,
    DataDriftDetector, ConceptDriftDetector, PredictionDriftDetector,
    DataQualityMonitor, FrequencyCheck, RangeCheck,
    ModelMonitor, PerformanceMonitor,
    PlotReport,
    RetrainingRule, RetrainingTrigger,
    RuleEngine, RuleViolation,
    SQLiteStore,
    register_rule,
)

from pipelines.run_hpf_forecast import generate_hpf_data, build_ml_features


# =====================================================================
# 一、在框架启动时注册 HPF 专属规则
#     这些规则只在本 example 里登记 (进程级). 如果你想跨项目复用,
#     把它们挪到 tsf_frame.business.hpf_rules 之类的模块, 然后在
#     包 __init__ 里 import 即可自动注册.
# =====================================================================

@register_rule('R_HPF_NON_NEGATIVE')
def rule_hpf_non_negative(
    *, prediction, target: str = 'monthly_deposit',
    non_negative_cols: Optional[List[str]] = None,
    severity: str = AlertLevel.CRITICAL, **_,
) -> List[RuleViolation]:
    """HPF 专属: 缴存/提取/贷款等列一律不能为负。"""
    cols = non_negative_cols or []
    if target not in cols:
        return []
    arr = np.atleast_1d(np.asarray(prediction, dtype=float))
    neg = int(np.sum(arr < 0))
    if neg == 0:
        return []
    return [RuleViolation(
        rule_id='R_HPF_NON_NEGATIVE',
        severity=severity,
        message=(f'HPF 目标 "{target}" 预测出现 {neg} 个负值 '
                 f'(min={float(arr.min()):.2f})'),
        value=float(arr.min()),
        details={'target': target},
    )]


@register_rule('R_HPF_SUDDEN_CHANGE')
def rule_hpf_sudden_change(
    *, prediction, last_actual: Optional[float] = None,
    sudden_change_ratio: float = 0.30,
    target: str = 'monthly_deposit',
    severity: str = AlertLevel.WARNING, **_,
) -> List[RuleViolation]:
    """HPF 月环比突变: 缴存额相比上月变化超过 sudden_change_ratio 即告警。"""
    if last_actual is None or last_actual == 0:
        return []
    arr = np.atleast_1d(np.asarray(prediction, dtype=float))
    if arr.size == 0:
        return []
    first = float(arr[0])
    ratio = abs(first - last_actual) / abs(last_actual)
    if ratio <= sudden_change_ratio:
        return []
    return [RuleViolation(
        rule_id='R_HPF_SUDDEN_CHANGE',
        severity=severity,
        message=(f'HPF "{target}" 月环比变化 {ratio:.1%} '
                 f'超阈值 {sudden_change_ratio:.0%} '
                 f'(pred={first:,.0f} vs last={last_actual:,.0f})'),
        value=ratio,
        details={'target': target, 'pred': first,
                 'last_actual': last_actual},
    )]


@register_rule('R_HPF_OUTLIER_ECHO')
def rule_hpf_outlier_echo(
    *, prediction, recent_actuals: Optional[List[float]] = None,
    outlier_sigma_ratio: float = 3.0,
    target: str = 'monthly_deposit',
    severity: str = AlertLevel.WARNING, **_,
) -> List[RuleViolation]:
    """预测相对历史均值的 z-score 超过阈值即警告 (异常回声)。"""
    if not recent_actuals or len(recent_actuals) < 6:
        return []
    arr = np.atleast_1d(np.asarray(prediction, dtype=float))
    if arr.size == 0:
        return []
    ref = np.asarray(recent_actuals, dtype=float)
    mu, sigma = float(ref.mean()), float(ref.std(ddof=0))
    if sigma < 1e-9:
        return []
    z = abs(float(arr[0]) - mu) / sigma
    if z <= outlier_sigma_ratio:
        return []
    return [RuleViolation(
        rule_id='R_HPF_OUTLIER_ECHO',
        severity=severity,
        message=(f'HPF "{target}" 预测 z={z:.2f} 超阈值 '
                 f'{outlier_sigma_ratio:.1f}σ (pred={float(arr[0]):,.0f}, '
                 f'历史 μ={mu:,.0f} σ={sigma:,.0f})'),
        value=z,
        details={'target': target, 'z': z, 'mu': mu, 'sigma': sigma},
    )]


# =====================================================================
# 二、监控栈构造工厂 — 把配置翻译成装配好的 ModelMonitor
# =====================================================================

def build_hpf_monitor(
    *,
    model_id: str,
    cfg_mon,                          # HPFMonitoringConfig 实例
    target_col: str,
    reference_features: np.ndarray,
    reference_predictions: Optional[np.ndarray],
    performance_baseline: dict,
    logger,
) -> ModelMonitor:
    """把 HPFMonitoringConfig 的阈值映射到新组件实例。"""

    # --- 持久化: SQLite ---
    store = SQLiteStore(db_path=cfg_mon.sqlite_path)

    # --- 性能监控: 窗口 = window_months, 全项目 MAPE 量纲统一为小数 ---
    # mape 指标 (monitoring.mape / MetricsCalculator.mape) 和
    # HPFMonitoringConfig.mape_warning / mape_critical 都是**小数形式** (0.10=10%),
    # 不再需要 ×100 换算. 展示时用 f'{value:.2%}'.
    perf = PerformanceMonitor(
        model_id=model_id,
        window_size=cfg_mon.window_months,
        metrics=['mae', 'mape', 'rmse', 'r2', 'smape'],
        baseline=performance_baseline,
        probabilistic=True,               # 自动加入 picp/miw/winkler
    )

    # --- 漂移 ---
    data_drift = DataDriftDetector(
        reference=reference_features,
        window_size=cfg_mon.window_months * 2,  # 漂移检测窗口是性能窗口的2倍
        psi_warn=cfg_mon.psi_warning,
        psi_crit=cfg_mon.psi_critical,
        ks_alpha=cfg_mon.ks_pvalue,
    )
    concept_drift = ConceptDriftDetector(
        window_size=cfg_mon.window_months,
        mean_shift_std=3.0,
    )
    prediction_drift = (
        PredictionDriftDetector(
            reference=reference_predictions,
            window_size=cfg_mon.window_months,
            psi_warn=cfg_mon.psi_warning,
            psi_crit=cfg_mon.psi_critical,
        )
        if reference_predictions is not None else None
    )

    # --- 原始数据质量 (频率连续 + 非负) ---
    data_quality = (DataQualityMonitor()
                    .add(FrequencyCheck(expected_freq='MS'))
                    .add(RangeCheck(non_negative=cfg_mon.non_negative_cols)))

    # --- 业务规则引擎: 用上面注册的 3 个 R_HPF_* 规则 ---
    rule_engine = RuleEngine(
        rule_ids=[
            'R_HPF_NON_NEGATIVE',
            'R_HPF_SUDDEN_CHANGE',
            'R_HPF_OUTLIER_ECHO',
        ],
        params={
            'R_HPF_NON_NEGATIVE': {
                'target': target_col,
                'non_negative_cols': cfg_mon.non_negative_cols,
            },
            'R_HPF_SUDDEN_CHANGE': {
                'target': target_col,
                'sudden_change_ratio': cfg_mon.sudden_change_ratio,
            },
            'R_HPF_OUTLIER_ECHO': {
                'target': target_col,
                'outlier_sigma_ratio': cfg_mon.outlier_sigma_ratio,
            },
        },
    )

    # --- 重训触发: 用 HPF 特化阈值替换默认 mape_hard ---
    # 量纲约定: cfg_mon.mape_critical / mape_warning 都是**小数** (0.10 / 0.15),
    # metrics['mape'] 也是小数, 直接比较, 无需手工 *100 转换.
    # / Both thresholds and metrics are decimals; direct comparison.
    retrain = RetrainingTrigger([
        RetrainingRule(
            rule_id='mape_hpf_critical',
            kind='performance',
            predicate=lambda c: (
                c.get('performance', {}).get('mape', 0)
                > cfg_mon.mape_critical),
            reason=f'MAPE > {cfg_mon.mape_critical:.0%}',
        ),
        RetrainingRule(
            rule_id='concept_drift',
            kind='drift',
            predicate=lambda c: bool(c.get('concept_drift')),
            reason='检测到概念漂移',
        ),
        RetrainingRule(
            rule_id='data_drift_and_mape',
            kind='drift',
            predicate=lambda c: (
                bool(c.get('data_drift')) and
                c.get('performance', {}).get('mape', 0)
                > cfg_mon.mape_warning),
            reason='数据漂移 + MAPE 超警戒',
        ),
    ], cooldown_hours=24.0)

    # --- 构造 ModelMonitor ---
    mon = ModelMonitor(
        model_id=model_id,
        store=store,
        performance_monitor=perf,
        data_quality=data_quality,
        data_drift=data_drift,
        concept_drift=concept_drift,
        prediction_drift=prediction_drift,
        rule_engine=rule_engine,
        retraining_trigger=retrain,
        target_name=target_col,
        cold_start_samples=cfg_mon.cold_start_months,
        window_size=cfg_mon.window_months,
    )

    # --- 告警通道 (3 条): 终端 + logger + 专用文件 ---
    os.makedirs(os.path.dirname(cfg_mon.alert_log_file) or '.', exist_ok=True)
    (mon.alert_manager
        .add_channel(ConsoleChannel(min_level=AlertLevel.WARNING))
        .add_channel(LoggingChannel(
            logger=logger, min_level=AlertLevel.INFO))
        .add_channel(FileChannel(
            cfg_mon.alert_log_file, min_level=AlertLevel.WARNING)))

    logger.info(f'监控栈已装配: store={cfg_mon.sqlite_path}, '
                f'alert_file={cfg_mon.alert_log_file}, '
                f'rules={rule_engine.rule_ids}, '
                f'retrain_rules={retrain.list_rules()}')
    return mon


# =====================================================================
# 三、主流程
# =====================================================================

def main():
    logger = get_logger(
        'hpf_monitoring_example',
        log_dir=str(_PROJECT_ROOT / 'logs' / 'runs'),
    )
    logger.info('=' * 70)
    logger.info('  HPF 端到端监控演示 (generic ModelMonitor stack)')
    logger.info('=' * 70)

    # ── 1) 配置 + 模拟数据 ────────────────────────────────────────
    cfg = HPFConfig()
    cfg.data.target_columns = ['monthly_deposit']
    cfg.model.model_name = 'ridge'
    cfg.model.probabilistic = True
    cfg.model.probabilistic_method = 'residual'
    cfg.monitoring.cold_start_months = 6     # 演示时放宽冷启动

    target_col = cfg.data.target_columns[0]
    logger.info('[1/7] 生成 12 年模拟月度公积金数据')
    raw = generate_hpf_data(years=12)
    logger.info(f'     shape={raw.shape}  '
                f'{raw.index[0].date()}..{raw.index[-1].date()}')

    # ── 2) 用 DataQualityMonitor 先扫一遍原始数据 ─────────────────
    logger.info('[2/7] 原始数据质量扫描')
    pre_dqm = (DataQualityMonitor()
               .add(FrequencyCheck(expected_freq='MS'))
               .add(RangeCheck(non_negative=cfg.monitoring.non_negative_cols)))
    pre_issues = pre_dqm.check(raw)
    logger.info(f'     质量问题数: {len(pre_issues)}')
    for iss in pre_issues[:5]:
        logger.info(f'       {iss.severity.upper():<8} {iss.issue_id} '
                    f'[{iss.column}] {iss.message}')

    # ── 3) 预处理 + 特征工程 ─────────────────────────────────────
    logger.info('[3/7] 预处理 + 特征工程')
    adapter = HPFAdapter(cfg.to_adapter_config())
    ok, msg = adapter.validate_data(raw)
    logger.info(f'     validate: {msg}')
    processed, metadata = adapter.preprocess(raw)
    X, y, feat_dates, feat_cols = build_ml_features(
        processed, target_col,
        feature_config=cfg.to_feature_config(),
    )
    logger.info(f'     features={len(feat_cols)}  samples={len(X)}')

    # ── 4) 划分 + 训练 ────────────────────────────────────────────
    n_test = max(18, int(len(X) * 0.20))
    n_val = max(6, int(len(X) * 0.10))
    n_train = len(X) - n_test - n_val
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    test_dates = feat_dates[n_train + n_val:]
    logger.info(f'     train={n_train}  val={n_val}  test={n_test}')

    logger.info('[4/7] 训练 Ridge')
    model = get_ml_model('ridge', cfg.to_model_config())
    model.fit((X_train, y_train))

    # 反归一化小工具
    def denorm(arr: np.ndarray) -> np.ndarray:
        tmp = pd.DataFrame(arr.reshape(-1, 1), columns=[target_col])
        return adapter._denormalize(tmp, metadata)[target_col].values

    # 验证集预测: 用于做 PredictionDriftDetector 的参考基线
    val_preds_level = model.predict(X_val).flatten()
    val_preds_orig = denorm(val_preds_level)
    val_y_orig = denorm(y_val)
    val_mape = float(np.mean(np.abs((val_y_orig - val_preds_orig)
                                    / np.where(val_y_orig == 0, 1,
                                               val_y_orig))))
    logger.info(f'     val MAPE (orig scale) = {val_mape:.2%}')

    # ── 5) 装配监控栈 ────────────────────────────────────────────
    logger.info('[5/7] 装配监控栈 (ModelMonitor + HPF 规则注入)')
    monitor = build_hpf_monitor(
        model_id='hpf_ridge_deposit_demo',
        cfg_mon=cfg.monitoring,
        target_col=target_col,
        reference_features=X_train,        # 归一化空间的训练特征
        reference_predictions=val_preds_orig,
        performance_baseline={
            'mape': val_mape,              # 用验证集 MAPE 做基线
            'mae': float(np.mean(np.abs(val_y_orig - val_preds_orig))),
        },
        logger=logger,
    )

    # ── 6) 回放测试集作为"生产"预测流 ────────────────────────────
    logger.info('[6/7] 回放测试集 (逐月喂入 monitor)')
    y_pred_level = model.predict(X_test).flatten()
    prob = model.predict_probabilistic(X_test)
    lower_orig = (denorm(prob.lower.flatten())
                  if prob.lower is not None else None)
    upper_orig = (denorm(prob.upper.flatten())
                  if prob.upper is not None else None)
    y_pred_orig = denorm(y_pred_level)
    y_test_orig = denorm(y_test)

    for i, ts in enumerate(test_dates):
        monitor.record_prediction(
            timestamp=pd.Timestamp(ts).to_pydatetime(),
            prediction=float(y_pred_orig[i]),
            actual=float(y_test_orig[i]),
            features=X_test[i],
            y_lower=None if lower_orig is None else float(lower_orig[i]),
            y_upper=None if upper_orig is None else float(upper_orig[i]),
        )

    status = monitor.check_status()
    logger.info(f'     alert_level        = {status.alert_level}')
    logger.info(f'     needs_retraining   = {status.needs_retraining}')
    logger.info(f'     data_drift         = '
                f'{status.data_drift and status.data_drift.detected}')
    logger.info(f'     concept_drift      = '
                f'{status.concept_drift and status.concept_drift.detected}')
    logger.info(f'     prediction_drift   = '
                f'{status.prediction_drift and status.prediction_drift.detected}')
    for k, v in status.performance_metrics.items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            logger.info(f'       metric.{k:<10} = {v:.4f}')

    # ── 7) 人工注入 3 种异常, 验证 HPF 规则触发 ──────────────────
    logger.info('[7/7] 人工注入异常 → 触发 HPF 规则告警')
    last_ts = pd.Timestamp(test_dates[-1])
    last_actual = float(y_test_orig[-1])

    # (a) R_HPF_NON_NEGATIVE — CRITICAL
    t1 = (last_ts + pd.DateOffset(months=1)).to_pydatetime()
    monitor.record_prediction(
        timestamp=t1, prediction=-500.0,
        actual=last_actual, features=X_test[-1],
    )
    s_a = monitor.check_status()
    logger.info(f'     [a] 负值注入       level={s_a.alert_level}  '
                f'viol={len(s_a.rule_violations)}')

    # (b) R_HPF_SUDDEN_CHANGE — WARNING: 预测翻 3 倍
    t2 = (last_ts + pd.DateOffset(months=2)).to_pydatetime()
    monitor.record_prediction(
        timestamp=t2, prediction=last_actual * 3.0,
        actual=last_actual, features=X_test[-1],
    )
    s_b = monitor.check_status()
    logger.info(f'     [b] 环比 3x        level={s_b.alert_level}  '
                f'viol={len(s_b.rule_violations)}')

    # (c) R_HPF_OUTLIER_ECHO — WARNING: 预测 5σ 外
    recent = [float(v) for v in y_test_orig[-12:]]
    mu, sigma = float(np.mean(recent)), float(np.std(recent))
    t3 = (last_ts + pd.DateOffset(months=3)).to_pydatetime()
    outlier_val = mu + 6 * sigma
    # 监控器只知 last_actual, 不知 recent_actuals, 所以手动调用
    # rule_engine 一次作示例
    extra_viol = monitor.rule_engine.check(
        prediction=[outlier_val],
        context={'target': target_col,
                 'last_actual': last_actual,
                 'recent_actuals': recent},
    )
    logger.info(f'     [c] 6σ 外预测 手动规则扫描 → {len(extra_viol)} 条:')
    for v in extra_viol:
        logger.info(f'         {v.severity.upper():<8} {v.rule_id}: '
                    f'{v.message}')

    # (d) 数据质量: 频率不连续 → 直接喂进 DataQualityMonitor
    discontinuous = raw.copy().drop(raw.index[5:7])
    dq_issues = monitor.record_data_batch(discontinuous)
    logger.info(f'     [d] 频率不连续注入 → 质量问题 {len(dq_issues)} 条')

    final = monitor.check_status()
    logger.info(f'     final alert_level = {final.alert_level}')
    if final.needs_retraining:
        logger.info('     触发重训建议:')
        for r in final.recommendations[:5]:
            logger.info(f'       - {r}')

    # ── 8) 生成静态 PNG 报表 ──────────────────────────────────────
    os.makedirs(cfg.monitoring.report_dir, exist_ok=True)
    out_path = os.path.join(
        cfg.monitoring.report_dir,
        f'hpf_report_{monitor.model_id}_'
        f'{datetime.now():%Y%m%d_%H%M%S}.png',
    )
    report_path = PlotReport().generate(
        model_id=monitor.model_id,
        store=monitor.store,
        out_path=out_path,
        days=400,
    )
    logger.info(f'     报表: {report_path}')

    # ── 9) SQLite 快照 ────────────────────────────────────────────
    preds = monitor.store.query_predictions(model_id=monitor.model_id)
    metrics = monitor.store.query_metrics(model_id=monitor.model_id)
    alerts = monitor.store.query_alerts(model_id=monitor.model_id)
    logger.info('=' * 70)
    logger.info(f'SQLite 快照 @ {cfg.monitoring.sqlite_path}')
    logger.info(f'  predictions      : {len(preds)}')
    logger.info(f'  metrics_snapshot : {len(metrics)}')
    logger.info(f'  alerts           : {len(alerts)}')
    dist = {}
    for a in alerts:
        dist[a.level] = dist.get(a.level, 0) + 1
    for lv in ('info', 'warning', 'error', 'critical'):
        if lv in dist:
            logger.info(f'    {lv:<9} {dist[lv]}')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
