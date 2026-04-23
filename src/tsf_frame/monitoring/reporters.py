"""
监控报表 / Monitoring reports
==============================

两种开箱即用的报表生成器, 全部实现 ``ReportGenerator`` 接口:

==================  =====================================================
TextReport          纯文本 (``.txt``), 便于 CI / 邮件正文 / 终端查看
PlotReport          静态 PNG (matplotlib 3x2 子图), 预测/指标/告警
==================  =====================================================

报表数据源是任意 ``MetricStore`` 实例, 所以无论你用 SQLite, JSONL,
还是 InMemory, 都可以生成同样的报表——这就是"插拔"。

自定义报表? 继承 ``ReportGenerator``, 用 ``@register_report('my_name')``
注册, 即可通过 ``create_report('my_name')`` 获取。

注意: PlotReport 使用非交互 Agg 后端, 适合 headless CI;  中文字体
走项目统一的 ``visualization.base_visualizer._CN_FONTS`` (若可用)。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .interfaces import (
    Alert,
    AlertLevel,
    LEVEL_ORDER,
    MetricStore,
    ReportGenerator,
    register_report,
)

__all__ = ['TextReport', 'PlotReport']


def _default_window(
    days: int = 90,
) -> Tuple[datetime, datetime]:
    """默认报表窗口: 近 N 天。"""
    end = datetime.now()
    start = end - timedelta(days=days)
    return start, end


# ==========================================================================
# TextReport
# ==========================================================================

@register_report('text')
class TextReport(ReportGenerator):
    """
    文本报表 / Plain-text report.

    结构:
    1. Header (model_id / 时间窗 / 生成时间)
    2. 预测统计 (n, y_pred/y_actual 基础统计)
    3. 指标最新快照 (最多 20 条)
    4. 告警摘要 (按级别分桶)
    5. 最新 10 条告警明细
    """

    def generate(
        self,
        *,
        model_id: str,
        store: MetricStore,
        out_path: Optional[str] = None,
        days: int = 90,
        **_: Any,
    ) -> str:
        start, end = _default_window(days)
        preds = store.query_predictions(model_id=model_id,
                                        start=start, end=end)
        metrics = store.query_metrics(model_id=model_id,
                                      start=start, end=end)
        alerts = store.query_alerts(model_id=model_id,
                                    start=start, end=end)

        lines: List[str] = []
        sep = '=' * 70
        lines.append(sep)
        lines.append(f' Monitoring Report — {model_id}')
        lines.append(sep)
        lines.append(f'Generated at : {datetime.now():%Y-%m-%d %H:%M:%S}')
        lines.append(f'Window       : {start:%Y-%m-%d}  →  {end:%Y-%m-%d}  '
                     f'(近 {days} 天)')
        lines.append('')

        # 1) 预测统计
        lines.append('-' * 70)
        lines.append(f'Predictions  : {len(preds)} 条')
        if preds:
            y_preds = [p['y_pred'] for p in preds]
            actuals = [p['y_actual'] for p in preds
                       if p.get('y_actual') is not None]
            lines.append(f'  y_pred    min/mean/max = '
                         f'{min(y_preds):.3f} / '
                         f'{sum(y_preds)/len(y_preds):.3f} / '
                         f'{max(y_preds):.3f}')
            if actuals:
                lines.append(f'  y_actual  min/mean/max = '
                             f'{min(actuals):.3f} / '
                             f'{sum(actuals)/len(actuals):.3f} / '
                             f'{max(actuals):.3f}  (n={len(actuals)})')
        lines.append('')

        # 2) 指标最近快照
        lines.append('-' * 70)
        lines.append(f'Metric snapshots (最近 20 条): ')
        for m in metrics[-20:]:
            lines.append(f"  {m['timestamp']:%Y-%m-%d %H:%M:%S}  "
                         f"{m['name']:<15} = {m['value']:.4f}"
                         + (f"  (window={m['window']})"
                            if m.get('window') else ''))
        lines.append('')

        # 3) 告警分级
        lines.append('-' * 70)
        lines.append(f'Alerts       : {len(alerts)} 条')
        dist: Dict[str, int] = {}
        for a in alerts:
            dist[a.level] = dist.get(a.level, 0) + 1
        for lvl in (AlertLevel.INFO, AlertLevel.WARNING,
                    AlertLevel.ERROR, AlertLevel.CRITICAL):
            if dist.get(lvl):
                lines.append(f'  {lvl:<9} {dist[lvl]}')
        lines.append('')

        # 4) 最近 10 条告警详情
        if alerts:
            lines.append('Latest 10 alerts:')
            for a in alerts[-10:]:
                lines.append(
                    f"  [{a.timestamp:%Y-%m-%d %H:%M:%S}] "
                    f'{a.level.upper():<8} {a.source or "-":<12} '
                    f'{a.message}'
                )

        lines.append(sep)
        content = '\n'.join(lines)

        if out_path is None:
            out_dir = os.path.abspath('./logs/reports')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir,
                f'report_{model_id}_{datetime.now():%Y%m%d_%H%M%S}.txt',
            )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(content, encoding='utf-8')
        return out_path


# ==========================================================================
# PlotReport
# ==========================================================================

@register_report('plot')
class PlotReport(ReportGenerator):
    """
    静态图形报表 / Static PNG report (3x2 subplots).

    子图布局:
        [1] 预测 vs 实际时序 (含 95% 区间阴影)
        [2] 残差序列
        [3] 指标趋势 (前 3 个最常见的 metric_name)
        [4] 告警时间线 (按级别着色)
        [5] 告警级别直方图
        [6] 模型预测分布直方图
    """

    def generate(
        self,
        *,
        model_id: str,
        store: MetricStore,
        out_path: Optional[str] = None,
        days: int = 90,
        **_: Any,
    ) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 中文字体 (若可用)
        try:
            from ..visualization.base_visualizer import _CN_FONTS
            plt.rcParams['font.sans-serif'] = _CN_FONTS
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass

        start, end = _default_window(days)
        preds = store.query_predictions(model_id=model_id,
                                        start=start, end=end)
        metrics = store.query_metrics(model_id=model_id,
                                      start=start, end=end)
        alerts = store.query_alerts(model_id=model_id,
                                    start=start, end=end)

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(
            f'Monitoring Report — {model_id}  '
            f'({start:%Y-%m-%d}~{end:%Y-%m-%d})',
            fontsize=14, fontweight='bold',
        )

        # [1] 预测 vs 实际
        ax = axes[0, 0]
        if preds:
            xs = [p['timestamp'] for p in preds]
            yp = [p['y_pred'] for p in preds]
            ya = [p.get('y_actual') for p in preds]
            lo = [p.get('y_lower') for p in preds]
            hi = [p.get('y_upper') for p in preds]
            ax.plot(xs, yp, '-', color='tab:blue', label='预测', linewidth=1.2)
            if all(v is not None for v in lo + hi):
                ax.fill_between(xs, lo, hi, color='tab:blue',
                                alpha=0.15, label='区间')
            if any(v is not None for v in ya):
                xs2 = [x for x, v in zip(xs, ya) if v is not None]
                ys2 = [v for v in ya if v is not None]
                ax.plot(xs2, ys2, 'o-', color='tab:orange',
                        markersize=3, label='实际')
            ax.legend(loc='best', fontsize=8)
        ax.set_title('[1] 预测 vs 实际', fontsize=10)
        ax.tick_params(axis='x', rotation=30, labelsize=7)

        # [2] 残差
        ax = axes[0, 1]
        resid = [(p['timestamp'], p['y_actual'] - p['y_pred'])
                 for p in preds if p.get('y_actual') is not None]
        if resid:
            xs, rs = zip(*resid)
            ax.plot(xs, rs, 'o-', color='tab:red', markersize=2, linewidth=0.8)
            ax.axhline(0, linestyle='--', color='gray', linewidth=0.8)
        ax.set_title('[2] 残差 (actual - pred)', fontsize=10)
        ax.tick_params(axis='x', rotation=30, labelsize=7)

        # [3] 指标趋势
        ax = axes[1, 0]
        by_name: Dict[str, List[Tuple[datetime, float]]] = {}
        for m in metrics:
            by_name.setdefault(m['name'], []).append(
                (m['timestamp'], m['value']))
        top_names = sorted(by_name, key=lambda k: -len(by_name[k]))[:3]
        for name in top_names:
            pts = by_name[name]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, 'o-', label=name, markersize=3, linewidth=1.0)
        if top_names:
            ax.legend(loc='best', fontsize=8)
        ax.set_title('[3] 指标趋势 (Top 3)', fontsize=10)
        ax.tick_params(axis='x', rotation=30, labelsize=7)

        # [4] 告警时间线
        ax = axes[1, 1]
        color_map = {
            AlertLevel.INFO: 'tab:gray',
            AlertLevel.WARNING: 'tab:orange',
            AlertLevel.ERROR: 'tab:red',
            AlertLevel.CRITICAL: 'darkred',
        }
        if alerts:
            for a in alerts:
                ax.scatter(a.timestamp,
                           LEVEL_ORDER.get(a.level, 0),
                           color=color_map.get(a.level, 'tab:gray'),
                           s=25, alpha=0.7)
            ax.set_yticks(list(LEVEL_ORDER.values()))
            ax.set_yticklabels(list(LEVEL_ORDER.keys()))
        ax.set_title('[4] 告警时间线', fontsize=10)
        ax.tick_params(axis='x', rotation=30, labelsize=7)

        # [5] 告警级别直方图
        ax = axes[2, 0]
        dist = {lv: 0 for lv in LEVEL_ORDER}
        for a in alerts:
            dist[a.level] = dist.get(a.level, 0) + 1
        names = list(dist.keys())
        values = [dist[n] for n in names]
        colors = [color_map.get(n, 'gray') for n in names]
        ax.bar(names, values, color=colors)
        for i, v in enumerate(values):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=8)
        ax.set_title('[5] 告警分级计数', fontsize=10)

        # [6] 预测分布
        ax = axes[2, 1]
        if preds:
            ax.hist([p['y_pred'] for p in preds], bins=30,
                    color='tab:blue', alpha=0.6, edgecolor='white')
        ax.set_title('[6] 预测值分布', fontsize=10)

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if out_path is None:
            out_dir = os.path.abspath('./logs/reports')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir,
                f'report_{model_id}_{datetime.now():%Y%m%d_%H%M%S}.png',
            )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_path


# ==========================================================================
# main — 生成一张 sample report
# ==========================================================================

def main() -> None:
    """用 InMemoryStore 编造几条记录, 生成两份报表."""
    import numpy as np
    from .stores import InMemoryStore

    print('=' * 70)
    print(' reporters — text + plot demo')
    print('=' * 70)

    store = InMemoryStore()
    rng = np.random.default_rng(7)
    now = datetime.now()
    for i in range(60):
        ts = now - timedelta(days=60 - i)
        yp = 100 + i + rng.normal(0, 3)
        ya = 100 + i + rng.normal(0, 4)
        store.insert_prediction(
            model_id='demo', timestamp=ts, target='y',
            y_pred=yp, y_lower=yp - 6, y_upper=yp + 6, y_actual=ya,
        )
        if i % 5 == 0:
            store.insert_metrics_snapshot(
                model_id='demo', timestamp=ts,
                metrics={'mape': 0.03 + i * 0.001,
                         'mae': 2.0 + i * 0.02},
                window=20,
            )

    # 告警几条
    for i, lv in enumerate([AlertLevel.INFO, AlertLevel.WARNING,
                            AlertLevel.ERROR, AlertLevel.CRITICAL]):
        store.insert_alert(Alert(
            alert_id=f'demo_{i}',
            model_id='demo',
            level=lv,
            message=f'sample alert #{i}',
            timestamp=now - timedelta(days=10 - i),
            source='demo',
        ))

    text_path = TextReport().generate(model_id='demo', store=store)
    print(f'\n文本报表:  {text_path}')

    try:
        plot_path = PlotReport().generate(model_id='demo', store=store)
        print(f'图形报表:  {plot_path}')
    except Exception as exc:
        print(f'图形报表生成跳过: {exc}')


if __name__ == '__main__':
    main()
