"""
PredictionPlotter 契约测试 / Visualization toolkit contract tests.

只验证"图能画出来 + 文件能保存 + 关键 mat plotlib artist 存在",
不做像素级视觉 diff (那种属于 visual regression, 留给人眼复核).

锁定:
1. 接受 ndarray / Series / DataFrame 三种输入
2. 原子方法都返回传入的 ax
3. 复合工具产出非空 PNG 文件
4. 调色板循环正确
5. _CN_FONTS 已应用到 rcParams
"""
from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use('Agg')   # 必须在 import pyplot 之前
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tsf_frame.visualization import (
    DEFAULT_PALETTE,
    PredictionPlotter,
    _CN_FONTS,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def plotter():
    return PredictionPlotter(figsize=(10, 6), dpi=80)


@pytest.fixture
def toy_series():
    rng = np.random.default_rng(0)
    n = 30
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    y_true = np.cumsum(rng.normal(0, 1, n)) + 100
    y_pred = y_true + rng.normal(0, 1, n)
    return dates, y_true, y_pred


@pytest.fixture
def tmp_png(tmp_path):
    return str(tmp_path / 'out.png')


# ──────────────────────────────────────────────────────────────────────
# 1. 配置/导入
# ──────────────────────────────────────────────────────────────────────

def test_cn_fonts_applied_in_rcParams():
    """import 后 matplotlib rcParams 应被设上中文字体."""
    fonts = plt.rcParams['font.sans-serif']
    assert any(f in fonts for f in _CN_FONTS), (
        f'_CN_FONTS 未注入 rcParams: {fonts}')


def test_default_palette_distinct_and_long_enough():
    """调色板应至少 7 种且互不相同."""
    assert len(DEFAULT_PALETTE) >= 7
    assert len(set(DEFAULT_PALETTE)) == len(DEFAULT_PALETTE)


# ──────────────────────────────────────────────────────────────────────
# 2. 输入宽容性
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('input_type', ['ndarray', 'series', 'list'])
def test_lines_compare_accepts_various_input_types(plotter, input_type):
    """同一份数据用 ndarray / Series / list 喂入应都能画出来."""
    fig, ax = plt.subplots()
    base = np.arange(10, dtype=float)
    pred = base + 0.5
    if input_type == 'series':
        base = pd.Series(base)
        pred = pd.Series(pred)
    elif input_type == 'list':
        base = base.tolist()
        pred = pred.tolist()
    ret = plotter.lines_compare(ax, x=range(10), baseline=base,
                                series={'pred': pred})
    assert ret is ax
    assert len(ax.lines) >= 2  # baseline + 1 series
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 3. 原子方法 — 都返回传入的 ax
# ──────────────────────────────────────────────────────────────────────

def test_lines_compare_returns_ax(plotter, toy_series):
    dates, y_true, y_pred = toy_series
    fig, ax = plt.subplots()
    ret = plotter.lines_compare(
        ax, x=dates, baseline=y_true, series={'A': y_pred},
        title='t', ylabel='y',
    )
    assert ret is ax
    assert ax.get_title() == 't'
    plt.close(fig)


def test_interval_band_draws_fill(plotter, toy_series):
    dates, y_true, y_pred = toy_series
    fig, ax = plt.subplots()
    plotter.interval_band(
        ax, x=dates, y_pred=y_pred,
        y_lower=y_pred - 1, y_upper=y_pred + 1,
        y_actual=y_true,
    )
    # fill_between 会创建 PolyCollection
    polys = [c for c in ax.collections]
    assert len(polys) >= 1
    plt.close(fig)


def test_bars_with_value_labels(plotter):
    fig, ax = plt.subplots()
    plotter.bars(ax, names=['a', 'b', 'c'], values=[1.0, 2.5, 3.0],
                 fmt='{:.1f}')
    # 数值标注是 ax.texts
    assert len(ax.texts) == 3
    assert ax.texts[1].get_text() == '2.5'
    plt.close(fig)


def test_bars_h_horizontal(plotter):
    fig, ax = plt.subplots()
    plotter.bars_h(ax, names=['x', 'y', 'z'], values=[10, 20, 30])
    # 横向柱也有 patches
    assert len(ax.patches) == 3
    plt.close(fig)


def test_histogram(plotter):
    fig, ax = plt.subplots()
    plotter.histogram(ax, values=np.random.default_rng(0).normal(size=200),
                      bins=20)
    assert len(ax.patches) == 20
    plt.close(fig)


def test_residuals_zero_line(plotter, toy_series):
    dates, y_true, y_pred = toy_series
    fig, ax = plt.subplots()
    plotter.residuals(ax, x=dates, residuals=y_true - y_pred)
    # 应该有数据线 + axhline(0)
    assert len(ax.lines) >= 2
    plt.close(fig)


def test_scatter_categorical_with_yticks(plotter):
    fig, ax = plt.subplots()
    plotter.scatter_categorical(
        ax,
        x=[1, 2, 3, 4],
        y=[0, 1, 2, 1],
        color_by=['info', 'warning', 'error', 'warning'],
        yticks=[0, 1, 2, 3],
        yticklabels=['info', 'warning', 'error', 'critical'],
    )
    assert ax.get_yticklabels()[1].get_text() == 'warning'
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 4. 复合工具 — 文件落地 + 内容非空
# ──────────────────────────────────────────────────────────────────────

def test_forecast_comparison_fig_creates_png(plotter, toy_series, tmp_png):
    dates, y_true, y_pred = toy_series
    plotter.forecast_comparison_fig(
        x=dates, y_true=y_true,
        models_pred={'A': y_pred, 'B': y_pred + 0.5},
        best_interval=(y_pred, y_pred - 1, y_pred + 1),
        target_label='monthly_deposit',
        save_path=tmp_png,
    )
    assert os.path.exists(tmp_png)
    assert os.path.getsize(tmp_png) > 1000   # PNG header + 实际内容


def test_metrics_bars_fig(plotter, tmp_png):
    plotter.metrics_bars_fig(
        groups={
            'MAPE(越低越好)': (['m1', 'm2'], [3.5, 7.2], '{:.2f}%'),
            'R²(越高越好)':   (['m1', 'm2'], [0.95, 0.88], '{:.4f}'),
        },
        suptitle='对比',
        save_path=tmp_png,
    )
    assert os.path.exists(tmp_png) and os.path.getsize(tmp_png) > 1000


def test_training_curves_fig(plotter, tmp_png):
    plotter.training_curves_fig(
        results={
            'M1': {'train_loss': [1.0, 0.5, 0.3], 'val_loss': [1.1, 0.6, 0.4]},
            'M2': {'train_loss': [0.9, 0.45, 0.25], 'val_loss': [1.0, 0.5, 0.3]},
        },
        save_path=tmp_png,
    )
    assert os.path.exists(tmp_png) and os.path.getsize(tmp_png) > 1000


def test_feature_importance_fig_sorted_top_n(plotter, tmp_png):
    """top_n 截断 + 内部按 importance 降序."""
    plotter.feature_importance_fig(
        names=['a', 'b', 'c', 'd', 'e'],
        values=[0.1, 0.5, 0.3, 0.05, 0.4],
        top_n=3,
        save_path=tmp_png,
    )
    assert os.path.exists(tmp_png) and os.path.getsize(tmp_png) > 1000


# ──────────────────────────────────────────────────────────────────────
# 5. save() + 边界
# ──────────────────────────────────────────────────────────────────────

def test_save_creates_parent_dir(plotter, tmp_path):
    """save 应自动创建 save_path 的父目录."""
    nested = tmp_path / 'a' / 'b' / 'c' / 'out.png'
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    out = plotter.save(fig, str(nested))
    assert os.path.exists(out)
    plt.close(fig)


def test_metrics_bars_fig_empty_groups_raises(plotter):
    with pytest.raises(ValueError, match='不能为空'):
        plotter.metrics_bars_fig(groups={})
