"""
统一绘图工具 / Unified prediction plotter
=========================================

这是 TSF_Frame 项目所有图表的"标准库"。设计目标:

* **原子 + 复合两层** — 7 个原子方法画到任意 ``ax`` 上, 4 个复合工具
  自建 ``figure`` 并保存; 调用者既可以快速出图, 也可以自己拼复杂网格.
* **输入宽容** — 接受 ``np.ndarray`` / ``pd.Series`` / ``list``, 不强制
  ``DataFrame``.
* **风格统一** — 共用调色板 / 字体 / 网格 / 数值标注样式; 全项目 PNG 一致.
* **零 plotly 依赖** — 纯 ``matplotlib``, headless / CI 都能跑.
* **路径可控** — ``save_path`` 由调用方决定, 不再硬编码文件名.

七个原子方法 (都接 ``ax=``):

    lines_compare      多 series 折线对比 + 可选 baseline (黑实线)
    interval_band      预测均值 + 概率区间 (fill_between) + 可选 actual
    bars               垂直柱状图, 自动数值标注
    bars_h             水平柱状图 (适合特征重要性)
    histogram          直方图
    scatter_categorical 类别散点 (告警时间线场景)
    residuals          残差时序 (零参考线 + 散点连线)

四个复合工具 (自建 fig, 返回 ``Figure`` 或保存路径):

    forecast_comparison_fig  2x1: 多模型对比 + 最优模型概率区间
    metrics_bars_fig         1xN: 多组指标柱状对比
    training_curves_fig      1x2: train/val loss 曲线 (DL 场景)
    feature_importance_fig   1x1: 特征重要性 Top-N 横向柱状

使用示例::

    from tsf_frame.visualization import PredictionPlotter

    plotter = PredictionPlotter(figsize=(14, 10), dpi=120)

    # 复合工具一行出图
    plotter.forecast_comparison_fig(
        x=dates, y_true=y_test,
        models_pred={'XGBoost': pred1, 'LightGBM': pred2},
        best_interval=(pred1, lo, hi),
        target_label='monthly_deposit',
        save_path='./logs/outputs/hpf/forecast.png',
    )

    # 原子方法自由组合
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plotter.lines_compare(axes[0, 0], x=dates, baseline=y_true,
                          series={'pred': y_pred}, title='预测对比')
    plotter.bars(axes[0, 1], names=models, values=mapes, fmt='{:.2f}%',
                 title='MAPE')
    ...
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base_visualizer import _CN_FONTS  # 保留以触发 rcParams 副作用

__all__ = ['PredictionPlotter', 'DEFAULT_PALETTE']


#: 全项目统一调色板 (按"先用最显眼的"顺序)
DEFAULT_PALETTE: List[str] = [
    '#E74C3C',  # 红
    '#3498DB',  # 蓝
    '#2ECC71',  # 绿
    '#F39C12',  # 橙
    '#9B59B6',  # 紫
    '#1ABC9C',  # 青
    '#E67E22',  # 橘
    '#34495E',  # 深灰蓝
]


def _to_array(x: Any) -> np.ndarray:
    """把任意一维输入(list / ndarray / Series)转成 1D ndarray."""
    if isinstance(x, pd.Series):
        return x.to_numpy()
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                f'DataFrame 应只有一列, 实际 {x.shape[1]}; 想多 series 请用 dict')
        return x.iloc[:, 0].to_numpy()
    return np.asarray(x).ravel()


class PredictionPlotter:
    """
    时序预测画图工具集 / Unified plotter for forecasting projects.

    Args:
        figsize:        默认画布尺寸 (复合工具用)
        dpi:            默认 DPI
        palette:        颜色循环列表; None 用 DEFAULT_PALETTE
        apply_cn_fonts: True 时确保 matplotlib 中文字体已激活
                        (实际激活逻辑在 base_visualizer.py 顶层副作用,
                        此处只是显式校验/兜底)
        show:           方法默认是否 plt.show(); CI / headless 应保持 False

    Attributes:
        figsize:        默认画布尺寸 (h, w)
        dpi:            默认 DPI
        palette:        颜色循环, lines_compare / bars / etc 都用它
        show:           是否在保存后还 plt.show()
    """

    def __init__(
        self,
        *,
        figsize: Tuple[float, float] = (12, 8),
        dpi: int = 120,
        palette: Optional[Sequence[str]] = None,
        apply_cn_fonts: bool = True,
        show: bool = False,
    ):
        # 默认画布尺寸 (复合工具调用 plt.subplots 时用)
        # / Default figure size for composite plotters
        self.figsize: Tuple[float, float] = tuple(figsize)
        # 输出 DPI; 推荐 120~150 (清晰但不太大)
        # / Output DPI
        self.dpi: int = int(dpi)
        # 调色板; 多 series 时按 i % len(palette) 循环取色
        # / Color cycle list
        self.palette: List[str] = (
            list(palette) if palette is not None else list(DEFAULT_PALETTE)
        )
        # 是否在 save 后再 plt.show; 通常 CI / 批量产图设 False
        # / Whether to call plt.show after save
        self.show: bool = bool(apply_cn_fonts and show)  # show 默认就关
        # 中文字体激活状态 (导入 base_visualizer 时已经写入 rcParams)
        # / CN fonts activated flag (informational)
        self.apply_cn_fonts: bool = apply_cn_fonts
        if apply_cn_fonts:
            plt.rcParams['font.sans-serif'] = _CN_FONTS
            plt.rcParams['axes.unicode_minus'] = False

    # ==================================================================
    # 原子方法 (画到给定 ax)
    # ==================================================================

    def lines_compare(
        self,
        ax: plt.Axes,
        *,
        x: Any,
        baseline: Optional[Any] = None,
        series: Mapping[str, Any],
        baseline_label: str = '实际值',
        baseline_color: str = 'black',
        baseline_linewidth: float = 2.0,
        series_linestyle: str = '--',
        series_alpha: float = 0.85,
        ylabel: str = '',
        title: str = '',
        legend: bool = True,
        grid: bool = True,
    ) -> plt.Axes:
        """
        多 series 折线对比 / Compare multiple series with optional baseline.

        Args:
            ax:        matplotlib Axes (如果是 None 请用复合工具)
            x:         共同的 x 轴 (dates / index / range)
            baseline:  基准序列 (如真实值), 黑色实线粗画; None 不画
            series:    {label: y_series} 多条对比线, 按 palette 循环上色
            其余参数:    样式控制

        Returns:
            ax (便于链式调用)
        """
        if baseline is not None:
            ax.plot(
                x, _to_array(baseline),
                color=baseline_color, linewidth=baseline_linewidth,
                label=baseline_label, zorder=5,
            )
        for i, (label, y) in enumerate(series.items()):
            ax.plot(
                x, _to_array(y),
                linestyle=series_linestyle,
                color=self.palette[i % len(self.palette)],
                alpha=series_alpha,
                label=label,
            )
        if title:
            ax.set_title(title, fontsize=13)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
        if grid:
            ax.grid(True, alpha=0.3)
        return ax

    def interval_band(
        self,
        ax: plt.Axes,
        *,
        x: Any,
        y_pred: Any,
        y_lower: Any,
        y_upper: Any,
        y_actual: Optional[Any] = None,
        actual_label: str = '实际值',
        actual_color: str = 'black',
        mean_label: str = '预测均值',
        mean_color: Optional[str] = None,
        band_label: str = '95% 置信区间',
        band_alpha: float = 0.25,
        ylabel: str = '',
        title: str = '',
        legend: bool = True,
        grid: bool = True,
    ) -> plt.Axes:
        """
        概率预测区间图 / Mean prediction with uncertainty band.

        Args:
            ax:         画到这个 Axes
            x:          x 轴
            y_pred:     预测均值
            y_lower:    区间下界
            y_upper:    区间上界
            y_actual:   可选, 实际值 (黑实线对比)
            其余参数:     样式控制
        """
        mean_color = mean_color or self.palette[0]

        if y_actual is not None:
            ax.plot(x, _to_array(y_actual), color=actual_color,
                    linewidth=2.0, label=actual_label)
        ax.plot(x, _to_array(y_pred), linestyle='--',
                color=mean_color, linewidth=1.5, label=mean_label)
        ax.fill_between(
            x, _to_array(y_lower), _to_array(y_upper),
            color=mean_color, alpha=band_alpha, label=band_label,
        )
        if title:
            ax.set_title(title, fontsize=13)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
        if grid:
            ax.grid(True, alpha=0.3)
        return ax

    def bars(
        self,
        ax: plt.Axes,
        *,
        names: Sequence[str],
        values: Sequence[float],
        fmt: str = '{:.2f}',
        color: Optional[Union[str, Sequence[str]]] = None,
        alpha: float = 0.85,
        text_fontsize: int = 10,
        text_offset_factor: float = 0.02,
        title: str = '',
        ylabel: str = '',
        grid_y: bool = True,
    ) -> plt.Axes:
        """
        垂直柱状图 + 数值标注 / Vertical bars with auto value labels.

        Args:
            ax:         画到这个 Axes
            names:      x 轴标签
            values:     柱高
            fmt:        数值显示格式 (例 '{:.2f}%' 加百分号)
            color:      单色字符串(全部同色) 或 颜色列表; None 用 palette
            text_offset_factor: 数值上方偏移比例 (相对最大值)
        """
        n = len(names)
        if color is None:
            colors = [self.palette[i % len(self.palette)] for i in range(n)]
        elif isinstance(color, str):
            colors = [color] * n
        else:
            colors = list(color)
        bars = ax.bar(list(names), list(values), color=colors, alpha=alpha)
        if values:
            offset = max(abs(v) for v in values) * text_offset_factor or 0.01
        else:
            offset = 0.01
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + offset,
                fmt.format(v),
                ha='center', va='bottom', fontsize=text_fontsize,
            )
        if title:
            ax.set_title(title, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel)
        if grid_y:
            ax.grid(True, alpha=0.3, axis='y')
        return ax

    def bars_h(
        self,
        ax: plt.Axes,
        *,
        names: Sequence[str],
        values: Sequence[float],
        fmt: Optional[str] = None,
        color: Optional[Union[str, Sequence[str]]] = None,
        alpha: float = 0.85,
        title: str = '',
        xlabel: str = '',
        grid_x: bool = True,
    ) -> plt.Axes:
        """
        水平柱状图 (Top-N 特征重要性常用) / Horizontal bars.

        names / values 顺序就是从下到上的显示顺序; 调用方需自行按重要性排序。
        """
        n = len(names)
        if color is None:
            colors = [self.palette[1 % len(self.palette)]] * n  # 默认蓝
        elif isinstance(color, str):
            colors = [color] * n
        else:
            colors = list(color)
        ax.barh(list(names), list(values), color=colors, alpha=alpha)
        if fmt:
            for i, v in enumerate(values):
                ax.text(v, i, fmt.format(v), va='center', fontsize=9)
        if title:
            ax.set_title(title, fontsize=12)
        if xlabel:
            ax.set_xlabel(xlabel)
        if grid_x:
            ax.grid(True, alpha=0.3, axis='x')
        return ax

    def histogram(
        self,
        ax: plt.Axes,
        *,
        values: Any,
        bins: int = 30,
        color: Optional[str] = None,
        alpha: float = 0.85,
        title: str = '',
        xlabel: str = '',
        grid: bool = True,
    ) -> plt.Axes:
        """直方图 / Plain histogram."""
        ax.hist(_to_array(values), bins=bins,
                color=color or self.palette[1], alpha=alpha)
        if title:
            ax.set_title(title, fontsize=10)
        if xlabel:
            ax.set_xlabel(xlabel)
        if grid:
            ax.grid(True, alpha=0.3)
        return ax

    def scatter_categorical(
        self,
        ax: plt.Axes,
        *,
        x: Sequence[Any],
        y: Sequence[Any],
        color_by: Optional[Sequence[Any]] = None,
        yticks: Optional[Sequence[Any]] = None,
        yticklabels: Optional[Sequence[str]] = None,
        title: str = '',
        x_rotation: int = 30,
        marker_size: int = 30,
    ) -> plt.Axes:
        """
        类别散点 (告警时间线常用) / Scatter on a categorical y-axis.

        Args:
            x:           时间戳序列
            y:           类别值序列 (会被 yticks/yticklabels 替换显示)
            color_by:    可选, 与 x 等长的着色键 (按值映射 palette)
            yticks:      y 轴刻度位置
            yticklabels: y 轴刻度文字
        """
        if color_by is not None:
            # 把每个唯一值映射到 palette 一种颜色
            uniq = list(dict.fromkeys(color_by))
            color_map = {v: self.palette[i % len(self.palette)]
                         for i, v in enumerate(uniq)}
            colors = [color_map[v] for v in color_by]
            ax.scatter(list(x), list(y), c=colors, s=marker_size)
        else:
            ax.scatter(list(x), list(y), s=marker_size)
        if yticks is not None:
            ax.set_yticks(list(yticks))
            if yticklabels is not None:
                ax.set_yticklabels(list(yticklabels))
        if title:
            ax.set_title(title, fontsize=10)
        if x_rotation:
            ax.tick_params(axis='x', rotation=x_rotation, labelsize=7)
        return ax

    def residuals(
        self,
        ax: plt.Axes,
        *,
        x: Any,
        residuals: Any,
        color: Optional[str] = None,
        markersize: float = 2,
        linewidth: float = 0.8,
        zero_line: bool = True,
        title: str = '',
        x_rotation: int = 30,
        grid: bool = True,
    ) -> plt.Axes:
        """残差时序图 (默认带零参考线) / Residual time-series with zero ref line."""
        ax.plot(
            x, _to_array(residuals), 'o-',
            color=color or self.palette[0],
            markersize=markersize, linewidth=linewidth,
        )
        if zero_line:
            ax.axhline(0, linestyle='--', color='gray', linewidth=0.8)
        if title:
            ax.set_title(title, fontsize=10)
        if x_rotation:
            ax.tick_params(axis='x', rotation=x_rotation, labelsize=7)
        if grid:
            ax.grid(True, alpha=0.3)
        return ax

    # ==================================================================
    # 复合工具 (自建 figure, 一行出整张图)
    # ==================================================================

    def forecast_comparison_fig(
        self,
        *,
        x: Any,
        y_true: Any,
        models_pred: Mapping[str, Any],
        best_interval: Optional[Tuple[Any, Any, Any]] = None,
        target_label: str = '',
        ylabel: str = '',
        suptitle: Optional[str] = None,
        save_path: Optional[str] = None,
        close: bool = True,
    ) -> plt.Figure:
        """
        预测对比图 (2x1) / Forecast comparison: top=multi-model, bottom=best+CI.

        Args:
            x:             共同 x 轴 (dates)
            y_true:        实际值 (黑实线)
            models_pred:   {model_name: y_pred} 多模型预测
            best_interval: (y_pred_best, y_lower, y_upper); None 时下半图省略
            target_label:  目标列名 (用于上半图标题)
            ylabel:        y 轴单位
            suptitle:      整张图大标题; None 自动从 target_label 拼
            save_path:     保存路径; None 不保存
            close:         保存后关闭 figure (大批量产图时建议 True)
        """
        nrows = 2 if best_interval is not None else 1
        fig, axes = plt.subplots(
            nrows, 1, figsize=(self.figsize[0], 5 * nrows),
        )
        axes = np.atleast_1d(axes)

        # 上图: 多模型对比
        self.lines_compare(
            axes[0],
            x=x, baseline=y_true, series=dict(models_pred),
            ylabel=ylabel,
            title=(f'公积金 {target_label} 预测对比' if target_label
                   else '预测对比'),
        )

        # 下图: 最优模型 + 区间
        if best_interval is not None:
            best_pred, lo, hi = best_interval
            best_name = (
                next(iter(models_pred)) if not models_pred
                else min(
                    models_pred,
                    key=lambda k: float(np.mean(np.abs(
                        _to_array(models_pred[k]) - _to_array(y_true)
                    ))),
                )
            )
            self.interval_band(
                axes[1],
                x=x, y_pred=best_pred, y_lower=lo, y_upper=hi,
                y_actual=y_true,
                mean_label=f'{best_name} 预测均值',
                ylabel=ylabel,
                title=f'最优模型({best_name})概率预测',
            )

        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout()
        if save_path:
            self.save(fig, save_path)
            if close:
                plt.close(fig)
        elif self.show:
            plt.show()
        return fig

    def metrics_bars_fig(
        self,
        *,
        groups: Mapping[str, Tuple[Sequence[str], Sequence[float], str]],
        suptitle: str = '',
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        close: bool = True,
    ) -> plt.Figure:
        """
        多组柱状指标对比 / Multiple bar charts side by side.

        Args:
            groups:   {subplot_title: (names, values, fmt)}
                      e.g. {'MAPE(越低越好)': (models, mape_vals, '{:.2f}%'),
                            'R²(越高越好)':   (models, r2_vals,  '{:.4f}')}
            suptitle: 总标题
            figsize:  覆盖默认 figsize
        """
        n = len(groups)
        if n == 0:
            raise ValueError('groups 不能为空')
        fig, axes = plt.subplots(
            1, n, figsize=figsize or (6 * n, 5),
        )
        axes = np.atleast_1d(axes)
        for ax, (title, (names, values, fmt)) in zip(axes, groups.items()):
            self.bars(ax, names=names, values=values, fmt=fmt, title=title)
        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout()
        if save_path:
            self.save(fig, save_path)
            if close:
                plt.close(fig)
        elif self.show:
            plt.show()
        return fig

    def training_curves_fig(
        self,
        *,
        results: Mapping[str, Mapping[str, Sequence[float]]],
        train_key: str = 'train_loss',
        val_key: str = 'val_loss',
        log_scale: bool = True,
        ylabel: str = 'Loss',
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        close: bool = True,
    ) -> plt.Figure:
        """
        DL 训练曲线 (1x2) / Train + Val loss curves for multiple models.

        Args:
            results:   {model_name: {train_key: [...], val_key: [...]}}
                       通常是 ``{name: {'history': {...}}}`` 的 history 字段
            log_scale: y 轴是否对数
        """
        fig, axes = plt.subplots(
            1, 2, figsize=figsize or (14, 5),
        )

        for i, (name, hist) in enumerate(results.items()):
            color = self.palette[i % len(self.palette)]
            tr = hist.get(train_key)
            va = hist.get(val_key)
            if tr:
                axes[0].plot(tr, color=color, label=name)
            if va:
                axes[1].plot(va, color=color, label=name)

        for ax, title in zip(axes, ['训练 Loss 曲线', '验证 Loss 曲线']):
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            if log_scale:
                ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if save_path:
            self.save(fig, save_path)
            if close:
                plt.close(fig)
        elif self.show:
            plt.show()
        return fig

    def feature_importance_fig(
        self,
        *,
        names: Sequence[str],
        values: Sequence[float],
        top_n: Optional[int] = 10,
        title: str = '特征重要性',
        xlabel: str = 'Importance',
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        close: bool = True,
    ) -> plt.Figure:
        """
        特征重要性 Top-N / Feature importance bar chart.

        names / values 不要求预排序; 函数内部按 values 降序取 top_n,
        然后倒序画(最高的在最上)。
        """
        order = np.argsort(values)[::-1]
        if top_n is not None:
            order = order[:top_n]
        names_sorted = [names[i] for i in order][::-1]
        values_sorted = [values[i] for i in order][::-1]

        fig, ax = plt.subplots(figsize=figsize or (10, 6))
        self.bars_h(
            ax, names=names_sorted, values=values_sorted,
            title=title, xlabel=xlabel,
        )
        fig.tight_layout()
        if save_path:
            self.save(fig, save_path)
            if close:
                plt.close(fig)
        elif self.show:
            plt.show()
        return fig

    # ==================================================================
    # I/O
    # ==================================================================

    def save(
        self,
        fig: plt.Figure,
        save_path: str,
        *,
        dpi: Optional[int] = None,
        bbox_inches: str = 'tight',
    ) -> str:
        """
        保存 figure / Save figure with auto-mkdir.

        Returns:
            实际保存的绝对路径
        """
        out = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        fig.savefig(out, dpi=dpi or self.dpi, bbox_inches=bbox_inches)
        return out


# ==========================================================================
# main — 端到端 demo (生成一张组合图)
# ==========================================================================

def main() -> None:  # pragma: no cover
    """演示原子方法 + 复合工具一起用。"""
    import tempfile
    rng = np.random.default_rng(0)
    n = 60
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    y_true = np.cumsum(rng.normal(0, 1, n)) + 100
    pred1 = y_true + rng.normal(0, 1, n)
    pred2 = y_true + rng.normal(0, 1.5, n) + 0.5
    lo = pred1 - 2.5
    hi = pred1 + 2.5

    plotter = PredictionPlotter(figsize=(14, 8), dpi=120)

    out = os.path.join(tempfile.gettempdir(), 'plotter_demo.png')
    plotter.forecast_comparison_fig(
        x=dates, y_true=y_true,
        models_pred={'ModelA': pred1, 'ModelB': pred2},
        best_interval=(pred1, lo, hi),
        target_label='monthly_deposit', ylabel='亿元',
        suptitle='Demo: forecast comparison',
        save_path=out,
    )
    print(f'Demo saved → {out}')


if __name__ == '__main__':
    main()
