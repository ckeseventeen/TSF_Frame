"""
目标变换 / Target transformations
==================================

把"绝对值预测"转化为更容易学习的"相对量预测", 解决树模型/Transformer/LSTM
**无法外推训练分布**的根本问题. 训练侧把 y 变换到稳定值域学习, 推理侧
再 inverse_transform 还原到业务量纲.

设计哲学 / Design philosophy:
  * 无状态优先: 多数变换不需要 fit 状态 (Identity/Diff/Log/Log1p);
    Box-Cox 等需要 lambda 的高级变换可在 P1 加上.
  * **对齐对调用方透明**: ``transform(y, X)`` 在差分场景同时返回对齐后的 X
    (差分会让 y 少 1 行, X 需要 [1:] 切片), 避免上层手动 bookkeeping.
  * **anchor 显式传入**: 差分还原需要"测试集开始前最后一个真实水平值",
    这是业务侧才知道的 (例如 ``y_val[-1]``), 不能 transform 时缓存
    (因为推理时点和训练时点之间可能隔了任意时长).
  * 工厂函数 ``get_target_transform(name_or_obj)`` 让 config 字段可以是字符串.

可用变换 / Available:
  Identity   : 不做任何变换 (默认)
  Diff       : 一阶差分 y_diff[i] = y[i] - y[i-1], 推理累加还原
               → 适合: 长趋势数据 (HPF deposit/loan_balance), ARIMA(.,1,.)
  Log        : 自然对数 (要求 y > 0)
               → 适合: 乘性增长数据, log-normal 分布
  Log1p      : log(1 + y) (要求 y >= 0, 容忍 0)
               → 适合: 计数数据 / 含 0 的正值数据

  组合需求请用 ``ComposeTransform([Log(), Diff()])`` (按序应用).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List

import numpy as np

__all__ = [
    'TargetTransform',
    'IdentityTransform',
    'DiffTransform',
    'LogTransform',
    'Log1pTransform',
    'ComposeTransform',
    'get_target_transform',
]


class TargetTransform(ABC):
    """目标变换抽象基类 / Abstract base class for target transformations.

    生命周期 / Lifecycle:
        1. ``fit(y_train)``    (可选, 无状态变换可跳过)
        2. ``transform(y, X)`` (训练侧, 同时对齐 X)
        3. ``inverse_transform(y_pred, anchor=...)`` (推理侧, 还原到原尺度)

    所有子类都必须实现 ``transform`` 和 ``inverse_transform``.
    """

    #: 推理时是否需要 anchor 值 (差分类变换需要)
    #: / Whether inverse_transform requires an anchor argument
    needs_anchor: bool = False

    #: 变换后 y 的长度变化 (-1 表示比输入少 1 行, 0 表示等长)
    #: 用于 X 对齐 bookkeeping
    #: / Length delta after transform (-1 means output is 1 row shorter)
    length_delta: int = 0

    def fit(self, y_train: np.ndarray) -> 'TargetTransform':
        """子类按需覆盖. 默认无状态."""
        return self

    @abstractmethod
    def transform(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """对 y 做变换, 同时返回对齐后的 X (若 X 为 None 则返回 None)."""
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(
        self,
        y_t: np.ndarray,
        anchor: Optional[float] = None,
    ) -> np.ndarray:
        """从变换空间还原到原 y 空间."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────
# Identity
# ──────────────────────────────────────────────────────────────────────────

class IdentityTransform(TargetTransform):
    """不做任何变换 / No-op transform (default)."""

    needs_anchor = False
    length_delta = 0

    def transform(self, y, X=None):
        return np.asarray(y), X

    def inverse_transform(self, y_t, anchor=None):
        return np.asarray(y_t)


# ──────────────────────────────────────────────────────────────────────────
# Diff (差分)
# ──────────────────────────────────────────────────────────────────────────

class DiffTransform(TargetTransform):
    """
    一阶差分 / First-order difference.

    transform:
        y_diff[i] = y[i+1] - y[i]            (numpy.diff)
        X 同步切片: X_aligned = X[1:]         (与 y[i+1] 对齐)
        输出长度 = len(y) - 1

    inverse_transform (anchor 必传):
        y_pred[t] = anchor + Σ_{k=0..t} y_diff_pred[k]

    业务含义 / Business meaning:
        模型学的是"相对前一期的变化量"而不是"绝对水平". 变化量值域
        通常远小于水平值且**跨训练/测试集稳定**, 因此树模型/Transformer
        都能在它的值域内插值, 累加还原后水平值能突破训练集见过的上界.

        这等价于 ARIMA(p, 1, q) 里的 "1" — 一阶整合 / 去单位根.
    """

    needs_anchor = True
    length_delta = -1   # 差分让 y 少 1 行

    def transform(self, y, X=None):
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            # 多目标场景, 各列独立差分
            y_diff = np.diff(y_arr, axis=0)
        else:
            y_diff = np.diff(y_arr)
        X_aligned = X[1:] if X is not None else None
        return y_diff, X_aligned

    def inverse_transform(self, y_t, anchor=None):
        if anchor is None:
            raise ValueError(
                "DiffTransform.inverse_transform 需要 anchor 参数 "
                "(测试集开始前最后一个真实水平值, 如 y_val[-1]). "
                "/ DiffTransform.inverse_transform requires `anchor`."
            )
        y_t = np.asarray(y_t)
        # 沿样本维 (axis=0) 累加, 兼容 (N,) 和 (N, M) 多目标场景
        if y_t.ndim == 1:
            return float(anchor) + np.cumsum(y_t)
        return np.asarray(anchor) + np.cumsum(y_t, axis=0)


# ──────────────────────────────────────────────────────────────────────────
# Log
# ──────────────────────────────────────────────────────────────────────────

class LogTransform(TargetTransform):
    """
    自然对数 / Natural logarithm.

    transform:        y_log = log(y)        (要求 y > 0)
    inverse_transform: y_pred = exp(y_log_pred)

    业务含义 / Business meaning:
        把"乘性变化"线性化, 对**长期指数增长**或 log-normal 分布的目标
        友好 (如 GDP / 用户数 / 累计存款). 模型在 log 空间学加性结构.
    """

    needs_anchor = False
    length_delta = 0

    def fit(self, y_train):
        arr = np.asarray(y_train)
        if np.any(arr <= 0):
            raise ValueError(
                f"LogTransform 要求 y > 0, 但 y_train 最小值 = "
                f"{float(arr.min()):.4f}. 含 0/负值请用 Log1pTransform "
                f"或先平移. / LogTransform requires y > 0."
            )
        return self

    def transform(self, y, X=None):
        return np.log(np.asarray(y)), X

    def inverse_transform(self, y_t, anchor=None):
        return np.exp(np.asarray(y_t))


# ──────────────────────────────────────────────────────────────────────────
# Log1p
# ──────────────────────────────────────────────────────────────────────────

class Log1pTransform(TargetTransform):
    """
    log(1 + y) / log1p transform.

    transform:        y_l = log1p(y)        (要求 y >= -1, 实务中 y >= 0)
    inverse_transform: y_pred = expm1(y_l_pred)

    业务含义 / Business meaning:
        Log 的"含 0 友好版", 适用于:
          * 计数数据 (订单数 / 投诉数, 可能为 0)
          * 大量稀疏正值 (如某月某产品销量, 大部分日子为 0)
        在 0 附近近似线性, 在大 y 处近似 log.
    """

    needs_anchor = False
    length_delta = 0

    def fit(self, y_train):
        arr = np.asarray(y_train)
        if np.any(arr < -1):
            raise ValueError(
                f"Log1pTransform 要求 y >= -1, 但 y_train 最小值 = "
                f"{float(arr.min()):.4f}. / Log1pTransform requires y >= -1."
            )
        return self

    def transform(self, y, X=None):
        return np.log1p(np.asarray(y)), X

    def inverse_transform(self, y_t, anchor=None):
        return np.expm1(np.asarray(y_t))


# ──────────────────────────────────────────────────────────────────────────
# Compose (按序复合)
# ──────────────────────────────────────────────────────────────────────────

class ComposeTransform(TargetTransform):
    """
    按序复合多个 Transform / Compose multiple transforms in order.

    transform: 依次应用 t1.transform → t2.transform → ...
    inverse_transform: 逆序应用 tN.inverse_transform → ... → t1.inverse_transform

    典型组合: ``ComposeTransform([LogTransform(), DiffTransform()])`` —
    先取对数稳定方差, 再差分去趋势. 对**指数增长 + 长趋势**双重特性
    数据(如累计余额) 特别有效.

    注意 / Caveat:
        * needs_anchor: 任一子变换需要 anchor 则整体需要
        * length_delta: 所有子变换 length_delta 之和 (例如 [Log, Diff] → -1)
        * anchor 的语义: 是**最外层(变换链最后一步)** 还原所需的锚点,
          即 anchor 是在**已经被前面变换处理过的空间**里的值. 调用方
          需要自己把"原始 y_val[-1]"先 forward 变换到中间空间作为 anchor.
          调用复杂度由此上升, 简单场景建议直接选单一变换.
    """

    def __init__(self, transforms: List[TargetTransform]):
        if not transforms:
            raise ValueError("ComposeTransform 至少需要一个子 transform")
        self.transforms: List[TargetTransform] = list(transforms)
        self.needs_anchor = any(t.needs_anchor for t in self.transforms)
        self.length_delta = sum(t.length_delta for t in self.transforms)

    def fit(self, y_train):
        y_cur = np.asarray(y_train)
        for t in self.transforms:
            t.fit(y_cur)
            # 用 transform 后的 y 继续 fit 下一个 (无 X 上下文, X=None 让位置占位)
            y_cur, _ = t.transform(y_cur)
        return self

    def transform(self, y, X=None):
        y_cur, X_cur = np.asarray(y), X
        for t in self.transforms:
            y_cur, X_cur = t.transform(y_cur, X_cur)
        return y_cur, X_cur

    def inverse_transform(self, y_t, anchor=None):
        y_cur = np.asarray(y_t)
        for t in reversed(self.transforms):
            y_cur = t.inverse_transform(y_cur, anchor=anchor)
        return y_cur


# ──────────────────────────────────────────────────────────────────────────
# 工厂 / Factory
# ──────────────────────────────────────────────────────────────────────────

_REGISTRY = {
    'identity': IdentityTransform,
    'none':     IdentityTransform,
    'diff':     DiffTransform,
    'log':      LogTransform,
    'log1p':    Log1pTransform,
}


def get_target_transform(
    name_or_obj: Union[str, TargetTransform, None],
) -> TargetTransform:
    """
    工厂函数 / Factory.

    支持:
        None / 'identity' / 'none'  → IdentityTransform
        'diff'                       → DiffTransform
        'log'                        → LogTransform
        'log1p'                      → Log1pTransform
        TargetTransform 实例         → 原样返回

    用法:
        from tsf_frame.utils.target_transforms import get_target_transform
        tt = get_target_transform(config.get('target_transform', 'identity'))
        y_t, X_t = tt.fit(y_train).transform(y_train, X_train)
        ...
        y_pred = tt.inverse_transform(model.predict(X_test),
                                       anchor=y_val[-1])
    """
    if name_or_obj is None:
        return IdentityTransform()
    if isinstance(name_or_obj, TargetTransform):
        return name_or_obj
    if isinstance(name_or_obj, str):
        key = name_or_obj.lower()
        if key in _REGISTRY:
            return _REGISTRY[key]()
        raise ValueError(
            f"未知 target_transform: {name_or_obj!r}. "
            f"可选: {list(_REGISTRY)}. / Unknown transform name."
        )
    raise TypeError(
        f"target_transform 必须是 str / TargetTransform / None, "
        f"got {type(name_or_obj).__name__}"
    )
