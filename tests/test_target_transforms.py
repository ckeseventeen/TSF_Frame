"""
单元测试 / Unit tests for target_transforms.
"""

import numpy as np
import pytest

from tsf_frame.utils.target_transforms import (
    IdentityTransform,
    DiffTransform,
    LogTransform,
    Log1pTransform,
    ComposeTransform,
    get_target_transform,
)


# ─── Identity ────────────────────────────────────────────────────────────

class TestIdentity:
    def test_transform_returns_y_unchanged(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        t = IdentityTransform()
        y_t, X_t = t.transform(y, X=None)
        np.testing.assert_array_equal(y_t, y)
        assert X_t is None

    def test_inverse_is_identity(self):
        y_pred = np.array([10.0, 11.0, 12.0])
        t = IdentityTransform()
        out = t.inverse_transform(y_pred)
        np.testing.assert_array_equal(out, y_pred)

    def test_x_passthrough(self):
        X = np.arange(8).reshape(4, 2)
        t = IdentityTransform()
        _, X_t = t.transform(np.zeros(4), X=X)
        np.testing.assert_array_equal(X_t, X)


# ─── Diff ────────────────────────────────────────────────────────────────

class TestDiff:
    def test_transform_length_minus_one(self):
        y = np.array([10.0, 12.0, 15.0, 14.0, 20.0])
        X = np.arange(10).reshape(5, 2)
        t = DiffTransform()
        y_d, X_d = t.transform(y, X)
        np.testing.assert_array_equal(y_d, np.array([2.0, 3.0, -1.0, 6.0]))
        # X 必须切片到 [1:] 与 y_diff 对齐
        np.testing.assert_array_equal(X_d, X[1:])
        assert len(X_d) == len(y_d)

    def test_inverse_requires_anchor(self):
        t = DiffTransform()
        with pytest.raises(ValueError, match="anchor"):
            t.inverse_transform(np.array([1.0, 2.0]))

    def test_inverse_cumsum_reconstruction(self):
        # 原序列: [10, 12, 15, 14, 20]
        # 差分:   [2, 3, -1, 6]
        # 假设 anchor 是序列首项 10, cumsum 应还原后 4 项 [12, 15, 14, 20]
        y_diff = np.array([2.0, 3.0, -1.0, 6.0])
        t = DiffTransform()
        out = t.inverse_transform(y_diff, anchor=10.0)
        np.testing.assert_allclose(out, np.array([12.0, 15.0, 14.0, 20.0]))

    def test_diff_inverse_roundtrip(self):
        """transform → inverse_transform 完整还原 (除首项被 anchor 替代)."""
        rng = np.random.default_rng(0)
        y = np.cumsum(rng.normal(0, 1, 50)) + 100.0
        anchor = float(y[0])
        t = DiffTransform()
        y_d, _ = t.transform(y)
        y_reconstructed = t.inverse_transform(y_d, anchor=anchor)
        # y_reconstructed 应等于 y[1:]
        np.testing.assert_allclose(y_reconstructed, y[1:], rtol=1e-10)

    def test_2d_multi_target_diff(self):
        """多目标 (N, M) 场景, 每列独立差分."""
        y = np.array([[1.0, 10.0], [3.0, 11.0], [6.0, 9.0], [4.0, 12.0]])
        t = DiffTransform()
        y_d, _ = t.transform(y)
        # 各列独立差分
        np.testing.assert_allclose(y_d, np.array([[2.0, 1.0], [3.0, -2.0], [-2.0, 3.0]]))


# ─── Log ─────────────────────────────────────────────────────────────────

class TestLog:
    def test_basic_forward_inverse_roundtrip(self):
        y = np.array([1.0, 2.0, 5.0, 10.0])
        t = LogTransform().fit(y)
        y_t, _ = t.transform(y)
        np.testing.assert_allclose(y_t, np.log(y))
        y_back = t.inverse_transform(y_t)
        np.testing.assert_allclose(y_back, y, rtol=1e-12)

    def test_rejects_zero_or_negative_on_fit(self):
        t = LogTransform()
        with pytest.raises(ValueError, match="y > 0"):
            t.fit(np.array([1.0, 0.0, 3.0]))
        with pytest.raises(ValueError, match="y > 0"):
            t.fit(np.array([-1.0, 2.0, 3.0]))


# ─── Log1p ───────────────────────────────────────────────────────────────

class TestLog1p:
    def test_handles_zero_values(self):
        y = np.array([0.0, 1.0, 10.0, 100.0])
        t = Log1pTransform().fit(y)
        y_t, _ = t.transform(y)
        np.testing.assert_allclose(y_t, np.log1p(y))
        y_back = t.inverse_transform(y_t)
        np.testing.assert_allclose(y_back, y, rtol=1e-12)

    def test_rejects_below_minus_one(self):
        t = Log1pTransform()
        with pytest.raises(ValueError, match="y >= -1"):
            t.fit(np.array([-2.0, 0.0, 5.0]))


# ─── Compose ─────────────────────────────────────────────────────────────

class TestCompose:
    def test_log_then_diff(self):
        """Log → Diff: 先对数稳定方差, 再差分去趋势."""
        # 指数增长序列, log 后线性, 再差分得到常量
        y = np.exp(np.arange(1, 6) * 0.5)   # [1.65, 2.72, 4.48, 7.39, 12.18]
        t = ComposeTransform([LogTransform(), DiffTransform()])
        t.fit(y)
        y_t, _ = t.transform(y)
        # log(y) = [0.5, 1.0, 1.5, 2.0, 2.5], diff = [0.5, 0.5, 0.5, 0.5]
        np.testing.assert_allclose(y_t, np.array([0.5, 0.5, 0.5, 0.5]), atol=1e-10)
        assert t.needs_anchor is True
        assert t.length_delta == -1

    def test_compose_inverse_chain(self):
        """组合变换的 inverse_transform 走逆序链."""
        y = np.array([1.0, 2.0, 4.0, 8.0, 16.0])  # 等比 2 (log 后线性)
        t = ComposeTransform([LogTransform(), DiffTransform()])
        t.fit(y)
        y_t, _ = t.transform(y)
        # anchor 必须是 log(y[0]) = log(1) = 0 (差分后空间的锚点)
        anchor_in_log = float(np.log(y[0]))
        y_back = t.inverse_transform(y_t, anchor=anchor_in_log)
        # 还原后是 y[1:] = [2, 4, 8, 16]
        np.testing.assert_allclose(y_back, y[1:], rtol=1e-12)


# ─── Factory ─────────────────────────────────────────────────────────────

class TestFactory:
    @pytest.mark.parametrize("key,expected_cls", [
        (None,        IdentityTransform),
        ('none',      IdentityTransform),
        ('identity',  IdentityTransform),
        ('diff',      DiffTransform),
        ('log',       LogTransform),
        ('log1p',     Log1pTransform),
        ('DIFF',      DiffTransform),  # 大小写不敏感
    ])
    def test_factory_creates_correct_class(self, key, expected_cls):
        t = get_target_transform(key)
        assert isinstance(t, expected_cls)

    def test_factory_passthrough_for_instance(self):
        inst = DiffTransform()
        out = get_target_transform(inst)
        assert out is inst

    def test_factory_rejects_unknown_string(self):
        with pytest.raises(ValueError, match="未知 target_transform"):
            get_target_transform('boxcox')   # 暂未实现

    def test_factory_rejects_bad_type(self):
        with pytest.raises(TypeError):
            get_target_transform(42)
