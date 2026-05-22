"""
InferenceRunner — 端到端推理 wrapper / End-to-end inference wrapper
======================================================================

把训练时分散在多个对象里的 5 个状态打包成单一 artifact, 推理时一行调用即出预测.
解决"训练流程 vs 推理流程"间状态散落、容易漏传 ``fit=False`` 等 anti-data-leakage
诉求的工程问题.

打包的 5 个状态:
  1. ``adapter``         — BaseBusinessAdapter (含 _scalers + _is_fitted)
  2. ``feat_eng``        — (Composite)FeatureEngineer (含 fit 状态)
  3. ``mfh``             — MixedFeatureHandler (含 static_values + _is_fitted)
  4. ``model``           — BaseModel 子类实例 (含 state_dict)
  5. ``model_config``    — 重建 model 时需要的 config dict

推理一行流:
    runner = InferenceRunner.load('hpf_model.pkl', model_cls=XGBoostModel)
    pred   = runner.predict(latest_df, target_col='monthly_deposit')
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd

from ..models.base_model import BaseModel, ProbabilisticPrediction

logger = logging.getLogger(__name__)

__all__ = ['InferenceRunner']


class InferenceRunner:
    """
    端到端推理 wrapper / End-to-end inference wrapper.

    Args:
        adapter:        BusinessAdapter 实例 (训练完, _is_fitted=True)
        feat_eng:       特征工程实例 (fit_transform 过)
        mfh:            MixedFeatureHandler 实例 (fit 过)
        model:          BaseModel 子类实例 (fit 过)
        model_config:   重建 model 用的 config dict (传 _build_from_config 用)
        target_col:     默认目标列 (predict 时可覆盖)
    """

    ARTIFACT_VERSION = 1

    def __init__(
        self,
        *,
        adapter,
        feat_eng,
        mfh,
        model: BaseModel,
        model_config: Dict[str, Any],
        target_col: Optional[str] = None,
    ):
        self.adapter = adapter
        self.feat_eng = feat_eng
        self.mfh = mfh
        self.model = model
        self.model_config = dict(model_config)
        self.target_col = target_col

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """
        把 5 个状态一次性 pickle 落盘.

        Args:
            path: 目标文件路径 (建议 .pkl 后缀)

        Returns:
            实际写入的绝对路径
        """
        path = str(Path(path).resolve())
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        # 整个 model 对象直接 pickle, 对 ML (sklearn 包装) 和 DL (nn.Module) 都通用.
        # DL 模型若需要跨 PyTorch 版本恢复, 也可以同时存 state_dict (双保险).
        # / Pickle whole model object; sklearn wrappers + nn.Module both supported.
        try:
            model_state = self.model.state_dict()
        except Exception as exc:
            # 多数 sklearn 包装模型 (RidgeModel/XGBoostModel) 不支持 state_dict, 走整对象 pickle.
            # 留 debug 日志而非吞错, 便于排查"为什么 state_dict 字段为空".
            model_state = None
            logger.debug(
                "InferenceRunner.save: model.state_dict() 不支持 (%s: %s); "
                "走整对象 pickle 路径.", type(exc).__name__, exc,
            )
        artifact = {
            'version': self.ARTIFACT_VERSION,
            'adapter': self.adapter,
            'feat_eng': self.feat_eng,
            'mfh': self.mfh,
            'model': self.model,                      # 整对象 pickle (主路径)
            'model_state_dict': model_state,          # 双保险, DL 模型用
            'model_config': self.model_config,
            'model_class_name': type(self.model).__name__,
            'target_col': self.target_col,
        }
        with open(path, 'wb') as f:
            pickle.dump(artifact, f)
        return path

    @classmethod
    def load(
        cls,
        path: str,
        *,
        model_cls: Optional[Type[BaseModel]] = None,
    ) -> 'InferenceRunner':
        """
        从 pickle artifact 重建 InferenceRunner.

        Args:
            path:       artifact 路径
            model_cls:  model 类 (调用方传入, 因 pickle 不存类引用避免依赖版本).
                        若 None, 尝试从 model_class_name 走 MODEL_REGISTRY / DL_MODEL_REGISTRY.

        Returns:
            重建好的 runner; adapter._is_fitted 强制为 True 防止误重新 fit.
        """
        with open(path, 'rb') as f:
            artifact = pickle.load(f)

        if artifact.get('version', 1) != cls.ARTIFACT_VERSION:
            import warnings
            warnings.warn(
                f"InferenceRunner artifact 版本 {artifact.get('version')} 与当前 "
                f"{cls.ARTIFACT_VERSION} 不一致, 加载可能失败.",
                UserWarning, stacklevel=2,
            )

        adapter = artifact['adapter']
        feat_eng = artifact['feat_eng']
        mfh = artifact['mfh']
        model_config = artifact['model_config']

        # 强制 adapter._is_fitted=True (防止 predict 时 fit=None 自动 fit 推理数据 → 数据泄露)
        # / Anti-data-leakage: force fit=True flag on reload.
        if hasattr(adapter, '_is_fitted'):
            adapter._is_fitted = True

        # 优先用 artifact 里直接 pickle 的整对象 model (兼容 ML/DL); 否则按 state_dict 重建.
        model = artifact.get('model')
        if model is None:
            if model_cls is None:
                model_cls = cls._resolve_model_class(artifact.get('model_class_name'))
            model = model_cls(model_config)
            model_state = artifact.get('model_state_dict')
            if model_state is not None:
                try:
                    model.load_state_dict(model_state)
                except Exception as exc:
                    import warnings
                    warnings.warn(
                        f"InferenceRunner.load: state_dict 恢复失败 ({exc}); "
                        f"artifact 也没存整对象 model. 推理可能出错.",
                        UserWarning, stacklevel=2,
                    )

        return cls(
            adapter=adapter,
            feat_eng=feat_eng,
            mfh=mfh,
            model=model,
            model_config=model_config,
            target_col=artifact.get('target_col'),
        )

    @staticmethod
    def _resolve_model_class(name: Optional[str]) -> Type[BaseModel]:
        """从 class name 解析 model class (走 MODEL_REGISTRY / DL_MODEL_REGISTRY)."""
        if name is None:
            raise ValueError(
                "InferenceRunner.load: 既没传 model_cls, artifact 里也没存 "
                "model_class_name. 请显式传 model_cls."
            )
        # 延迟 import 避免循环依赖
        try:
            from ..models.classical.ml_models import MODEL_REGISTRY
            for cls_obj in MODEL_REGISTRY.values():
                if cls_obj.__name__ == name:
                    return cls_obj
        except Exception as exc:
            # MODEL_REGISTRY 加载失败 (理论上不应发生); 留 warning 让用户排查
            logger.warning(
                "InferenceRunner: 解析 MODEL_REGISTRY 时失败 (%s: %s), "
                "尝试 DL_MODEL_REGISTRY...", type(exc).__name__, exc,
            )
        try:
            from ..models.transformer.transformer_models import DL_MODEL_REGISTRY
            for cls_obj in DL_MODEL_REGISTRY.values():
                if cls_obj.__name__ == name:
                    return cls_obj
        except Exception as exc:
            logger.warning(
                "InferenceRunner: 解析 DL_MODEL_REGISTRY 时失败 (%s: %s).",
                type(exc).__name__, exc,
            )
        raise ValueError(
            f"InferenceRunner.load: 无法从 MODEL_REGISTRY / DL_MODEL_REGISTRY "
            f"解析 model_class_name={name!r}. 请显式传 model_cls."
        )

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def predict(
        self,
        latest_df: pd.DataFrame,
        *,
        target_col: Optional[str] = None,
        return_raw: bool = False,
    ) -> ProbabilisticPrediction:
        """
        端到端单窗口推理: validate → preprocess(fit=False) → feat_eng.transform
        → mfh.create_single_input → model.predict_probabilistic → denormalize.

        Args:
            latest_df:   最近 N 行原始数据 (调用方按 SQL 拉, N >= mfh.min_required_rows)
            target_col:  目标列名, 默认走 self.target_col
            return_raw:  True 时返回 zscore 空间的预测, 默认 False 反归一化到 raw 量纲

        Returns:
            ProbabilisticPrediction (mean / lower / upper 在 raw 量纲下)

        Raises:
            ValueError: latest_df 行数不够; target_col 未提供也未在 self.target_col 设置
        """
        tgt = target_col or self.target_col
        if tgt is None:
            raise ValueError(
                "InferenceRunner.predict: target_col 既没传参也未在构造时设置."
            )

        # ── 1) 防御性行数检查 ──
        min_rows = self._min_required_rows()
        if len(latest_df) < min_rows:
            raise ValueError(
                f"InferenceRunner.predict: latest_df 只有 {len(latest_df)} 行, "
                f"需要 >= {min_rows} (mfh.seq_len + feature_engineer 最大回看)."
            )

        # ── 2) 数据校验 (若 adapter 支持) ──
        if hasattr(self.adapter, 'validate_data'):
            ok, msg = self.adapter.validate_data(latest_df)
            if not ok:
                raise ValueError(f"InferenceRunner.predict: adapter.validate_data 失败: {msg}")

        # ── 3) 预处理 (fit=False 防数据泄露) ──
        processed, _metadata = self.adapter.preprocess(latest_df, fit=False)

        # ── 4) 特征工程 (transform-only) ──
        if hasattr(self.feat_eng, 'transform'):
            df_feat = self.feat_eng.transform(processed)
        else:
            df_feat = processed

        # 丢弃前 N 行 NaN (lag/rolling 引入)
        df_feat = df_feat.dropna()
        if len(df_feat) < self.mfh.seq_len:
            raise ValueError(
                f"InferenceRunner.predict: 特征工程后剩 {len(df_feat)} 行, "
                f"不足 seq_len={self.mfh.seq_len}."
            )

        # ── 5) 单窗口推理 ──
        X_input = self.mfh.create_single_input(df_feat)  # (1, seq_len, F)

        # ML 模型期望 2D (N, F); seq_len=1 时 squeeze 时间维, 否则取最后一步.
        # DL 模型期望 3D (B, L, C), 保持原状.
        # 判定: BaseMLModel 才有 _build_model 方法 (sklearn 包装路径), DL 没有.
        # 不能用 isinstance(model, nn.Module) — BaseModel 已继承 nn.Module, ML/DL 都是 True.
        # / Duck-type: BaseMLModel exposes _build_model (sklearn wrapper); DL models don't.
        is_ml_model = hasattr(self.model, '_build_model')
        if is_ml_model and X_input.ndim == 3:
            X_input = X_input[:, -1, :]  # (1, F)

        if hasattr(self.model, 'predict_probabilistic'):
            prob = self.model.predict_probabilistic(X_input)
        else:
            point = self.model.predict(X_input)
            prob = ProbabilisticPrediction(mean=np.asarray(point))

        if return_raw:
            return prob

        # ── 6) 反归一化回 raw 量纲 ──
        return self._denormalize_prob(prob, tgt)

    def _denormalize_prob(
        self, prob: ProbabilisticPrediction, target_col: str,
    ) -> ProbabilisticPrediction:
        """
        对 mean / lower / upper 各自反归一化, 走 adapter.postprocess 公共 API.

        优先 postprocess (HPF: 反归一化 + 非负 clip);
        adapter 不支持 postprocess 时退化为 _denormalize 私有路径 (兼容自定义 adapter).
        """
        meta = {'scalers': getattr(self.adapter, '_scalers', {})}

        # 临时把 target_col 加进 adapter.target_columns 以便 postprocess 走非负 clip 分支
        # / target_col 可能未在 adapter.target_columns 中(多目标场景), 也要能反归一化
        has_postprocess = hasattr(self.adapter, 'postprocess')
        original_targets = getattr(self.adapter, 'target_columns', None)

        def _denorm(arr: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(np.asarray(arr).reshape(-1, 1), columns=[target_col])
            if has_postprocess:
                # postprocess 内部走 _denormalize + (HPF 特有) 非负 clip
                # 临时挪 target_columns 以让裁剪应用到当前列
                try:
                    if original_targets is not None and target_col not in original_targets:
                        self.adapter.target_columns = [target_col]
                    out = self.adapter.postprocess(
                        np.asarray(arr).reshape(-1, 1), meta,
                    )
                finally:
                    if original_targets is not None:
                        self.adapter.target_columns = original_targets
                return out[target_col].values
            # 退化路径: 自定义 adapter 没有 postprocess 时
            out = self.adapter._denormalize(df, meta)
            return out[target_col].values

        mean = _denorm(prob.mean)
        lower = _denorm(prob.lower) if prob.lower is not None else None
        upper = _denorm(prob.upper) if prob.upper is not None else None
        return ProbabilisticPrediction(
            mean=mean, lower=lower, upper=upper, std=prob.std, samples=prob.samples,
        )

    def _min_required_rows(self) -> int:
        """优先调 mfh.min_required_rows(feat_eng); 没有则 seq_len fallback."""
        if hasattr(self.mfh, 'min_required_rows'):
            try:
                return int(self.mfh.min_required_rows(feature_engineer=self.feat_eng))
            except TypeError:
                # 老版本 mfh.min_required_rows 是 property 不接受参数
                try:
                    return int(self.mfh.min_required_rows)
                except Exception:
                    pass
        return int(getattr(self.mfh, 'seq_len', 1))
