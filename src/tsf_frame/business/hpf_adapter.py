"""
住房公积金（Housing Provident Fund）业务适配器

支持的典型预测指标:
  - monthly_deposit      : 月缴存额（元）
  - monthly_withdrawal   : 月提取额（元）
  - loan_balance         : 贷款余额（元）
  - depositor_count      : 缴存人数（人）
  - loan_count           : 贷款笔数（笔）
  - loan_issued          : 当期贷款发放额（元）
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging

from .base_adapter import BaseBusinessAdapter
# 全项目 MAPE 收口在 MetricsCalculator (eps 保护 + 小数形式),
# 这里复用而不是重写, 避免三处 MAPE 计算逻辑分歧.
from ..utils.metrics import MetricsCalculator
logger = logging.getLogger(__name__)


class HPFAdapter(BaseBusinessAdapter):
    """Housing Provident Fund (HPF) business adapter.

    Note: This class is **not thread‑safe**; each instance should be used by a single thread.
    """
    """
    公积金业务适配器，提供:
      - preprocess: 异常值处理 + 归一化
      - postprocess: 反归一化 + 非负约束
      - validate_data: 业务规则校验
      - get_business_metrics: 公积金专项评估指标
      - get_policy_adjusted_forecast: 叠加政策调整效应
      原始数据
   ↓
validate_data（检查）
   ↓
preprocess
   ├─ 异常值处理
   └─ 归一化
   ↓
模型训练 / 预测
   ↓
postprocess
   ├─ 反归一化
   └─ 强制非负
   ↓
get_business_metrics（评估效果）
   ↓
get_policy_adjusted_forecast（政策模拟）


    """

    # 公积金业务中不应出现负值的指标
    NON_NEGATIVE_COLS: List[str] = [
        'monthly_deposit', 'monthly_withdrawal', 'loan_balance',
        'depositor_count', 'loan_count', 'loan_issued',
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        初始化公积金适配器 / Initialize HPF adapter

        Args:
            config: 配置字典 / Configuration dict
                - normalization (str): 'minmax' 或 'zscore'，默认 'zscore'（公积金金额数量级差异大，推荐 zscore）
                - handle_outliers (bool): 是否执行异常值处理，默认 True
                - outlier_method (str): 'iqr' 或 'sigma'，月度数据推荐 iqr
        """
        super().__init__(config)
        self.business_type = 'hpf'
        self.normalization: str = config.get('normalization', 'zscore')
        self.handle_outliers: bool = config.get('handle_outliers', True)
        # 'iqr'（四分位距法）或 'sigma'（3σ法），月度数据推荐 iqr
        self.outlier_method: str = config.get('outlier_method', 'iqr')
        # _scalers 结构 / _scalers structure:
        #   {col_name: {'type': 'minmax', 'min': float, 'max': float}}
        #   {col_name: {'type': 'zscore', 'mean': float, 'std': float}}
        # 由 fit_preprocess() 学习, 由 transform_preprocess() 复用,
        # 也用于 _denormalize() 反变换.
        self._scalers: Dict[str, Any] = {}
        # 是否已通过 fit_preprocess 学过 scaler;
        # transform_preprocess 必须在 _is_fitted=True 后调用, 否则报错
        # / Whether fit_preprocess has been called
        self._is_fitted: bool = False

# Validate NON_NEGATIVE_COLS are subset of target_columns
missing_non_neg = set(self.NON_NEGATIVE_COLS) - set(self.target_columns)
if missing_non_neg:
    raise ValueError(f"NON_NEGATIVE_COLS {missing_non_neg} not found in target_columns")

    # ── 预处理 ───────────────────────────────────────────────────────────────

    def preprocess(
        self, data: pd.DataFrame, fit: Optional[bool] = None, **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        公积金数据预处理 / HPF data preprocessing.

        流程: [可选]异常值处理 → 归一化.

        🔴 防数据泄露 / Anti data-leakage:
        - 训练集第一次调用: ``fit=True`` (或 ``fit=None`` 自动推断), 学习 scaler
          (mu/sigma 或 min/max) 并缓存到 ``self._scalers``.
        - 测试集调用: 必须 ``fit=False``, **复用训练集学到的 scaler**, 不能用
          测试集自己的统计量.

        Args:
            data: 原始数据.
            fit:
                - True  : 强制重新 fit scaler (覆盖之前的)
                - False : 强制只 transform (要求已经 fit 过, 否则 RuntimeError)
                - None  : 自动 — 第一次调用时 fit, 后续 transform-only

        Returns:
            (处理后的数据, 元数据字典). metadata 包含 scalers / outliers_info,
            供 postprocess 反变换.

        Raises:
            RuntimeError: ``fit=False`` 但 adapter 还没 fit 过.
        """
        # 决定是否 fit
        if fit is None:
            fit = not self._is_fitted   # 首次调用自动 fit, 之后 transform-only
        if not fit and not self._is_fitted:
            raise RuntimeError(
                'preprocess(fit=False) 要求 adapter 已经 fit 过. '
                '请先在训练集上调 preprocess(data, fit=True), '
                '再在测试集上调 preprocess(data, fit=False).'
            )

        data = data.copy()
        # Ensure deterministic order for outlier handling
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        metadata: Dict[str, Any] = {
            'original_columns': data.columns.tolist(),
            'normalization': self.normalization,
            'scalers': {},
            'outliers_info': {},
        }

        if self.handle_outliers:
            logger.debug("Handling outliers for columns: %s", self.target_columns)
            # 异常值处理只用当前数据自身的分布 (IQR / 3σ),
            # 这是局部插值, 不会跨 train/test 泄露统计量.
            data, outliers_info = self._handle_outliers(data)
            metadata['outliers_info'] = outliers_info

        if fit:
            logger.debug("Fitting scalers on training data")
            # 训练阶段: 重新学 scaler 并保存
            data = self._normalize(data, metadata)
            self._is_fitted = True
        else:
            # 推理阶段: 复用 self._scalers, 不再重算统计量 (防数据泄露)
            metadata['scalers'] = dict(self._scalers)
            data = self._apply_scaler(data, self._scalers)

        return data, metadata

    def _apply_scaler(
        self, data: pd.DataFrame, scalers: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        用已 fit 的 scaler 做 transform-only 归一化 (不重算统计量).
        / Transform-only normalization using a previously fitted scaler.
        """
        data = data.copy()
        for col, sc in scalers.items():
            if col not in data.columns:
                continue
            if sc['type'] == 'minmax':
                data[col] = (data[col] - sc['min']) / (sc['max'] - sc['min'] + 1e-8)
            elif sc['type'] == 'zscore':
                data[col] = (data[col] - sc['mean']) / (sc['std'] + 1e-8)
        return data

    def _handle_outliers(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        异常值检测与处理 / Outlier detection and handling

        将异常值置为 NaN 后通过插值填充，避免极端值对模型训练的干扰。
        Replaces outliers with NaN and fills via interpolation to prevent
        extreme values from distorting model training.

        Args:
            data: 输入数据 / Input DataFrame

        Returns:
            (处理后的数据, 异常值信息字典) / (Cleaned data, outlier info dict)
        """
        data = data.copy()
        info: Dict[str, Any] = {}

        for col in self.target_columns:
            if col not in data.columns:
                continue

            if self.outlier_method == 'iqr':
                # IQR 法: 超出 [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 范围视为异常值
                # IQR method: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are outliers
                Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            else:  # 3-sigma
                # 3-sigma 法: 超出 [mu - 3*sigma, mu + 3*sigma] 范围视为异常值
                # 3-sigma method: values outside [mu - 3*sigma, mu + 3*sigma] are outliers
                mu, sigma = data[col].mean(), data[col].std()
                lower, upper = mu - 3 * sigma, mu + 3 * sigma

            mask = (data[col] < lower) | (data[col] > upper)
            info[col] = {
                'outlier_count': int(mask.sum()),
                'lower_bound': float(lower),
                'upper_bound': float(upper),
            }

            # 异常值置 NaN，然后插值填充 / Set outliers to NaN, then interpolate
            data.loc[mask, col] = np.nan
            # 时间序列插值：有时间索引用 time 法，否则用线性插值
            # Time-series interpolation: use 'time' for DatetimeIndex, else 'linear'
            method = 'time' if isinstance(data.index, pd.DatetimeIndex) else 'linear'
            data[col] = data[col].interpolate(method=method).ffill().bfill()

        return data, info

    def _normalize(self, data: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        数值列归一化 / Normalize numeric columns

        MinMax: x' = (x - min) / (max - min + eps)，映射到 [0, 1]
        Z-Score: x' = (x - mean) / (std + eps)，均值 0 方差 1

        Args:
            data: 待归一化数据 / Data to normalize
            metadata: 元数据字典，归一化参数写入 scalers 字段 / Metadata dict

        Returns:
            归一化后的数据 / Normalized data
        """
        data = data.copy()
        for col in data.select_dtypes(include=[np.number]).columns:
            if self.normalization == 'minmax':
                vmin, vmax = data[col].min(), data[col].max()
                # 加 eps 防止 max == min 时除零
                # Add eps to prevent division by zero when max == min
                data[col] = (data[col] - vmin) / (vmax - vmin + 1e-8)
                metadata['scalers'][col] = {'type': 'minmax', 'min': float(vmin), 'max': float(vmax)}
            elif self.normalization == 'zscore':
                mu, sigma = data[col].mean(), data[col].std()
                # 加 eps 防止常数列 std == 0 时除零
                # Add eps to prevent division by zero on constant columns
                data[col] = (data[col] - mu) / (sigma + 1e-8)
                metadata['scalers'][col] = {'type': 'zscore', 'mean': float(mu), 'std': float(sigma)}
        self._scalers = metadata['scalers']
        return data

    def _denormalize(self, data: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        反归一化 / Denormalize

        MinMax 逆变换: x = x' * (max - min + eps) + min（与正变换对称）
        Z-Score 逆变换: x = x' * std + mean

        Args:
            data: 归一化后的数据 / Normalized data
            metadata: 包含归一化参数的元数据 / Metadata with scaler parameters

        Returns:
            反归一化后的数据 / Denormalized data
        """
        data = data.copy()
        scalers = metadata.get('scalers', {})
        for col in data.columns:
            if col not in scalers:
                continue
            sc = scalers[col]
            if sc['type'] == 'minmax':
                data[col] = data[col] * (sc['max'] - sc['min'] + 1e-8) + sc['min']
            elif sc['type'] == 'zscore':
                data[col] = data[col] * sc['std'] + sc['mean']
        return data

    # ── 后处理 ───────────────────────────────────────────────────────────────

    def postprocess(self, predictions: np.ndarray, metadata: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Postprocess predictions: denormalize and enforce non‑negative constraints.

        Logging added for debugging the steps.
        """
        logger.debug("Starting postprocess with predictions shape %s", predictions.shape)
        """
        公积金数据后处理 / HPF data postprocessing

        流程: 反归一化 → 非负约束裁剪
        Pipeline: denormalization → non-negative clipping

        Args:
            predictions: 模型原始预测输出(归一化空间) / Raw model predictions in normalized space
            metadata: 预处理阶段保存的元数据 / Metadata saved during preprocessing

        Returns:
            还原到原始尺度且满足非负约束的预测 DataFrame
            Predictions restored to original scale with non-negative constraint enforced
        """
        pred_df = pd.DataFrame(predictions, columns=self.target_columns)
        pred_df = self._denormalize(pred_df, metadata)

        # 公积金业务指标不能为负(缴存额/提取额/贷款余额等均非负)
        # HPF metrics must be non-negative (deposit/withdrawal/loan balance, etc.)
        for col in self.target_columns:
            if col in self.NON_NEGATIVE_COLS:
                pred_df[col] = pred_df[col].clip(lower=0)

        logger.debug("Postprocess completed, returning DataFrame with columns %s", pred_df.columns.tolist())
        return pred_df

    # ── 数据校验 ─────────────────────────────────────────────────────────────

    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        公积金业务规则校验 / HPF business-rule validation

        四级校验 / Four-level validation:
          1. 必要列存在 / Required columns present
          2. 非负约束(关键业务规则) / Non-negative constraint (key business rule)
          3. 缺失率 ≤ 10% / Missing ratio ≤ 10%
          4. 时间索引频率规则(月度数据应连续) / Regular time index frequency

        Args:
            data: 待校验数据 / Data to validate

        Returns:
            (是否通过, 描述信息) / (pass/fail, description message)
        """
        # 1. 必要列检查 / Required columns check
        missing = [c for c in self.target_columns if c not in data.columns]
        if missing:
            return False, f"缺少必要列: {missing}"

        # 2. 非负约束检查 / Non-negative constraint check
        for col in self.target_columns:
            if col in self.NON_NEGATIVE_COLS and (data[col] < 0).any():
                neg_count = (data[col] < 0).sum()
                return False, f"列 '{col}' 存在 {neg_count} 个负值，公积金业务指标不应为负"

        # 3. 高缺失率检查（超过 10%） / High missing-ratio check (> 10%)
        null_ratios = data[self.target_columns].isnull().mean()
        high_null = null_ratios[null_ratios > 0.1]
        if not high_null.empty:
            return False, f"以下列缺失率超过10%: {high_null.to_dict()}"

        # 4. 时间索引连续性检查（月度数据） / Time index continuity check (monthly data)
        # 仅当能明确判定为"重复时间戳"时才阻断; pd.infer_freq 在数据头尾有 NaT、
        # 或样本量过小时会误返 None, 不应据此一票否决. 严格频率检查交给
        # monitoring.data_quality.FrequencyCheck (有 tolerance 字段) 做.
        # / Only block on duplicate timestamps; defer strict frequency checking to
        #   monitoring.FrequencyCheck (which supports tolerance).
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            if data.index.duplicated().any():
                dup_count = int(data.index.duplicated().sum())
                return False, f"时间索引存在 {dup_count} 个重复时间戳"

        return True, "数据验证通过"

    # ── 业务指标 ─────────────────────────────────────────────────────────────

    def get_business_metrics(
        self, y_true: pd.DataFrame, y_pred: pd.DataFrame
    ) -> Dict[str, float]:
        """
        计算公积金业务专项指标 / Compute HPF-specific business metrics

        四大业务指标 / Four business metrics:
          1. MAPE      : 平均绝对百分比误差(避免 0 分母) / Mean absolute percentage error
          2. direction_accuracy : 环比方向准确率(t 对 t-1 的涨跌方向) / MoM direction accuracy
          3. yoy_mae   : 同比增长率 MAE(t 对 t-12,需 >12 个月数据) / YoY growth-rate MAE
          4. qoq_mae   : 季度环比 MAE(t 对 t-3) / Quarter-over-quarter MAE

        Args:
            y_true: 真实值 DataFrame / Actual values
            y_pred: 预测值 DataFrame / Predicted values

        Returns:
            {col_metric: value} 指标字典,key 形如 'monthly_deposit_mape'
            Dict mapping '{col}_{metric}' → value
        """
        metrics: Dict[str, float] = {}

        for col in self.target_columns:
            if col not in y_true.columns or col not in y_pred.columns:
                continue

            true_vals = y_true[col].values.astype(float)
            pred_vals = y_pred[col].values.astype(float)

            # MAPE: 直接用 MetricsCalculator (eps 保护 + 小数形式),
            # 避免本模块自己再写一份 mask/eps 逻辑造成三处 MAPE 不统一.
            # / Delegate to MetricsCalculator for project-wide MAPE consistency.
            metrics[f'{col}_mape'] = MetricsCalculator.mape(true_vals, pred_vals)

            # 趋势方向准确率（环比方向）
            if len(true_vals) > 1:
                dir_acc = np.mean(
                    np.sign(true_vals[1:] - true_vals[:-1]) ==
                    np.sign(pred_vals[1:] - pred_vals[:-1])
                )
                metrics[f'{col}_direction_accuracy'] = float(dir_acc)

            # 同比增长率误差（需要 12 个月以上数据）
            # Year-over-Year growth rate error (requires >12 months of data)
            # 注意:pred_yoy 需基于 pred 自身历史计算,与 true_yoy 对等,才能反映
            # "双方各自的同比增长率差异"。若两者共用 true 作分母会混淆概念。
            if len(true_vals) > 12:
                true_yoy = (true_vals[12:] - true_vals[:-12]) / (np.abs(true_vals[:-12]) + 1e-8)
                pred_yoy = (pred_vals[12:] - pred_vals[:-12]) / (np.abs(pred_vals[:-12]) + 1e-8)
                metrics[f'{col}_yoy_mae'] = float(np.mean(np.abs(true_yoy - pred_yoy)))

            # 季度环比误差: 与 yoy 对称 — 双方各自基于自己历史计算环比率
            # / Quarter-over-quarter MAE: each side uses its OWN history as denominator
            if len(true_vals) > 3:
                true_qoq = (true_vals[3:] - true_vals[:-3]) / (np.abs(true_vals[:-3]) + 1e-8)
                pred_qoq = (pred_vals[3:] - pred_vals[:-3]) / (np.abs(pred_vals[:-3]) + 1e-8)
                metrics[f'{col}_qoq_mae'] = float(np.mean(np.abs(true_qoq - pred_qoq)))

        return metrics

    # ── 政策调整 ─────────────────────────────────────────────────────────────

    def get_policy_adjusted_forecast(
        self,
        base_forecast: pd.DataFrame,
        policy_effects: Dict[str, float],
    ) -> pd.DataFrame:
        """
        在基础预测结果上叠加政策调整效应 / Apply policy adjustment effects on top of base forecast

        使用乘法调整：adjusted = base * (1 + effect)，
        正数表示增长，负数表示下降。
        Uses multiplicative adjustment: adjusted = base * (1 + effect),
        positive values indicate growth, negative values indicate decline.

        Args:
            base_forecast: 模型基础预测 DataFrame / Base forecast DataFrame from the model
            policy_effects: 政策效应系数字典 / Policy effect coefficient dict, e.g.:
                {'monthly_deposit': 0.05}  -> 政策使缴存额增长 5% / deposit grows 5%
                {'monthly_withdrawal': -0.03}  -> 政策使提取额减少 3% / withdrawal drops 3%

        Returns:
            调整后的预测 DataFrame / Policy-adjusted forecast DataFrame
        """
        adjusted = base_forecast.copy()
        for col, effect in policy_effects.items():
            if col in adjusted.columns:
                # 乘法调整: 原始值 * (1 + 政策效应系数)
                # Multiplicative adjustment: original * (1 + policy effect coefficient)
                # 公积金指标在业务上应非负(缴存额/提取额/贷款余额等不可能为负),
                # 若传入 effect <= -1 时 (1 + effect) 为非正,clip 保证下界为 0。
                # HPF metrics must be non-negative; clip to 0 when effect <= -1.
                adjusted[col] = (adjusted[col] * (1 + effect)).clip(lower=0)
        return adjusted

    def get_seasonal_decomposition(
        self, data: pd.Series, period: int = 12
    ) -> Dict[str, pd.Series]:
        """Simple additive seasonal decomposition.

        Returns a dict with ``trend``, ``seasonal`` and ``residual`` components.
        """
        # Ensure data is sorted by index for correct rolling calculations
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        # Trend via centered moving average
        trend = data.rolling(window=period, center=True, min_periods=1).mean()
        # Remove trend to obtain seasonal + residual signal
        seasonal_raw = data - trend
        # Determine seasonal index based on datetime month or positional period
        if isinstance(data.index, pd.DatetimeIndex):
            seasonal_idx = data.index.month
        else:
            seasonal_idx = np.arange(len(data)) % period
        # Compute seasonal means using groupby for efficiency
        seasonal_means = pd.Series(seasonal_raw.values, index=seasonal_idx).groupby(level=0).mean()
        # Map back to full seasonal series
        seasonal = pd.Series(
            seasonal_idx.map(seasonal_means).values,
            index=data.index
        )
        # Residual component
        residual = data - trend - seasonal
        return {'trend': trend, 'seasonal': seasonal, 'residual': residual}
