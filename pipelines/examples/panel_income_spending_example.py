"""
Panel Income & Spending Forecasting — ML vs DL 并排对比
========================================================

场景 / Scenario:
  100 个个体 × 36 个月的合成面板数据 (panel data),
  每人有静态属性 (身高/性别/行业) 和时变目标 (收入/花销),
  用过去 12 个月的所有信息预测未来 6 个月的 (收入, 花销).

目的 / Goal:
  在**完全相同的数据/划分/编码**下并排对比两条路径:
    路径 A — LightGBM   (ML):  flatten 特征 + MultiOutputRegressor + 残差法 OOS 区间
    路径 B — DLinear    (DL):  3D 时序张量 (B, L, C) + 多目标多步 + 分位数概率预测

  让"为什么这种场景该选哪条路径"用数字直接说话.

特征处理 / Feature handling:
  身高    — 数值,        zscore 标准化
  性别    — 类别 (M/F),   1 列 OneHot (drop_first=True)
  行业    — 类别 (5 类),  4 列 OneHot (drop_first=True)
  收入花销 — 数值时变,     训练集统计 zscore + 时变 lookback

  ⚠ 静态属性在 DL 路径里**会在每个时间步重复同样的值**(把 (B, L, C) 的
    L 维填满), 这是必须的 — 否则它们在 (B, L, C) 张量里没法表达.

y 顺序 / Target-major flatten:
  forward 输出 / fit 输入的 y 都按 **target-major, step-minor** 组织:
    y[:] = [income_h0, income_h1, ..., income_h5,
            spending_h0, spending_h1, ..., spending_h5]
  这与 DLinear forward 末尾 `out[:, :T, :].view(B, -1)` 的内存顺序一致.
  传错顺序训练照样收敛, 但每个目标对不齐 → silent failure. 见 pack_y().

运行方式 / Run:
    python pipelines/examples/panel_income_spending_example.py
"""

from __future__ import annotations

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ─── 让 src-layout 包可被 import ──────────────────────────────────────────────
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
for _p in (_HERE.parent, *_HERE.parents):
    if (_p / 'configs').is_dir() and (_p / 'src').is_dir():
        for _q in (_p, _p / 'src'):
            if str(_q) not in sys.path:
                sys.path.insert(0, str(_q))
        break

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 项目复用 / Project utilities
from tsf_frame.visualization import PredictionPlotter  # 触发中文字体 rcParams
from tsf_frame.visualization.base_visualizer import _CN_FONTS  # noqa: F401
from tsf_frame.utils.logger import get_logger
from tsf_frame.utils.metrics import MetricsCalculator
from tsf_frame.models.classical.ml_models import LightGBMModel
from tsf_frame.models.transformer.transformer_models import DLinear

logger = get_logger(name='panel_demo')

# ─── 全局常量 ────────────────────────────────────────────────────────────────
N_INDIVIDUALS = 100
N_MONTHS      = 36
LOOKBACK      = 12       # 用过去 12 个月
HORIZON       = 6        # 预测未来 6 个月
N_TARGETS     = 2        # 收入 + 花销
TRAIN_RATIO   = 0.80     # 80 个个体训练
VAL_RATIO     = 0.10     # 10 个个体验证 (LightGBM 残差法的 OOS 拟合)
                         # 余下 0.10 测试
SEED          = 42

# 行业列表 / Industry categories
INDUSTRIES = ['IT', '金融', '教育', '医疗', '制造']
# 各行业月收入基线倍率 (相对中位数), 用于让数据有结构差异
# / Industry baseline multipliers — adds structure to synthetic data
INDUSTRY_MULT = {'IT': 1.4, '金融': 1.6, '教育': 0.9,
                 '医疗': 1.1, '制造': 0.8}
# 性别基线 (差异故意做小, 真实场景也应避免大歧视性差距)
GENDER_MULT = {'男': 1.05, '女': 1.00}


# ============================================================================
# 1. 合成数据 / Synthesize panel data
# ============================================================================

def generate_panel_data(seed: int = SEED) -> pd.DataFrame:
    """
    生成 N_INDIVIDUALS × N_MONTHS 行的 long-format 面板数据.

    每个个体内的时序由以下叠加而来:
      baseline   = 行业系数 × 性别系数 × 1 万元
      trend      = 月度 0.3% 增长 (代表通胀/职业晋升)
      seasonal   = 12 月余弦, 12 月有 ~5% 的"年终奖"凸起
      AR(1)      = 0.4 * (上月偏离均值)
      noise      = 高斯, σ ~ baseline 的 8%
    花销 spending = 0.55 * income + season_food(12 月+消费季) + noise
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_INDIVIDUALS):
        # 静态属性 / Static attributes per individual
        height = float(rng.normal(170, 8))           # 身高
        gender = rng.choice(['男', '女'])
        industry = rng.choice(INDUSTRIES,
                              p=[0.25, 0.20, 0.15, 0.15, 0.25])

        baseline = (10000.0
                    * INDUSTRY_MULT[industry]
                    * GENDER_MULT[gender])

        # 时序 / Time series
        income = np.zeros(N_MONTHS)
        spending = np.zeros(N_MONTHS)
        income[0] = baseline + rng.normal(0, baseline * 0.05)
        for t in range(1, N_MONTHS):
            trend = baseline * (1.0 + 0.003) ** t
            season_inc = baseline * 0.05 * np.cos(2 * np.pi * t / 12)
            year_end = baseline * 0.5 if (t % 12 == 11) else 0.0  # 年终奖
            ar = 0.4 * (income[t - 1] - baseline * (1.0 + 0.003) ** (t - 1))
            noise = rng.normal(0, baseline * 0.08)
            income[t] = trend + season_inc + year_end + ar + noise

        # 花销: 与收入正相关, 带独立季节
        for t in range(N_MONTHS):
            base_spend = 0.55 * income[t]
            season_sp = baseline * 0.04 * np.cos(2 * np.pi * (t - 6) / 12)
            shop_peak = baseline * 0.30 if (t % 12 == 10) else 0.0  # 双 11
            spending[t] = (base_spend + season_sp + shop_peak
                           + rng.normal(0, baseline * 0.04))

        for t in range(N_MONTHS):
            rows.append({
                'person_id': i,
                'month_idx': t,
                'height':    height,
                'gender':    gender,
                'industry':  industry,
                'income':    float(income[t]),
                'spending':  float(spending[t]),
            })
    return pd.DataFrame(rows)


# ============================================================================
# 2. 编码 + 归一化 (训练集统计) / Encode and standardize
# ============================================================================

def encode_static(df: pd.DataFrame, *,
                  fit: bool, ref: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    OneHot + zscore 静态特征.
    fit=True: 用本 df 拟合 (gender/industry 类别表 + height 均值方差)
    fit=False: 用 ref 里的统计量 transform.
    """
    if fit:
        gender_cats   = sorted(df['gender'].unique())
        industry_cats = sorted(df['industry'].unique())
        height_mu     = float(df['height'].mean())
        height_sd     = float(df['height'].std() or 1.0)
        ref = {'gender_cats': gender_cats,
               'industry_cats': industry_cats,
               'height_mu': height_mu, 'height_sd': height_sd}
    else:
        assert ref is not None
        gender_cats   = ref['gender_cats']
        industry_cats = ref['industry_cats']
        height_mu     = ref['height_mu']
        height_sd     = ref['height_sd']

    out = pd.DataFrame(index=df.index)
    out['height_z'] = (df['height'] - height_mu) / height_sd
    # OneHot, drop_first 减少冗余维度
    for cat in gender_cats[1:]:
        out[f'gender_{cat}'] = (df['gender'] == cat).astype(float)
    for cat in industry_cats[1:]:
        out[f'industry_{cat}'] = (df['industry'] == cat).astype(float)
    return out, ref


def standardize_targets(df: pd.DataFrame, *, fit: bool,
                        stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    收入/花销 zscore (训练集统计). 模型在标准化空间训练,
    评估时 inverse_transform 回原尺度算 MAPE 才有可解释性.
    """
    if fit:
        stats = {
            'income_mu':  float(df['income'].mean()),
            'income_sd':  float(df['income'].std() or 1.0),
            'spending_mu': float(df['spending'].mean()),
            'spending_sd': float(df['spending'].std() or 1.0),
        }
    assert stats is not None
    out = df.copy()
    out['income_z']   = (df['income'] - stats['income_mu']) / stats['income_sd']
    out['spending_z'] = (df['spending'] - stats['spending_mu']) / stats['spending_sd']
    return out, stats


def inv_targets(arr: np.ndarray, stats: dict, target: str) -> np.ndarray:
    """zscore → 原尺度 / Inverse zscore."""
    mu = stats[f'{target}_mu']
    sd = stats[f'{target}_sd']
    return arr * sd + mu


# ============================================================================
# 3. 滑窗 + 路径专属特征布局 / Window construction
# ============================================================================

def build_windows(df: pd.DataFrame, static_cols: list[str],
                  ref_target_stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对每个 person_id 内部按时间滑窗.

    Returns:
        X3d   : (N, L, C_in)   DL 路径直接用; ML 路径 flatten 后再用
        X_ml  : (N, F)         ML 路径用的 flatten 特征 (静态只放一次, 时变 lookback)
        y_flat: (N, T*H)       两路径共用的 y, **target-major** flatten
    """
    L, H, T = LOOKBACK, HORIZON, N_TARGETS
    static_arrs: list[np.ndarray] = []
    timev_seqs: list[np.ndarray] = []
    y_seqs: list[np.ndarray] = []
    for pid, g in df.groupby('person_id'):
        g = g.sort_values('month_idx').reset_index(drop=True)
        n = len(g)
        # 静态: 每人一份, 后面再按需要重复到时间步
        static_vec = g[static_cols].iloc[0].values.astype(float)  # (C_static,)
        # 时变: 每月一行, (n, 2) — income_z, spending_z
        timev = g[['income_z', 'spending_z']].values.astype(float)
        for start in range(n - L - H + 1):
            x_time = timev[start: start + L]                   # (L, 2)
            # DL 路径: 把静态在每个时间步重复, 拼到时变右侧 → (L, C_static + 2)
            x_full = np.concatenate(
                [np.tile(static_vec, (L, 1)), x_time], axis=1)
            # y: 未来 H 步 (income_z, spending_z), 拆成两条再 target-major 拼接
            y_inc  = timev[start + L: start + L + H, 0]        # (H,)
            y_spe  = timev[start + L: start + L + H, 1]        # (H,)
            y_flat = np.concatenate([y_inc, y_spe])             # (T*H,)

            timev_seqs.append(x_full)
            static_arrs.append(static_vec)
            y_seqs.append(y_flat)

    X3d = np.stack(timev_seqs, axis=0).astype(np.float32)        # (N, L, C_in)
    static_mat = np.stack(static_arrs, axis=0).astype(np.float32)
    # ML flatten: 静态只放一次 + 时变 (L 步 × 2 目标) flatten
    timev_flat = X3d[:, :, len(static_cols):].reshape(X3d.shape[0], -1)
    X_ml = np.concatenate([static_mat, timev_flat], axis=1)
    y = np.stack(y_seqs, axis=0).astype(np.float32)
    return X3d, X_ml, y


def split_individuals(df: pd.DataFrame, seed: int = SEED
                      ) -> tuple[list[int], list[int], list[int]]:
    """按 person_id 切分 — 训练/验证/测试**完全不重叠**, 防止个体级泄漏."""
    rng = np.random.default_rng(seed)
    ids = np.arange(N_INDIVIDUALS)
    rng.shuffle(ids)
    n_tr = int(N_INDIVIDUALS * TRAIN_RATIO)
    n_va = int(N_INDIVIDUALS * VAL_RATIO)
    return ids[:n_tr].tolist(), ids[n_tr:n_tr + n_va].tolist(), ids[n_tr + n_va:].tolist()


# ============================================================================
# 4. 评估工具 / Evaluation
# ============================================================================

def reshape_to_TH(arr_flat: np.ndarray) -> np.ndarray:
    """(N, T*H) → (N, T, H)  按 target-major 解读."""
    N = arr_flat.shape[0]
    return arr_flat.reshape(N, N_TARGETS, HORIZON)


def evaluate(y_true_flat: np.ndarray, y_pred_flat: np.ndarray,
             stats: dict, label: str,
             y_lower_flat: np.ndarray | None = None,
             y_upper_flat: np.ndarray | None = None) -> dict:
    """
    返回 dict, 同时打印分目标的 MAPE / 区间覆盖率(PICP).
    所有指标都在**原尺度**上算, 才有"百分误差"的可解释含义.
    """
    yt = reshape_to_TH(y_true_flat)
    yp = reshape_to_TH(y_pred_flat)
    # 反归一化
    yt_inc = inv_targets(yt[:, 0, :], stats, 'income')
    yp_inc = inv_targets(yp[:, 0, :], stats, 'income')
    yt_spe = inv_targets(yt[:, 1, :], stats, 'spending')
    yp_spe = inv_targets(yp[:, 1, :], stats, 'spending')

    mape_inc = MetricsCalculator.mape(yt_inc, yp_inc)
    mape_spe = MetricsCalculator.mape(yt_spe, yp_spe)
    mape_all = MetricsCalculator.mape(
        np.concatenate([yt_inc, yt_spe]),
        np.concatenate([yp_inc, yp_spe]),
    )
    rmse_inc = MetricsCalculator.rmse(yt_inc, yp_inc)
    rmse_spe = MetricsCalculator.rmse(yt_spe, yp_spe)

    out = {'mape_income': mape_inc, 'mape_spending': mape_spe,
           'mape_overall': mape_all,
           'rmse_income': rmse_inc, 'rmse_spending': rmse_spe}

    # 概率区间覆盖率 (PICP) — 真值落在 [lower, upper] 内的比例
    if y_lower_flat is not None and y_upper_flat is not None:
        lo = reshape_to_TH(y_lower_flat)
        hi = reshape_to_TH(y_upper_flat)
        lo_inc = inv_targets(lo[:, 0, :], stats, 'income')
        hi_inc = inv_targets(hi[:, 0, :], stats, 'income')
        lo_spe = inv_targets(lo[:, 1, :], stats, 'spending')
        hi_spe = inv_targets(hi[:, 1, :], stats, 'spending')
        picp_inc = float(np.mean((yt_inc >= lo_inc) & (yt_inc <= hi_inc)))
        picp_spe = float(np.mean((yt_spe >= lo_spe) & (yt_spe <= hi_spe)))
        miw_inc = float(np.mean(hi_inc - lo_inc))
        miw_spe = float(np.mean(hi_spe - lo_spe))
        out.update({'picp_income': picp_inc, 'picp_spending': picp_spe,
                    'miw_income': miw_inc, 'miw_spending': miw_spe})

    logger.info(f'  [{label}]')
    logger.info(f'    收入  MAPE={mape_inc:.2%}  RMSE={rmse_inc:.0f} 元')
    logger.info(f'    花销  MAPE={mape_spe:.2%}  RMSE={rmse_spe:.0f} 元')
    logger.info(f'    总体  MAPE={mape_all:.2%}')
    if 'picp_income' in out:
        logger.info(f'    收入区间  PICP={out["picp_income"]:.1%}  '
                    f'宽度均值={out["miw_income"]:.0f} 元')
        logger.info(f'    花销区间  PICP={out["picp_spending"]:.1%}  '
                    f'宽度均值={out["miw_spending"]:.0f} 元')
    return out


# ============================================================================
# 5. 主流程 / Main
# ============================================================================

def main() -> None:
    logger.info('=' * 70)
    logger.info(' Panel Income & Spending — ML(LightGBM) vs DL(DLinear)')
    logger.info('=' * 70)

    # ── 5.1 数据生成 ─────────────────────────────────────────────────────────
    logger.info('[1/6] 合成 100 个体 × 36 个月面板数据...')
    df_raw = generate_panel_data()
    logger.info(f'  形状: {df_raw.shape}, 字段: {list(df_raw.columns)}')

    # ── 5.2 划分 (按个体, 防止泄漏) ──────────────────────────────────────────
    logger.info('[2/6] 按 person_id 切分 (训练/验证/测试)...')
    tr_ids, va_ids, te_ids = split_individuals(df_raw)
    logger.info(f'  训练个体={len(tr_ids)}  验证个体={len(va_ids)}  '
                f'测试个体={len(te_ids)}')
    df_tr = df_raw[df_raw['person_id'].isin(tr_ids)].copy()
    df_va = df_raw[df_raw['person_id'].isin(va_ids)].copy()
    df_te = df_raw[df_raw['person_id'].isin(te_ids)].copy()

    # ── 5.3 编码 + 归一化 (用训练集统计, 严格防泄漏) ─────────────────────────
    logger.info('[3/6] OneHot + zscore (用训练集统计)...')
    s_tr, ref_static = encode_static(df_tr, fit=True)
    s_va, _ = encode_static(df_va, fit=False, ref=ref_static)
    s_te, _ = encode_static(df_te, fit=False, ref=ref_static)
    static_cols = list(s_tr.columns)
    df_tr = pd.concat([df_tr.reset_index(drop=True),
                       s_tr.reset_index(drop=True)], axis=1)
    df_va = pd.concat([df_va.reset_index(drop=True),
                       s_va.reset_index(drop=True)], axis=1)
    df_te = pd.concat([df_te.reset_index(drop=True),
                       s_te.reset_index(drop=True)], axis=1)

    df_tr, target_stats = standardize_targets(df_tr, fit=True)
    df_va, _ = standardize_targets(df_va, fit=False, stats=target_stats)
    df_te, _ = standardize_targets(df_te, fit=False, stats=target_stats)
    logger.info(f'  静态列 ({len(static_cols)}): {static_cols}')
    logger.info(f'  目标 zscore 统计: '
                f"income μ={target_stats['income_mu']:.0f} σ={target_stats['income_sd']:.0f}"
                f"  spending μ={target_stats['spending_mu']:.0f} σ={target_stats['spending_sd']:.0f}")

    # ── 5.4 构造滑窗 ─────────────────────────────────────────────────────────
    logger.info('[4/6] 构造滑窗 (lookback=12, horizon=6, target-major y)...')
    X3d_tr, Xml_tr, y_tr = build_windows(df_tr, static_cols, target_stats)
    X3d_va, Xml_va, y_va = build_windows(df_va, static_cols, target_stats)
    X3d_te, Xml_te, y_te = build_windows(df_te, static_cols, target_stats)
    C_in = X3d_tr.shape[2]
    logger.info(f'  X3d (DL):  {X3d_tr.shape}  (B, L={LOOKBACK}, C_in={C_in})')
    logger.info(f'  Xml (ML):  {Xml_tr.shape}  flatten 后 = '
                f'{len(static_cols)} 静态 + {LOOKBACK}*{N_TARGETS} 时变')
    logger.info(f'  y      :  {y_tr.shape}  target-major flat = '
                f'{N_TARGETS}*{HORIZON} = {N_TARGETS * HORIZON}')

    # ════════════════════════════════════════════════════════════════════════
    # 路径 A — LightGBM (ML, flatten 特征 + 残差法 OOS 区间)
    # ════════════════════════════════════════════════════════════════════════
    logger.info('=' * 70)
    logger.info('[5/6] 路径 A — LightGBM (ML)')
    logger.info('=' * 70)
    ml_model = LightGBMModel(config={
        'model_name': 'lgb_panel',
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_seed': SEED,
        'probabilistic': True,
        'probabilistic_method': 'residual',
        'confidence_level': 0.95,
    })
    # val_data 喂给 fit, 走我们刚修过的 OOS 残差路径
    ml_model.fit(train_data=(Xml_tr, y_tr), val_data=(Xml_va, y_va))
    logger.info(f'  ✓ 训练完成. 残差来源: '
                f'{getattr(ml_model, "_residual_source", "unknown")} '
                f'(val=验证集 OOS, 区间更可信)')
    ml_pred = ml_model.predict_probabilistic(Xml_te)
    logger.info('  → 测试集评估:')
    ml_metrics = evaluate(
        y_true_flat=y_te,
        y_pred_flat=ml_pred.mean,
        stats=target_stats,
        label='LightGBM 测试集',
        y_lower_flat=ml_pred.lower,
        y_upper_flat=ml_pred.upper,
    )

    # ════════════════════════════════════════════════════════════════════════
    # 路径 B — DLinear (DL, 3D + 多目标多步 + 分位数)
    # ════════════════════════════════════════════════════════════════════════
    logger.info('=' * 70)
    logger.info('[5/6] 路径 B — DLinear (DL, 多目标多步分位数模式)')
    logger.info('=' * 70)
    dl_model = DLinear(config={
        'model_name': 'dlinear_panel',
        'input_size':  C_in,             # 6 静态 + 2 时变 = 8
        'num_targets': N_TARGETS,        # 取前 2 个 channel 做目标 (= 时变 channel)
        'pred_len':    HORIZON,
        'seq_len':     LOOKBACK,
        'moving_avg_kernel': 5,          # 短窗下 25 太大, 改 5
        'probabilistic_method': 'quantile',
        'quantiles':  [0.025, 0.5, 0.975],
        'confidence_level': 0.95,
        'train_epochs': 60,
        'batch_size':   64,
        'learning_rate': 0.01,
        'device': 'cpu',
    })
    # ⚠ 关键: DLinear channel-independent 架构下,
    #   `num_targets=2` 取的是 input 的**前 2 个 channel** 做目标.
    #   build_windows 把静态放在前面、时变 (income, spending) 放在后面 →
    #   这里**前 2 channel 是静态**, 不是目标 channel! 训练会失效.
    #   必须把目标 channel 放在 input 的前面.
    # 修正: 重新组织 X 让前 2 个 channel = (income_z, spending_z), 静态在后面.
    def reorder_targets_first(X3d: np.ndarray, n_static: int) -> np.ndarray:
        # 原顺序: [static_0..static_{S-1}, income_z, spending_z]
        # 新顺序: [income_z, spending_z, static_0..static_{S-1}]
        S = n_static
        return np.concatenate([X3d[:, :, S:], X3d[:, :, :S]], axis=2)

    n_static = len(static_cols)
    X3d_tr_dl = reorder_targets_first(X3d_tr, n_static)
    X3d_va_dl = reorder_targets_first(X3d_va, n_static)
    X3d_te_dl = reorder_targets_first(X3d_te, n_static)
    logger.info(f'  channel 顺序已重排: 前 {N_TARGETS} 个 = [income_z, spending_z], '
                f'后 {n_static} 个 = 静态属性')

    # _DLBaseModel.fit 走 _dl_fit
    dl_model.fit(train_data=(X3d_tr_dl, y_tr), val_data=(X3d_va_dl, y_va))

    dl_prob = dl_model.predict_probabilistic(X3d_te_dl)
    logger.info('  → 测试集评估:')
    dl_metrics = evaluate(
        y_true_flat=y_te,
        y_pred_flat=dl_prob.mean,
        stats=target_stats,
        label='DLinear 测试集',
        y_lower_flat=dl_prob.lower,
        y_upper_flat=dl_prob.upper,
    )

    # ════════════════════════════════════════════════════════════════════════
    # 6. 并排对比 + 可视化
    # ════════════════════════════════════════════════════════════════════════
    logger.info('=' * 70)
    logger.info('[6/6] 并排对比')
    logger.info('=' * 70)
    rows = [
        ('收入 MAPE',     ml_metrics['mape_income'],     dl_metrics['mape_income']),
        ('花销 MAPE',     ml_metrics['mape_spending'],   dl_metrics['mape_spending']),
        ('总体 MAPE',     ml_metrics['mape_overall'],    dl_metrics['mape_overall']),
        ('收入 RMSE (元)', ml_metrics['rmse_income'],     dl_metrics['rmse_income']),
        ('花销 RMSE (元)', ml_metrics['rmse_spending'],   dl_metrics['rmse_spending']),
        ('收入 PICP',     ml_metrics.get('picp_income', float('nan')),
                         dl_metrics.get('picp_income', float('nan'))),
        ('花销 PICP',     ml_metrics.get('picp_spending', float('nan')),
                         dl_metrics.get('picp_spending', float('nan'))),
        ('收入区间宽度 (元)', ml_metrics.get('miw_income', float('nan')),
                            dl_metrics.get('miw_income', float('nan'))),
        ('花销区间宽度 (元)', ml_metrics.get('miw_spending', float('nan')),
                            dl_metrics.get('miw_spending', float('nan'))),
    ]
    logger.info(f'  {"指标":<22} {"LightGBM (ML)":>18} {"DLinear (DL)":>18}')
    logger.info('  ' + '-' * 60)
    for name, vml, vdl in rows:
        if 'MAPE' in name or 'PICP' in name:
            logger.info(f'  {name:<22} {vml:>17.2%} {vdl:>17.2%}')
        else:
            logger.info(f'  {name:<22} {vml:>18.0f} {vdl:>18.0f}')

    # ── 可视化: 取测试集第 0 个个体的"收入"轨迹 ─────────────────────────────
    out_dir = _Path(__file__).resolve().parents[2] / 'logs' / 'outputs' / 'panel_demo'
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_one_individual(
        y_true_flat=y_te[0],
        ml_mean=ml_pred.mean[0], ml_lo=ml_pred.lower[0], ml_hi=ml_pred.upper[0],
        dl_mean=dl_prob.mean[0], dl_lo=dl_prob.lower[0], dl_hi=dl_prob.upper[0],
        stats=target_stats,
        save_path=str(out_dir / 'panel_compare_individual0.png'),
    )
    plot_metric_bars(rows, save_path=str(out_dir / 'panel_compare_metrics.png'))
    logger.info(f'\n  输出图保存到: {out_dir}')
    logger.info('=' * 70)


# ============================================================================
# 可视化 / Plotting
# ============================================================================

def plot_one_individual(y_true_flat: np.ndarray,
                        ml_mean: np.ndarray, ml_lo: np.ndarray, ml_hi: np.ndarray,
                        dl_mean: np.ndarray, dl_lo: np.ndarray, dl_hi: np.ndarray,
                        stats: dict, save_path: str) -> None:
    """画测试集首个个体的"收入"和"花销"未来 6 步预测 + 95% 区间."""
    yt = y_true_flat.reshape(N_TARGETS, HORIZON)
    mlm = ml_mean.reshape(N_TARGETS, HORIZON)
    mll = ml_lo.reshape(N_TARGETS, HORIZON)
    mlh = ml_hi.reshape(N_TARGETS, HORIZON)
    dlm = dl_mean.reshape(N_TARGETS, HORIZON)
    dll = dl_lo.reshape(N_TARGETS, HORIZON)
    dlh = dl_hi.reshape(N_TARGETS, HORIZON)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    targets = ['income', 'spending']
    titles = ['收入 income (元/月)', '花销 spending (元/月)']
    x = np.arange(1, HORIZON + 1)
    for k, (tgt, ax) in enumerate(zip(targets, axes)):
        yt_o = inv_targets(yt[k], stats, tgt)
        ml_m = inv_targets(mlm[k], stats, tgt)
        ml_l = inv_targets(mll[k], stats, tgt)
        ml_h = inv_targets(mlh[k], stats, tgt)
        dl_m = inv_targets(dlm[k], stats, tgt)
        dl_l = inv_targets(dll[k], stats, tgt)
        dl_h = inv_targets(dlh[k], stats, tgt)
        ax.plot(x, yt_o, 'k-o', label='真值', linewidth=2)
        ax.plot(x, ml_m, 'C0--s', label='LightGBM mean')
        ax.fill_between(x, ml_l, ml_h, alpha=0.15, color='C0',
                        label='LightGBM 95% 残差区间')
        ax.plot(x, dl_m, 'C3--^', label='DLinear mean')
        ax.fill_between(x, dl_l, dl_h, alpha=0.15, color='C3',
                        label='DLinear 95% 分位数区间')
        ax.set_xlabel('未来第 N 个月')
        ax.set_ylabel('元/月')
        ax.set_title(titles[k])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle('测试集第 0 个个体: 未来 6 个月预测对比', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_metric_bars(rows, save_path: str) -> None:
    """两路径的关键指标并排柱状图."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左: MAPE / PICP 比例类
    pct_rows = [(n, m, d) for (n, m, d) in rows if 'MAPE' in n or 'PICP' in n]
    names = [r[0] for r in pct_rows]
    ml_v = [r[1] * 100 for r in pct_rows]
    dl_v = [r[2] * 100 for r in pct_rows]
    xpos = np.arange(len(names))
    w = 0.36
    axes[0].bar(xpos - w / 2, ml_v, w, label='LightGBM', color='C0')
    axes[0].bar(xpos + w / 2, dl_v, w, label='DLinear', color='C3')
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_ylabel('百分比 (%)')
    axes[0].set_title('误差 / 区间覆盖率')
    axes[0].legend(); axes[0].grid(alpha=0.3, axis='y')

    # 右: 元为单位的指标
    yuan_rows = [(n, m, d) for (n, m, d) in rows
                 if ('RMSE' in n) or ('宽度' in n)]
    names = [r[0] for r in yuan_rows]
    ml_v = [r[1] for r in yuan_rows]
    dl_v = [r[2] for r in yuan_rows]
    xpos = np.arange(len(names))
    axes[1].bar(xpos - w / 2, ml_v, w, label='LightGBM', color='C0')
    axes[1].bar(xpos + w / 2, dl_v, w, label='DLinear', color='C3')
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(names, rotation=20)
    axes[1].set_ylabel('元')
    axes[1].set_title('误差 / 区间宽度 (绝对值)')
    axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
