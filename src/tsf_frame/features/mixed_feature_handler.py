"""
混合特征处理模块 / Mixed feature handling module

将时序特征（随时间变化）和静态特征（不随时间变化）合并处理，
生成可供深度学习模型使用的滑动窗口序列。
Combines time-varying features and static features, producing
sliding-window sequences for deep learning models.

设计模式（类比 sklearn Scaler）:
──────────────────────────────────────────────────────────────
  Scaler                    MixedFeatureHandler
  ──────                    ────────────────────
  fit(train)                fit(train_df)
    → 学习 mean/std            → 提取静态特征值 static_values
  transform(any_data)       transform(any_df)
    → 归一化                   → 合并时序+静态特征，返回 (X_2d, y_1d)
  fit_transform(train)      fit_transform(train_df)
    → fit + transform          → fit + transform

  额外方法（Scaler 没有的）:
  create_sequences(df)      → transform + 滑动窗口，返回 (X_3d, y_2d)
                               适用场景: 训练/测试（批量数据）
  create_single_input(df)   → 取最近 seq_len 行，拼接特征，返回 (1, seq_len, F)
                               适用场景: 生产预测（实时单条推理）
──────────────────────────────────────────────────────────────

静态特征说明:
  静态特征 = 在整个时间序列中保持不变的属性（如个人性别、企业行业类型）。
  仅适用于 "个体×时间" 的面板数据场景。
  若所有列都随时间变化（如全市公积金月报），则 static_cols 传空列表 []。
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union


class MixedFeatureHandler:
    """
    混合特征处理器 / Mixed feature handler

    将静态特征在每个时间步重复拼接到时序特征后，
    然后按滑动窗口切分为 (X_seq, y_seq) 序列对。
    Repeats static features at each time step, concatenates with
    time-varying features, then slices into (X_seq, y_seq) pairs.

    使用模式（类比 sklearn Scaler）:
        训练阶段:  handler.fit(train_df)  → 提取静态值
                   X, y = handler.create_sequences(train_df)
        测试阶段:  X, y = handler.create_sequences(test_df)   # 不要重新 fit
        生产阶段:  X = handler.create_single_input(recent_df)  # 最近 seq_len 行

    Args:
        time_varying_cols: 时序特征列名 / Time-varying feature column names
        static_cols: 静态特征列名（不随时间变化的列，如性别/行业）。
                     如果没有静态特征，传空列表 []。
        target_col: 目标列名 / Target column name
        seq_len: 输入序列长度 / Input sequence length
        pred_len: 预测长度 / Prediction horizon
        label_len: Decoder 启动长度，仅 Informer/Autoformer 等 Encoder-Decoder
                   模型需要。默认 0 对普通模型透明。
    """
    def __init__(self,
                 time_varying_cols: List[str],
                 static_cols: List[str],
                 target_col: str,
                 seq_len: int,
                 pred_len: int = 1,
                 label_len: int = 0):
        self.time_varying_cols = time_varying_cols
        self.static_cols = static_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.static_values = None       # fit() 后填充
        self._is_fitted = False         # 标记是否已 fit

    def fit(self, df: pd.DataFrame) -> 'MixedFeatureHandler':
        """
        从训练集提取静态特征值（类比 scaler.fit 只用训练集）。

        静态特征按定义在整个数据集中保持不变,因此只需任取一行即可。
        使用 dropna() 保证取到完整行,避免首行因上游差分/滚动窗口产生的 NaN。

        Args:
            df: 训练集 DataFrame（必须包含 static_cols 中的列）。

        Returns:
            self（支持链式调用: handler.fit(df).create_sequences(df)）
        """
        if self.static_cols:
            static_df = df[self.static_cols].dropna()
            if len(static_df) == 0:
                raise ValueError(
                    f"静态特征列 {self.static_cols} 全部为 NaN,无法提取静态值。\n"
                    f"Static columns {self.static_cols} are all NaN; cannot extract values."
                )
            # 取首个完整(非 NaN)行作为静态特征值
            self.static_values = static_df.iloc[0].values
        self._is_fitted = True
        return self

    def _check_fitted(self):
        """检查是否已 fit，未 fit 时给出明确提示。"""
        if not self._is_fitted:
            raise RuntimeError(
                "MixedFeatureHandler 尚未 fit。"
                "请先调用 handler.fit(train_df) 提取静态特征值。\n"
                "Handler has not been fitted. Call handler.fit(train_df) first."
            )

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        合并时序特征与静态特征（类比 scaler.transform，可用于任意数据集）。

        策略: 把 static_values 在每个时间步重复拼接到时序特征右侧,形成 2D 宽矩阵。

        Args:
            df: 任意阶段的 DataFrame（训练/测试/生产皆可）。

        Returns:
            (X, y) 元组:
              X: shape (N, n_time + n_static) — 合并后的 2D 特征矩阵
              y: shape (N,) — 目标序列
        """
        self._check_fitted()

        X_time = df[self.time_varying_cols].values

        if self.static_cols and self.static_values is not None:
            X_static = np.tile(self.static_values, (len(df), 1))
            X = np.concatenate([X_time, X_static], axis=1)
        else:
            X = X_time

        y = df[self.target_col].values

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        fit + transform 一步到位（仅用于训练集）。

        等价于:
            handler.fit(df)
            X, y = handler.transform(df)

        Returns:
            (X, y): 2D 合并特征矩阵 + 目标数组
        """
        self.fit(df)
        return self.transform(df)

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成滑动窗口序列对（训练/测试阶段使用）。

        内部流程: transform(df) → 滑动窗口切片 → 3D 序列
        训练/测试都可直接调用，但必须先 fit 过训练集。

        label_len 说明
        ─────────────────────────────────────────────────────────
        │←── seq_len ──→│←─label_len─→│←── pred_len ──→│
              X_seq          ↑               y_seq
                        decoder 启动段    （含启动段 + 预测段）
        ─────────────────────────────────────────────────────────
        当 label_len=0 时，y_seq 仅包含 pred_len 步。

        Args:
            df: 训练集或测试集 DataFrame。

        Returns:
            (X_seq, y_seq):
              X_seq: shape (N, seq_len, n_features)    — Encoder 输入
              y_seq: shape (N, label_len + pred_len)   — 目标序列
        """
        total_len = self.seq_len + self.pred_len
        if len(df) < total_len:
            raise ValueError(
                f"数据长度 {len(df)} 不足，至少需要 seq_len({self.seq_len}) "
                f"+ pred_len({self.pred_len}) = {total_len} 行。\n"
                f"Data length {len(df)} < required {total_len}."
            )

        X, y = self.transform(df)

        X_seq, y_seq = [], []
        label_len = self.label_len

        for i in range(len(X) - self.seq_len - self.pred_len + 1):
            X_seq.append(X[i: i + self.seq_len])
            y_start = i + self.seq_len - label_len
            y_end = i + self.seq_len + self.pred_len
            y_seq.append(y[y_start: y_end])

        return np.array(X_seq), np.array(y_seq)

    def create_single_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        生产环境专用：将最近 seq_len 行数据转为单个模型输入。

        与 create_sequences 的区别:
          - create_sequences: 批量数据 → 多个窗口 (N, seq_len, F)，含目标 y
          - create_single_input: 最近 seq_len 行 → 单个窗口 (1, seq_len, F)，无目标 y

        Args:
            df: 最近的数据 DataFrame，至少 seq_len 行。
                如果超过 seq_len 行，只取最后 seq_len 行。

        Returns:
            X_input: shape (1, seq_len, n_features) — 可直接喂给 model.predict()
        """
        self._check_fitted()

        if len(df) < self.seq_len:
            raise ValueError(
                f"生产预测需要至少 {self.seq_len} 行数据，当前只有 {len(df)} 行。\n"
                f"Production prediction requires at least {self.seq_len} rows, "
                f"got {len(df)}."
            )

        # 只取最后 seq_len 行
        recent = df.iloc[-self.seq_len:]

        X_time = recent[self.time_varying_cols].values

        if self.static_cols and self.static_values is not None:
            X_static = np.tile(self.static_values, (self.seq_len, 1))
            X = np.concatenate([X_time, X_static], axis=1)
        else:
            X = X_time

        return X.reshape(1, self.seq_len, -1)


class AdvancedMixedFeatureHandler:
    """
    高级混合特征处理器（分离式）/ Advanced mixed feature handler

    与 MixedFeatureHandler 不同，此处理器将时序特征和静态特征分开返回，
    适用于具有双输入分支的模型架构（一支处理时序、另一支处理静态）。

    使用模式:
        训练阶段:  handler.fit(train_df)
                   X_time, X_static, y = handler.create_sequences(train_df)
        测试阶段:  X_time, X_static, y = handler.create_sequences(test_df)
        生产阶段:  X_time, X_static = handler.create_single_input(recent_df)
    """
    def __init__(self,
                 time_varying_cols: List[str],
                 static_cols: List[str],
                 target_col: str,
                 seq_len: int,
                 pred_len: int = 1):
        self.time_varying_cols = time_varying_cols
        self.static_cols = static_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.static_values = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'AdvancedMixedFeatureHandler':
        """
        提取静态特征值（同 MixedFeatureHandler.fit）。

        Returns:
            self（支持链式调用）
        """
        if self.static_cols:
            static_df = df[self.static_cols].dropna()
            if len(static_df) == 0:
                raise ValueError(
                    f"静态特征列 {self.static_cols} 全部为 NaN。\n"
                    f"Static columns {self.static_cols} are all NaN."
                )
            self.static_values = static_df.iloc[0].values
        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "AdvancedMixedFeatureHandler 尚未 fit。"
                "请先调用 handler.fit(train_df)。"
            )

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成时序/静态分离的序列（训练/测试阶段使用）。

        Args:
            df: 输入 DataFrame

        Returns:
            三元组（顺序固定）:
              X_time_seq:   (N, seq_len, n_time)   时序特征序列
              X_static_seq: (N, n_static)          静态向量（每样本相同）
              y_seq:        (N, pred_len)          目标序列

            当 static_cols 为空时，X_static_seq 为 (N, 0) 空数组。
        """
        self._check_fitted()

        if len(df) < self.seq_len + self.pred_len:
            raise ValueError(
                f"数据长度 {len(df)} 不足,至少需要 {self.seq_len + self.pred_len} 行。\n"
                f"Data length {len(df)} < required {self.seq_len + self.pred_len}."
            )

        X_time = df[self.time_varying_cols].values
        y = df[self.target_col].values

        X_time_seq, y_seq = [], []
        for i in range(len(X_time) - self.seq_len - self.pred_len + 1):
            X_time_seq.append(X_time[i:i + self.seq_len])
            y_seq.append(y[i + self.seq_len:i + self.seq_len + self.pred_len])

        X_time_seq = np.array(X_time_seq)
        y_seq = np.array(y_seq)

        if self.static_cols and self.static_values is not None:
            X_static_seq = np.tile(self.static_values, (len(X_time_seq), 1))
        else:
            X_static_seq = np.empty((len(X_time_seq), 0))

        return X_time_seq, X_static_seq, y_seq

    def create_single_input(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        生产环境专用：将最近 seq_len 行数据转为双分支模型输入。

        Args:
            df: 最近的数据 DataFrame，至少 seq_len 行。

        Returns:
            (X_time_input, X_static_input):
              X_time_input:   (1, seq_len, n_time)   时序分支输入
              X_static_input: (1, n_static)           静态分支输入
        """
        self._check_fitted()

        if len(df) < self.seq_len:
            raise ValueError(
                f"生产预测需要至少 {self.seq_len} 行数据，当前只有 {len(df)} 行。"
            )

        recent = df.iloc[-self.seq_len:]
        X_time = recent[self.time_varying_cols].values.reshape(1, self.seq_len, -1)

        if self.static_cols and self.static_values is not None:
            X_static = self.static_values.reshape(1, -1)
        else:
            X_static = np.empty((1, 0))

        return X_time, X_static


# ─── 便捷函数 ─────────────────────────────────────────────────────────────────

def prepare_mixed_features(df: pd.DataFrame,
                           time_varying_cols: List[str],
                           static_cols: List[str],
                           target_col: str,
                           seq_len: int,
                           pred_len: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：一步完成 fit + create_sequences。

    注意: 此函数不返回 handler 对象，因此无法复用于测试/生产阶段。
    如需跨阶段使用，请直接创建 MixedFeatureHandler 实例。
    """
    handler = MixedFeatureHandler(
        time_varying_cols=time_varying_cols,
        static_cols=static_cols,
        target_col=target_col,
        seq_len=seq_len,
        pred_len=pred_len
    )
    handler.fit(df)
    return handler.create_sequences(df)


# ─── 示例数据 ─────────────────────────────────────────────────────────────────

def create_mixed_feature_example():
    """
    生成模拟面板数据：跟踪一个人的收入变化。
    - age, income: 时序特征（随时间变化）
    - gender, education, industry: 静态特征（不变）
    """
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'age': np.linspace(25, 27, 100),
        'income': 5000 + np.cumsum(np.random.randn(100) * 100),
        'gender': np.random.choice([0, 1], 100),
        'education': np.random.choice([0, 1, 2], 100),
        'industry': np.random.choice([0, 1, 2, 3], 100)
    })
    df.set_index('date', inplace=True)

    # 静态特征：全列赋同一个值（模拟"这个人"的不变属性）
    df['gender'] = df['gender'].iloc[0]
    df['education'] = df['education'].iloc[0]
    df['industry'] = df['industry'].iloc[0]

    return df


# ─── 完整三阶段示例 ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 80)
    print("MixedFeatureHandler 三阶段完整示例")
    print("训练 → 测试 → 生产预测")
    print("=" * 80)

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║  0. 准备数据                                                     ║
    # ╚════════════════════════════════════════════════════════════════════╝
    df = create_mixed_feature_example()
    print(f"\n[数据] 总行数: {len(df)}")
    print(df.head(3))

    # 划分：前 80 行训练，中间 10 行测试，最后 10 行模拟"生产新数据"
    train_df = df.iloc[:80]
    test_df = df.iloc[80:90]
    production_df = df.iloc[90:]
    print(f"\n[划分] 训练: {len(train_df)} 行, 测试: {len(test_df)} 行, "
          f"生产: {len(production_df)} 行")

    # 特征定义
    time_varying_cols = ['age', 'income']
    static_cols = ['gender', 'education', 'industry']
    target_col = 'income'
    seq_len = 7
    pred_len = 1

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║  1. 训练阶段 — fit + create_sequences                           ║
    # ╚════════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 80)
    print("阶段 1: 训练")
    print("=" * 80)

    handler = MixedFeatureHandler(
        time_varying_cols=time_varying_cols,
        static_cols=static_cols,
        target_col=target_col,
        seq_len=seq_len,
        pred_len=pred_len,
    )

    # ★ fit 只在训练集上调用一次，类比 scaler.fit(train_data)
    handler.fit(train_df)
    print(f"\n[fit] 提取的静态特征值: {handler.static_values}")
    print(f"       对应列名: {static_cols}")

    # ★ create_sequences 生成 3D 训练序列
    X_train, y_train = handler.create_sequences(train_df)
    print(f"\n[create_sequences] X_train shape: {X_train.shape}")
    print(f"                    y_train shape: {y_train.shape}")
    print(f"  → 每个样本: {seq_len} 步 × {X_train.shape[-1]} 特征"
          f"({len(time_varying_cols)} 时序 + {len(static_cols)} 静态)")

    # 假设模型训练: model.fit((X_train, y_train))
    print("\n  [模型训练] model.fit((X_train, y_train)) ← 此处省略具体模型")

    # ★ 保存 handler（生产环境必须）
    # import joblib
    # joblib.dump(handler, 'handler.pkl')
    print("  [保存] handler 需要与模型一起持久化 (joblib.dump)")

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║  2. 测试阶段 — 不要重新 fit，直接 create_sequences              ║
    # ╚════════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 80)
    print("阶段 2: 测试")
    print("=" * 80)

    # ★ 重点：不调用 handler.fit(test_df)！沿用训练时的 static_values
    # handler = joblib.load('handler.pkl')  # 生产中从文件加载

    # 测试集可能不够 seq_len + pred_len，需要拼接一点训练尾部数据
    # 实际中测试集通常足够长，这里因为只有 10 行需要特殊处理
    test_with_context = pd.concat([train_df.iloc[-(seq_len - 1):], test_df])
    print(f"\n[拼接上下文] 测试数据补齐: {len(test_with_context)} 行"
          f"（测试 {len(test_df)} + 上下文 {seq_len - 1}）")

    X_test, y_test = handler.create_sequences(test_with_context)
    print(f"\n[create_sequences] X_test shape: {X_test.shape}")
    print(f"                    y_test shape: {y_test.shape}")

    # 假设模型预测: y_pred = model.predict(X_test)
    y_pred = y_test * 1.05   # 模拟预测结果
    mae = np.mean(np.abs(y_pred.flatten() - y_test.flatten()))
    print(f"\n  [模型评估] MAE = {mae:.4f} (模拟值)")

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║  3. 生产阶段 — create_single_input (单窗口推理)                  ║
    # ╚════════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 80)
    print("阶段 3: 生产预测")
    print("=" * 80)

    # 模拟：从数据库获取最近 seq_len 条记录
    recent_data = production_df.iloc[-seq_len:]  # 最近 7 行
    print(f"\n[获取最新数据] 最近 {len(recent_data)} 行:")
    print(recent_data)

    # ★ create_single_input: 生产环境专用方法
    X_input = handler.create_single_input(recent_data)
    print(f"\n[create_single_input] X_input shape: {X_input.shape}")
    print(f"  → 可直接喂给 model.predict(X_input)")

    # 假设模型预测: y_next = model.predict(X_input)
    y_next = 5500.0   # 模拟预测值
    print(f"\n  [预测结果] 下一步预测值: {y_next:.2f}")

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║  对比: 与 transform() 的关系                                     ║
    # ╚════════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 80)
    print("附录: transform() vs create_sequences() vs create_single_input()")
    print("=" * 80)

    X_2d, y_1d = handler.transform(train_df)
    print(f"\n  transform(df)          → X: {X_2d.shape} (2D), y: {y_1d.shape} (1D)")
    print(f"  create_sequences(df)   → X: {X_train.shape} (3D), y: {y_train.shape} (2D)")
    print(f"  create_single_input(df)→ X: {X_input.shape} (3D, batch=1, 无 y)")
    print()
    print("  transform:           底层方法，拼接特征，返回 2D（一般不直接用）")
    print("  create_sequences:    训练/测试用，批量滑窗，返回 3D + 目标")
    print("  create_single_input: 生产用，单窗口，返回 3D，无目标")

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║  方案 2: AdvancedMixedFeatureHandler（双分支模型）                ║
    # ╚════════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 80)
    print("方案 2: AdvancedMixedFeatureHandler（双分支模型）")
    print("=" * 80)

    adv_handler = AdvancedMixedFeatureHandler(
        time_varying_cols=time_varying_cols,
        static_cols=static_cols,
        target_col=target_col,
        seq_len=seq_len,
        pred_len=pred_len,
    )

    # 训练
    adv_handler.fit(train_df)
    X_time, X_static, y = adv_handler.create_sequences(train_df)
    print(f"\n  [训练] X_time: {X_time.shape}, X_static: {X_static.shape}, y: {y.shape}")

    # 生产
    X_time_prod, X_static_prod = adv_handler.create_single_input(recent_data)
    print(f"  [生产] X_time: {X_time_prod.shape}, X_static: {X_static_prod.shape}")
    print(f"         → model.predict(X_time_prod, X_static_prod)")

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║  方案 3: 纯时序场景（无静态特征，如公积金月报）                   ║
    # ╚════════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 80)
    print("方案 3: 纯时序（无静态特征，static_cols=[]）")
    print("=" * 80)

    # 公积金月报场景：每个字段每月都在变，没有静态特征
    pure_ts_handler = MixedFeatureHandler(
        time_varying_cols=['age', 'income'],   # 全部都是时序
        static_cols=[],                         # ★ 空列表，无静态特征
        target_col='income',
        seq_len=seq_len,
        pred_len=pred_len,
    )
    pure_ts_handler.fit(train_df)
    X_pure, y_pure = pure_ts_handler.create_sequences(train_df)
    X_pure_prod = pure_ts_handler.create_single_input(recent_data)

    print(f"\n  [训练] X: {X_pure.shape} (无静态特征，只有 {len(['age', 'income'])} 列)")
    print(f"  [生产] X: {X_pure_prod.shape}")
    print(f"\n  → static_cols=[] 时，handler 退化为纯滑动窗口工具，效果等同于手写循环")

    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)
