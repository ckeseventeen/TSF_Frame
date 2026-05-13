"""
生产数据 IO —— 三个原子函数, CSV / Hive / MySQL 可切换

切换方式见 configs/hpf/data_source_config.py 顶部说明.

公开 API:
  fetch_data(sql_path, ..., config)            读取历史训练/推理输入
  save_predictions(task_name, df, ..., config) 落库预测结果
  fetch_last_prediction(task_id, ..., config)  查上月对本月的预测 (监控比对)

向后兼容别名 (旧代码用):
  fetch_hive_data = fetch_data
  save_to_mysql   = save_predictions
"""

from __future__ import annotations
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from configs.hpf.data_source_config import DataSourceConfig

logger = logging.getLogger('tsf_frame.production_loader')

# 进程级默认配置 —— 首次使用时按环境变量初始化, 也可通过 set_default_config() 覆盖.
_DEFAULT_CONFIG: Optional[DataSourceConfig] = None


def get_default_config() -> DataSourceConfig:
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = DataSourceConfig()
        logger.info(f"DataSourceConfig 已初始化: {_DEFAULT_CONFIG.to_dict()}")
    return _DEFAULT_CONFIG


def set_default_config(cfg: DataSourceConfig) -> None:
    """显式覆盖进程级默认配置 (通常在 main() 开头调一次)."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = cfg
    logger.info(f"DataSourceConfig 已显式设置: {cfg.to_dict()}")


# ── 1. 读取输入数据 ───────────────────────────────────────────────────────

def fetch_data(
    sql_path: str,
    data_path: Optional[str] = None,
    config: Optional[DataSourceConfig] = None,
) -> pd.DataFrame:
    """
    根据 config.source 从 CSV 或 Hive 读取历史数据.

    Args:
        sql_path:  SQL 模板路径 (Hive 模式实际执行; CSV 模式只用其 stem 找同名 csv)
        data_path: CSV 模式下显式覆盖路径 (None 时按 stem 在 csv_dir 下查找)
        config:    None 时用进程默认 (由环境变量初始化)

    Returns:
        DataFrame; 找不到数据时返回空 DataFrame (caller 应判 empty 跳过)
    """
    cfg = config or get_default_config()

    if cfg.source == 'csv':
        return _fetch_from_csv(sql_path, data_path, cfg)
    if cfg.source == 'hive':
        return _fetch_from_hive(sql_path, cfg)
    raise ValueError(f"未知 source: {cfg.source}")


def _fetch_from_csv(
    sql_path: str, data_path: Optional[str], cfg: DataSourceConfig,
) -> pd.DataFrame:
    if data_path is not None:
        csv_path = Path(data_path)
    else:
        # 约定: sql 文件 stem 即 csv 文件名
        # e.g. 'req_01_collection_amount.sql' → 'req_01_collection_amount.csv'
        csv_path = Path(cfg.csv_dir) / f"{Path(sql_path).stem}.csv"

    if not csv_path.exists():
        logger.warning(
            f"[CSV] 数据文件不存在: {csv_path}. "
            f"请在 {cfg.csv_dir} 下放置同名 csv, 或显式传 data_path."
        )
        return pd.DataFrame()

    logger.info(f"[CSV] 读取: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"[CSV] 加载 {len(df)} 行 × {len(df.columns)} 列")
    return df


def _fetch_from_hive(sql_path: str, cfg: DataSourceConfig) -> pd.DataFrame:
    try:
        from pyhive import hive  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Hive 模式需要 pyhive: pip install 'pyhive[hive]' thrift thrift-sasl"
        ) from e

    with open(sql_path, 'r', encoding='utf-8') as f:
        sql = f.read()

    logger.info(
        f"[Hive] 连接 {cfg.hive_host}:{cfg.hive_port}/{cfg.hive_database} "
        f"(auth={cfg.hive_auth})"
    )
    conn = hive.Connection(
        host=cfg.hive_host,
        port=cfg.hive_port,
        username=cfg.hive_username,
        password=cfg.hive_password if cfg.hive_auth != 'NONE' else None,
        database=cfg.hive_database,
        auth=cfg.hive_auth,
    )
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()
    logger.info(f"[Hive] 加载 {len(df)} 行 × {len(df.columns)} 列")
    return df


# ── 2. 预测结果落库 ───────────────────────────────────────────────────────

def save_predictions(
    task_name: str,
    df: pd.DataFrame,
    config: Optional[DataSourceConfig] = None,
    table_name: Optional[str] = None,
) -> str:
    """
    根据 config.sink 把预测结果写到 CSV 文件或 MySQL 表.

    Args:
        task_name:  任务标识 (e.g. 'REQ_01'), 用作 CSV 子目录或 MySQL 表名前缀
        df:         预测 DataFrame (索引应为 future_dates)
        config:     None 时用进程默认
        table_name: MySQL 模式下显式表名 (None 时用 'forecast_{task_name.lower()}')

    Returns:
        写入位置 (CSV 文件绝对路径 或 'mysql://<table>') —— 便于 caller 记录
    """
    cfg = config or get_default_config()

    if cfg.sink == 'csv':
        return _save_to_csv(task_name, df, cfg)
    if cfg.sink == 'mysql':
        return _save_to_mysql(task_name, df, cfg, table_name)
    raise ValueError(f"未知 sink: {cfg.sink}")


def _save_to_csv(task_name: str, df: pd.DataFrame, cfg: DataSourceConfig) -> str:
    out_dir = Path(cfg.output_dir) / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f"forecast_{timestamp}.csv"
    df.to_csv(out_path, index=True)
    logger.info(f"[CSV] 落盘 {len(df)} 行 → {out_path}")
    return str(out_path)


def _save_to_mysql(
    task_name: str, df: pd.DataFrame, cfg: DataSourceConfig,
    table_name: Optional[str],
) -> str:
    try:
        from sqlalchemy import create_engine  # type: ignore
    except ImportError as e:
        raise ImportError(
            "MySQL 模式需要 sqlalchemy + pymysql: pip install sqlalchemy pymysql"
        ) from e

    table = table_name or f"forecast_{task_name.lower()}"
    engine = create_engine(cfg.mysql_url)
    # 给落库行加 created_at 列, 便于查"上月预测"
    out = df.copy()
    out['created_at'] = datetime.now()
    out.to_sql(table, engine, if_exists='append', index=True, index_label='target_ts')
    logger.info(f"[MySQL] 落库 {len(df)} 行 → {table}")
    return f"mysql://{table}"


# ── 3. 查上月对本月的预测 (监控比对用) ───────────────────────────────────

def fetch_last_prediction(
    task_id: str,
    target_col: str,
    config: Optional[DataSourceConfig] = None,
) -> Optional[float]:
    """
    返回上月跑批时对"本月"做出的预测值 (即上次预测序列的第一个时间步).
    找不到时返回 None — caller 据此跳过监控比对.

    CSV 模式: 读 output_dir/<task_id>/forecast_*.csv 中倒数第二份 (最后一份是当次跑批刚写的)
    MySQL 模式: 按 created_at 倒序取上一次跑批结果
    """
    cfg = config or get_default_config()

    if cfg.sink == 'csv':
        return _fetch_last_prediction_csv(task_id, target_col, cfg)
    if cfg.sink == 'mysql':
        return _fetch_last_prediction_mysql(task_id, target_col, cfg)
    raise ValueError(f"未知 sink: {cfg.sink}")


def _fetch_last_prediction_csv(
    task_id: str, target_col: str, cfg: DataSourceConfig,
) -> Optional[float]:
    out_dir = Path(cfg.output_dir) / task_id
    if not out_dir.exists():
        return None
    files = sorted(out_dir.glob('forecast_*.csv'))
    # 月度跑批正常顺序: 上次跑批写过一份, 本次跑批走到这里时还没写,
    # 所以 files[-1] 就是上月那份; 但为了健壮起见,
    # 如果本次已写 (files >= 2), 取倒数第二份.
    if not files:
        return None
    pick = files[-2] if len(files) >= 2 else files[-1]
    try:
        df = pd.read_csv(pick, index_col=0, parse_dates=True)
    except Exception as e:
        logger.warning(f"[CSV] 读取 {pick} 失败: {e}")
        return None
    if target_col not in df.columns or df.empty:
        return None
    # 上月预测的第 1 行 = 上月对本月的预测
    return float(df[target_col].iloc[0])


def _fetch_last_prediction_mysql(
    task_id: str, target_col: str, cfg: DataSourceConfig,
) -> Optional[float]:
    try:
        from sqlalchemy import create_engine, text  # type: ignore
    except ImportError as e:
        raise ImportError(
            "MySQL 模式需要 sqlalchemy + pymysql: pip install sqlalchemy pymysql"
        ) from e

    table = f"forecast_{task_id.lower()}"
    engine = create_engine(cfg.mysql_url)
    try:
        # 取上一次跑批 (created_at 倒序第二个 batch) 的第一个 target_ts 行
        sql = text(
            f"""
            SELECT {target_col}
              FROM {table}
             WHERE created_at = (
                SELECT DISTINCT created_at FROM {table}
                 ORDER BY created_at DESC LIMIT 1 OFFSET 1
             )
             ORDER BY target_ts ASC
             LIMIT 1
            """
        )
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        return float(row[0]) if row else None
    except Exception as e:
        logger.warning(f"[MySQL] 查询上月预测失败: {e}")
        return None


# ── 向后兼容别名 ─────────────────────────────────────────────────────────
fetch_hive_data = fetch_data
save_to_mysql = save_predictions
