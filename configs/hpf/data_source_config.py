"""
生产数据源配置 —— CSV 兜底 / Hive 真接入二选一

切换方式 (优先级: 显式构造 > 环境变量 > 默认值):
  1. 代码里显式: DataSourceConfig(source='hive', hive_host='...')
  2. 环境变量:   TSF_HPF_SOURCE=hive
                TSF_HPF_HIVE_HOST=hive-prod-01
                TSF_HPF_MYSQL_URL=mysql+pymysql://user:pw@host:3306/db
  3. 默认:      source='csv', 从 data/hpf/<sql_stem>.csv 读,
                结果落到 logs/outputs/<task_id>/forecast_*.csv

依赖:
  - CSV 模式:   仅 pandas (默认已安装)
  - Hive 模式:  pip install 'pyhive[hive]' thrift
  - MySQL 模式: pip install sqlalchemy pymysql
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 项目根: configs/hpf/data_source_config.py → parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _root(*parts: str) -> str:
    return str(_PROJECT_ROOT.joinpath(*parts))


@dataclass
class DataSourceConfig:
    """生产数据源配置 —— 三个核心 IO (读取/落库/查上月预测) 的统一开关."""

    # ── 输入数据源 ───────────────────────────────────────────────────────
    # 'csv' (本地兜底, 默认) 或 'hive' (生产真接入)
    source: str = field(
        default_factory=lambda: os.getenv('TSF_HPF_SOURCE', 'csv').lower()
    )

    # CSV 模式: 数据根目录, 默认按 sql 文件名找同名 csv
    # 例: sql_path='configs/hpf/sql_templates/req_01_collection_amount.sql'
    #     → csv_dir + 'req_01_collection_amount.csv'
    csv_dir: str = field(
        default_factory=lambda: os.getenv('TSF_HPF_CSV_DIR', _root('data', 'hpf'))
    )

    # ── 结果落库 ─────────────────────────────────────────────────────────
    # 'csv' 写 output_dir/<task>/forecast_<timestamp>.csv
    # 'mysql' 写 mysql_url 指定库 (需要 sqlalchemy + pymysql)
    sink: str = field(
        default_factory=lambda: os.getenv('TSF_HPF_SINK', 'csv').lower()
    )

    output_dir: str = field(
        default_factory=lambda: os.getenv('TSF_HPF_OUTPUT_DIR', _root('logs', 'outputs'))
    )

    # ── Hive 连接 (source='hive' 时使用) ────────────────────────────────
    hive_host: Optional[str] = field(
        default_factory=lambda: os.getenv('TSF_HPF_HIVE_HOST')
    )
    hive_port: int = field(
        default_factory=lambda: int(os.getenv('TSF_HPF_HIVE_PORT', '10000'))
    )
    hive_username: Optional[str] = field(
        default_factory=lambda: os.getenv('TSF_HPF_HIVE_USER')
    )
    hive_password: Optional[str] = field(
        default_factory=lambda: os.getenv('TSF_HPF_HIVE_PASSWORD')
    )
    hive_database: str = field(
        default_factory=lambda: os.getenv('TSF_HPF_HIVE_DATABASE', 'default')
    )
    # 'NONE' / 'NOSASL' / 'LDAP' / 'KERBEROS' / 'CUSTOM'
    hive_auth: str = field(
        default_factory=lambda: os.getenv('TSF_HPF_HIVE_AUTH', 'NONE')
    )

    # ── MySQL 连接 (sink='mysql' 时使用) ────────────────────────────────
    # 形如: mysql+pymysql://user:pw@host:3306/db
    mysql_url: Optional[str] = field(
        default_factory=lambda: os.getenv('TSF_HPF_MYSQL_URL')
    )

    def __post_init__(self) -> None:
        if self.source not in ('csv', 'hive'):
            raise ValueError(
                f"source 必须是 'csv' 或 'hive', 收到: {self.source!r}"
            )
        if self.sink not in ('csv', 'mysql'):
            raise ValueError(
                f"sink 必须是 'csv' 或 'mysql', 收到: {self.sink!r}"
            )
        if self.source == 'hive' and not self.hive_host:
            raise ValueError(
                "source='hive' 但 hive_host 未配置. "
                "请设置环境变量 TSF_HPF_HIVE_HOST 或显式传入."
            )
        if self.sink == 'mysql' and not self.mysql_url:
            raise ValueError(
                "sink='mysql' 但 mysql_url 未配置. "
                "请设置环境变量 TSF_HPF_MYSQL_URL 或显式传入."
            )

    def to_dict(self) -> dict:
        d = {
            'source': self.source, 'sink': self.sink,
            'csv_dir': self.csv_dir, 'output_dir': self.output_dir,
        }
        if self.source == 'hive':
            d.update({
                'hive_host': self.hive_host, 'hive_port': self.hive_port,
                'hive_database': self.hive_database, 'hive_auth': self.hive_auth,
                # 不打印密码
            })
        if self.sink == 'mysql':
            # mysql_url 含密码, 打印时遮掩
            d['mysql_url'] = self._mask_url(self.mysql_url)
        return d

    @staticmethod
    def _mask_url(url: Optional[str]) -> Optional[str]:
        if not url or '@' not in url:
            return url
        head, tail = url.rsplit('@', 1)
        if '://' in head:
            scheme, creds = head.split('://', 1)
            return f"{scheme}://***@{tail}"
        return f"***@{tail}"
