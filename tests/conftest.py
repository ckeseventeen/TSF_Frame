"""pytest 配置：把项目根目录和 src/ 塞进 sys.path, 便于直接 `pytest` 无需安装。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / 'src'):
    s = str(_p)
    if _p.is_dir() and s not in sys.path:
        sys.path.insert(0, s)
