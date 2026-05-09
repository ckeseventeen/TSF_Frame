"""
日志管理模块 / Logger management module

基于 colorlog 提供统一的日志管理，支持控制台彩色输出和文件持久化。
采用单例模式确保全局共享同一个日志管理器实例。
Provides unified logging with colored console output (via colorlog) and
file persistence. Uses the Singleton pattern so all callers share one manager.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Optional
import colorlog


class LoggerManager:
    """
    单例日志管理器 / Singleton logger manager

    维护一个全局的 logger 注册表，避免重复创建 handler。
    Maintains a global logger registry to prevent duplicate handlers.
    """
    _instance = None
    _loggers = {}
    # 记录每个 logger 首次创建时的参数, 用于检测参数变化
    # / Args used to first-create each logger; for change detection
    _logger_args: Dict[str, tuple] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True

    def get_logger(self, name: str = 'tsf_frame',
                   log_dir: Optional[str] = None,
                   level: int = logging.INFO,
                   console: bool = True,
                   file: bool = True) -> logging.Logger:
        """
        获取或创建指定名称的 logger / Get or create a logger with the given name

        Args:
            name: logger 名称 / Logger name
            log_dir: 日志文件输出目录（为 None 则不写文件）/ Log file directory (None = no file output)
            level: 日志级别 / Logging level
            console: 是否输出到控制台 / Whether to output to console
            file: 是否输出到文件 / Whether to output to file

        Returns:
            配置好的 logging.Logger 实例 / Configured logging.Logger instance

        ⚠ 同名 logger 被第二次请求时, 若参数 (log_dir/level/console/file)
        与首次不同, 会发出 RuntimeWarning 提示用户 — 单例缓存返回的仍是
        **第一次的配置**. 想真正改配置请先 ``close_logger(name)`` 再重建.
        """
        # 当前调用的参数指纹 (用 None 表示日志目录默认)
        current_args = (log_dir, level, console, file)
        if name in self._loggers:
            # 命中缓存: 校验参数是否与首次创建一致
            cached_args = self._logger_args.get(name)
            if cached_args is not None and cached_args != current_args:
                import warnings
                warnings.warn(
                    f"LoggerManager: logger '{name}' 已用参数 {cached_args} 创建过, "
                    f"本次请求参数 {current_args} 被忽略 (返回缓存实例). "
                    f"如需用新参数, 请先调 close_logger('{name}').",
                    RuntimeWarning, stacklevel=3,
                )
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)
        # 关闭向 root logger 的冒泡,防止双重输出
        # Disable propagation to root to prevent duplicated output
        logger.propagate = False

        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level)
            logger.addHandler(console_handler)
        
        if file and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        
        self._loggers[name] = logger
        self._logger_args[name] = current_args
        return logger
    
    def close_logger(self, name: str):
        """关闭并移除指定 logger 的所有 handler / Close and remove all handlers for the named logger"""
        if name in self._loggers:
            logger = self._loggers[name]
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)
            del self._loggers[name]
            # 同步清掉参数指纹, 这样下次 get_logger(name, ...) 不会误触发警告
            self._logger_args.pop(name, None)
    
    def close_all(self):
        """关闭所有已注册的 logger / Close all registered loggers"""
        for name in list(self._loggers.keys()):
            self.close_logger(name)


def get_logger(name: str = 'tsf_frame', **kwargs) -> logging.Logger:
    """模块级便捷函数，委托给 LoggerManager 单例 / Module-level shortcut delegating to the LoggerManager singleton"""
    return LoggerManager().get_logger(name, **kwargs)
