"""
日志配置工具
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "financial_data",
    level: int = logging.INFO,
    log_file: str = None,
    console: bool = True,
) -> logging.Logger:
    """
    配置日志记录器

    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径
        console: 是否输出到控制台

    Returns:
        Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    if console and not any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    ):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
