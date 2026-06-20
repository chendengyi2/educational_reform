"""
数据加载工具
支持 CSV、Excel、JSON 等格式的自动识别与加载
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def load_csv(
    filepath: str,
    encoding: str = "utf-8",
    **kwargs,
) -> pd.DataFrame:
    """
    加载CSV文件

    Args:
        filepath: 文件路径
        encoding: 文件编码
        **kwargs: 传递给 pd.read_csv 的额外参数

    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    # 尝试常见中文编码
    for enc in [encoding, "utf-8", "gbk", "gb2312", "utf-8-sig"]:
        try:
            df = pd.read_csv(filepath, encoding=enc, **kwargs)
            logger.info(f"加载CSV: {filepath}，编码: {enc}，形状: {df.shape}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError(f"无法识别文件编码: {filepath}")


def load_excel(
    filepath: str,
    sheet_name: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    加载Excel文件

    Args:
        filepath: 文件路径
        sheet_name: 工作表名
        **kwargs: 传递给 pd.read_excel 的额外参数

    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    df = pd.read_excel(filepath, sheet_name=sheet_name or 0, **kwargs)
    logger.info(f"加载Excel: {filepath}，形状: {df.shape}")
    return df


def load_json(
    filepath: str,
    orient: str = "records",
    **kwargs,
) -> pd.DataFrame:
    """
    加载JSON文件

    Args:
        filepath: 文件路径
        orient: JSON格式方向
        **kwargs: 传递给 pd.read_json 的额外参数

    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    df = pd.read_json(filepath, orient=orient, **kwargs)
    logger.info(f"加载JSON: {filepath}，形状: {df.shape}")
    return df


def auto_load(filepath: str, **kwargs) -> pd.DataFrame:
    """
    根据文件扩展名自动选择加载方式

    Args:
        filepath: 文件路径

    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    loaders = {
        ".csv": load_csv,
        ".xlsx": load_excel,
        ".xls": load_excel,
        ".json": load_json,
    }

    if suffix not in loaders:
        raise ValueError(f"不支持的文件格式: {suffix}，支持: {list(loaders.keys())}")

    return loaders[suffix](filepath, **kwargs)
