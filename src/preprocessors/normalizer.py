"""
数据标准化模块
支持：Min-Max标准化、Z-Score标准化、RobustScaler、Log变换
适用于金融数据中不同量纲特征的统一处理
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import json


logger = logging.getLogger(__name__)


class DataNormalizer:
    """数据标准化处理器"""

    def __init__(self):
        self.params_ = {}  # 存储标准化参数（用于逆变换和新数据一致性处理）

    def minmax_scale(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        feature_range: tuple = (0, 1),
    ) -> pd.DataFrame:
        """
        Min-Max 标准化

        X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min

        Args:
            df: 输入DataFrame
            columns: 标准化列，None表示所有数值列
            feature_range: 目标范围

        Returns:
            标准化后的DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in df.columns:
                continue

            xmin = df[col].min()
            xmax = df[col].max()

            if xmax == xmin:
                logger.warning(f"列 '{col}' 最大值=最小值，无法进行Min-Max标准化，跳过")
                df[col] = feature_range[0]
                continue

            df[col] = (
                (df[col] - xmin) / (xmax - xmin)
                * (feature_range[1] - feature_range[0])
                + feature_range[0]
            )

            self.params_[col] = {
                "method": "minmax",
                "min": float(xmin),
                "max": float(xmax),
                "feature_range": list(feature_range),
            }
            logger.info(f"列 '{col}' Min-Max标准化完成，范围{feature_range}")

        return df

    def zscore_scale(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        with_mean: bool = True,
        with_std: bool = True,
    ) -> pd.DataFrame:
        """
        Z-Score 标准化（零均值单位方差）

        X_scaled = (X - mean) / std

        Args:
            df: 输入DataFrame
            columns: 标准化列
            with_mean: 是否减去均值
            with_std: 是否除以标准差

        Returns:
            标准化后的DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in df.columns:
                continue

            mean = df[col].mean() if with_mean else 0
            std = df[col].std() if with_std else 1

            if std == 0:
                logger.warning(f"列 '{col}' 标准差为0，跳过Z-Score标准化")
                df[col] = 0 if with_mean else df[col]
                continue

            df[col] = (df[col] - mean) / std

            self.params_[col] = {
                "method": "zscore",
                "mean": float(mean),
                "std": float(std),
                "with_mean": with_mean,
                "with_std": with_std,
            }
            logger.info(f"列 '{col}' Z-Score标准化完成")

        return df

    def robust_scale(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        quantile_range: tuple = (25.0, 75.0),
    ) -> pd.DataFrame:
        """
        Robust 标准化（基于分位数，对异常值鲁棒）

        X_scaled = (X - median) / IQR

        Args:
            df: 输入DataFrame
            columns: 标准化列
            quantile_range: 分位数范围

        Returns:
            标准化后的DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in df.columns:
                continue

            median = df[col].median()
            q_low = df[col].quantile(quantile_range[0] / 100)
            q_high = df[col].quantile(quantile_range[1] / 100)
            iqr = q_high - q_low

            if iqr == 0:
                logger.warning(f"列 '{col}' IQR为0，跳过Robust标准化")
                df[col] = 0
                continue

            df[col] = (df[col] - median) / iqr

            self.params_[col] = {
                "method": "robust",
                "median": float(median),
                "iqr": float(iqr),
                "quantile_range": list(quantile_range),
            }
            logger.info(f"列 '{col}' Robust标准化完成")

        return df

    def log_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        base: str = "e",
        offset: float = 1.0,
    ) -> pd.DataFrame:
        """
        对数变换（适用于右偏分布，如收入、交易金额等）

        Args:
            df: 输入DataFrame
            columns: 变换列
            base: 对数底（e=自然对数, 10=常用对数, 2=二进制对数）
            offset: 偏移量（避免 log(0)）

        Returns:
            变换后的DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in df.columns:
                continue

            # 检查非正值
            if (df[col] < 0).any():
                logger.warning(f"列 '{col}' 包含负值，对数变换前自动偏移")
                min_val = df[col].min()
                offset = abs(min_val) + offset

            transformed = df[col] + offset
            if base == "e":
                df[col] = np.log(transformed)
            elif base == "10":
                df[col] = np.log10(transformed)
            elif base == "2":
                df[col] = np.log2(transformed)

            self.params_[col] = {
                "method": "log",
                "base": base,
                "offset": float(offset),
            }
            logger.info(f"列 '{col}' 对数变换完成（base={base}, offset={offset}）")

        return df

    def inverse_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        逆标准化（将标准化数据还原）

        Args:
            df: 标准化后的DataFrame
            columns: 指定列

        Returns:
            还原后的DataFrame
        """
        df = df.copy()
        cols = columns or [c for c in df.columns if c in self.params_]

        for col in cols:
            if col not in self.params_:
                logger.warning(f"列 '{col}' 无标准化参数，跳过逆变换")
                continue

            params = self.params_[col]
            method = params["method"]

            if method == "minmax":
                xmin, xmax = params["min"], params["max"]
                fr = params["feature_range"]
                df[col] = (df[col] - fr[0]) / (fr[1] - fr[0]) * (xmax - xmin) + xmin

            elif method == "zscore":
                mean = params["mean"] if params["with_mean"] else 0
                std = params["std"] if params["with_std"] else 1
                df[col] = df[col] * std + mean

            elif method == "robust":
                df[col] = df[col] * params["iqr"] + params["median"]

            elif method == "log":
                base = params["base"]
                offset = params["offset"]
                if base == "e":
                    df[col] = np.exp(df[col]) - offset
                elif base == "10":
                    df[col] = 10 ** df[col] - offset
                elif base == "2":
                    df[col] = 2 ** df[col] - offset

            logger.info(f"列 '{col}' 逆{method}变换完成")

        return df

    def save_params(self, filepath: str = "config/normalizer_params.json"):
        """保存标准化参数（用于生产环境一致性）"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.params_, f, ensure_ascii=False, indent=2)
        logger.info(f"标准化参数已保存到: {filepath}")

    def load_params(self, filepath: str = "config/normalizer_params.json"):
        """加载标准化参数"""
        with open(filepath, "r", encoding="utf-8") as f:
            self.params_ = json.load(f)
        logger.info(f"已加载标准化参数: {filepath}")
