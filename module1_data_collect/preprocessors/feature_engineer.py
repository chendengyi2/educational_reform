"""
特征工程模块
支持：时间特征提取、交叉特征、统计特征、编码转换、特征选择
针对财经数据场景设计
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Callable


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程处理器"""

    def __init__(self):
        self.encoders_ = {}     # 编码器映射
        self.feature_names_ = []  # 生成的新特征名

    # ========== 时间特征 ==========

    def extract_time_features(
        self,
        df: pd.DataFrame,
        time_col: str,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        从时间列提取特征

        Args:
            df: 输入DataFrame
            time_col: 时间列名
            features: 需要提取的特征列表，None表示全部

        Returns:
            添加时间特征后的DataFrame
        """
        df = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        all_features = [
            "year", "month", "day", "hour", "minute",
            "dayofweek", "quarter", "is_weekend",
            "is_month_start", "is_month_end",
            "weekofyear", "days_in_month",
        ]
        features = features or all_features

        dt = df[time_col].dt
        new_cols = {}

        feature_map = {
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "minute": dt.minute,
            "dayofweek": dt.dayofweek,
            "quarter": dt.quarter,
            "is_weekend": (dt.dayofweek >= 5).astype(int),
            "is_month_start": dt.is_month_start.astype(int),
            "is_month_end": dt.is_month_end.astype(int),
            "weekofyear": dt.isocalendar().week.astype(int),
            "days_in_month": dt.days_in_month,
        }

        for feat in features:
            if feat in feature_map:
                col_name = f"{time_col}_{feat}"
                new_cols[col_name] = feature_map[feat]
                logger.info(f"提取时间特征: {col_name}")

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 3, 5, 10, 20],
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        创建滞后特征（适用于股票价格等时间序列）

        Args:
            df: 输入DataFrame
            columns: 需要创建滞后特征的列
            lags: 滞后期数列表
            group_col: 分组列（如股票代码），确保组内滞后

        Returns:
            添加滞后特征后的DataFrame
        """
        df = df.copy()

        for col in columns:
            for lag in lags:
                new_col = f"{col}_lag{lag}"
                if group_col:
                    df[new_col] = df.groupby(group_col)[col].shift(lag)
                else:
                    df[new_col] = df[col].shift(lag)
                logger.info(f"创建滞后特征: {new_col}")

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20],
        stats: List[str] = ["mean", "std", "min", "max"],
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        创建滚动统计特征（适用于金融时间序列）

        Args:
            df: 输入DataFrame
            columns: 目标列
            windows: 滚动窗口大小列表
            stats: 统计量列表
            group_col: 分组列

        Returns:
            添加滚动特征后的DataFrame
        """
        df = df.copy()

        for col in columns:
            for window in windows:
                for stat in stats:
                    new_col = f"{col}_roll{window}_{stat}"

                    if group_col:
                        group = df.groupby(group_col)[col]
                    else:
                        group = df[col]

                    roller = group.rolling(window=window, min_periods=1)

                    if stat == "mean":
                        df[new_col] = roller.mean().reset_index(level=0, drop=True) if group_col else roller.mean()
                    elif stat == "std":
                        df[new_col] = roller.std().reset_index(level=0, drop=True) if group_col else roller.std()
                    elif stat == "min":
                        df[new_col] = roller.min().reset_index(level=0, drop=True) if group_col else roller.min()
                    elif stat == "max":
                        df[new_col] = roller.max().reset_index(level=0, drop=True) if group_col else roller.max()

                    logger.info(f"创建滚动特征: {new_col}")

        return df

    # ========== 交叉特征 ==========

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        column_pairs: List[tuple],
        operations: List[str] = None,
    ) -> pd.DataFrame:
        """
        创建交叉特征

        Args:
            df: 输入DataFrame
            column_pairs: 列对列表，如 [("col_a", "col_b")]
            operations: 操作列表（multiply, divide, add, subtract）

        Returns:
            添加交叉特征后的DataFrame
        """
        df = df.copy()
        operations = operations or ["multiply", "divide"]

        op_map = {
            "multiply": lambda a, b: a * b,
            "divide": lambda a, b: a / (b + 1e-8),  # 避免除零
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
        }

        for col_a, col_b in column_pairs:
            if col_a not in df.columns or col_b not in df.columns:
                logger.warning(f"列对 ({col_a}, {col_b}) 中存在不存在的列，跳过")
                continue

            for op in operations:
                if op not in op_map:
                    continue
                new_col = f"{col_a}_{op}_{col_b}"
                df[new_col] = op_map[op](df[col_a], df[col_b])
                logger.info(f"创建交叉特征: {new_col}")

        return df

    def create_ratio_features(
        self,
        df: pd.DataFrame,
        numerator: str,
        denominator: str,
        feature_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        创建比率特征（如 市盈率 = 股价/每股收益）

        Args:
            df: 输入DataFrame
            numerator: 分子列名
            denominator: 分母列名
            feature_name: 新特征名

        Returns:
            添加比率特征后的DataFrame
        """
        df = df.copy()
        name = feature_name or f"{numerator}_to_{denominator}"
        df[name] = df[numerator] / (df[denominator] + 1e-8)
        df[name] = df[name].replace([np.inf, -np.inf], np.nan)
        logger.info(f"创建比率特征: {name}")
        return df

    # ========== 编码 ==========

    def label_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> pd.DataFrame:
        """
        标签编码（适用于有序分类变量）

        Args:
            df: 输入DataFrame
            columns: 编码列

        Returns:
            编码后的DataFrame
        """
        df = df.copy()

        for col in columns:
            unique_vals = df[col].dropna().unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            df[col] = df[col].map(mapping)
            self.encoders_[col] = mapping
            logger.info(f"列 '{col}' 标签编码完成，{len(mapping)} 个类别")

        return df

    def onehot_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drop_first: bool = True,
    ) -> pd.DataFrame:
        """
        独热编码（适用于无序分类变量）

        Args:
            df: 输入DataFrame
            columns: 编码列
            drop_first: 是否删除第一个类别（避免共线性）

        Returns:
            编码后的DataFrame
        """
        df = df.copy()

        for col in columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            self.encoders_[col] = {"method": "onehot", "columns": dummies.columns.tolist()}
            logger.info(f"列 '{col}' 独热编码完成，生成 {len(dummies.columns)} 个新列")

        return df

    def target_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_col: str,
    ) -> pd.DataFrame:
        """
        目标编码（用目标变量的均值替代类别，适用于高基数分类变量）

        Args:
            df: 输入DataFrame
            columns: 编码列
            target_col: 目标列

        Returns:
            编码后的DataFrame
        """
        df = df.copy()
        global_mean = df[target_col].mean()

        for col in columns:
            encoding_map = df.groupby(col)[target_col].mean().to_dict()
            df[f"{col}_target_enc"] = df[col].map(encoding_map).fillna(global_mean)
            self.encoders_[col] = {"method": "target", "map": encoding_map, "global_mean": global_mean}
            logger.info(f"列 '{col}' 目标编码完成")

        return df

    # ========== 特征选择 ==========

    def select_features_by_correlation(
        self,
        df: pd.DataFrame,
        target_col: str,
        threshold: float = 0.05,
    ) -> List[str]:
        """
        基于相关性筛选特征

        Args:
            df: 输入DataFrame
            target_col: 目标列
            threshold: 相关性绝对值阈值

        Returns:
            筛选后的特征列名列表
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col not in numeric_df.columns:
            logger.error(f"目标列 '{target_col}' 不是数值型")
            return []

        corr = numeric_df.corr()[target_col].abs()
        selected = corr[corr >= threshold].index.tolist()
        selected = [c for c in selected if c != target_col]

        logger.info(f"相关性筛选: 选中 {len(selected)} 个特征（阈值={threshold}）")
        return selected

    def select_features_by_variance(
        self,
        df: pd.DataFrame,
        threshold: float = 0.01,
    ) -> List[str]:
        """
        基于方差筛选特征（过滤低方差特征）

        Args:
            df: 输入DataFrame
            threshold: 方差阈值

        Returns:
            方差大于阈值的特征列名列表
        """
        numeric_df = df.select_dtypes(include=[np.number])
        variances = numeric_df.var()
        selected = variances[variances > threshold].index.tolist()

        removed = len(numeric_df.columns) - len(selected)
        logger.info(f"方差筛选: 选中 {len(selected)} 个特征，移除 {removed} 个低方差特征")
        return selected
