"""
缺失值处理模块
支持多种缺失值填充策略，适用于数值型、分类型、时间序列型数据
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List


logger = logging.getLogger(__name__)


class MissingValueHandler:
    """缺失值处理器"""

    def __init__(self):
        self.fill_values_ = {}  # 记录各列的填充值（用于后续数据的一致性处理）
        self.report_ = {}

    def analyze_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        分析数据缺失情况

        Args:
            df: 输入DataFrame

        Returns:
            缺失分析报告
        """
        total = len(df)
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / total * 100).round(2)

        report = pd.DataFrame({
            "列名": df.columns,
            "缺失数量": missing_count.values,
            "缺失比例(%)": missing_pct.values,
            "数据类型": df.dtypes.values,
            "非缺失数量": total - missing_count.values,
        })
        report = report[report["缺失数量"] > 0].sort_values(
            "缺失比例(%)", ascending=False
        ).reset_index(drop=True)

        self.report_ = report

        if not report.empty:
            logger.info(f"发现 {len(report)} 列存在缺失值")
            print(report.to_string(index=False))
        else:
            logger.info("数据无缺失值")

        return report

    def fill_constant(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        fill_value: Union[int, float, str] = 0,
    ) -> pd.DataFrame:
        """
        常量填充

        Args:
            df: 输入DataFrame
            columns: 指定列，None表示所有含缺失值的列
            fill_value: 填充值

        Returns:
            填充后的DataFrame
        """
        df = df.copy()
        cols = columns or df.columns[df.isnull().any()].tolist()

        for col in cols:
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)
                self.fill_values_[col] = fill_value
                logger.info(f"列 '{col}' 使用常量 {fill_value} 填充缺失值")

        return df

    def fill_mean(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        均值填充（适用于数值型列）

        Args:
            df: 输入DataFrame
            columns: 指定列

        Returns:
            填充后的DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if df[c].isnull().any()]

        for col in cols:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            self.fill_values_[col] = mean_val
            logger.info(f"列 '{col}' 使用均值 {mean_val:.4f} 填充缺失值")

        return df

    def fill_median(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        中位数填充（对异常值更鲁棒）

        Args:
            df: 输入DataFrame
            columns: 指定列

        Returns:
            填充后的DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if df[c].isnull().any()]

        for col in cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            self.fill_values_[col] = median_val
            logger.info(f"列 '{col}' 使用中位数 {median_val:.4f} 填充缺失值")

        return df

    def fill_mode(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        众数填充（适用于分类型列）

        Args:
            df: 输入DataFrame
            columns: 指定列

        Returns:
            填充后的DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
        cols = [c for c in cols if df[c].isnull().any()]

        for col in cols:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].fillna(mode_val)
            self.fill_values_[col] = mode_val
            logger.info(f"列 '{col}' 使用众数 '{mode_val}' 填充缺失值")

        return df

    def fill_forward(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        前向填充（适用于时间序列数据）

        Args:
            df: 输入DataFrame
            columns: 指定列

        Returns:
            填充后的DataFrame
        """
        df = df.copy()
        cols = columns or df.columns[df.isnull().any()].tolist()

        for col in cols:
            if col in df.columns:
                df[col] = df[col].ffill()
                logger.info(f"列 '{col}' 使用前向填充")

        # 剩余缺失值用后向填充
        df = df.bfill()

        return df

    def fill_interpolate(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "linear",
    ) -> pd.DataFrame:
        """
        插值填充（适用于时间序列）

        Args:
            df: 输入DataFrame
            columns: 指定列
            method: 插值方法（linear, quadratic, cubic, spline等）

        Returns:
            填充后的DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if df[c].isnull().any()]

        for col in cols:
            df[col] = df[col].interpolate(method=method)
            logger.info(f"列 '{col}' 使用 {method} 插值填充缺失值")

        return df

    def fill_group(
        self,
        df: pd.DataFrame,
        group_col: str,
        fill_method: str = "mean",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        分组填充（按某列分组后，组内使用统计量填充）

        Args:
            df: 输入DataFrame
            group_col: 分组依据列
            fill_method: 组内填充方法（mean, median, mode）
            columns: 需要填充的列

        Returns:
            填充后的DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if df[c].isnull().any()]

        for col in cols:
            if fill_method == "mean":
                fill_vals = df.groupby(group_col)[col].transform("mean")
            elif fill_method == "median":
                fill_vals = df.groupby(group_col)[col].transform("median")
            else:
                fill_vals = df.groupby(group_col)[col].transform(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                )

            df[col] = df[col].fillna(fill_vals)
            logger.info(f"列 '{col}' 使用分组{fill_method}填充缺失值（按 '{group_col}' 分组）")

        return df

    def drop_missing(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        axis: int = 1,
    ) -> pd.DataFrame:
        """
        删除缺失值过多的行/列

        Args:
            df: 输入DataFrame
            threshold: 缺失比例阈值（超过则删除），0~1
            axis: 0=删除行，1=删除列

        Returns:
            处理后的DataFrame
        """
        df = df.copy()

        if axis == 1:
            # 删除缺失比例超过阈值的列
            missing_pct = df.isnull().mean()
            drop_cols = missing_pct[missing_pct > threshold].index.tolist()
            df = df.drop(columns=drop_cols)
            if drop_cols:
                logger.info(f"删除缺失比例 >{threshold*100}% 的列: {drop_cols}")
        else:
            # 删除缺失比例超过阈值的行
            missing_pct = df.isnull().mean(axis=1)
            df = df[missing_pct <= threshold]
            logger.info(f"删除缺失比例 >{threshold*100}% 的行，剩余 {len(df)} 行")

        return df

    def auto_fill(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        drop_threshold: float = 0.8,
    ) -> pd.DataFrame:
        """
        自动填充：根据数据类型自动选择填充策略

        策略:
        - 缺失比例 > drop_threshold 的列直接删除
        - 数值型列: 使用 median（对异常值鲁棒）
        - 分类型列: 使用 mode
        - 时间序列列: 使用前向填充

        Args:
            df: 输入DataFrame
            numeric_strategy: 数值列填充策略
            categorical_strategy: 分类列填充策略
            drop_threshold: 列删除缺失比例阈值

        Returns:
            填充后的DataFrame
        """
        df = df.copy()

        # 1. 删除缺失过多的列
        df = self.drop_missing(df, threshold=drop_threshold, axis=1)

        # 2. 数值列填充
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols_missing = [c for c in num_cols if df[c].isnull().any()]
        if num_cols_missing:
            if numeric_strategy == "mean":
                df = self.fill_mean(df, columns=num_cols_missing)
            elif numeric_strategy == "median":
                df = self.fill_median(df, columns=num_cols_missing)
            elif numeric_strategy == "zero":
                df = self.fill_constant(df, columns=num_cols_missing, fill_value=0)

        # 3. 分类列填充
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols_missing = [c for c in cat_cols if df[c].isnull().any()]
        if cat_cols_missing:
            if categorical_strategy == "mode":
                df = self.fill_mode(df, columns=cat_cols_missing)
            elif categorical_strategy == "unknown":
                df = self.fill_constant(df, columns=cat_cols_missing, fill_value="unknown")

        # 4. 检查是否还有缺失
        remaining = df.isnull().sum().sum()
        if remaining > 0:
            logger.warning(f"自动填充后仍有 {remaining} 个缺失值，使用前向填充处理")
            df = self.fill_forward(df)

        logger.info("自动缺失值处理完成")
        return df
