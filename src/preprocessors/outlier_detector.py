"""
异常值检测模块
支持：IQR法、Z-Score法、孤立森林、基于分位数的方法
适用于金融数据中异常交易、异常价格等检测场景
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union


logger = logging.getLogger(__name__)


class OutlierDetector:
    """异常值检测器"""

    def __init__(self):
        self.bounds_ = {}       # 各列的异常值边界
        self.outlier_info_ = {}  # 各列的异常值统计信息

    def detect_iqr(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        factor: float = 1.5,
        return_mask: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, Tuple[pd.Series, dict]]]:
        """
        IQR（四分位距）法检测异常值

        异常值定义: < Q1 - factor*IQR 或 > Q3 + factor*IQR
        factor=1.5 为标准异常值，factor=3 为极端异常值

        Args:
            df: 输入DataFrame
            columns: 检测列，None表示所有数值列
            factor: IQR倍数因子
            return_mask: 是否返回布尔掩码

        Returns:
            异常值DataFrame 或 {列名: (掩码, 统计信息)} 字典
        """
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        results = {}

        for col in cols:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - factor * iqr
            upper = q3 + factor * iqr

            mask = (df[col] < lower) | (df[col] > upper)
            outlier_count = mask.sum()

            info = {
                "method": "IQR",
                "factor": factor,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(outlier_count),
                "outlier_pct": round(outlier_count / len(df) * 100, 2),
            }

            self.bounds_[col] = {"lower": lower, "upper": upper}
            self.outlier_info_[col] = info
            results[col] = (mask, info)

            logger.info(
                f"列 '{col}' IQR检测: 范围[{lower:.4f}, {upper:.4f}]，"
                f"异常值 {outlier_count} 个 ({info['outlier_pct']}%)"
            )

        if return_mask:
            mask_df = pd.DataFrame(
                {col: mask for col, (mask, _) in results.items()}, index=df.index
            )
            return mask_df

        return results

    def detect_zscore(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 3.0,
        return_mask: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, Tuple[pd.Series, dict]]]:
        """
        Z-Score 法检测异常值

        异常值定义: |Z| > threshold
        适用于近似正态分布的数据

        Args:
            df: 输入DataFrame
            columns: 检测列
            threshold: Z-Score阈值
            return_mask: 是否返回布尔掩码

        Returns:
            异常值检测结果
        """
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        results = {}

        for col in cols:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            mean = series.mean()
            std = series.std()

            if std == 0:
                logger.warning(f"列 '{col}' 标准差为0，跳过Z-Score检测")
                continue

            z_scores = (df[col] - mean) / std
            mask = z_scores.abs() > threshold
            outlier_count = mask.sum()

            info = {
                "method": "Z-Score",
                "threshold": threshold,
                "mean": mean,
                "std": std,
                "outlier_count": int(outlier_count),
                "outlier_pct": round(outlier_count / len(df) * 100, 2),
            }

            self.outlier_info_[col] = info
            results[col] = (mask, info)

            logger.info(
                f"列 '{col}' Z-Score检测: 阈值={threshold}，"
                f"异常值 {outlier_count} 个 ({info['outlier_pct']}%)"
            )

        if return_mask:
            mask_df = pd.DataFrame(
                {col: mask for col, (mask, _) in results.items()}, index=df.index
            )
            return mask_df

        return results

    def detect_isolation_forest(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> pd.Series:
        """
        孤立森林法检测异常值

        适用于高维数据和多变量异常检测

        Args:
            df: 输入DataFrame
            columns: 检测列
            contamination: 异常比例估计
            random_state: 随机种子

        Returns:
            异常值标签 Series（-1=异常, 1=正常）
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.error("需要 scikit-learn: pip install scikit-learn")
            return pd.Series(dtype=int)

        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[cols].copy()

        # 处理缺失值（孤立森林不支持NaN）
        data = data.fillna(data.median())

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        labels = iso_forest.fit_predict(data)
        result = pd.Series(labels, index=df.index, name="is_outlier")

        outlier_count = (labels == -1).sum()
        logger.info(
            f"孤立森林检测: 异常值 {outlier_count} 个 "
            f"({outlier_count / len(df) * 100:.2f}%)"
        )

        return result

    def clip_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        factor: float = 1.5,
    ) -> pd.DataFrame:
        """
        截断异常值（Winsorize）

        将异常值截断到边界值，而非删除

        Args:
            df: 输入DataFrame
            columns: 处理列
            method: 检测方法（iqr / quantile）
            factor: IQR因子（仅 method=iqr 时使用）

        Returns:
            处理后的DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in df.columns:
                continue

            if method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
            elif method == "quantile":
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
            else:
                continue

            clipped_count = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)

            self.bounds_[col] = {"lower": lower, "upper": upper}
            logger.info(
                f"列 '{col}' 截断异常值: 范围[{lower:.4f}, {upper:.4f}]，"
                f"截断 {clipped_count} 个值"
            )

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        factor: float = 1.5,
    ) -> pd.DataFrame:
        """
        删除异常值所在行

        Args:
            df: 输入DataFrame
            columns: 处理列
            method: 检测方法（iqr / zscore）
            factor: 检测参数

        Returns:
            处理后的DataFrame
        """
        if method == "iqr":
            results = self.detect_iqr(df, columns=columns, factor=factor)
        elif method == "zscore":
            results = self.detect_zscore(df, columns=columns, threshold=factor)
        else:
            logger.error(f"不支持的检测方法: {method}")
            return df

        # 合并所有列的异常掩码
        mask = pd.Series(False, index=df.index)
        for col, (col_mask, info) in results.items():
            mask = mask | col_mask

        removed = mask.sum()
        df_clean = df[~mask].reset_index(drop=True)
        logger.info(f"删除异常值行 {removed} 行，剩余 {len(df_clean)} 行")

        return df_clean

    def get_outlier_report(self) -> pd.DataFrame:
        """获取异常值检测报告"""
        if not self.outlier_info_:
            logger.warning("尚未执行异常值检测")
            return pd.DataFrame()

        report = pd.DataFrame(self.outlier_info_).T
        print(report.to_string())
        return report
