"""
数据预处理流水线
将缺失值处理、异常值检测、标准化、特征工程串联为可配置的流水线
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from .missing_handler import MissingValueHandler
from .outlier_detector import OutlierDetector
from .normalizer import DataNormalizer
from .feature_engineer import FeatureEngineer


logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """数据预处理流水线"""

    DEFAULT_CONFIG = {
        "missing_value": {
            "enabled": True,
            "strategy": "auto",
            "numeric_strategy": "median",
            "categorical_strategy": "mode",
            "drop_threshold": 0.8,
        },
        "outlier": {
            "enabled": True,
            "method": "iqr",
            "action": "clip",
            "factor": 1.5,
            "columns": None,
        },
        "normalization": {
            "enabled": True,
            "method": "robust",
            "columns": None,
        },
        "feature_engineering": {
            "enabled": True,
            "time_features": None,
            "lag_features": None,
            "rolling_features": None,
            "encode_columns": None,
            "encode_method": "label",
        },
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 流水线配置，None使用默认配置
        """
        self.config = self._merge_config(config or {})
        self.missing_handler = MissingValueHandler()
        self.outlier_detector = OutlierDetector()
        self.normalizer = DataNormalizer()
        self.feature_engineer = FeatureEngineer()
        self.step_log = []

    def _merge_config(self, user_config: Dict) -> Dict:
        """合并用户配置与默认配置"""
        config = {}
        for key, default_val in self.DEFAULT_CONFIG.items():
            if key in user_config:
                if isinstance(default_val, dict):
                    config[key] = {**default_val, **user_config[key]}
                else:
                    config[key] = user_config[key]
            else:
                config[key] = default_val
        return config

    def _log_step(self, step_name: str, shape_before: tuple, shape_after: tuple, details: str = ""):
        """记录流水线步骤日志"""
        entry = {
            "step": step_name,
            "shape_before": list(shape_before),
            "shape_after": list(shape_after),
            "timestamp": datetime.now().isoformat(),
            "details": details,
        }
        self.step_log.append(entry)
        logger.info(
            f"[{step_name}] 形状变化: {shape_before} -> {shape_after} {details}"
        )

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整预处理流水线

        流程:
        1. 缺失值处理
        2. 异常值检测与处理
        3. 数据标准化
        4. 特征工程

        Args:
            df: 原始数据

        Returns:
            预处理后的数据
        """
        logger.info("=" * 50)
        logger.info("开始执行数据预处理流水线")
        logger.info(f"原始数据形状: {df.shape}")
        logger.info("=" * 50)

        # Step 1: 缺失值处理
        if self.config["missing_value"]["enabled"]:
            shape_before = df.shape
            cfg = self.config["missing_value"]

            if cfg["strategy"] == "auto":
                df = self.missing_handler.auto_fill(
                    df,
                    numeric_strategy=cfg["numeric_strategy"],
                    categorical_strategy=cfg["categorical_strategy"],
                    drop_threshold=cfg["drop_threshold"],
                )
            elif cfg["strategy"] == "drop":
                df = self.missing_handler.drop_missing(df, threshold=cfg["drop_threshold"])

            self._log_step("缺失值处理", shape_before, df.shape)

        # Step 2: 异常值处理
        if self.config["outlier"]["enabled"]:
            shape_before = df.shape
            cfg = self.config["outlier"]

            if cfg["action"] == "clip":
                df = self.outlier_detector.clip_outliers(
                    df,
                    columns=cfg.get("columns"),
                    method=cfg["method"],
                    factor=cfg["factor"],
                )
            elif cfg["action"] == "remove":
                df = self.outlier_detector.remove_outliers(
                    df,
                    columns=cfg.get("columns"),
                    method=cfg["method"],
                    factor=cfg["factor"],
                )
            elif cfg["action"] == "detect_only":
                if cfg["method"] == "iqr":
                    self.outlier_detector.detect_iqr(df, columns=cfg.get("columns"), factor=cfg["factor"])
                elif cfg["method"] == "zscore":
                    self.outlier_detector.detect_zscore(df, columns=cfg.get("columns"), threshold=cfg["factor"])

            self._log_step("异常值处理", shape_before, df.shape, f"method={cfg['method']}, action={cfg['action']}")

        # Step 3: 数据标准化
        if self.config["normalization"]["enabled"]:
            shape_before = df.shape
            cfg = self.config["normalization"]

            if cfg["method"] == "minmax":
                df = self.normalizer.minmax_scale(df, columns=cfg.get("columns"))
            elif cfg["method"] == "zscore":
                df = self.normalizer.zscore_scale(df, columns=cfg.get("columns"))
            elif cfg["method"] == "robust":
                df = self.normalizer.robust_scale(df, columns=cfg.get("columns"))
            elif cfg["method"] == "log":
                df = self.normalizer.log_transform(df, columns=cfg.get("columns"))

            self._log_step("数据标准化", shape_before, df.shape, f"method={cfg['method']}")

        # Step 4: 特征工程
        if self.config["feature_engineering"]["enabled"]:
            shape_before = df.shape
            cfg = self.config["feature_engineering"]

            # 时间特征
            if cfg.get("time_features"):
                time_cfg = cfg["time_features"]
                df = self.feature_engineer.extract_time_features(
                    df,
                    time_col=time_cfg["column"],
                    features=time_cfg.get("features"),
                )

            # 滞后特征
            if cfg.get("lag_features"):
                lag_cfg = cfg["lag_features"]
                df = self.feature_engineer.create_lag_features(
                    df,
                    columns=lag_cfg["columns"],
                    lags=lag_cfg.get("lags", [1, 5, 10]),
                    group_col=lag_cfg.get("group_col"),
                )

            # 滚动特征
            if cfg.get("rolling_features"):
                roll_cfg = cfg["rolling_features"]
                df = self.feature_engineer.create_rolling_features(
                    df,
                    columns=roll_cfg["columns"],
                    windows=roll_cfg.get("windows", [5, 10, 20]),
                    stats=roll_cfg.get("stats", ["mean", "std"]),
                    group_col=roll_cfg.get("group_col"),
                )

            # 编码
            if cfg.get("encode_columns"):
                if cfg["encode_method"] == "label":
                    df = self.feature_engineer.label_encode(df, columns=cfg["encode_columns"])
                elif cfg["encode_method"] == "onehot":
                    df = self.feature_engineer.onehot_encode(df, columns=cfg["encode_columns"])

            self._log_step("特征工程", shape_before, df.shape)

        logger.info("=" * 50)
        logger.info(f"预处理流水线执行完成，最终数据形状: {df.shape}")
        logger.info("=" * 50)

        return df

    def get_pipeline_report(self) -> str:
        """生成流水线执行报告"""
        if not self.step_log:
            return "流水线尚未执行"

        report_lines = [
            "=" * 60,
            "数据预处理流水线执行报告",
            "=" * 60,
        ]

        for i, entry in enumerate(self.step_log, 1):
            report_lines.append(
                f"\n步骤 {i}: {entry['step']}\n"
                f"  形状变化: {tuple(entry['shape_before'])} -> {tuple(entry['shape_after'])}\n"
                f"  时间: {entry['timestamp']}\n"
                f"  详情: {entry['details']}"
            )

        report_lines.append(f"\n{'=' * 60}")
        return "\n".join(report_lines)

    def save_pipeline_report(self, filepath: str = "logs/pipeline_report.txt"):
        """保存流水线报告"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        report = self.get_pipeline_report()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"流水线报告已保存到: {filepath}")

    def save_config(self, filepath: str = "config/pipeline_config.json"):
        """保存当前流水线配置"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        logger.info(f"流水线配置已保存到: {filepath}")

    @classmethod
    def from_config(cls, filepath: str) -> "PreprocessingPipeline":
        """从配置文件加载流水线"""
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        pipeline = cls(config=config)
        logger.info(f"已从配置文件加载流水线: {filepath}")
        return pipeline
