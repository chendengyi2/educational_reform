"""
Kaggle 数据采集器
支持下载 Kaggle 金融风控、股票预测等数据集
"""

import os
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime


logger = logging.getLogger(__name__)


class KaggleCollector:
    """Kaggle 数据集采集器"""

    # Kaggle 财经相关数据集（slug 格式: owner/dataset-name）
    DATASETS = {
        "home_credit_default": {
            "slug": "home-credit-default-risk",
            "name": "Home Credit Default Risk",
            "description": "房屋贷款违约风险预测，包含申请信息、历史信用等",
            "size": "约 300MB",
        },
        "credit_card_fraud": {
            "slug": "mlg-ulb/creditcardfraud",
            "name": "Credit Card Fraud Detection",
            "description": "信用卡欺诈检测，包含284807条交易记录",
            "size": "约 69MB",
        },
        "stock_market_dataset": {
            "slug": "jacksoncrow/stock-market-dataset",
            "name": "Stock Market Dataset",
            "description": "全球主要股票市场历史数据",
            "size": "约 500MB",
        },
        "loan_prediction": {
            "slug": "itsmesunil/bank-loan-prediction",
            "name": "Bank Loan Prediction",
            "description": "银行贷款审批预测数据集",
            "size": "约 50KB",
        },
        "financial_sentiment": {
            "slug": "sbhatti/financial-sentiment-analysis",
            "name": "Financial Sentiment Analysis",
            "description": "财经文本情感分析数据集",
            "size": "约 1MB",
        },
    }

    def __init__(self, save_dir: str = "data/raw/kaggle"):
        """
        Args:
            save_dir: 数据保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._kaggle_available = self._check_kaggle_api()

    def _check_kaggle_api(self) -> bool:
        """检查 Kaggle API 是否可用"""
        try:
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info(f"Kaggle API 可用: {result.stdout.strip()}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        logger.warning(
            "Kaggle API 不可用。请安装: pip install kaggle，"
            "并配置 ~/.kaggle/kaggle.json（从 https://www.kaggle.com/settings 获取 API Token）"
        )
        return False

    def list_datasets(self) -> dict:
        """列出可用的 Kaggle 财经数据集"""
        print("=" * 60)
        print("Kaggle - 可用财经数据集")
        print("=" * 60)
        for key, info in self.DATASETS.items():
            print(f"\n[{key}]")
            print(f"  名称: {info['name']}")
            print(f"  Slug: {info['slug']}")
            print(f"  描述: {info['description']}")
            print(f"  大小: {info['size']}")
        return self.DATASETS

    def download_dataset(
        self,
        dataset_key: str,
        unzip: bool = True,
    ) -> Optional[Path]:
        """
        通过 Kaggle API 下载数据集

        Args:
            dataset_key: 数据集标识（如 'home_credit_default'）
            unzip: 是否自动解压

        Returns:
            数据集保存路径
        """
        if dataset_key not in self.DATASETS:
            logger.error(f"未知数据集: {dataset_key}")
            return None

        if not self._kaggle_available:
            logger.error("Kaggle API 不可用，无法下载。请先配置 API。")
            return None

        dataset = self.DATASETS[dataset_key]
        slug = dataset["slug"]
        dest_dir = self.save_dir / dataset_key
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "kaggle", "datasets", "download",
                "-d", slug,
                "-p", str(dest_dir),
            ]
            if unzip:
                cmd.append("--unzip")

            logger.info(f"正在下载数据集: {dataset['name']} ({slug})...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error(f"下载失败: {result.stderr}")
                return None

            # 记录元信息
            self._save_metadata(dataset_key, dataset, dest_dir)

            logger.info(f"数据集已下载到: {dest_dir}")
            return dest_dir

        except subprocess.TimeoutExpired:
            logger.error("下载超时（>600s），请检查网络连接")
            return None
        except Exception as e:
            logger.error(f"下载异常: {e}")
            return None

    def search_datasets(self, keyword: str, max_results: int = 10) -> List[dict]:
        """
        在 Kaggle 上搜索财经相关数据集

        Args:
            keyword: 搜索关键词
            max_results: 最大返回数量

        Returns:
            搜索结果列表
        """
        if not self._kaggle_available:
            logger.error("Kaggle API 不可用")
            return []

        try:
            cmd = [
                "kaggle", "datasets", "list",
                "-s", keyword,
                "--csv",
                "-p", str(max_results),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                import csv
                import io
                reader = csv.DictReader(io.StringIO(result.stdout))
                datasets = list(reader)
                for ds in datasets:
                    print(f"- {ds.get('title', 'N/A')}: {ds.get('ref', 'N/A')}")
                return datasets
            else:
                logger.error(f"搜索失败: {result.stderr}")
                return []

        except Exception as e:
            logger.error(f"搜索异常: {e}")
            return []

    def download_competition(
        self,
        competition_slug: str,
        unzip: bool = True,
    ) -> Optional[Path]:
        """
        下载 Kaggle 竞赛数据

        Args:
            competition_slug: 竞赛标识（如 'home-credit-default-risk'）
            unzip: 是否自动解压

        Returns:
            数据保存路径
        """
        if not self._kaggle_available:
            logger.error("Kaggle API 不可用")
            return None

        dest_dir = self.save_dir / "competitions" / competition_slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "kaggle", "competitions", "download",
                "-c", competition_slug,
                "-p", str(dest_dir),
            ]
            if unzip:
                cmd.append("--unzip")

            logger.info(f"正在下载竞赛数据: {competition_slug}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error(f"下载失败: {result.stderr}")
                return None

            logger.info(f"竞赛数据已下载到: {dest_dir}")
            return dest_dir

        except Exception as e:
            logger.error(f"下载异常: {e}")
            return None

    def _save_metadata(self, dataset_key: str, dataset_info: dict, dest_dir: Path):
        """保存数据集元信息"""
        meta = {
            "dataset_key": dataset_key,
            "slug": dataset_info["slug"],
            "name": dataset_info["name"],
            "description": dataset_info["description"],
            "download_time": datetime.now().isoformat(),
        }
        meta_path = dest_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
