"""
阿里云天池平台数据采集器
支持下载天池公开财经数据集（如贷款违约预测、股票走势等）
"""

import os
import logging
import hashlib
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class TianchiCollector:
    """阿里云天池平台数据采集器"""

    BASE_URL = "https://tianchi.aliyun.com/dataset"

    # 天池平台常见财经数据集元信息（示例，实际使用时需根据天池最新数据集更新）
    DATASETS = {
        "loan_default": {
            "name": "贷款违约预测数据集",
            "description": "包含用户贷款信息与违约标签，适用于金融风控建模",
            "url": "https://tianchi.aliyun.com/dataset/dataDetail?dataId=94022",
            "format": "csv",
        },
        "stock_prediction": {
            "name": "A股股票走势预测数据集",
            "description": "包含沪深A股历史行情数据，适用于股价预测",
            "url": "https://tianchi.aliyun.com/dataset/dataDetail?dataId=87477",
            "format": "csv",
        },
        "credit_risk": {
            "name": "信用风险评估数据集",
            "description": "包含用户信用相关信息，适用于信用评分建模",
            "url": "https://tianchi.aliyun.com/dataset/dataDetail?dataId=79846",
            "format": "csv",
        },
    }

    def __init__(self, save_dir: str = "data/raw/tianchi"):
        """
        Args:
            save_dir: 数据保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def list_datasets(self) -> dict:
        """列出可用的天池财经数据集"""
        print("=" * 60)
        print("阿里云天池平台 - 可用财经数据集")
        print("=" * 60)
        for key, info in self.DATASETS.items():
            print(f"\n[{key}]")
            print(f"  名称: {info['name']}")
            print(f"  描述: {info['description']}")
            print(f"  格式: {info['format']}")
            print(f"  链接: {info['url']}")
        return self.DATASETS

    def download_dataset(
        self,
        dataset_key: str,
        local_filename: Optional[str] = None,
        verify_hash: bool = False,
    ) -> Optional[Path]:
        """
        下载天池数据集

        注意：天池数据集需要登录后下载，此方法提供交互式下载流程指引，
        并支持从本地已下载文件导入到项目数据目录。

        Args:
            dataset_key: 数据集标识（如 'loan_default'）
            local_filename: 已下载到本地的文件路径，用于导入
            verify_hash: 是否校验文件完整性

        Returns:
            保存的文件路径，失败返回 None
        """
        if dataset_key not in self.DATASETS:
            logger.error(f"未知数据集: {dataset_key}，可用: {list(self.DATASETS.keys())}")
            return None

        dataset = self.DATASETS[dataset_key]

        # 如果提供了本地文件，直接导入
        if local_filename:
            return self._import_local_file(local_filename, dataset_key, verify_hash)

        # 否则提供下载指引
        logger.info(f"数据集 '{dataset['name']}' 需要在天池平台登录后下载")
        logger.info(f"请访问: {dataset['url']}")
        logger.info(f"下载后请将文件放到: {self.save_dir / dataset_key}")
        logger.info(f"然后使用 local_filename 参数导入")

        # 创建数据集目录
        dataset_dir = self.save_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # 生成导入说明文件
        readme_path = dataset_dir / "README.txt"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"数据集: {dataset['name']}\n")
            f.write(f"下载地址: {dataset['url']}\n")
            f.write(f"格式: {dataset['format']}\n")
            f.write(f"说明: 请将下载的数据文件放入此目录\n")

        return dataset_dir

    def _import_local_file(
        self, local_filename: str, dataset_key: str, verify_hash: bool
    ) -> Optional[Path]:
        """从本地文件导入数据集到项目目录"""
        src_path = Path(local_filename)
        if not src_path.exists():
            logger.error(f"本地文件不存在: {local_filename}")
            return None

        dataset_dir = self.save_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dataset_dir / src_path.name

        # 复制文件
        import shutil
        shutil.copy2(src_path, dest_path)

        # 记录元信息
        meta_path = dataset_dir / "meta.json"
        meta = {
            "dataset_key": dataset_key,
            "filename": src_path.name,
            "import_time": datetime.now().isoformat(),
            "file_size": dest_path.stat().st_size,
        }
        if verify_hash:
            meta["md5"] = self._compute_md5(dest_path)

        import json
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"数据集 '{dataset_key}' 已导入到: {dest_path}")
        return dest_path

    @staticmethod
    def _compute_md5(filepath: Path) -> str:
        """计算文件MD5"""
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def batch_import(self, file_mapping: dict) -> dict:
        """
        批量导入本地数据文件

        Args:
            file_mapping: {dataset_key: local_file_path} 映射

        Returns:
            导入结果字典
        """
        results = {}
        for key, filepath in file_mapping.items():
            result = self.download_dataset(key, local_filename=filepath, verify_hash=True)
            results[key] = {
                "success": result is not None,
                "path": str(result) if result else None,
            }
        return results
