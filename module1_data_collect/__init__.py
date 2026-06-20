"""
模块一：财经大数据收集与预处理

功能：
- 多源数据采集：阿里云天池、Kaggle、Python爬虫
- 数据清洗与预处理流水线：缺失值处理、异常值检测、数据标准化、特征工程
"""

from .collectors import TianchiCollector, KaggleCollector, FinanceCrawler
from .preprocessors import PreprocessingPipeline
from .utils import auto_load, setup_logger

__all__ = [
    'TianchiCollector', 'KaggleCollector', 'FinanceCrawler',
    'PreprocessingPipeline', 'auto_load', 'setup_logger',
]
