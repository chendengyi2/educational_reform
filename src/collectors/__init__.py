"""
财经大数据采集模块
支持多源数据采集：阿里云天池、Kaggle、爬虫
"""

from .tianchi_collector import TianchiCollector
from .kaggle_collector import KaggleCollector
from .crawler_collector import FinanceCrawler

__all__ = ['TianchiCollector', 'KaggleCollector', 'FinanceCrawler']
