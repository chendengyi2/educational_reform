"""
财经大数据收集与预处理 - 项目入口
"""

from src.collectors import TianchiCollector, KaggleCollector, FinanceCrawler
from src.preprocessors import PreprocessingPipeline
from src.utils import auto_load, setup_logger
