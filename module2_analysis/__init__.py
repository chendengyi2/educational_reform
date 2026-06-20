"""
模块二：财经大数据分析与挖掘

功能：
- 分类算法：逻辑回归、随机森林、XGBoost、SVM
- 聚类算法：K-Means、DBSCAN、层次聚类
- 回归算法：线性回归、岭回归、Lasso、GBDT
- 推荐算法：协同过滤(User-based/Item-based)、ALS矩阵分解
"""

from .classify import ClassificationModels
from .cluster import ClusteringModels
from .regress import RegressionModels
from .recommend import RecommendationModels

__all__ = [
    'ClassificationModels', 'ClusteringModels',
    'RegressionModels', 'RecommendationModels',
]
