"""
聚类算法模块
支持：K-Means、DBSCAN、层次聚类
应用场景：客户分群、市场细分
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA


logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """聚类结果数据结构"""
    model_name: str
    n_clusters: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    labels: Optional[np.ndarray] = None
    cluster_sizes: Optional[Dict[int, int]] = None
    cluster_centers: Optional[np.ndarray] = None
    inertia: float = 0.0
    model: Any = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_clusters": self.n_clusters,
            "silhouette": round(self.silhouette, 4),
            "calinski_harabasz": round(self.calinski_harabasz, 4),
            "davies_bouldin": round(self.davies_bouldin, 4),
            "cluster_sizes": self.cluster_sizes,
            "inertia": round(self.inertia, 4),
        }


class ClusteringModels:
    """聚类算法集合"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models_ = {}
        self.results_ = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        scale: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """准备聚类数据"""
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        X = df[feature_cols].values

        if scale:
            X = self.scaler.fit_transform(X)

        logger.info(f"聚类数据准备完成: {X.shape}")
        return X, feature_cols

    def _evaluate(
        self,
        labels: np.ndarray,
        X: np.ndarray,
        model_name: str,
        model: Any = None,
    ) -> ClusteringResult:
        """评估聚类结果"""
        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})  # 排除噪声点

        # 聚类指标
        sil = -1.0
        ch = 0.0
        db = float("inf")

        if n_clusters >= 2:
            # 过滤噪声点计算指标
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil = silhouette_score(X[mask], labels[mask])
                ch = calinski_harabasz_score(X[mask], labels[mask])
                db = davies_bouldin_score(X[mask], labels[mask])

        # 各簇样本数
        cluster_sizes = {}
        for label in sorted(unique_labels):
            cluster_sizes[int(label)] = int(np.sum(labels == label))

        # 簇中心
        centers = None
        inertia = 0.0
        if hasattr(model, "cluster_centers_"):
            centers = model.cluster_centers_
            inertia = model.inertia_ if hasattr(model, "inertia_") else 0.0

        result = ClusteringResult(
            model_name=model_name,
            n_clusters=n_clusters,
            silhouette=sil,
            calinski_harabasz=ch,
            davies_bouldin=db,
            labels=labels,
            cluster_sizes=cluster_sizes,
            cluster_centers=centers,
            inertia=inertia,
            model=model,
        )

        logger.info(
            f"[{model_name}] 簇数={n_clusters}, 轮廓系数={sil:.4f}, CH={ch:.2f}"
        )
        return result

    def kmeans(
        self,
        X: np.ndarray,
        n_clusters: int = 3,
        **kwargs,
    ) -> ClusteringResult:
        """K-Means 聚类"""
        default_params = {
            "n_clusters": n_clusters,
            "random_state": self.random_state,
            "n_init": 10,
        }
        default_params.update(kwargs)

        model = KMeans(**default_params)
        labels = model.fit_predict(X)
        self.models_["KMeans"] = model

        result = self._evaluate(labels, X, "KMeans", model)
        self.results_["KMeans"] = result
        return result

    def dbscan(
        self,
        X: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5,
        **kwargs,
    ) -> ClusteringResult:
        """DBSCAN 密度聚类"""
        default_params = {"eps": eps, "min_samples": min_samples}
        default_params.update(kwargs)

        model = DBSCAN(**default_params)
        labels = model.fit_predict(X)
        self.models_["DBSCAN"] = model

        result = self._evaluate(labels, X, "DBSCAN", model)
        self.results_["DBSCAN"] = result
        return result

    def hierarchical(
        self,
        X: np.ndarray,
        n_clusters: int = 3,
        linkage: str = "ward",
        **kwargs,
    ) -> ClusteringResult:
        """层次聚类"""
        default_params = {
            "n_clusters": n_clusters,
            "linkage": linkage,
        }
        default_params.update(kwargs)

        model = AgglomerativeClustering(**default_params)
        labels = model.fit_predict(X)
        self.models_["Hierarchical"] = model

        result = self._evaluate(labels, X, "Hierarchical", model)
        self.results_["Hierarchical"] = result
        return result

    def find_optimal_k(
        self,
        X: np.ndarray,
        k_range: range = range(2, 11),
        method: str = "silhouette",
    ) -> Tuple[int, Dict]:
        """
        寻找最优K值

        Args:
            X: 数据
            k_range: K值搜索范围
            method: 评估方法（silhouette / elbow）

        Returns:
            (最优K值, 评估指标字典)
        """
        scores = {}

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X)

            if method == "silhouette":
                score = silhouette_score(X, labels)
            elif method == "calinski":
                score = calinski_harabasz_score(X, labels)
            else:  # elbow
                score = model.inertia_

            scores[k] = score
            logger.info(f"K={k}, {method}={score:.4f}")

        if method == "elbow":
            optimal_k = max(scores, key=scores.get)
        else:
            optimal_k = max(scores, key=scores.get)

        logger.info(f"最优K值: {optimal_k} ({method}={scores[optimal_k]:.4f})")
        return optimal_k, scores

    def compare_all(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        n_clusters: int = 3,
    ) -> pd.DataFrame:
        """对比所有聚类算法"""
        X, features = self.prepare_data(df, feature_cols)

        self.kmeans(X, n_clusters=n_clusters)
        self.dbscan(X)
        self.hierarchical(X, n_clusters=n_clusters)

        comparison = pd.DataFrame([r.to_dict() for r in self.results_.values()])
        comparison = comparison.sort_values("silhouette", ascending=False).reset_index(drop=True)

        print("\n" + "=" * 60)
        print("聚类算法对比结果")
        print("=" * 60)
        print(comparison[["model_name", "n_clusters", "silhouette", "calinski_harabasz", "davies_bouldin"]].to_string(index=False))
        print("=" * 60)

        return comparison

    def get_cluster_profile(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        生成聚类画像（各簇的特征均值）

        Args:
            df: 原始DataFrame
            labels: 聚类标签
            feature_cols: 特征列

        Returns:
            各簇特征均值表
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        df_copy = df[feature_cols].copy()
        df_copy["cluster"] = labels

        profile = df_copy.groupby("cluster")[feature_cols].mean().round(4)
        profile["count"] = df_copy.groupby("cluster")["cluster"].count()

        print("\n聚类画像:")
        print(profile.to_string())
        return profile

    def reduce_dimensions(
        self,
        X: np.ndarray,
        n_components: int = 2,
    ) -> np.ndarray:
        """PCA降维（用于可视化）"""
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA降维到 {n_components}D，解释方差比: {explained:.4f}")
        return X_reduced
