"""
推荐算法模块
支持：协同过滤（User-based / Item-based）、ALS矩阵分解
应用场景：理财产品推荐、投资组合推荐
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """推荐结果数据结构"""
    user_id: int
    recommendations: List[Tuple[int, float]] = field(default_factory=list)  # [(item_id, score), ...]

    def top_n(self, n: int = 10) -> List[Tuple[int, float]]:
        return self.recommendations[:n]


class UserBasedCF:
    """基于用户的协同过滤"""

    def __init__(self, k: int = 20):
        """
        Args:
            k: 最近邻数量
        """
        self.k = k
        self.user_similarity_ = None
        self.rating_matrix_ = None

    def fit(self, rating_matrix: pd.DataFrame):
        """
        训练模型

        Args:
            rating_matrix: 评分矩阵 (行=用户, 列=物品)
        """
        self.rating_matrix_ = rating_matrix.fillna(0)

        # 计算用户相似度矩阵
        self.user_similarity_ = pd.DataFrame(
            cosine_similarity(self.rating_matrix_),
            index=rating_matrix.index,
            columns=rating_matrix.index,
        )
        logger.info(f"User-Based CF 训练完成，用户数: {len(rating_matrix)}")
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        if user_id not in self.rating_matrix_.index:
            return 0.0
        if item_id not in self.rating_matrix_.columns:
            return 0.0

        # 找到对物品评过分的用户
        item_ratings = self.rating_matrix_[item_id]
        rated_users = item_ratings[item_ratings > 0].index

        if len(rated_users) == 0:
            return self.rating_matrix_.loc[user_id].mean()

        # 计算相似度并选Top-K
        sims = self.user_similarity_.loc[user_id, rated_users]
        top_k_users = sims.nlargest(self.k)

        if top_k_users.sum() == 0:
            return self.rating_matrix_.loc[user_id].mean()

        # 加权预测
        scores = item_ratings[top_k_users.index]
        pred = np.dot(top_k_users.values, scores.values) / top_k_users.sum()
        return pred

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """为用户推荐Top-N物品"""
        if user_id not in self.rating_matrix_.index:
            return []

        # 找到用户未评过分的物品
        user_ratings = self.rating_matrix_.loc[user_id]
        unrated = user_ratings[user_ratings == 0].index

        predictions = []
        for item_id in unrated:
            pred_score = self.predict(user_id, item_id)
            predictions.append((item_id, round(pred_score, 4)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class ItemBasedCF:
    """基于物品的协同过滤"""

    def __init__(self, k: int = 20):
        self.k = k
        self.item_similarity_ = None
        self.rating_matrix_ = None

    def fit(self, rating_matrix: pd.DataFrame):
        """训练模型"""
        self.rating_matrix_ = rating_matrix.fillna(0)

        # 计算物品相似度矩阵
        self.item_similarity_ = pd.DataFrame(
            cosine_similarity(self.rating_matrix_.T),
            index=rating_matrix_.columns,
            columns=rating_matrix_.columns,
        )
        logger.info(f"Item-Based CF 训练完成，物品数: {len(rating_matrix_.columns)}")
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        if user_id not in self.rating_matrix_.index:
            return 0.0
        if item_id not in self.item_similarity_.index:
            return 0.0

        # 找到用户评过分的物品
        user_ratings = self.rating_matrix_.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index

        if len(rated_items) == 0:
            return 0.0

        # 物品相似度 Top-K
        sims = self.item_similarity_.loc[item_id, rated_items]
        top_k_items = sims.nlargest(self.k)

        if top_k_items.sum() == 0:
            return user_ratings.mean()

        scores = user_ratings[top_k_items.index]
        pred = np.dot(top_k_items.values, scores.values) / top_k_items.sum()
        return pred

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """为用户推荐Top-N物品"""
        if user_id not in self.rating_matrix_.index:
            return []

        user_ratings = self.rating_matrix_.loc[user_id]
        unrated = user_ratings[user_ratings == 0].index

        predictions = []
        for item_id in unrated:
            pred_score = self.predict(user_id, item_id)
            predictions.append((item_id, round(pred_score, 4)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class ALSMatrixFactorization:
    """ALS 矩阵分解推荐"""

    def __init__(self, n_factors: int = 10, n_iterations: int = 20, reg: float = 0.1):
        """
        Args:
            n_factors: 隐因子维度
            n_iterations: 迭代次数
            reg: 正则化参数
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self.user_factors_ = None
        self.item_factors_ = None
        self.rating_matrix_ = None

    def fit(self, rating_matrix: pd.DataFrame):
        """
        交替最小二乘法训练

        Args:
            rating_matrix: 评分矩阵
        """
        R = rating_matrix.fillna(0).values
        n_users, n_items = R.shape

        # 随机初始化
        np.random.seed(42)
        self.user_factors_ = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors_ = np.random.normal(0, 0.1, (n_items, self.n_factors))

        self.rating_matrix_ = rating_matrix

        for iteration in range(self.n_iterations):
            # 固定 V，更新 U
            for u in range(n_users):
                rated = R[u, :] > 0
                if rated.sum() == 0:
                    continue
                V_rated = self.item_factors_[rated]
                R_rated = R[u, rated]
                A = V_rated.T @ V_rated + self.reg * np.eye(self.n_factors)
                b = V_rated.T @ R_rated
                self.user_factors_[u] = np.linalg.solve(A, b)

            # 固定 U，更新 V
            for i in range(n_items):
                rated = R[:, i] > 0
                if rated.sum() == 0:
                    continue
                U_rated = self.user_factors_[rated]
                R_rated = R[rated, i]
                A = U_rated.T @ U_rated + self.reg * np.eye(self.n_factors)
                b = U_rated.T @ R_rated
                self.item_factors_[i] = np.linalg.solve(A, b)

            # 计算损失
            R_pred = self.user_factors_ @ self.item_factors_.T
            mask = R > 0
            loss = np.sum((R[mask] - R_pred[mask]) ** 2)

            if (iteration + 1) % 5 == 0:
                logger.info(f"ALS 迭代 {iteration + 1}/{self.n_iterations}, loss={loss:.4f}")

        logger.info("ALS 矩阵分解训练完成")
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """预测评分"""
        user_idx = self.rating_matrix_.index.get_loc(user_id) if user_id in self.rating_matrix_.index else None
        item_idx = self.rating_matrix_.columns.get_loc(item_id) if item_id in self.rating_matrix_.columns else None

        if user_idx is None or item_idx is None:
            return 0.0

        pred = np.dot(self.user_factors_[user_idx], self.item_factors_[item_idx])
        return float(pred)

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """为用户推荐Top-N物品"""
        if user_id not in self.rating_matrix_.index:
            return []

        user_idx = self.rating_matrix_.index.get_loc(user_id)
        user_ratings = self.rating_matrix_.loc[user_id]
        unrated = user_ratings[user_ratings == 0].index

        predictions = []
        for item_id in unrated:
            pred_score = self.predict(user_id, item_id)
            predictions.append((item_id, round(pred_score, 4)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class RecommendationModels:
    """推荐算法集合"""

    def __init__(self):
        self.models_ = {}
        self.results_ = {}

    def user_based_cf(self, rating_matrix: pd.DataFrame, k: int = 20) -> UserBasedCF:
        """User-based 协同过滤"""
        model = UserBasedCF(k=k)
        model.fit(rating_matrix)
        self.models_["UserBasedCF"] = model
        return model

    def item_based_cf(self, rating_matrix: pd.DataFrame, k: int = 20) -> ItemBasedCF:
        """Item-based 协同过滤"""
        model = ItemBasedCF(k=k)
        model.fit(rating_matrix)
        self.models_["ItemBasedCF"] = model
        return model

    def als(self, rating_matrix: pd.DataFrame, n_factors: int = 10, n_iterations: int = 20, reg: float = 0.1) -> ALSMatrixFactorization:
        """ALS 矩阵分解"""
        model = ALSMatrixFactorization(n_factors=n_factors, n_iterations=n_iterations, reg=reg)
        model.fit(rating_matrix)
        self.models_["ALS"] = model
        return model

    def evaluate(
        self,
        model,
        rating_matrix: pd.DataFrame,
        test_ratio: float = 0.2,
    ) -> Dict[str, float]:
        """
        评估推荐模型（RMSE / MAE）

        Args:
            model: 推荐模型
            rating_matrix: 评分矩阵
            test_ratio: 测试集比例

        Returns:
            评估指标
        """
        R = rating_matrix.fillna(0)
        mask = R.values > 0
        n_ratings = mask.sum()

        # 随机划分测试集
        np.random.seed(42)
        indices = np.argwhere(mask)
        np.random.shuffle(indices)
        n_test = int(n_ratings * test_ratio)
        test_indices = indices[:n_test]

        errors = []
        abs_errors = []

        for u_idx, i_idx in test_indices:
            user_id = R.index[u_idx]
            item_id = R.columns[i_idx]
            actual = R.iloc[u_idx, i_idx]
            pred = model.predict(user_id, item_id)

            errors.append((actual - pred) ** 2)
            abs_errors.append(abs(actual - pred))

        rmse = np.sqrt(np.mean(errors))
        mae = np.mean(abs_errors)

        model_name = type(model).__name__
        result = {"model_name": model_name, "rmse": round(rmse, 4), "mae": round(mae, 4)}
        self.results_[model_name] = result

        logger.info(f"[{model_name}] RMSE={rmse:.4f}, MAE={mae:.4f}")
        return result

    def compare_all(
        self,
        rating_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """对比所有推荐算法"""
        self.user_based_cf(rating_matrix)
        self.item_based_cf(rating_matrix)
        self.als(rating_matrix)

        for model in self.models_.values():
            self.evaluate(model, rating_matrix)

        comparison = pd.DataFrame(list(self.results_.values()))
        comparison = comparison.sort_values("rmse").reset_index(drop=True)

        print("\n" + "=" * 40)
        print("推荐算法对比结果")
        print("=" * 40)
        print(comparison.to_string(index=False))

        return comparison

    def recommend(self, model_name: str, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """使用指定模型推荐"""
        if model_name not in self.models_:
            raise ValueError(f"模型 '{model_name}' 不存在")
        return self.models_[model_name].recommend(user_id, n)
