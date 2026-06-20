"""
模块二：财经大数据分析与挖掘 - 主程序

使用方式：
  python run_module2.py --demo          # 运行完整演示（分类+聚类+回归+推荐）
  python run_module2.py --classify      # 仅运行分类算法演示
  python run_module2.py --cluster       # 仅运行聚类算法演示
  python run_module2.py --regress       # 仅运行回归算法演示
  python run_module2.py --recommend     # 仅运行推荐算法演示
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from module2_analysis.classify import ClassificationModels
from module2_analysis.cluster import ClusteringModels
from module2_analysis.regress import RegressionModels
from module2_analysis.recommend import RecommendationModels

import pandas as pd
import numpy as np


def generate_classification_data(n=500):
    """生成分类演示数据"""
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(22, 65, n),
        "income": np.random.lognormal(10.5, 0.8, n),
        "credit_score": np.clip(np.random.normal(680, 80, n), 300, 850),
        "debt_ratio": np.random.beta(2, 5, n),
        "employment_years": np.random.exponential(5, n).round(1),
        "default": np.random.binomial(1, 0.15, n),
    })


def generate_clustering_data(n=300):
    """生成聚类演示数据"""
    np.random.seed(42)
    return pd.DataFrame({
        "revenue": np.random.lognormal(10, 1.5, n),
        "net_profit": np.random.lognormal(8, 1.2, n),
        "roe": np.random.normal(12, 8, n),
        "debt_ratio": np.random.beta(2, 5, n),
        "market_cap": np.random.lognormal(12, 1, n),
        "pe_ratio": np.random.lognormal(3, 0.8, n),
    })


def generate_regression_data(n=500):
    """生成回归演示数据"""
    np.random.seed(42)
    X = np.random.randn(n, 4)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n) * 0.5
    return pd.DataFrame(X, columns=["feat1", "feat2", "feat3", "feat4"]).assign(target=y)


def generate_recommendation_data():
    """生成推荐演示数据"""
    np.random.seed(42)
    n_users, n_items = 20, 15
    ratings = np.random.choice([0, 1, 2, 3, 4, 5], size=(n_users, n_items), p=[0.5, 0.05, 0.1, 0.1, 0.15, 0.1])
    return pd.DataFrame(ratings, index=range(n_users), columns=[f"product_{i}" for i in range(n_items)])


def demo_classification():
    print("\n" + "=" * 60)
    print("  分类算法演示：信用风险评估")
    print("=" * 60)
    df = generate_classification_data()
    cm = ClassificationModels()
    result = cm.compare_all(df, target_col="default")
    return result


def demo_clustering():
    print("\n" + "=" * 60)
    print("  聚类算法演示：客户分群/市场细分")
    print("=" * 60)
    df = generate_clustering_data()
    clm = ClusteringModels()
    result = clm.compare_all(df, n_clusters=3)
    # 聚类画像
    if "KMeans" in clm.models_:
        X, _ = clm.prepare_data(df)
        labels = clm.models_["KMeans"].predict(X)
        clm.get_cluster_profile(df, labels)
    return result


def demo_regression():
    print("\n" + "=" * 60)
    print("  回归算法演示：股票价格预测")
    print("=" * 60)
    df = generate_regression_data()
    rm = RegressionModels()
    result = rm.compare_all(df, target_col="target")
    return result


def demo_recommendation():
    print("\n" + "=" * 60)
    print("  推荐算法演示：理财产品推荐")
    print("=" * 60)
    rating_matrix = generate_recommendation_data()
    rec = RecommendationModels()
    result = rec.compare_all(rating_matrix)

    # 为用户0推荐
    for model_name in rec.models_:
        recs = rec.recommend(model_name, user_id=0, n=5)
        print(f"\n{model_name} 推荐结果(用户0): {recs}")
    return result


def main():
    parser = argparse.ArgumentParser(description="模块二：财经大数据分析与挖掘")
    parser.add_argument("--demo", action="store_true", help="运行全部算法演示")
    parser.add_argument("--classify", action="store_true", help="分类算法演示")
    parser.add_argument("--cluster", action="store_true", help="聚类算法演示")
    parser.add_argument("--regress", action="store_true", help="回归算法演示")
    parser.add_argument("--recommend", action="store_true", help="推荐算法演示")
    args = parser.parse_args()

    if args.demo:
        demo_classification()
        demo_clustering()
        demo_regression()
        demo_recommendation()
    elif args.classify:
        demo_classification()
    elif args.cluster:
        demo_clustering()
    elif args.regress:
        demo_regression()
    elif args.recommend:
        demo_recommendation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
