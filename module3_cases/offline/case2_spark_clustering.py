#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案例2：基于Spark MLlib的上市公司财务指标聚类分析

模拟Spark DataFrame操作，使用pandas实现等价逻辑，
完成上市公司财务数据的聚类分析与市场细分。

大数据技术栈说明：
- 实际环境：数据存储在Hive，通过Spark SQL读取，使用Spark MLlib进行聚类
- 本地模拟：使用pandas DataFrame模拟Spark DataFrame的转换和操作
- 数据规模：实际环境处理百万级上市公司年报数据，此处模拟500家
- 集群环境：实际运行在YARN集群，此处单机pandas模拟
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 第一部分：模拟Spark DataFrame生成上市公司财务数据
# 实际环境：Spark读取Hive表 spark.read.table("dw.finance_report")
# ============================================================

def simulate_spark_dataframe(n=500):
    """
    模拟Spark DataFrame操作，生成上市公司财务数据
    等价于 spark.read.table("dw.annual_finance_report")
    """
    np.random.seed(42)

    industries = ["银行", "房地产", "医药", "科技", "制造", "能源", "消费", "金融"]
    market_types = ["主板", "创业板", "科创板"]

    # 模拟三种典型公司画像：蓝筹/成长/小微
    n_bluechip = n // 3
    n_growth = n // 3
    n_small = n - n_bluechip - n_growth

    revenue_blue = np.random.lognormal(12, 0.4, n_bluechip)
    revenue_growth = np.random.lognormal(10.5, 0.5, n_growth)
    revenue_small = np.random.lognormal(9, 0.6, n_small)
    revenue = np.concatenate([revenue_blue, revenue_growth, revenue_small])

    npm_blue = np.random.normal(0.15, 0.05, n_bluechip).clip(0.02, 0.4)
    npm_growth = np.random.normal(0.08, 0.06, n_growth).clip(-0.1, 0.35)
    npm_small = np.random.normal(0.04, 0.04, n_small).clip(-0.15, 0.2)
    net_profit_margin = np.concatenate([npm_blue, npm_growth, npm_small])

    roe_blue = np.random.normal(0.12, 0.04, n_bluechip).clip(0.02, 0.25)
    roe_growth = np.random.normal(0.08, 0.06, n_growth).clip(-0.1, 0.3)
    roe_small = np.random.normal(0.05, 0.05, n_small).clip(-0.15, 0.15)
    roe = np.concatenate([roe_blue, roe_growth, roe_small])

    alr_blue = np.random.normal(0.65, 0.1, n_bluechip).clip(0.3, 0.9)
    alr_growth = np.random.normal(0.45, 0.12, n_growth).clip(0.15, 0.75)
    alr_small = np.random.normal(0.55, 0.15, n_small).clip(0.2, 0.85)
    asset_liability_ratio = np.concatenate([alr_blue, alr_growth, alr_small])

    data = {
        "stock_code": [f"{np.random.randint(600000, 605000)}" for _ in range(n)],
        "company_name": [f"公司_{i:04d}" for i in range(n)],
        "industry": np.random.choice(industries, n),
        "market_type": np.random.choice(market_types, n, p=[0.5, 0.3, 0.2]),
        "revenue": np.round(revenue, 2),
        "net_profit_margin": np.round(net_profit_margin, 4),
        "roe": np.round(roe, 4),
        "asset_liability_ratio": np.round(asset_liability_ratio, 4),
        "current_ratio": np.round(np.random.lognormal(0.5, 0.5, n), 2),
        "revenue_growth_rate": np.round(np.random.normal(0.1, 0.3, n), 4),
        "total_assets": np.round(revenue * np.random.uniform(1.5, 4.0, n), 2),
        "market_cap": np.round(revenue * np.random.uniform(0.5, 3.0, n), 2),
    }

    # 模拟Spark DataFrame创建: spark.createDataFrame(data)
    df = pd.DataFrame(data)
    logger.info(f"[Spark] 创建DataFrame，记录数: {len(df)}, 列数: {len(df.columns)}")
    return df


# ============================================================
# 第二部分：模拟Spark SQL特征工程
# 实际环境：spark.sql("SELECT ... FROM finance_report WHERE ...")
# ============================================================

def spark_sql_feature_engineering(df):
    """
    模拟Spark SQL特征工程操作
    等价于:
    spark.sql(\"\"\"
        SELECT *, 
            revenue / total_assets AS asset_turnover,
            market_cap / revenue AS ps_ratio,
            CASE WHEN revenue_growth_rate > 0.2 THEN '高增长'
                 WHEN revenue_growth_rate > 0 THEN '正增长'
                 ELSE '负增长' END AS growth_category
        FROM finance_report
    \"\"\")
    """
    df = df.copy()
    df["asset_turnover"] = (df["revenue"] / df["total_assets"]).round(4)
    df["ps_ratio"] = (df["market_cap"] / df["revenue"]).round(4)

    # 行业编码
    industry_map = {ind: i for i, ind in enumerate(sorted(df["industry"].unique()))}
    df["industry_code"] = df["industry"].map(industry_map)

    logger.info(f"[Spark SQL] 特征工程完成，特征数: {len(df.columns)}")
    return df


# ============================================================
# 第三部分：K-Means最优K值搜索
# ============================================================

def search_optimal_k(X_scaled, k_range=range(2, 11)):
    """通过轮廓系数和CH指数搜索最优K值"""
    logger.info("[Spark MLlib] 开始K-Means最优K值搜索...")
    silhouette_scores = {}
    ch_scores = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        silhouette_scores[k] = sil
        ch_scores[k] = ch
        logger.info(f"  K={k}: 轮廓系数={sil:.4f}, CH指数={ch:.2f}")

    best_k_sil = max(silhouette_scores, key=silhouette_scores.get)
    best_k_ch = max(ch_scores, key=ch_scores.get)
    logger.info(f"  轮廓系数最优K={best_k_sil}, CH指数最优K={best_k_ch}")

    optimal_k = best_k_sil
    return optimal_k, silhouette_scores, ch_scores


# ============================================================
# 第四部分：多聚类算法对比
# ============================================================

def run_clustering_comparison(X_scaled, optimal_k):
    """运行K-Means、DBSCAN、层次聚类对比"""
    results = {}

    # K-Means聚类
    logger.info(f"[Spark MLlib] 运行K-Means聚类 (K={optimal_k})...")
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    results["K-Means"] = {
        "labels": km_labels,
        "silhouette": silhouette_score(X_scaled, km_labels),
        "n_clusters": optimal_k,
    }
    logger.info(f"  K-Means: 轮廓系数={results['K-Means']['silhouette']:.4f}")

    # DBSCAN聚类
    logger.info("[Spark MLlib] 运行DBSCAN聚类...")
    db = DBSCAN(eps=1.2, min_samples=5)
    db_labels = db.fit_predict(X_scaled)
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    if n_db_clusters >= 2:
        mask = db_labels != -1
        db_sil = silhouette_score(X_scaled[mask], db_labels[mask]) if mask.sum() > n_db_clusters else -1
    else:
        db_sil = -1
    results["DBSCAN"] = {
        "labels": db_labels,
        "silhouette": db_sil,
        "n_clusters": n_db_clusters,
        "noise_ratio": (db_labels == -1).sum() / len(db_labels),
    }
    logger.info(f"  DBSCAN: 簇数={n_db_clusters}, 轮廓系数={db_sil:.4f}, 噪声比={results['DBSCAN']['noise_ratio']:.2%}")

    # 层次聚类
    logger.info("[Spark MLlib] 运行层次聚类...")
    hc = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
    hc_labels = hc.fit_predict(X_scaled)
    results["层次聚类"] = {
        "labels": hc_labels,
        "silhouette": silhouette_score(X_scaled, hc_labels),
        "n_clusters": optimal_k,
    }
    logger.info(f"  层次聚类: 轮廓系数={results['层次聚类']['silhouette']:.4f}")

    return results


# ============================================================
# 第五部分：聚类画像分析
# ============================================================

def generate_cluster_profiles(df, labels, feature_cols, algorithm_name="K-Means"):
    """生成各聚类簇的特征画像"""
    df = df.copy()
    df["cluster"] = labels

    logger.info(f"\n[{algorithm_name}] 聚类画像分析:")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    profiles = {}
    for c in range(n_clusters):
        cluster_data = df[df["cluster"] == c]
        profile = {}
        for col in feature_cols:
            profile[col] = {
                "mean": cluster_data[col].mean(),
                "std": cluster_data[col].std(),
                "min": cluster_data[col].min(),
                "max": cluster_data[col].max(),
            }
        profiles[c] = profile
        logger.info(f"  簇 {c}: 样本数={len(cluster_data)}")

    return profiles


# ============================================================
# 第六部分：结果输出
# ============================================================

def print_results(df, clustering_results, profiles, silhouette_scores, ch_scores, feature_cols):
    """输出完整的分析结果"""
    print("\n" + "=" * 70)
    print("  基于Spark MLlib的上市公司财务指标聚类分析 - 结果报告")
    print("=" * 70)

    # K值搜索结果
    print("\n【一】K-Means最优K值搜索")
    print("-" * 45)
    print(f"{'K值':<6} {'轮廓系数':<12} {'CH指数':<12}")
    print("-" * 45)
    for k in sorted(silhouette_scores.keys()):
        print(f"{k:<6} {silhouette_scores[k]:<12.4f} {ch_scores[k]:<12.2f}")

    # 算法对比
    print("\n【二】聚类算法对比")
    print("-" * 55)
    print(f"{'算法':<12} {'簇数':<8} {'轮廓系数':<12} {'备注':<20}")
    print("-" * 55)
    for name, res in clustering_results.items():
        note = ""
        if name == "DBSCAN":
            note = f"噪声比: {res['noise_ratio']:.1%}"
        print(f"{name:<12} {res['n_clusters']:<8} {res['silhouette']:<12.4f} {note:<20}")

    # 聚类画像
    best_labels = clustering_results["K-Means"]["labels"]
    df_copy = df.copy()
    df_copy["cluster"] = best_labels
    n_clusters = clustering_results["K-Means"]["n_clusters"]

    print("\n【三】K-Means聚类画像（各簇均值）")
    print("-" * 90)
    header = f"{'特征':<24}" + "".join([f"{'簇'+str(c):<14}" for c in range(n_clusters)])
    print(header)
    print("-" * 90)
    for col in feature_cols:
        row = f"{col:<24}"
        for c in range(n_clusters):
            val = df_copy[df_copy["cluster"] == c][col].mean()
            row += f"{val:<14.4f}"
        print(row)

    # 市场细分结论
    print("\n【四】市场细分结论")
    print("-" * 55)
    cluster_names = {}
    for c in range(n_clusters):
        cdata = df_copy[df_copy["cluster"] == c]
        avg_revenue = cdata["revenue"].mean()
        avg_roe = cdata["roe"].mean()
        avg_alr = cdata["asset_liability_ratio"].mean()
        if avg_revenue > 3e5 and avg_roe > 0.10:
            label = "蓝筹龙头股"
        elif avg_roe > 0.08:
            label = "高成长潜力股"
        else:
            label = "小微风险股"
        cluster_names[c] = label
        print(f"  簇 {c} ({label}): {len(cdata)}家公司")
        print(f"    平均营收: {avg_revenue:,.0f}, 平均ROE: {avg_roe:.2%}, 资产负债率: {avg_alr:.2%}")

    print("\n" + "=" * 70)


# ============================================================
# 主函数
# ============================================================

def main():
    logger.info("=" * 50)
    logger.info("开始执行：基于Spark MLlib的上市公司财务指标聚类分析")
    logger.info("=" * 50)

    # Step 1: 模拟Spark DataFrame加载数据
    logger.info("\n--- Step 1: 从Hive加载财务数据到Spark DataFrame ---")
    df = simulate_spark_dataframe(n=500)

    # Step 2: Spark SQL特征工程
    logger.info("\n--- Step 2: Spark SQL特征工程 ---")
    df = spark_sql_feature_engineering(df)

    # Step 3: 数据标准化
    feature_cols = [
        "revenue", "net_profit_margin", "roe", "asset_liability_ratio",
        "current_ratio", "revenue_growth_rate", "asset_turnover", "ps_ratio",
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)
    logger.info(f"[Spark MLlib] 特征标准化完成，特征数: {len(feature_cols)}")

    # Step 4: K-Means最优K值搜索
    logger.info("\n--- Step 3: K-Means最优K值搜索 ---")
    optimal_k, silhouette_scores, ch_scores = search_optimal_k(X_scaled)

    # Step 5: 多聚类算法对比
    logger.info("\n--- Step 4: 多聚类算法对比 ---")
    clustering_results = run_clustering_comparison(X_scaled, optimal_k)

    # Step 6: 聚类画像
    logger.info("\n--- Step 5: 聚类画像分析 ---")
    profiles = generate_cluster_profiles(df, clustering_results["K-Means"]["labels"], feature_cols)

    # Step 7: 结果输出
    print_results(df, clustering_results, profiles, silhouette_scores, ch_scores, feature_cols)

    logger.info("分析完成！")


if __name__ == "__main__":
    main()
