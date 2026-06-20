#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案例1：基于Hive的银行客户信用评分分析

模拟Hive数据仓库环境，使用pandas模拟Hive SQL查询操作，
完成客户信用评分的特征工程与多模型分类对比。

大数据技术栈说明：
- 实际环境：数据存储在Hive数据仓库中，通过HiveSQL进行ETL和特征提取
- 本地模拟：使用pandas DataFrame模拟Hive表，用pandas操作模拟HiveSQL查询
- 数据导入：实际环境通过Sqoop从MySQL/Oracle导入Hive，此处用numpy生成模拟数据
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 第一部分：模拟Hive数据仓库表结构
# 实际环境中这些表存储在Hive数据仓库，通过Sqoop从业务库导入
# ============================================================

def simulate_hive_customer_table(n=5000):
    """
    模拟Hive客户信息表 dw.dim_customer
    实际HiveSQL: CREATE TABLE dw.dim_customer (customer_id STRING, age INT, ...)
    """
    np.random.seed(42)
    data = {
        "customer_id": [f"C{str(i).zfill(6)}" for i in range(1, n + 1)],
        "age": np.random.randint(22, 70, n),
        "gender": np.random.choice(["M", "F"], n),
        "education": np.random.choice(["高中", "本科", "硕士", "博士"], n, p=[0.3, 0.45, 0.2, 0.05]),
        "occupation": np.random.choice(["公务员", "企业员工", "个体经营", "自由职业", "其他"], n, p=[0.15, 0.4, 0.2, 0.15, 0.1]),
        "annual_income": np.random.lognormal(10.5, 0.6, n).round(2),
        "num_dependents": np.random.poisson(1, n),
    }
    df = pd.DataFrame(data)
    logger.info(f"[Hive] 加载客户信息表 dw.dim_customer，记录数: {len(df)}")
    return df


def simulate_hive_credit_record_table(customer_ids):
    """
    模拟Hive信用记录表 dw.fact_credit_record
    实际HiveSQL: CREATE TABLE dw.fact_credit_record (customer_id STRING, ...)
    """
    np.random.seed(43)
    n = len(customer_ids)
    data = {
        "customer_id": customer_ids,
        "credit_history_months": np.random.randint(6, 240, n),
        "num_overdue_30d": np.random.poisson(0.5, n),
        "num_overdue_60d": np.random.poisson(0.2, n),
        "num_overdue_90d": np.random.poisson(0.1, n),
        "max_overdue_days": np.random.exponential(15, n).astype(int).clip(0, 180),
        "num_credit_cards": np.random.poisson(2, n),
        "credit_card_utilization": np.random.beta(2, 5, n).round(4),
    }
    df = pd.DataFrame(data)
    logger.info(f"[Hive] 加载信用记录表 dw.fact_credit_record，记录数: {len(df)}")
    return df


def simulate_hive_loan_table(customer_ids):
    """
    模拟Hive贷款信息表 dw.fact_loan
    实际HiveSQL: CREATE TABLE dw.fact_loan (customer_id STRING, ...)
    """
    np.random.seed(44)
    n = len(customer_ids)
    loan_amount = np.random.lognormal(11, 0.8, n).round(2)
    monthly_payment = (loan_amount * np.random.uniform(0.01, 0.03, n)).round(2)
    data = {
        "customer_id": customer_ids,
        "num_loans": np.random.poisson(1.5, n),
        "total_loan_amount": loan_amount,
        "monthly_payment": monthly_payment,
        "loan_purpose": np.random.choice(["购房", "购车", "经营", "消费", "教育"], n, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
        "has_mortgage": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    }
    df = pd.DataFrame(data)
    logger.info(f"[Hive] 加载贷款信息表 dw.fact_loan，记录数: {len(df)}")
    return df


# ============================================================
# 第二部分：模拟HiveSQL查询进行特征工程
# 实际环境通过HiveSQL进行多表JOIN和特征计算
# ============================================================

def hive_sql_feature_engineering(customer_df, credit_df, loan_df):
    """
    模拟HiveSQL特征工程，等价于以下HiveSQL：

    SELECT
        c.customer_id,
        c.age,
        c.annual_income,
        cr.credit_history_months,
        cr.num_overdue_30d + cr.num_overdue_60d + cr.num_overdue_90d AS total_overdue,
        l.total_loan_amount / c.annual_income AS debt_income_ratio,
        l.monthly_payment / (c.annual_income / 12) AS payment_income_ratio,
        cr.credit_card_utilization,
        ...
    FROM dw.dim_customer c
    JOIN dw.fact_credit_record cr ON c.customer_id = cr.customer_id
    JOIN dw.fact_loan l ON c.customer_id = l.customer_id
    """
    # 模拟Hive的多表JOIN操作
    merged = customer_df.merge(credit_df, on="customer_id", how="left")
    merged = merged.merge(loan_df, on="customer_id", how="left")
    merged = merged.fillna(0)
    logger.info(f"[Hive SQL] 多表JOIN完成，记录数: {len(merged)}")

    # 特征工程：计算衍生特征
    merged["total_overdue"] = merged["num_overdue_30d"] + merged["num_overdue_60d"] + merged["num_overdue_90d"]
    merged["debt_income_ratio"] = (merged["total_loan_amount"] / merged["annual_income"]).clip(0, 10)
    merged["payment_income_ratio"] = (merged["monthly_payment"] / (merged["annual_income"] / 12)).clip(0, 1)
    merged["credit_history_length"] = merged["credit_history_months"] / 12
    merged["has_overdue"] = (merged["total_overdue"] > 0).astype(int)
    merged["severe_overdue"] = (merged["num_overdue_90d"] > 0).astype(int)

    # 性别编码
    merged["gender_code"] = (merged["gender"] == "M").astype(int)

    # 学历编码（有序）
    edu_map = {"高中": 1, "本科": 2, "硕士": 3, "博士": 4}
    merged["education_code"] = merged["education"].map(edu_map)

    # 生成标签：违约概率与特征相关
    risk_score = (
        merged["debt_income_ratio"] * 0.3
        + merged["total_overdue"] * 0.25
        + merged["credit_card_utilization"] * 0.2
        + merged["payment_income_ratio"] * 0.15
        - merged["credit_history_length"] * 0.05
        - merged["education_code"] * 0.03
        + np.random.normal(0, 0.15, len(merged))
    )
    merged["default_label"] = (risk_score > risk_score.quantile(0.3)).astype(int)

    logger.info(f"[Hive SQL] 特征工程完成，特征数: {len(merged.columns)}")
    return merged


# ============================================================
# 第三部分：模型训练与对比
# ============================================================

def train_and_evaluate_models(df):
    """训练逻辑回归、随机森林、XGBoost并对比"""
    feature_cols = [
        "age", "annual_income", "gender_code", "education_code",
        "credit_history_length", "total_overdue", "num_overdue_90d",
        "debt_income_ratio", "payment_income_ratio", "credit_card_utilization",
        "has_mortgage", "num_credit_cards", "num_dependents", "num_loans",
        "has_overdue", "severe_overdue",
    ]
    X = df[feature_cols].values
    y = df["default_label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}, 正样本率: {y_train.mean():.3f}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # 模型1：逻辑回归
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    results["逻辑回归"] = {
        "accuracy": accuracy_score(y_test, lr_pred),
        "report": classification_report(y_test, lr_pred, output_dict=True),
    }
    logger.info(f"逻辑回归 准确率: {results['逻辑回归']['accuracy']:.4f}")

    # 模型2：随机森林
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["随机森林"] = {
        "accuracy": accuracy_score(y_test, rf_pred),
        "report": classification_report(y_test, rf_pred, output_dict=True),
        "feature_importance": dict(zip(feature_cols, rf.feature_importances_)),
    }
    logger.info(f"随机森林 准确率: {results['随机森林']['accuracy']:.4f}")

    # 模型3：XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        results["XGBoost"] = {
            "accuracy": accuracy_score(y_test, xgb_pred),
            "report": classification_report(y_test, xgb_pred, output_dict=True),
            "feature_importance": dict(zip(feature_cols, xgb.feature_importances_)),
        }
        logger.info(f"XGBoost 准确率: {results['XGBoost']['accuracy']:.4f}")
    except ImportError:
        logger.warning("XGBoost未安装，跳过XGBoost模型")

    return results, feature_cols


# ============================================================
# 第四部分：信用评分等级映射
# ============================================================

def assign_credit_grade(results, df, feature_cols):
    """根据最优模型的预测概率映射信用等级 A/B/C/D/E"""
    from sklearn.ensemble import RandomForestClassifier

    X = df[feature_cols].values
    y = df["default_label"].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    best_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)

    # 预测违约概率
    prob_default = best_model.predict_proba(X)[:, 1]

    # 按违约概率分箱映射信用等级
    grade_bounds = [0, 0.15, 0.30, 0.50, 0.70, 1.0]
    grade_labels = ["A", "B", "C", "D", "E"]
    df["default_prob"] = prob_default
    df["credit_grade"] = pd.cut(df["default_prob"], bins=grade_bounds, labels=grade_labels, include_lowest=True)

    grade_dist = df["credit_grade"].value_counts().sort_index()
    logger.info("===== 信用评分等级分布 =====")
    for grade, count in grade_dist.items():
        pct = count / len(df) * 100
        logger.info(f"  等级 {grade}: {count}人 ({pct:.1f}%)")

    return df


# ============================================================
# 第五部分：结果输出
# ============================================================

def print_results(results, df):
    """输出完整的分析结果"""
    print("\n" + "=" * 70)
    print("  基于Hive的银行客户信用评分分析 - 结果报告")
    print("=" * 70)

    # 模型对比
    print("\n【一】模型性能对比")
    print("-" * 55)
    print(f"{'模型':<12} {'准确率':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 55)
    for name, res in results.items():
        rpt = res["report"]["weighted avg"]
        print(f"{name:<12} {res['accuracy']:<10.4f} {rpt['precision']:<10.4f} {rpt['recall']:<10.4f} {rpt['f1-score']:<10.4f}")

    # 特征重要性
    print("\n【二】特征重要性排名（随机森林）")
    print("-" * 40)
    if "随机森林" in results and "feature_importance" in results["随机森林"]:
        fi = results["随机森林"]["feature_importance"]
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_fi[:10], 1):
            print(f"  {i}. {feat:<28} {imp:.4f}")

    # 信用等级分布
    print("\n【三】信用评分等级分布")
    print("-" * 40)
    grade_desc = {
        "A": "优质客户，违约风险极低",
        "B": "良好客户，违约风险较低",
        "C": "一般客户，违约风险中等",
        "D": "关注客户，违约风险较高",
        "E": "高风险客户，需重点监控",
    }
    for grade in ["A", "B", "C", "D", "E"]:
        count = (df["credit_grade"] == grade).sum()
        pct = count / len(df) * 100
        print(f"  等级 {grade}: {count:>5}人 ({pct:>5.1f}%) - {grade_desc[grade]}")

    print("\n" + "=" * 70)


# ============================================================
# 主函数
# ============================================================

def main():
    logger.info("=" * 50)
    logger.info("开始执行：基于Hive的银行客户信用评分分析")
    logger.info("=" * 50)

    # Step 1: 模拟Hive数据仓库加载（实际通过Sqoop从业务库导入）
    logger.info("\n--- Step 1: 从Hive数据仓库加载数据 ---")
    customer_df = simulate_hive_customer_table(n=5000)
    credit_df = simulate_hive_credit_record_table(customer_df["customer_id"].tolist())
    loan_df = simulate_hive_loan_table(customer_df["customer_id"].tolist())

    # Step 2: HiveSQL特征工程（多表JOIN + 衍生特征计算）
    logger.info("\n--- Step 2: 执行HiveSQL特征工程 ---")
    merged_df = hive_sql_feature_engineering(customer_df, credit_df, loan_df)

    # Step 3: 模型训练与对比
    logger.info("\n--- Step 3: 模型训练与对比 ---")
    results, feature_cols = train_and_evaluate_models(merged_df)

    # Step 4: 信用评分等级映射
    logger.info("\n--- Step 4: 信用评分等级映射 ---")
    final_df = assign_credit_grade(results, merged_df, feature_cols)

    # Step 5: 结果输出
    print_results(results, final_df)

    logger.info("分析完成！")


if __name__ == "__main__":
    main()
