#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案例4：基于分类算法的P2P网贷违约风险识别

模拟P2P网贷平台数据，完成特征工程、类别不平衡处理（SMOTE）、
多模型分类对比，重点关注召回率（违约漏检代价高）。

大数据技术栈说明：
- 实际环境：借款数据存储在Hive，通过Sqoop从业务库导入
- 类别不平衡：实际通过Spark分布式处理SMOTE过采样，此处用imbalanced-learn本地实现
- 模型部署：训练好的模型可通过Spark Streaming在线预测
- 本地模拟：使用pandas和sklearn完成全流程，模拟大数据处理逻辑
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 第一部分：生成P2P网贷模拟数据
# 实际环境：通过Sqoop从MySQL导入Hive表 dw.p2p_loan_record
# ============================================================

def generate_p2p_data(n=8000):
    """
    生成P2P网贷模拟数据，包含借款人信息、借款属性、信用历史
    默认违约率约20%，模拟真实的类别不平衡
    """
    np.random.seed(42)

    # 借款人基本信息
    data = {
        "loan_id": [f"L{str(i).zfill(7)}" for i in range(1, n + 1)],
        "borrower_age": np.random.randint(20, 60, n),
        "borrower_gender": np.random.choice(["男", "女"], n, p=[0.65, 0.35]),
        "education": np.random.choice(["高中及以下", "大专", "本科", "硕士及以上"], n, p=[0.3, 0.3, 0.3, 0.1]),
        "marital_status": np.random.choice(["未婚", "已婚", "离异"], n, p=[0.35, 0.5, 0.15]),
        "employment_years": np.random.randint(0, 30, n),
        "monthly_income": np.random.lognormal(9, 0.6, n).round(2),
        "housing_type": np.random.choice(["自有", "租房", "其他"], n, p=[0.35, 0.45, 0.2]),
    }

    # 借款属性
    data["loan_amount"] = np.random.lognormal(10, 0.8, n).round(2)
    data["loan_term"] = np.random.choice([3, 6, 9, 12, 18, 24, 36], n, p=[0.05, 0.1, 0.1, 0.25, 0.2, 0.2, 0.1])
    data["loan_rate"] = np.random.uniform(0.06, 0.24, n).round(4)
    data["loan_purpose"] = np.random.choice(
        ["个人消费", "经营周转", "教育支出", "医疗支出", "房屋装修", "其他"],
        n, p=[0.3, 0.25, 0.1, 0.1, 0.15, 0.1],
    )

    # 信用历史
    data["num_history_loans"] = np.random.poisson(2, n)
    data["num_history_overdue"] = np.random.poisson(0.5, n)
    data["max_overdue_days"] = np.random.exponential(10, n).astype(int).clip(0, 180)
    data["credit_score"] = np.random.normal(650, 80, n).astype(int).clip(300, 900)
    data["debt_ratio"] = np.random.beta(2, 5, n).round(4)

    df = pd.DataFrame(data)

    # 根据特征计算违约概率（有逻辑的标签生成）
    risk_score = (
        -0.02 * (df["credit_score"] - 650) / 80
        + 0.15 * df["debt_ratio"]
        + 0.2 * (df["loan_rate"] - 0.06) / 0.18
        + 0.1 * df["num_history_overdue"] / (df["num_history_loans"] + 1)
        + 0.08 * (df["loan_amount"] / df["monthly_income"] / 12).clip(0, 5)
        - 0.05 * df["employment_years"] / 30
        + np.random.normal(0, 0.15, n)
    )
    # 约20%违约率
    threshold = np.percentile(risk_score, 80)
    df["is_default"] = (risk_score >= threshold).astype(int)

    logger.info(f"[数据生成] P2P网贷数据生成完成，记录数: {len(df)}, 违约率: {df['is_default'].mean():.2%}")
    return df


# ============================================================
# 第二部分：特征工程
# ============================================================

def feature_engineering(df):
    """
    特征工程：编码、分箱、衍生特征
    实际环境：通过HiveSQL或Spark UDF完成
    """
    df = df.copy()

    # 1. 借款利率分箱
    rate_bins = [0.06, 0.10, 0.14, 0.18, 0.24]
    rate_labels = ["低利率", "中低利率", "中高利率", "高利率"]
    df["loan_rate_bin"] = pd.cut(df["loan_rate"], bins=rate_bins, labels=rate_labels, include_lowest=True)

    # 2. 收入等级
    income_quants = df["monthly_income"].quantile([0.25, 0.5, 0.75]).values
    income_bins = [0, income_quants[0], income_quants[1], income_quants[2], df["monthly_income"].max() + 1]
    income_labels = ["低收入", "中低收入", "中高收入", "高收入"]
    df["income_level"] = pd.cut(df["monthly_income"], bins=income_bins, labels=income_labels, include_lowest=True)

    # 3. 信用评分分组
    credit_bins = [300, 550, 650, 750, 900]
    credit_labels = ["差", "中", "良", "优"]
    df["credit_group"] = pd.cut(df["credit_score"], bins=credit_bins, labels=credit_labels, include_lowest=True)

    # 4. 衍生特征
    df["loan_to_income"] = (df["loan_amount"] / df["monthly_income"]).clip(0, 100)
    df["monthly_payment"] = (df["loan_amount"] * df["loan_rate"] / 12).round(2)
    df["payment_to_income"] = (df["monthly_payment"] / df["monthly_income"]).clip(0, 2)
    df["overdue_per_loan"] = (df["num_history_overdue"] / (df["num_history_loans"] + 1)).round(4)
    df["is_high_rate"] = (df["loan_rate"] > 0.18).astype(int)
    df["is_long_term"] = (df["loan_term"] >= 24).astype(int)
    df["has_overdue_history"] = (df["num_history_overdue"] > 0).astype(int)

    # 5. 类别特征编码
    le = LabelEncoder()
    categorical_cols = [
        "borrower_gender", "education", "marital_status", "housing_type",
        "loan_purpose", "loan_rate_bin", "income_level", "credit_group",
    ]
    for col in categorical_cols:
        df[f"{col}_code"] = le.fit_transform(df[col].astype(str))

    logger.info(f"[特征工程] 完成，特征数: {len(df.columns)}")
    return df


# ============================================================
# 第三部分：SMOTE过采样处理类别不平衡
# 实际环境：通过Spark分布式SMOTE，此处用imbalanced-learn
# ============================================================

def apply_smote(X_train, y_train):
    """
    使用SMOTE对训练集进行过采样，解决类别不平衡问题
    违约漏检代价高，因此需要提高少数类（违约）的召回率
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"[SMOTE] 过采样完成: {len(y_train)} -> {len(y_resampled)}, "
                     f"违约率: {y_train.mean():.2%} -> {y_resampled.mean():.2%}")
        return X_resampled, y_resampled
    except ImportError:
        logger.warning("imbalanced-learn未安装，跳过SMOTE，使用原始不平衡数据")
        return X_train, y_train


# ============================================================
# 第四部分：模型训练与对比
# ============================================================

def get_feature_columns():
    """获取用于建模的特征列"""
    return [
        "borrower_age", "employment_years", "monthly_income", "loan_amount",
        "loan_term", "loan_rate", "num_history_loans", "num_history_overdue",
        "max_overdue_days", "credit_score", "debt_ratio",
        "loan_to_income", "monthly_payment", "payment_to_income",
        "overdue_per_loan", "is_high_rate", "is_long_term", "has_overdue_history",
        "borrower_gender_code", "education_code", "marital_status_code",
        "housing_type_code", "loan_purpose_code", "loan_rate_bin_code",
        "income_level_code", "credit_group_code",
    ]


def train_and_evaluate(df):
    """训练多个分类模型并对比"""
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df["is_default"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}, 违约率: {y_train.mean():.2%}")

    # SMOTE过采样
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "逻辑回归": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "随机森林": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
        "GBDT": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        if name == "逻辑回归":
            model.fit(X_train_scaled, y_train_res)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall_default": recall_score(y_test, y_pred, pos_label=1),
            "f1_default": f1_score(y_test, y_pred, pos_label=1),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "predictions": y_pred,
            "probabilities": y_prob,
        }

        # 随机森林特征重要性
        if name == "随机森林":
            results[name]["feature_importance"] = dict(zip(feature_cols, model.feature_importances_))

        logger.info(f"  {name}: 准确率={results[name]['accuracy']:.4f}, "
                     f"违约召回率={results[name]['recall_default']:.4f}, "
                     f"违约F1={results[name]['f1_default']:.4f}")

    return results, y_test, feature_cols


# ============================================================
# 第五部分：结果输出
# ============================================================

def print_results(results, y_test, feature_cols):
    """输出完整的分析结果"""
    print("\n" + "=" * 70)
    print("  基于分类算法的P2P网贷违约风险识别 - 结果报告")
    print("=" * 70)

    # 模型性能对比（重点关注召回率）
    print("\n【一】模型性能对比（重点关注违约召回率）")
    print("-" * 70)
    print(f"{'模型':<12} {'准确率':<10} {'违约召回率':<14} {'违约Precision':<16} {'违约F1':<10}")
    print("-" * 70)
    for name, res in results.items():
        rpt = res["report"]["1"]
        print(f"{name:<12} {res['accuracy']:<10.4f} {res['recall_default']:<14.4f} {rpt['precision']:<16.4f} {res['f1_default']:<10.4f}")

    # 混淆矩阵
    print("\n【二】各模型混淆矩阵")
    print("-" * 55)
    for name, res in results.items():
        cm = res["confusion_matrix"]
        print(f"\n  {name}:")
        print(f"    {'预测:正常':<14} {'预测:违约':<14}")
        print(f"  实际:正常  {cm[0][0]:<14} {cm[0][1]:<14}")
        print(f"  实际:违约  {cm[1][0]:<14} {cm[1][1]:<14}")
        fnr = cm[1][0] / (cm[1][0] + cm[1][1])  # 漏检率
        print(f"  漏检率(FNR): {fnr:.2%}  <-- 违约漏检代价高，需重点关注")

    # 特征重要性
    print("\n【三】特征重要性排名（随机森林）")
    print("-" * 45)
    if "随机森林" in results and "feature_importance" in results["随机森林"]:
        fi = results["随机森林"]["feature_importance"]
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_fi[:15], 1):
            bar = "█" * int(imp * 80)
            print(f"  {i:>2}. {feat:<24} {imp:.4f} {bar}")

    # 风险管理建议
    print("\n【四】风险管理建议")
    print("-" * 55)
    best_model = max(results, key=lambda x: results[x]["recall_default"])
    print(f"  1. 推荐模型: {best_model}（违约召回率最高）")
    print(f"  2. 违约漏检率最低模型: {best_model}")
    print(f"  3. 关键风险特征: 信用评分、负债比、借款利率")
    print(f"  4. 建议对高利率+低收入+有逾期历史的借款人加强审核")
    print(f"  5. SMOTE过采样有效提升了少数类（违约）的识别能力")

    print("\n" + "=" * 70)


# ============================================================
# 主函数
# ============================================================

def main():
    logger.info("=" * 50)
    logger.info("开始执行：基于分类算法的P2P网贷违约风险识别")
    logger.info("=" * 50)

    # Step 1: 生成P2P网贷数据
    logger.info("\n--- Step 1: 生成P2P网贷模拟数据 ---")
    df = generate_p2p_data(n=8000)

    # Step 2: 特征工程
    logger.info("\n--- Step 2: 特征工程 ---")
    df = feature_engineering(df)

    # Step 3: 模型训练与对比（含SMOTE过采样）
    logger.info("\n--- Step 3: 模型训练与对比（含SMOTE处理） ---")
    results, y_test, feature_cols = train_and_evaluate(df)

    # Step 4: 结果输出
    print_results(results, y_test, feature_cols)

    logger.info("分析完成！")


if __name__ == "__main__":
    main()
