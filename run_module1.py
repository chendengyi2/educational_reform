"""
模块一：财经大数据收集与预处理 - 主程序

使用方式：
  python run_module1.py --demo          # 运行完整演示
  python run_module1.py --collect       # 查看可采集数据集
  python run_module1.py --preprocess --input data.csv  # 预处理指定文件
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from module1_data_collect.utils import setup_logger, auto_load
from module1_data_collect.collectors import TianchiCollector, KaggleCollector, FinanceCrawler
from module1_data_collect.preprocessors import PreprocessingPipeline

logger = setup_logger("module1", log_file="logs/module1.log")


def run_demo():
    """运行完整演示"""
    import pandas as pd
    import numpy as np

    print("\n" + "=" * 60)
    print("  模块一：财经大数据收集与预处理 - 完整演示")
    print("=" * 60)

    # 1. 展示数据采集能力
    print("\n[1] 数据采集模块")
    tc = TianchiCollector()
    tc.list_datasets()
    kc = KaggleCollector()
    kc.list_datasets()
    print("\n爬虫模块: 支持 新浪财经、东方财富 采集")

    # 2. 生成示例数据并预处理
    print("\n[2] 数据预处理流水线演示")
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "loan_id": range(1, n + 1),
        "age": np.random.randint(22, 65, n),
        "income": np.random.lognormal(10.5, 0.8, n).round(2),
        "loan_amount": np.random.lognormal(11, 0.6, n).round(2),
        "credit_score": np.clip(np.random.normal(680, 80, n), 300, 850).astype(int),
        "debt_ratio": np.clip(np.random.beta(2, 5, n), 0, 1).round(4),
        "employment_years": np.random.exponential(5, n).round(1),
        "default": np.random.binomial(1, 0.15, n),
    })

    # 注入缺失值和异常值
    for col in ["income", "credit_score", "debt_ratio"]:
        mask = np.random.random(n) < 0.05
        df.loc[mask, col] = np.nan
    df.loc[np.random.choice(n, 5), "income"] = df["income"].mean() * 50

    print(f"原始数据形状: {df.shape}，缺失值: {df.isnull().sum().sum()}")

    # 执行流水线
    pipeline = PreprocessingPipeline()
    df_processed = pipeline.run(df)

    print(f"\n处理后数据形状: {df_processed.shape}，缺失值: {df_processed.isnull().sum().sum()}")
    print(pipeline.get_pipeline_report())

    # 保存
    output = "data/processed/module1_demo_result.csv"
    df_processed.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {output}")


def main():
    parser = argparse.ArgumentParser(description="模块一：财经大数据收集与预处理")
    parser.add_argument("--demo", action="store_true", help="运行完整演示")
    parser.add_argument("--collect", action="store_true", help="查看可采集数据集")
    parser.add_argument("--preprocess", action="store_true", help="预处理数据")
    parser.add_argument("--input", type=str, help="输入数据文件路径")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.collect:
        TianchiCollector().list_datasets()
        KaggleCollector().list_datasets()
    elif args.preprocess and args.input:
        df = auto_load(args.input)
        pipeline = PreprocessingPipeline()
        result = pipeline.run(df)
        print(pipeline.get_pipeline_report())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()   # 运行
