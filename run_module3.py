"""
模块三：案例资源库建设 - 主程序

使用方式：
  python run_module3.py --list              # 列出所有案例
  python run_module3.py --run 1             # 运行案例1
  python run_module3.py --run 5             # 运行案例5
  python run_module3.py --run-all-offline   # 运行所有离线案例
  python run_module3.py --run-all-realtime  # 运行所有实时案例
"""

import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

CASES = {
    1: ("offline", "case1_hive_credit_scoring.py", "基于Hive的银行客户信用评分分析"),
    2: ("offline", "case2_spark_clustering.py", "基于Spark MLlib的上市公司财务指标聚类分析"),
    3: ("offline", "case3_hbase_stock_regression.py", "基于HBase的股票行情数据存储与回归预测"),
    4: ("offline", "case4_p2p_default_risk.py", "基于分类算法的P2P网贷违约风险识别"),
    5: ("realtime", "case5_kafka_stock_monitor.py", "基于Kafka+Spark Streaming的实时股票行情监控"),
    6: ("realtime", "case6_flink_anomaly_detection.py", "基于Flink的实时交易异常检测"),
    7: ("realtime", "case7_redis_realtime_recommend.py", "基于Redis的实时推荐系统"),
    8: ("realtime", "case8_storm_sentiment_analysis.py", "基于Storm的实时舆情情感分析"),
}


def list_cases():
    print("\n" + "=" * 70)
    print("  案例资源库")
    print("=" * 70)
    print("\n【离线分析案例】")
    for cid, (cat, fname, desc) in CASES.items():
        if cat == "offline":
            print(f"  案例{cid}: {desc}")
            print(f"         技术栈: Sqoop + Spark + Hive/HBase + Echarts + Flask + 数据挖掘算法")
            print(f"         文件: module3_cases/offline/{fname}")
    print("\n【实时分析案例】")
    for cid, (cat, fname, desc) in CASES.items():
        if cat == "realtime":
            print(f"  案例{cid}: {desc}")
            print(f"         技术栈: Kafka + Spark Streaming/Flink + Redis + Flask + 数据挖掘算法")
            print(f"         文件: module3_cases/realtime/{fname}")


def run_case(case_id: int):
    if case_id not in CASES:
        print(f"错误: 案例{case_id}不存在，可用范围 1-8")
        return
    cat, fname, desc = CASES[case_id]
    filepath = PROJECT_ROOT / "module3_cases" / cat / fname
    print(f"\n运行案例{case_id}: {desc}")
    print(f"文件: {filepath}")
    print("-" * 60)
    subprocess.run([sys.executable, str(filepath)], check=False)


def run_category(category: str):
    for cid, (cat, fname, desc) in CASES.items():
        if cat == category:
            run_case(cid)
            print()


def main():
    parser = argparse.ArgumentParser(description="模块三：案例资源库建设")
    parser.add_argument("--list", action="store_true", help="列出所有案例")
    parser.add_argument("--run", type=int, help="运行指定案例(1-8)")
    parser.add_argument("--run-all-offline", action="store_true", help="运行所有离线案例")
    parser.add_argument("--run-all-realtime", action="store_true", help="运行所有实时案例")
    args = parser.parse_args()

    if args.list:
        list_cases()
    elif args.run:
        run_case(args.run)
    elif args.run_all_offline:
        run_category("offline")
    elif args.run_all_realtime:
        run_category("realtime")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
