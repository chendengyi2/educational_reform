"""
模块三：案例资源库建设

离线分析案例：
- 案例1：基于Hive的银行客户信用评分分析
- 案例2：基于Spark MLlib的上市公司财务指标聚类分析
- 案例3：基于HBase的股票行情数据存储与回归预测
- 案例4：基于分类算法的P2P网贷违约风险识别

实时分析案例：
- 案例5：基于Kafka+Spark Streaming的实时股票行情监控
- 案例6：基于Flink的实时交易异常检测
- 案例7：基于Redis的实时推荐系统
- 案例8：基于Storm的实时舆情情感分析
"""

CASES_INFO = {
    "offline": [
        {"id": 1, "name": "银行客户信用评分分析", "tech": "Hive + 分类算法", "file": "case1_hive_credit_scoring.py"},
        {"id": 2, "name": "上市公司财务指标聚类分析", "tech": "Spark MLlib + 聚类算法", "file": "case2_spark_clustering.py"},
        {"id": 3, "name": "股票行情数据存储与回归预测", "tech": "HBase + 回归算法", "file": "case3_hbase_stock_regression.py"},
        {"id": 4, "name": "P2P网贷违约风险识别", "tech": "分类算法 + SMOTE", "file": "case4_p2p_default_risk.py"},
    ],
    "realtime": [
        {"id": 5, "name": "实时股票行情监控", "tech": "Kafka + Spark Streaming + Redis", "file": "case5_kafka_stock_monitor.py"},
        {"id": 6, "name": "实时交易异常检测", "tech": "Flink + 规则引擎", "file": "case6_flink_anomaly_detection.py"},
        {"id": 7, "name": "实时推荐系统", "tech": "Redis + 协同过滤", "file": "case7_redis_realtime_recommend.py"},
        {"id": 8, "name": "实时舆情情感分析", "tech": "Storm + 情感词典", "file": "case8_storm_sentiment_analysis.py"},
    ],
}
