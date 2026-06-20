#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案例6：基于Flink的实时交易异常检测

模拟架构：
- 交易数据源 → Flink Source → 窗口函数 → 异常检测规则引擎 → 告警输出

模拟内容：
1. 实时交易数据流（交易ID、账户、金额、时间、地点、类型）
2. Flink窗口函数模拟：滚动窗口统计交易频率和金额
3. 异常检测规则：
   - 短时间高频交易
   - 单笔大额交易
   - 异常时段交易
   - 同账户多地点交易
4. 输出异常交易告警列表和统计

生产环境替换：
- Flink Source → 真实Flink DataStream（from_kafka / from_socket）
- 窗口函数 → 真实Flink key_by + window 操作
- 状态管理 → 真实Flink State Backend
"""

import random
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AnomalyDetection")


# ============================================================
# 模拟交易数据源 - 生成实时交易数据流
# ============================================================
class SimulatedTransactionSource:
    """
    模拟交易数据源，生成实时交易记录。
    生产环境替换为：Flink KafkaSource / SocketSource
    """

    ACCOUNTS = [f"ACC{str(i).zfill(6)}" for i in range(1, 21)]
    CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]
    TRANSACTION_TYPES = ["转账", "消费", "提现", "充值", "理财购买"]
    # 大额交易模板（偶尔触发）
    LARGE_AMOUNT_ACCOUNTS = ["ACC000003", "ACC000007"]

    def __init__(self):
        self.transaction_counter = 0
        self.simulated_time = datetime(2026, 6, 14, 8, 0, 0)  # 从早上8点开始

    def generate_transaction(self):
        """生成单条交易记录，含正常和异常模式"""
        self.transaction_counter += 1
        # 推进模拟时间（每次0~5分钟随机推进）
        self.simulated_time += timedelta(seconds=random.randint(10, 300))

        account = random.choice(self.ACCOUNTS)
        city = random.choice(self.CITIES)
        tx_type = random.choice(self.TRANSACTION_TYPES)

        # 金额分布：大部分小额，少数大额
        if random.random() < 0.03:  # 3%大额交易
            amount = round(random.uniform(50000, 500000), 2)
        else:
            amount = round(random.uniform(10, 10000), 2)

        return {
            "transaction_id": f"TX{self.transaction_counter:08d}",
            "account": account,
            "amount": amount,
            "timestamp": self.simulated_time.strftime("%Y-%m-%d %H:%M:%S"),
            "city": city,
            "type": tx_type
        }

    def generate_batch(self, batch_size=20):
        """批量生成交易记录"""
        return [self.generate_transaction() for _ in range(batch_size)]


# ============================================================
# 模拟Flink窗口函数 - 滚动窗口统计
# ============================================================
class SimulatedFlinkWindow:
    """
    模拟Flink滚动窗口，对交易数据按账户分组并统计窗口内指标。
    生产环境替换为：datastream.key_by(lambda x: x['account']).window(TumblingEventTimeWindows.of(Time.minutes(5)))
    """

    def __init__(self, window_size_minutes=5):
        self.window_size_minutes = window_size_minutes
        # 每个账户的交易记录窗口
        self.account_windows = defaultdict(lambda: deque(maxlen=100))
        # 全局窗口（所有交易）
        self.global_window = deque(maxlen=500)

    def process_batch(self, batch_data):
        """
        处理一个批次的交易数据，按账户分组加入窗口。
        模拟Flink的 key_by + window + aggregate 操作。
        """
        for tx in batch_data:
            self.account_windows[tx["account"]].append(tx)
            self.global_window.append(tx)

        return self._compute_window_stats()

    def _compute_window_stats(self):
        """
        计算各账户在窗口内的统计指标：
        - 交易次数
        - 总金额
        - 平均金额
        - 最大单笔金额
        - 交易城市集合
        - 交易时段
        """
        stats = {}
        for account, txs in self.account_windows.items():
            if not txs:
                continue
            amounts = [tx["amount"] for tx in txs]
            cities = set(tx["city"] for tx in txs)
            hours = [datetime.strptime(tx["timestamp"], "%Y-%m-%d %H:%M:%S").hour for tx in txs]

            stats[account] = {
                "tx_count": len(txs),
                "total_amount": round(sum(amounts), 2),
                "avg_amount": round(sum(amounts) / len(amounts), 2),
                "max_amount": max(amounts),
                "cities": cities,
                "city_count": len(cities),
                "hour_range": (min(hours), max(hours)),
                "latest_tx": txs[-1]["transaction_id"],
                "types": set(tx["type"] for tx in txs)
            }
        return stats


# ============================================================
# 异常检测规则引擎
# ============================================================
class AnomalyRuleEngine:
    """
    基于规则的实时异常检测引擎。
    在Flink中对应 ProcessFunction / KeyedProcessFunction。
    """

    def __init__(self):
        # 规则参数
        self.high_freq_threshold = 8       # 5分钟内超过8笔视为高频
        self.large_amount_threshold = 50000  # 单笔超过5万视为大额
        self.abnormal_hours = set(range(0, 6))  # 凌晨0-5点视为异常时段
        self.multi_city_threshold = 3       # 3个以上不同城市视为异常

    def detect(self, tx_data_list, account_stats):
        """
        对交易数据和窗口统计进行异常检测。
        - tx_data_list: 本批次原始交易列表
        - account_stats: 窗口统计结果
        """
        anomalies = []

        for tx in tx_data_list:
            # 规则2：单笔大额交易（逐条检查）
            if tx["amount"] >= self.large_amount_threshold:
                anomalies.append({
                    "rule": "LARGE_AMOUNT",
                    "severity": "HIGH",
                    "account": tx["account"],
                    "transaction_id": tx["transaction_id"],
                    "detail": f"单笔大额交易 {tx['amount']:.2f}元，"
                              f"类型:{tx['type']}，城市:{tx['city']}",
                    "timestamp": tx["timestamp"]
                })

            # 规则3：异常时段交易（逐条检查）
            tx_hour = datetime.strptime(tx["timestamp"], "%Y-%m-%d %H:%M:%S").hour
            if tx_hour in self.abnormal_hours:
                anomalies.append({
                    "rule": "ABNORMAL_HOUR",
                    "severity": "MEDIUM",
                    "account": tx["account"],
                    "transaction_id": tx["transaction_id"],
                    "detail": f"异常时段交易({tx_hour}:00)，"
                              f"金额:{tx['amount']:.2f}元，城市:{tx['city']}",
                    "timestamp": tx["timestamp"]
                })

        # 基于窗口统计的规则
        for account, stats in account_stats.items():
            # 规则1：短时间高频交易
            if stats["tx_count"] >= self.high_freq_threshold:
                anomalies.append({
                    "rule": "HIGH_FREQUENCY",
                    "severity": "HIGH",
                    "account": account,
                    "transaction_id": stats["latest_tx"],
                    "detail": f"高频交易：{stats['tx_count']}笔/"
                              f"{self.high_freq_threshold}阈值，"
                              f"总金额:{stats['total_amount']:.2f}元",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            # 规则4：同账户多地点交易
            if stats["city_count"] >= self.multi_city_threshold:
                anomalies.append({
                    "rule": "MULTI_CITY",
                    "severity": "HIGH",
                    "account": account,
                    "transaction_id": stats["latest_tx"],
                    "detail": f"多地点交易：{stats['city_count']}个城市 "
                              f"({','.join(stats['cities'])})，"
                              f"交易{stats['tx_count']}笔",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        return anomalies


# ============================================================
# 主流程
# ============================================================
def main():
    logger.info("=" * 70)
    logger.info("实时交易异常检测系统启动 (Flink 模拟)")
    logger.info("=" * 70)

    # 初始化各组件
    source = SimulatedTransactionSource()
    flink_window = SimulatedFlinkWindow(window_size_minutes=5)
    rule_engine = AnomalyRuleEngine()

    # 模拟运行参数
    total_rounds = 25
    batch_size = 25

    all_anomalies = []
    total_transactions = 0
    anomaly_by_rule = defaultdict(int)
    anomaly_by_account = defaultdict(int)

    for round_idx in range(1, total_rounds + 1):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"第 {round_idx}/{total_rounds} 轮处理")

        # Step 1: 模拟数据源产生交易
        batch = source.generate_batch(batch_size)
        total_transactions += len(batch)
        logger.info(f"  [Flink Source] 接收 {len(batch)} 条交易记录")

        # Step 2: 窗口函数处理
        account_stats = flink_window.process_batch(batch)
        logger.info(f"  [Flink Window] 统计 {len(account_stats)} 个活跃账户")

        # Step 3: 异常检测
        anomalies = rule_engine.detect(batch, account_stats)
        all_anomalies.extend(anomalies)

        # 统计
        for a in anomalies:
            anomaly_by_rule[a["rule"]] += 1
            anomaly_by_account[a["account"]] += 1

        # 输出本批次异常
        if anomalies:
            for a in anomalies:
                severity_tag = "[!!!]" if a["severity"] == "HIGH" else "[!! ]"
                logger.warning(f"  {severity_tag} [{a['rule']}] {a['account']}: {a['detail']}")
        else:
            logger.info("  本轮无异常交易")

        # 偶尔输出活跃账户概况
        if round_idx % 5 == 0:
            top_accounts = sorted(account_stats.items(), key=lambda x: x[1]["tx_count"], reverse=True)[:3]
            logger.info("  活跃账户TOP3:")
            for acc, stats in top_accounts:
                logger.info(f"    {acc}: {stats['tx_count']}笔, "
                            f"总额{stats['total_amount']:.0f}元, "
                            f"{stats['city_count']}个城市")

    # ============================================================
    # 汇总报告
    # ============================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("实时交易异常检测 - 运行汇总")
    logger.info(f"{'=' * 70}")
    logger.info(f"总交易笔数: {total_transactions}")
    logger.info(f"总异常数量: {len(all_anomalies)}")
    logger.info(f"异常率: {len(all_anomalies)/total_transactions*100:.2f}%")

    logger.info("\n异常规则分布:")
    rule_names = {
        "HIGH_FREQUENCY": "高频交易",
        "LARGE_AMOUNT": "大额交易",
        "ABNORMAL_HOUR": "异常时段",
        "MULTI_CITY": "多地点交易"
    }
    for rule, count in sorted(anomaly_by_rule.items(), key=lambda x: -x[1]):
        logger.info(f"  {rule_names.get(rule, rule)}: {count}次")

    logger.info("\n异常账户TOP5:")
    for acc, count in sorted(anomaly_by_account.items(), key=lambda x: -x[1])[:5]:
        logger.info(f"  {acc}: {count}次异常")

    # 输出高风险告警列表
    high_severity = [a for a in all_anomalies if a["severity"] == "HIGH"]
    if high_severity:
        logger.info(f"\n高风险告警详情 (共{len(high_severity)}条):")
        for a in high_severity[:10]:
            logger.info(f"  [{a['rule']}] {a['account']} | {a['detail']}")


if __name__ == "__main__":
    main()
