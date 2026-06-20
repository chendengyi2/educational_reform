#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案例5：基于Kafka+Spark Streaming的实时股票行情监控

模拟架构：
- Kafka Producer → Kafka Topic → Spark Streaming → Redis存储 → 告警输出

模拟内容：
1. Kafka Producer：生成实时股票行情消息（股票代码、价格、成交量、时间戳）
2. Spark Streaming：滑动窗口计算均值、波动率、涨跌幅
3. 异常检测：价格突变（涨跌幅超过阈值）、成交量异常放大
4. Redis：存储实时指标（模拟dict实现）
5. 输出实时监控告警信息

生产环境替换：
- Kafka Producer → 真实Kafka生产者连接broker
- Spark Streaming → 真实Spark StreamingContext
- Redis → 真实Redis连接
"""

import random
import math
import logging
import time
from collections import defaultdict, deque
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("StockMonitor")


# ============================================================
# 模拟Kafka Producer - 生成实时股票行情消息
# ============================================================
class SimulatedKafkaProducer:
    """
    模拟Kafka Producer，生成实时股票行情数据。
    生产环境替换为：from kafka import KafkaProducer; producer = KafkaProducer(bootstrap_servers='broker:9092')
    """

    def __init__(self, stock_codes, initial_prices):
        self.stock_codes = stock_codes
        self.prices = {code: initial_prices[i] for i, code in enumerate(stock_codes)}
        self.base_volumes = {code: random.randint(5000, 50000) for code in stock_codes}

    def generate_tick(self):
        """生成一条股票行情消息，模拟价格随机游走"""
        code = random.choice(self.stock_codes)
        # 几何布朗运动模拟价格变动
        drift = 0.0001
        volatility = 0.02
        shock = random.gauss(0, 1)
        self.prices[code] *= math.exp(drift + volatility * shock)
        self.prices[code] = max(1.0, self.prices[code])

        # 成交量随机波动，偶尔出现异常放大
        volume_multiplier = random.uniform(0.5, 2.0)
        if random.random() < 0.05:  # 5%概率出现异常放量
            volume_multiplier = random.uniform(5.0, 15.0)
        volume = int(self.base_volumes[code] * volume_multiplier)

        message = {
            "stock_code": code,
            "price": round(self.prices[code], 2),
            "volume": volume,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }
        return message

    def produce_batch(self, batch_size=10):
        """批量生成行情消息，模拟一个时间窗口的数据"""
        return [self.generate_tick() for _ in range(batch_size)]


# ============================================================
# 模拟Spark Streaming - 滑动窗口计算
# ============================================================
class SimulatedSparkStreaming:
    """
    模拟Spark Streaming的滑动窗口计算。
    生产环境替换为：spark.streaming.StreamingContext + DStream窗口操作
    """

    def __init__(self, window_size=10, slide_interval=5):
        self.window_size = window_size      # 窗口大小（保留的数据批次）
        self.slide_interval = slide_interval  # 滑动间隔
        # 每只股票保留最近window_size个批次的数据
        self.windows = defaultdict(lambda: deque(maxlen=window_size))
        self.prev_metrics = {}  # 上一窗口的指标，用于计算涨跌幅

    def process_batch(self, batch_data):
        """
        处理一个批次的数据，按股票分组后加入滑动窗口。
        相当于Spark Streaming的 transform + reduceByKey + window 操作。
        """
        grouped = defaultdict(list)
        for msg in batch_data:
            grouped[msg["stock_code"]].append(msg)

        for code, ticks in grouped.items():
            self.windows[code].append(ticks)

        return self._compute_window_metrics()

    def _compute_window_metrics(self):
        """
        对滑动窗口内的数据计算统计指标：
        - 均价、最高价、最低价
        - 波动率（标准差/均值）
        - 涨跌幅（相对上一窗口）
        - 总成交量、平均成交量
        """
        metrics = {}
        for code, window in self.windows.items():
            all_ticks = [t for batch in window for t in batch]
            if not all_ticks:
                continue

            prices = [t["price"] for t in all_ticks]
            volumes = [t["volume"] for t in all_ticks]

            avg_price = sum(prices) / len(prices)
            max_price = max(prices)
            min_price = min(prices)
            std_price = math.sqrt(sum((p - avg_price) ** 2 for p in prices) / len(prices))
            volatility = std_price / avg_price if avg_price > 0 else 0

            total_volume = sum(volumes)
            avg_volume = total_volume / len(volumes)

            # 计算涨跌幅
            price_change = 0.0
            if code in self.prev_metrics:
                prev_avg = self.prev_metrics[code]["avg_price"]
                if prev_avg > 0:
                    price_change = (avg_price - prev_avg) / prev_avg * 100

            metrics[code] = {
                "avg_price": round(avg_price, 2),
                "max_price": round(max_price, 2),
                "min_price": round(min_price, 2),
                "volatility": round(volatility, 4),
                "price_change_pct": round(price_change, 2),
                "total_volume": total_volume,
                "avg_volume": round(avg_volume, 0),
                "tick_count": len(all_ticks),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        self.prev_metrics = metrics.copy()
        return metrics


# ============================================================
# 模拟Redis - 存储实时指标
# ============================================================
class SimulatedRedis:
    """
    模拟Redis存储实时指标。
    生产环境替换为：import redis; r = redis.Redis(host='localhost', port=6379)
    """

    def __init__(self):
        self.data = {}

    def set(self, key, value):
        """设置键值对，对应Redis SET"""
        self.data[key] = value

    def get(self, key):
        """获取键值，对应Redis GET"""
        return self.data.get(key)

    def hset(self, name, key, value):
        """设置哈希字段，对应Redis HSET"""
        if name not in self.data:
            self.data[name] = {}
        self.data[name][key] = value

    def hget(self, name, key):
        """获取哈希字段，对应Redis HGET"""
        return self.data.get(name, {}).get(key)

    def hgetall(self, name):
        """获取哈希所有字段，对应Redis HGETALL"""
        return self.data.get(name, {})


# ============================================================
# 异常检测器
# ============================================================
class AnomalyDetector:
    """实时异常检测：价格突变和成交量异常"""

    def __init__(self, price_change_threshold=3.0, volume_multiplier_threshold=5.0):
        self.price_change_threshold = price_change_threshold        # 涨跌幅阈值(%)
        self.volume_multiplier_threshold = volume_multiplier_threshold  # 成交量异常倍数
        self.baseline_volumes = {}  # 基线成交量

    def detect(self, metrics):
        """
        检测异常并生成告警。
        规则：
        1. 价格突变：涨跌幅超过阈值
        2. 成交量异常：当前平均成交量超过基线的N倍
        """
        alerts = []
        for code, m in metrics.items():
            # 更新基线成交量（首次或缓慢适应）
            if code not in self.baseline_volumes:
                self.baseline_volumes[code] = m["avg_volume"]
            else:
                self.baseline_volumes[code] = 0.8 * self.baseline_volumes[code] + 0.2 * m["avg_volume"]

            # 规则1：价格突变
            if abs(m["price_change_pct"]) > self.price_change_threshold:
                direction = "暴涨" if m["price_change_pct"] > 0 else "暴跌"
                alerts.append({
                    "alert_type": "PRICE_SURGE",
                    "stock_code": code,
                    "severity": "HIGH" if abs(m["price_change_pct"]) > 2 * self.price_change_threshold else "MEDIUM",
                    "message": f"{code} {direction} {abs(m['price_change_pct']):.2f}%，"
                               f"当前均价{m['avg_price']}，波动率{m['volatility']}",
                    "timestamp": m["timestamp"]
                })

            # 规则2：成交量异常放大
            baseline = self.baseline_volumes[code]
            if baseline > 0 and m["avg_volume"] > baseline * self.volume_multiplier_threshold:
                alerts.append({
                    "alert_type": "VOLUME_ANOMALY",
                    "stock_code": code,
                    "severity": "HIGH",
                    "message": f"{code} 成交量异常放大 {m['avg_volume']/baseline:.1f}倍，"
                               f"当前均量{m['avg_volume']:.0f}，基线{baseline:.0f}",
                    "timestamp": m["timestamp"]
                })

        return alerts


# ============================================================
# 主流程
# ============================================================
def main():
    logger.info("=" * 70)
    logger.info("实时股票行情监控系统启动 (Kafka + Spark Streaming 模拟)")
    logger.info("=" * 70)

    # 初始化各组件
    stock_codes = ["SH600519", "SH000001", "SZ000858", "SH601318", "SZ000333"]
    stock_names = ["贵州茅台", "上证指数", "五粮液", "中国平安", "美的集团"]
    initial_prices = [1800.0, 3200.0, 180.0, 50.0, 65.0]
    code_name_map = dict(zip(stock_codes, stock_names))

    producer = SimulatedKafkaProducer(stock_codes, initial_prices)
    streaming = SimulatedSparkStreaming(window_size=10, slide_interval=5)
    redis = SimulatedRedis()
    detector = AnomalyDetector(price_change_threshold=3.0, volume_multiplier_threshold=5.0)

    # 模拟运行参数
    total_rounds = 20  # 模拟轮次
    batch_size = 15    # 每批消息数

    all_alerts = []    # 收集所有告警

    for round_idx in range(1, total_rounds + 1):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"第 {round_idx}/{total_rounds} 轮处理")

        # Step 1: Kafka Producer 生成消息
        batch = producer.produce_batch(batch_size)
        logger.info(f"  [Kafka Producer] 生成 {len(batch)} 条行情消息")

        # Step 2: Spark Streaming 消费并处理
        metrics = streaming.process_batch(batch)

        # Step 3: 写入Redis存储实时指标
        for code, m in metrics.items():
            redis.hset(f"stock:realtime:{code}", "avg_price", m["avg_price"])
            redis.hset(f"stock:realtime:{code}", "volatility", m["volatility"])
            redis.hset(f"stock:realtime:{code}", "price_change_pct", m["price_change_pct"])
            redis.hset(f"stock:realtime:{code}", "total_volume", m["total_volume"])
            redis.hset(f"stock:realtime:{code}", "timestamp", m["timestamp"])

        # Step 4: 异常检测
        alerts = detector.detect(metrics)
        all_alerts.extend(alerts)

        # 输出当前窗口指标
        for code, m in metrics.items():
            name = code_name_map.get(code, code)
            change_icon = "↑" if m["price_change_pct"] > 0 else "↓" if m["price_change_pct"] < 0 else "→"
            logger.info(
                f"  [{name}] 均价:{m['avg_price']:>8.2f}  "
                f"波动率:{m['volatility']:.4f}  "
                f"涨跌:{change_icon}{abs(m['price_change_pct']):>5.2f}%  "
                f"成交量:{m['total_volume']:>8}"
            )

        # 输出告警
        for alert in alerts:
            severity_tag = "[!!!]" if alert["severity"] == "HIGH" else "[!! ]"
            logger.warning(f"  {severity_tag} 告警: {alert['message']}")

    # ============================================================
    # 汇总报告
    # ============================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("实时股票行情监控 - 运行汇总")
    logger.info(f"{'=' * 70}")
    logger.info(f"总处理轮次: {total_rounds}")
    logger.info(f"总告警数量: {len(all_alerts)}")

    # 按类型统计告警
    alert_by_type = defaultdict(int)
    alert_by_code = defaultdict(int)
    for a in all_alerts:
        alert_by_type[a["alert_type"]] += 1
        alert_by_code[a["stock_code"]] += 1

    logger.info("\n告警类型分布:")
    for atype, count in alert_by_type.items():
        logger.info(f"  {atype}: {count}次")

    logger.info("\n告警股票分布:")
    for code, count in sorted(alert_by_code.items(), key=lambda x: -x[1]):
        name = code_name_map.get(code, code)
        logger.info(f"  {name}({code}): {count}次")

    # 输出Redis中的最终指标
    logger.info("\nRedis存储的最终实时指标:")
    for code in stock_codes:
        data = redis.hgetall(f"stock:realtime:{code}")
        if data:
            name = code_name_map.get(code, code)
            logger.info(f"  {name}: 均价={data.get('avg_price', 'N/A')}, "
                        f"波动率={data.get('volatility', 'N/A')}, "
                        f"涨跌幅={data.get('price_change_pct', 'N/A')}%")


if __name__ == "__main__":
    main()
