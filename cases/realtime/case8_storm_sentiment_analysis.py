#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案例8：基于Storm的实时舆情情感分析

模拟架构：
- 舆情数据Spout → 分词Bolt → 情感分析Bolt → 热点统计Bolt → 输出

模拟内容：
1. 模拟实时财经舆情数据流（新闻标题、来源、时间）
2. 文本预处理（分词、停用词过滤、关键词提取）— 用简单规则模拟
3. 情感分析（基于金融情感词典的正负面判断）
4. 热点话题统计（滑动窗口关键词频率）
5. 输出情感分析结果和舆情趋势

生产环境替换：
- Storm Spout → 真实KafkaSpout / TwitterSpout
- Storm Bolt → 真实IRichBolt / BaseBasicBolt实现
- 拓扑提交 → StormSubmitter.submitTopology()
"""

import random
import re
import logging
from collections import defaultdict, deque
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SentimentAnalysis")


# ============================================================
# 金融情感词典
# ============================================================
POSITIVE_WORDS = {
    "增长": 2, "上涨": 2, "利好": 3, "盈利": 2, "突破": 2, "新高": 2,
    "反弹": 1, "回升": 1, "改善": 1, "强劲": 2, "繁荣": 2, "乐观": 2,
    "增长超预期": 3, "业绩大增": 3, "回暖": 1, "飙升": 2, "大涨": 3,
    "机遇": 1, "扩张": 1, "创新高": 3, "牛市": 2, "复苏": 1, "提振": 2,
    "红利": 1, "景气": 2, "超预期": 2, "领跑": 2, "龙头": 1
}

NEGATIVE_WORDS = {
    "下跌": -2, "暴跌": -3, "亏损": -2, "风险": -1, "衰退": -2, "危机": -3,
    "下滑": -1, "利空": -3, "低迷": -2, "疲软": -1, "违约": -3, "崩盘": -3,
    "腰斩": -3, "恐慌": -2, "暴跌潮": -3, "熊市": -2, "缩水": -2, "承压": -1,
    "下行": -1, "预警": -1, "黑天鹅": -3, "爆雷": -3, "跌停": -3, "缩量": -1,
    "减持": -2, "抛售": -2, "失守": -2, "破发": -2, "利空出尽": 1
}

# 停用词
STOP_WORDS = {"的", "了", "在", "是", "和", "与", "对", "为", "中", "将",
              "被", "从", "到", "以", "及", "等", "也", "但", "而", "或",
              "有", "其", "这", "那", "上", "下", "不", "一", "个", "更"}


# ============================================================
# 模拟舆情数据Spout
# ============================================================
class SimulatedNewsSpout:
    """
    模拟Storm Spout，产生实时财经舆情数据。
    生产环境替换为：storm.kafka.KafkaSpout
    """

    NEWS_TEMPLATES = [
        "{stock}业绩大增，净利润同比增长{pct}%",
        "{stock}暴跌{pct}%，投资者恐慌抛售",
        "{sector}板块整体上涨，市场情绪回暖",
        "{sector}行业利空频出，多只个股跌停",
        "央行宣布降准，利好{sector}板块",
        "{stock}发布业绩预警，股价承压下行",
        "市场乐观情绪提振，{sector}板块创新高",
        "{stock}爆雷！年报虚增利润被立案调查",
        "政策红利释放，{sector}迎来发展机遇",
        "{stock}增持回购提振信心，股价反弹",
        "外围市场暴跌，A股{sector}板块承压",
        "{stock}技术突破新高，机构纷纷上调评级",
        "经济数据不及预期，{sector}板块下行",
        "{stock}股东大幅减持，市场恐慌情绪蔓延",
        "国务院出台新政，{sector}景气度回升",
    ]

    STOCKS = ["贵州茅台", "中国平安", "宁德时代", "比亚迪", "招商银行",
              "腾讯控股", "阿里巴巴", "五粮液", "美的集团", "隆基绿能"]
    SECTORS = ["新能源", "半导体", "消费", "金融", "医药", "地产", "互联网", "军工"]

    SOURCES = ["新浪财经", "东方财富", "证券时报", "中国证券报", "第一财经", "财联社"]

    def __init__(self):
        self.news_counter = 0

    def generate_news(self):
        """生成一条财经新闻"""
        self.news_counter += 1
        template = random.choice(self.NEWS_TEMPLATES)
        stock = random.choice(self.STOCKS)
        sector = random.choice(self.SECTORS)
        pct = random.choice([5, 10, 15, 20, 30, 50])

        title = template.format(stock=stock, sector=sector, pct=pct)

        return {
            "news_id": f"NEWS{self.news_counter:06d}",
            "title": title,
            "source": random.choice(self.SOURCES),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stock": stock,
            "sector": sector
        }

    def generate_batch(self, batch_size=10):
        """批量生成新闻"""
        return [self.generate_news() for _ in range(batch_size)]


# ============================================================
# 分词Bolt - 文本预处理
# ============================================================
class TokenizerBolt:
    """
    模拟Storm分词Bolt，对新闻标题进行分词和关键词提取。
    生产环境替换为：BaseBasicBolt子类，使用jieba分词
    """

    # 关键词模式：匹配中文词汇（2-4字）
    KEYWORD_PATTERN = re.compile(r'[\u4e00-\u9fa5]{2,4}')

    # 行业关键词
    SECTOR_KEYWORDS = ["新能源", "半导体", "消费", "金融", "医药", "地产", "互联网", "军工"]

    def process(self, news):
        """
        对单条新闻进行分词处理：
        1. 从标题中提取中文词组
        2. 过滤停用词
        3. 标记关键词（情感词+行业词）
        """
        title = news["title"]

        # 简单分词：提取2-4字中文词组
        raw_tokens = self.KEYWORD_PATTERN.findall(title)

        # 过滤停用词和单字
        tokens = [t for t in raw_tokens if t not in STOP_WORDS and len(t) >= 2]

        # 识别情感关键词
        sentiment_keywords = []
        for token in tokens:
            if token in POSITIVE_WORDS or token in NEGATIVE_WORDS:
                sentiment_keywords.append(token)

        # 识别行业关键词
        sector_keywords = [t for t in tokens if t in self.SECTOR_KEYWORDS]

        return {
            **news,
            "tokens": tokens,
            "sentiment_keywords": sentiment_keywords,
            "sector_keywords": sector_keywords
        }


# ============================================================
# 情感分析Bolt
# ============================================================
class SentimentBolt:
    """
    模拟Storm情感分析Bolt，基于金融情感词典判断正负面。
    生产环境替换为：BaseBasicBolt子类，可接入NLP模型
    """

    def process(self, tokenized_news):
        """
        对分词后的新闻进行情感分析：
        1. 统计正负面情感词得分
        2. 计算综合情感分数
        3. 判定情感极性
        """
        keywords = tokenized_news["sentiment_keywords"]

        positive_score = 0
        negative_score = 0
        positive_words_found = []
        negative_words_found = []

        for kw in keywords:
            if kw in POSITIVE_WORDS:
                positive_score += POSITIVE_WORDS[kw]
                positive_words_found.append(kw)
            elif kw in NEGATIVE_WORDS:
                negative_score += abs(NEGATIVE_WORDS[kw])
                negative_words_found.append(kw)

        # 综合情感分：正-负，范围[-10, 10]
        sentiment_score = positive_score - negative_score

        # 判定极性
        if sentiment_score > 1:
            polarity = "正面"
        elif sentiment_score < -1:
            polarity = "负面"
        else:
            polarity = "中性"

        return {
            **tokenized_news,
            "sentiment_score": sentiment_score,
            "polarity": polarity,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "positive_words": positive_words_found,
            "negative_words": negative_words_found
        }


# ============================================================
# 热点统计Bolt - 滑动窗口关键词频率
# ============================================================
class HotTopicBolt:
    """
    模拟Storm热点统计Bolt，滑动窗口统计关键词频率。
    生产环境替换为：带窗口状态的Bolt，使用Storm的WindowedBolt
    """

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.keyword_window = deque(maxlen=window_size)
        self.polarity_window = deque(maxlen=window_size)

    def process(self, sentiment_news):
        """将分析结果加入滑动窗口，统计热点关键词和情感趋势"""
        # 收集关键词
        keywords = sentiment_news["sentiment_keywords"] + sentiment_news["sector_keywords"]
        self.keyword_window.append(keywords)

        # 收集情感极性
        self.polarity_window.append(sentiment_news["polarity"])

        # 统计窗口内关键词频率
        keyword_freq = defaultdict(int)
        for kw_list in self.keyword_window:
            for kw in kw_list:
                keyword_freq[kw] += 1

        # 统计窗口内情感分布
        polarity_dist = defaultdict(int)
        for p in self.polarity_window:
            polarity_dist[p] += 1

        # 排序取TOP关键词
        top_keywords = sorted(keyword_freq.items(), key=lambda x: -x[1])[:10]

        return {
            "top_keywords": top_keywords,
            "polarity_distribution": dict(polarity_dist),
            "total_news_in_window": len(self.polarity_window)
        }


# ============================================================
# 主流程
# ============================================================
def main():
    logger.info("=" * 70)
    logger.info("实时舆情情感分析系统启动 (Storm 模拟)")
    logger.info("=" * 70)

    # 初始化各组件（模拟Storm拓扑中的Spout和Bolt）
    spout = SimulatedNewsSpout()
    tokenizer_bolt = TokenizerBolt()
    sentiment_bolt = SentimentBolt()
    hot_topic_bolt = HotTopicBolt(window_size=5)

    # 模拟运行参数
    total_rounds = 20
    batch_size = 8

    all_results = []
    polarity_overall = defaultdict(int)

    for round_idx in range(1, total_rounds + 1):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"第 {round_idx}/{total_rounds} 轮处理")

        # Step 1: Spout产生舆情数据
        news_batch = spout.generate_batch(batch_size)
        logger.info(f"  [Spout] 接收 {len(news_batch)} 条财经新闻")

        # Step 2-4: 逐条通过Bolt流水线处理
        round_results = []
        for news in news_batch:
            # 分词Bolt
            tokenized = tokenizer_bolt.process(news)
            # 情感分析Bolt
            sentiment = sentiment_bolt.process(tokenized)
            # 热点统计Bolt
            hot_stats = hot_topic_bolt.process(sentiment)

            result = {
                "news_id": sentiment["news_id"],
                "title": sentiment["title"],
                "source": sentiment["source"],
                "polarity": sentiment["polarity"],
                "sentiment_score": sentiment["sentiment_score"],
                "positive_words": sentiment["positive_words"],
                "negative_words": sentiment["negative_words"],
            }
            round_results.append(result)
            polarity_overall[sentiment["polarity"]] += 1

        all_results.extend(round_results)

        # 输出本批次情感分析结果
        for r in round_results:
            polarity_icon = {"正面": "[+] ", "负面": "[-] ", "中性": "[=] "}.get(r["polarity"], "[?] ")
            logger.info(f"  {polarity_icon} {r['title']}")
            if r["positive_words"]:
                logger.info(f"       正面词: {', '.join(r['positive_words'])}")
            if r["negative_words"]:
                logger.info(f"       负面词: {', '.join(r['negative_words'])}")

        # 输出热点统计（使用最后一条完整分析结果获取窗口状态）
        last_hot_stats = hot_topic_bolt.process(sentiment) if sentiment else None
        if round_idx % 4 == 0 and last_hot_stats:
            logger.info("  热点关键词TOP5:")
            for kw, freq in last_hot_stats["top_keywords"][:5]:
                logger.info(f"    {kw}: {freq}次")
            logger.info(f"  情感分布: {last_hot_stats['polarity_distribution']}")

    # ============================================================
    # 汇总报告
    # ============================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("实时舆情情感分析 - 运行汇总")
    logger.info(f"{'=' * 70}")

    total_news = len(all_results)
    logger.info(f"总处理新闻数: {total_news}")
    logger.info(f"情感分布:")
    for polarity, count in sorted(polarity_overall.items()):
        pct = count / total_news * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  {polarity}: {count}条 ({pct:.1f}%) {bar}")

    # 情感分数分布
    scores = [r["sentiment_score"] for r in all_results]
    avg_score = sum(scores) / len(scores) if scores else 0
    logger.info(f"\n平均情感分数: {avg_score:.2f} (正值=正面, 负值=负面)")

    # 负面新闻统计
    negative_news = [r for r in all_results if r["polarity"] == "负面"]
    if negative_news:
        logger.info(f"\n负面新闻TOP5 (按情感分排序):")
        negative_news.sort(key=lambda x: x["sentiment_score"])
        for r in negative_news[:5]:
            logger.info(f"  [{r['source']}] {r['title']} (分数:{r['sentiment_score']})")

    # 正面新闻统计
    positive_news = [r for r in all_results if r["polarity"] == "正面"]
    if positive_news:
        logger.info(f"\n正面新闻TOP5 (按情感分排序):")
        positive_news.sort(key=lambda x: -x["sentiment_score"])
        for r in positive_news[:5]:
            logger.info(f"  [{r['source']}] {r['title']} (分数:+{r['sentiment_score']})")

    # 来源统计
    source_polarity = defaultdict(lambda: defaultdict(int))
    for r in all_results:
        source_polarity[r["source"]][r["polarity"]] += 1
    logger.info("\n各来源情感分布:")
    for source, dist in sorted(source_polarity.items()):
        total_src = sum(dist.values())
        pos_pct = dist.get("正面", 0) / total_src * 100
        neg_pct = dist.get("负面", 0) / total_src * 100
        logger.info(f"  {source}: 正面{pos_pct:.0f}% 负面{neg_pct:.0f}% (共{total_src}条)")


if __name__ == "__main__":
    main()
