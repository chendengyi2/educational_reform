#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案例7：基于Redis的实时推荐系统

模拟架构：
- 用户行为流 → 行为处理 → Redis缓存(用户画像/物品特征/热点) → 协同过滤推荐 → 输出

模拟内容：
1. 模拟用户行为流（浏览、点击、购买、评分）
2. 模拟Redis缓存（用户画像、物品特征、实时热点）
3. 协同过滤推荐算法（User-based）
4. 实时更新用户画像和推荐列表
5. 输出推荐结果和命中率统计

生产环境替换：
- Redis → 真实Redis连接 (import redis; r = redis.Redis(...))
- 行为流 → Kafka/RabbitMQ消费
- 推荐服务 → 微服务部署
"""

import random
import math
import logging
from collections import defaultdict
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RealtimeRecommend")


# ============================================================
# 模拟Redis缓存
# ============================================================
class SimulatedRedis:
    """
    模拟Redis缓存，存储用户画像、物品特征、实时热点等。
    生产环境替换为：import redis; r = redis.Redis(host='localhost', port=6379)
    """

    def __init__(self):
        self.string_data = {}
        self.hash_data = {}
        self.sorted_set_data = {}

    def set(self, key, value):
        self.string_data[key] = value

    def get(self, key):
        return self.string_data.get(key)

    def hset(self, name, key, value):
        if name not in self.hash_data:
            self.hash_data[name] = {}
        self.hash_data[name][key] = value

    def hget(self, name, key):
        return self.hash_data.get(name, {}).get(key)

    def hgetall(self, name):
        return self.hash_data.get(name, {})

    def zincrby(self, name, amount, member):
        """有序集合增加分数，对应Redis ZINCRBY"""
        if name not in self.sorted_set_data:
            self.sorted_set_data[name] = {}
        self.sorted_set_data[name][member] = self.sorted_set_data[name].get(member, 0) + amount

    def zrevrange(self, name, start, stop):
        """有序集合按分数倒序获取，对应Redis ZREVRANGE"""
        if name not in self.sorted_set_data:
            return []
        sorted_items = sorted(self.sorted_set_data[name].items(), key=lambda x: -x[1])
        return sorted_items[start:stop + 1] if stop != -1 else sorted_items[start:]


# ============================================================
# 模拟物品库
# ============================================================
ITEMS = {
    "ITEM001": {"name": "Python编程入门", "category": "编程", "price": 59.0},
    "ITEM002": {"name": "数据结构与算法", "category": "编程", "price": 79.0},
    "ITEM003": {"name": "机器学习实战", "category": "AI", "price": 89.0},
    "ITEM004": {"name": "深度学习原理", "category": "AI", "price": 99.0},
    "ITEM005": {"name": "数据库系统概论", "category": "编程", "price": 69.0},
    "ITEM006": {"name": "自然语言处理", "category": "AI", "price": 85.0},
    "ITEM007": {"name": "统计学基础", "category": "数学", "price": 55.0},
    "ITEM008": {"name": "线性代数", "category": "数学", "price": 45.0},
    "ITEM009": {"name": "云计算架构", "category": "运维", "price": 75.0},
    "ITEM010": {"name": "Docker实践指南", "category": "运维", "price": 65.0},
    "ITEM011": {"name": "计算机网络", "category": "编程", "price": 58.0},
    "ITEM012": {"name": "计算机视觉", "category": "AI", "price": 92.0},
}

CATEGORIES = list(set(item["category"] for item in ITEMS.values()))


# ============================================================
# 模拟用户行为流
# ============================================================
class SimulatedBehaviorStream:
    """
    模拟用户行为数据流。
    生产环境替换为：Kafka Consumer / Flink Source
    """

    USERS = [f"USER{str(i).zfill(3)}" for i in range(1, 16)]
    BEHAVIOR_TYPES = ["browse", "click", "purchase", "rate"]
    BEHAVIOR_WEIGHTS = [0.4, 0.3, 0.15, 0.15]  # 行为频率权重

    def __init__(self):
        # 用户兴趣偏好（每个用户对不同品类有不同的偏好概率）
        self.user_preferences = {}
        for user in self.USERS:
            prefs = {cat: random.uniform(0.1, 1.0) for cat in CATEGORIES}
            total = sum(prefs.values())
            self.user_preferences[user] = {cat: v / total for cat, v in prefs.items()}

    def generate_behavior(self):
        """生成一条用户行为记录"""
        user = random.choice(self.USERS)
        behavior = random.choices(self.BEHAVIOR_TYPES, weights=self.BEHAVIOR_WEIGHTS, k=1)[0]

        # 根据用户偏好选择品类的物品
        prefs = self.user_preferences[user]
        category = random.choices(list(prefs.keys()), weights=list(prefs.values()), k=1)[0]
        category_items = [iid for iid, item in ITEMS.items() if item["category"] == category]
        item_id = random.choice(category_items) if category_items else random.choice(list(ITEMS.keys()))

        # 评分行为附带评分值
        rating = round(random.uniform(1, 5), 1) if behavior == "rate" else None

        return {
            "user_id": user,
            "item_id": item_id,
            "behavior": behavior,
            "rating": rating,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def generate_batch(self, batch_size=15):
        """批量生成行为记录"""
        return [self.generate_behavior() for _ in range(batch_size)]


# ============================================================
# 协同过滤推荐算法 (User-based)
# ============================================================
class UserBasedCollaborativeFiltering:
    """
    User-based协同过滤推荐。
    核心思想：找到与目标用户兴趣相似的用户，推荐相似用户喜欢但目标用户未接触的物品。
    """

    def __init__(self):
        # 用户-物品评分矩阵（隐式反馈也转为评分）
        self.user_item_matrix = defaultdict(lambda: defaultdict(float))
        # 用户相似度缓存
        self.similarity_cache = {}

    def update_matrix(self, behaviors):
        """
        根据行为数据更新用户-物品矩阵。
        行为权重：浏览1分，点击2分，购买5分，评分为rating*1.5
        """
        weight_map = {"browse": 1.0, "click": 2.0, "purchase": 5.0}
        for b in behaviors:
            user = b["user_id"]
            item = b["item_id"]
            if b["behavior"] == "rate" and b["rating"] is not None:
                score = b["rating"] * 1.5
            else:
                score = weight_map.get(b["behavior"], 1.0)
            # 累加分数（体现用户对物品的兴趣程度）
            self.user_item_matrix[user][item] += score

        # 清空相似度缓存（用户画像更新后需重新计算）
        self.similarity_cache.clear()

    def _cosine_similarity(self, user_a, user_b):
        """计算两个用户的余弦相似度"""
        items_a = self.user_item_matrix[user_a]
        items_b = self.user_item_matrix[user_b]
        common_items = set(items_a.keys()) & set(items_b.keys())
        if not common_items:
            return 0.0

        dot_product = sum(items_a[i] * items_b[i] for i in common_items)
        norm_a = math.sqrt(sum(v ** 2 for v in items_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in items_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def get_similar_users(self, user_id, top_k=5):
        """获取与目标用户最相似的K个用户"""
        if user_id in self.similarity_cache:
            return self.similarity_cache[user_id]

        similarities = []
        for other_user in self.user_item_matrix:
            if other_user == user_id:
                continue
            sim = self._cosine_similarity(user_id, other_user)
            if sim > 0:
                similarities.append((other_user, sim))

        similarities.sort(key=lambda x: -x[1])
        result = similarities[:top_k]
        self.similarity_cache[user_id] = result
        return result

    def recommend(self, user_id, top_n=5):
        """
        为目标用户生成推荐列表。
        逻辑：找相似用户 → 收集相似用户高分物品 → 过滤已接触物品 → 排序输出
        """
        similar_users = self.get_similar_users(user_id, top_k=5)
        if not similar_users:
            return []

        candidate_items = defaultdict(float)
        user_items = set(self.user_item_matrix[user_id].keys())

        for sim_user, similarity in similar_users:
            for item, score in self.user_item_matrix[sim_user].items():
                if item not in user_items:  # 过滤已接触物品
                    candidate_items[item] += score * similarity

        # 按加权得分排序
        ranked = sorted(candidate_items.items(), key=lambda x: -x[1])
        return [(item, round(score, 2)) for item, score in ranked[:top_n]]


# ============================================================
# 实时推荐引擎
# ============================================================
class RealtimeRecommendEngine:
    """实时推荐引擎：整合行为处理、Redis缓存更新、推荐生成"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.cf = UserBasedCollaborativeFiltering()
        self.hit_count = 0       # 推荐命中次数
        self.recommend_count = 0  # 总推荐次数

    def process_behaviors(self, behaviors):
        """
        处理一批行为数据：
        1. 更新用户-物品矩阵
        2. 更新Redis中的用户画像和热点
        3. 生成推荐
        """
        # 更新协同过滤矩阵
        self.cf.update_matrix(behaviors)

        # 更新Redis
        for b in behaviors:
            user = b["user_id"]
            item = b["item_id"]

            # 更新用户画像（Redis Hash）
            self.redis.hset(f"user:profile:{user}", "last_active", b["timestamp"])
            self.redis.zincrby(f"user:interests:{user}", 1.0, ITEMS[item]["category"])

            # 更新物品热度（Redis Sorted Set）
            self.redis.zincrby("hot:items", 1.0, item)

            # 更新分类热度
            self.redis.zincrby("hot:categories", 1.0, ITEMS[item]["category"])

            # 检查推荐命中：如果该用户购买/评分的物品在之前的推荐列表中
            if b["behavior"] in ("purchase", "rate"):
                rec_list = self.redis.get(f"rec:latest:{user}")
                if rec_list and item in rec_list:
                    self.hit_count += 1

        # 为活跃用户生成推荐
        active_users = set(b["user_id"] for b in behaviors)
        recommendations = {}
        for user in active_users:
            recs = self.cf.recommend(user, top_n=5)
            recommendations[user] = recs
            self.recommend_count += 1
            # 缓存推荐列表到Redis
            self.redis.set(f"rec:latest:{user}", [item for item, _ in recs])

        return recommendations

    def get_hit_rate(self):
        """计算推荐命中率"""
        return self.hit_count / self.recommend_count if self.recommend_count > 0 else 0.0


# ============================================================
# 主流程
# ============================================================
def main():
    logger.info("=" * 70)
    logger.info("实时推荐系统启动 (Redis + User-based CF 模拟)")
    logger.info("=" * 70)

    # 初始化各组件
    redis = SimulatedRedis()
    stream = SimulatedBehaviorStream()
    engine = RealtimeRecommendEngine(redis)

    # 预填充一些行为数据（冷启动）
    logger.info("冷启动：预填充历史行为数据...")
    warmup_behaviors = stream.generate_batch(100)
    engine.process_behaviors(warmup_behaviors)
    logger.info(f"  已处理 {len(warmup_behaviors)} 条历史行为")

    # 模拟运行参数
    total_rounds = 20
    batch_size = 20

    all_recommendations = []

    for round_idx in range(1, total_rounds + 1):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"第 {round_idx}/{total_rounds} 轮处理")

        # Step 1: 接收行为流
        behaviors = stream.generate_batch(batch_size)
        behavior_summary = defaultdict(int)
        for b in behaviors:
            behavior_summary[b["behavior"]] += 1
        logger.info(f"  [行为流] 接收 {len(behaviors)} 条行为: "
                     + ", ".join(f"{k}={v}" for k, v in behavior_summary.items()))

        # Step 2: 处理行为并生成推荐
        recommendations = engine.process_behaviors(behaviors)
        all_recommendations.append(recommendations)

        # Step 3: 输出推荐结果
        for user, recs in recommendations.items():
            if recs:
                rec_items = [f"{ITEMS[iid]['name']}({score})" for iid, score in recs[:3]]
                logger.info(f"  推荐给 {user}: {', '.join(rec_items)}")

        # Step 4: 输出热点
        if round_idx % 5 == 0:
            hot_items = redis.zrevrange("hot:items", 0, 4)
            hot_cats = redis.zrevrange("hot:categories", 0, 2)
            logger.info("  热门物品TOP5:")
            for iid, score in hot_items:
                logger.info(f"    {ITEMS[iid]['name']}: 热度{score:.0f}")
            logger.info("  热门品类TOP3:")
            for cat, score in hot_cats:
                logger.info(f"    {cat}: 热度{score:.0f}")

    # ============================================================
    # 汇总报告
    # ============================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("实时推荐系统 - 运行汇总")
    logger.info(f"{'=' * 70}")

    logger.info(f"总推荐次数: {engine.recommend_count}")
    logger.info(f"推荐命中次数: {engine.hit_count}")
    logger.info(f"推荐命中率: {engine.get_hit_rate():.2%}")

    # 输出最终热门排行
    logger.info("\n最终热门物品TOP10:")
    hot_items = redis.zrevrange("hot:items", 0, 9)
    for iid, score in hot_items:
        logger.info(f"  {ITEMS[iid]['name']} ({ITEMS[iid]['category']}): 热度{score:.0f}")

    logger.info("\n最终热门品类:")
    hot_cats = redis.zrevrange("hot:categories", 0, -1)
    for cat, score in hot_cats:
        logger.info(f"  {cat}: 热度{score:.0f}")

    # 输出部分用户画像
    logger.info("\n部分用户兴趣画像:")
    for user in stream.USERS[:5]:
        interests = redis.zrevrange(f"user:interests:{user}", 0, 2)
        if interests:
            interest_str = ", ".join(f"{cat}({score:.0f})" for cat, score in interests)
            logger.info(f"  {user}: {interest_str}")

    # 输出推荐覆盖统计
    users_with_recs = sum(1 for recs in all_recommendations[-1].values() if recs)
    logger.info(f"\n最新一轮有推荐结果的用户数: {users_with_recs}")


if __name__ == "__main__":
    main()
