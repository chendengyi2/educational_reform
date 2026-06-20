"""
财经资讯爬虫采集器
从公开财经网站采集实时资讯数据
支持：新浪财经、东方财富、同花顺等

免责声明：本爬虫仅用于教改项目教学演示，请遵守目标网站 robots.txt 及相关法律法规，
控制采集频率，不得用于商业用途或大规模数据抓取。
"""

import re
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


@dataclass
class FinanceNews:
    """财经新闻数据结构"""
    title: str
    content: str
    source: str
    url: str
    publish_time: str
    crawl_time: str
    category: str = ""
    tags: list = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        return f"标题: {self.title}\n来源: {self.source}\n时间: {self.publish_time}\n内容: {self.content}"


class FinanceCrawler:
    """财经资讯爬虫"""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    def __init__(
        self,
        save_dir: str = "data/raw/crawler",
        request_delay: float = 2.0,
        timeout: int = 15,
        max_retries: int = 3,
    ):
        """
        Args:
            save_dir: 数据保存目录
            request_delay: 请求间隔（秒），避免对目标网站造成压力
            timeout: 请求超时时间
            max_retries: 最大重试次数
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        self._last_request_time = 0

    def _rate_limit(self):
        """请求频率控制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    def _fetch_page(self, url: str, encoding: Optional[str] = None) -> Optional[str]:
        """
        获取网页内容

        Args:
            url: 目标URL
            encoding: 指定编码（部分财经网站需要 gb2312/gbk）

        Returns:
            网页HTML文本，失败返回 None
        """
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()

                if encoding:
                    response.encoding = encoding
                else:
                    # 自动检测中文编码
                    content_type = response.headers.get("Content-Type", "")
                    if "gb2312" in content_type.lower() or "gbk" in content_type.lower():
                        response.encoding = "gbk"
                    else:
                        response.encoding = response.apparent_encoding

                return response.text

            except requests.RequestException as e:
                logger.warning(f"请求失败 (第{attempt + 1}次): {url}, 错误: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error(f"获取页面失败，已达最大重试次数: {url}")
        return None

    # ========== 新浪财经 ==========

    def crawl_sina_finance(
        self,
        category: str = "finance",
        max_pages: int = 3,
    ) -> List[FinanceNews]:
        """
        采集新浪财经资讯

        Args:
            category: 板块分类（finance-财经, stock-股票, money-理财）
            max_pages: 采集页数

        Returns:
            新闻列表
        """
        logger.info(f"开始采集新浪财经 [{category}] 板块，共 {max_pages} 页")
        news_list = []

        category_map = {
            "finance": "https://finance.sina.com.cn/roll/index.d.html",
            "stock": "https://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml",
        }

        base_url = category_map.get(category, category_map["finance"])

        for page in range(1, max_pages + 1):
            url = f"{base_url}?page={page}" if "?" not in base_url else f"{base_url}&page={page}"
            html = self._fetch_page(url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")

            # 解析新闻列表
            items = soup.select("div.listNete li, .news-item, .item")
            if not items:
                items = soup.find_all("a", href=re.compile(r"sina\.com\.cn.*\d{4}-\d{2}-\d{2}"))

            for item in items:
                try:
                    link_tag = item if item.name == "a" else item.find("a")
                    if not link_tag:
                        continue

                    title = link_tag.get_text(strip=True)
                    url = link_tag.get("href", "")

                    if not title or not url or not url.startswith("http"):
                        continue

                    news = FinanceNews(
                        title=title,
                        content="",  # 内容需要进一步抓取详情页
                        source="新浪财经",
                        url=url,
                        publish_time=datetime.now().strftime("%Y-%m-%d"),
                        crawl_time=datetime.now().isoformat(),
                        category=category,
                    )
                    news_list.append(news)

                except Exception as e:
                    logger.warning(f"解析新闻项失败: {e}")
                    continue

        logger.info(f"新浪财经采集完成，共获取 {len(news_list)} 条")
        return news_list

    def _fetch_news_content(self, news: FinanceNews) -> FinanceNews:
        """抓取新闻详情页内容"""
        html = self._fetch_page(news.url)
        if not html:
            return news

        soup = BeautifulSoup(html, "html.parser")

        # 常见正文选择器
        content_selectors = [
            "div.article-content", "div#artibody", "div.article",
            "div.content", "div#content", "div.text",
        ]
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                # 移除脚本和样式
                for tag in content_div(["script", "style", "img"]):
                    tag.decompose()
                news.content = content_div.get_text(strip=True, separator="\n")
                break

        # 尝试提取发布时间
        time_selectors = ["span.date", "span.time", "div.date", "time", "span.pub_time"]
        for selector in time_selectors:
            time_tag = soup.select_one(selector)
            if time_tag:
                news.publish_time = time_tag.get_text(strip=True)
                break

        return news

    # ========== 东方财富 ==========

    def crawl_eastmoney_news(
        self,
        max_pages: int = 3,
    ) -> List[FinanceNews]:
        """
        采集东方财富财经资讯（通过资讯列表API）

        Args:
            max_pages: 采集页数

        Returns:
            新闻列表
        """
        logger.info(f"开始采集东方财富资讯，共 {max_pages} 页")
        news_list = []

        # 东方财富资讯列表 API
        api_url = "https://np-listapi.eastmoney.com/comm/web/getNewsByColumns"

        for page in range(1, max_pages + 1):
            params = {
                "client": "web",
                "biz": "web_news_col",
                "column": "350",
                "order": "1",
                "needInteractData": "0",
                "page_index": str(page),
                "page_size": "20",
            }

            try:
                self._rate_limit()
                response = self.session.get(
                    api_url, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                items = data.get("data", {}).get("list", [])
                for item in items:
                    try:
                        title = item.get("title", "")
                        url = item.get("url", "")
                        content = item.get("content", item.get("digest", ""))
                        pub_time = item.get("showTime", "")

                        if not title:
                            continue

                        news = FinanceNews(
                            title=title,
                            content=content,
                            source="东方财富",
                            url=url,
                            publish_time=pub_time,
                            crawl_time=datetime.now().isoformat(),
                            category="财经资讯",
                        )
                        news_list.append(news)

                    except Exception as e:
                        logger.warning(f"解析东方财富新闻项失败: {e}")
                        continue

            except Exception as e:
                logger.warning(f"东方财富API请求失败 (第{page}页): {e}")
                continue

        logger.info(f"东方财富采集完成，共获取 {len(news_list)} 条")
        return news_list

    # ========== 通用爬虫 ==========

    def crawl_custom_url(
        self,
        url: str,
        title_selector: str = "h1",
        content_selector: str = "div.content, div.article, div.article-content",
        source_name: str = "自定义来源",
    ) -> Optional[FinanceNews]:
        """
        通用页面采集（适用于任意财经网站单页）

        Args:
            url: 目标页面URL
            title_selector: 标题CSS选择器
            content_selector: 正文CSS选择器（多个用逗号分隔）
            source_name: 数据来源名称

        Returns:
            新闻对象，失败返回 None
        """
        html = self._fetch_page(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        # 提取标题
        title_tag = soup.select_one(title_selector)
        title = title_tag.get_text(strip=True) if title_tag else ""

        # 提取正文
        content = ""
        for selector in content_selector.split(","):
            content_tag = soup.select_one(selector.strip())
            if content_tag:
                for tag in content_tag(["script", "style"]):
                    tag.decompose()
                content = content_tag.get_text(strip=True, separator="\n")
                break

        return FinanceNews(
            title=title,
            content=content,
            source=source_name,
            url=url,
            publish_time=datetime.now().strftime("%Y-%m-%d"),
            crawl_time=datetime.now().isoformat(),
        )

    # ========== 数据保存 ==========

    def save_news(
        self,
        news_list: List[FinanceNews],
        filename: Optional[str] = None,
        format: str = "json",
        fetch_content: bool = False,
    ) -> Path:
        """
        保存采集的新闻数据

        Args:
            news_list: 新闻列表
            filename: 文件名（不含扩展名），默认按日期生成
            format: 保存格式（json / csv / txt）
            fetch_content: 是否抓取详情页内容

        Returns:
            保存的文件路径
        """
        if not news_list:
            logger.warning("新闻列表为空，跳过保存")
            return None

        if fetch_content:
            logger.info("正在抓取新闻详情页内容...")
            for i, news in enumerate(news_list):
                if not news.content:
                    try:
                        news_list[i] = self._fetch_news_content(news)
                    except Exception as e:
                        logger.warning(f"抓取详情失败: {news.url}, 错误: {e}")

        if not filename:
            filename = f"finance_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        save_path = self.save_dir / f"{filename}.{format}"

        if format == "json":
            data = [n.to_dict() for n in news_list]
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        elif format == "csv":
            import csv
            with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(news_list[0]).keys()))
                writer.writeheader()
                for news in news_list:
                    row = news.to_dict()
                    row["tags"] = "|".join(row["tags"])
                    writer.writerow(row)

        elif format == "txt":
            with open(save_path, "w", encoding="utf-8") as f:
                for i, news in enumerate(news_list, 1):
                    f.write(f"--- 新闻 {i} ---\n")
                    f.write(news.to_text())
                    f.write("\n\n")

        logger.info(f"已保存 {len(news_list)} 条新闻到: {save_path}")
        return save_path

    def crawl_and_save(
        self,
        sources: List[str] = None,
        max_pages: int = 2,
        format: str = "json",
        fetch_content: bool = False,
    ) -> Dict[str, Path]:
        """
        一站式采集并保存

        Args:
            sources: 数据源列表（sina, eastmoney），None 表示全部
            max_pages: 每个源采集页数
            format: 保存格式
            fetch_content: 是否抓取详情页

        Returns:
            {数据源: 保存路径} 映射
        """
        if sources is None:
            sources = ["sina", "eastmoney"]

        results = {}

        for source in sources:
            try:
                if source == "sina":
                    news_list = self.crawl_sina_finance(max_pages=max_pages)
                elif source == "eastmoney":
                    news_list = self.crawl_eastmoney_news(max_pages=max_pages)
                else:
                    logger.warning(f"未知数据源: {source}")
                    continue

                if news_list:
                    path = self.save_news(
                        news_list,
                        filename=f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        format=format,
                        fetch_content=fetch_content,
                    )
                    results[source] = path

            except Exception as e:
                logger.error(f"采集数据源 '{source}' 失败: {e}")
                continue

        return results
