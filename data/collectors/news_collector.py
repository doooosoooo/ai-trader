"""뉴스 헤드라인 수집 — 네이버 금융 RSS 기반."""

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger


class NewsCollector:
    """뉴스 헤드라인 수집 및 요약."""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AITrader/1.0)",
        }

    def collect_stock_news(self, ticker: str, count: int = 10) -> list[dict]:
        """종목 관련 뉴스 수집 (네이버 금융)."""
        url = f"https://finance.naver.com/item/news_news.naver?code={ticker}&page=1"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            # 간단한 파싱 — 제목 추출
            titles = re.findall(
                r'class="tit">\s*<a[^>]*>([^<]+)</a>', resp.text
            )
            news = []
            for title in titles[:count]:
                title = title.strip()
                if title:
                    news.append({
                        "title": title,
                        "ticker": ticker,
                        "source": "naver",
                        "collected_at": datetime.now().isoformat(),
                    })
            logger.info(f"Collected {len(news)} news for {ticker}")
            return news
        except Exception as e:
            logger.warning(f"News collection failed for {ticker}: {e}")
            return []

    def collect_market_news(self, count: int = 15) -> list[dict]:
        """시장 전체 뉴스 수집."""
        url = "https://finance.naver.com/news/mainnews.naver"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            titles = re.findall(
                r'class="articleSubject">\s*<a[^>]*title="([^"]+)"', resp.text
            )
            news = []
            for title in titles[:count]:
                title = title.strip()
                if title:
                    news.append({
                        "title": title,
                        "source": "naver_market",
                        "collected_at": datetime.now().isoformat(),
                    })
            logger.info(f"Collected {len(news)} market news")
            return news
        except Exception as e:
            logger.warning(f"Market news collection failed: {e}")
            return []

    def summarize_for_llm(self, news_list: list[dict]) -> str:
        """뉴스 목록을 LLM에게 전달할 요약 텍스트로 변환."""
        if not news_list:
            return "뉴스 없음"

        lines = []
        for n in news_list[:20]:
            ticker_info = f"[{n['ticker']}] " if n.get("ticker") else ""
            lines.append(f"- {ticker_info}{n['title']}")

        return "\n".join(lines)
