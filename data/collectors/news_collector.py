"""뉴스 헤드라인 수집 — 네이버 금융 API + 뉴스 섹션."""

import html
import re
from datetime import datetime

import requests
from loguru import logger


class NewsCollector:
    """뉴스 헤드라인 수집 및 요약."""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

    def collect_stock_news(self, ticker: str, count: int = 10) -> list[dict]:
        """종목 관련 뉴스 수집 (네이버 모바일 주식 API)."""
        url = f"https://m.stock.naver.com/api/news/stock/{ticker}?pageSize={count}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            news = []
            for group in data:
                for item in group.get("items", []):
                    title = html.unescape(item.get("title", "")).strip()
                    if title:
                        news.append({
                            "title": title,
                            "ticker": ticker,
                            "source": item.get("officeName", "naver"),
                            "datetime": item.get("datetime", ""),
                            "collected_at": datetime.now().isoformat(),
                        })

            logger.info(f"Collected {len(news)} news for {ticker}")
            return news[:count]
        except Exception as e:
            logger.warning(f"News collection failed for {ticker}: {e}")
            return []

    def collect_market_news(self, count: int = 15) -> list[dict]:
        """시장 전체 뉴스 수집 (네이버 뉴스 증권 섹션)."""
        url = "https://news.naver.com/breakingnews/section/101/258"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            titles = re.findall(
                r'<strong class="sa_text_strong">([^<]+)</strong>', resp.text
            )
            news = []
            for title in titles[:count]:
                title = html.unescape(title.strip())
                if title:
                    news.append({
                        "title": title,
                        "source": "naver_economy",
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
            source = f" ({n['source']})" if n.get("source") else ""
            lines.append(f"- {ticker_info}{n['title']}{source}")

        return "\n".join(lines)
