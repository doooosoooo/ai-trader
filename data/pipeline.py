"""데이터 수집 파이프라인 — 전체 데이터 수집 오케스트레이션."""

from datetime import datetime

import pandas as pd
from loguru import logger

from data.collectors.price_collector import PriceCollector
from data.collectors.news_collector import NewsCollector
from data.collectors.dart_collector import DartCollector
from data.collectors.macro_collector import MacroCollector
from core.indicators import calculate_all_indicators, get_indicator_summary


class DataPipeline:
    """데이터 수집 파이프라인."""

    def __init__(self, market_client, config: dict, db_path: str | None = None):
        self.price_collector = PriceCollector(market_client, db_path)
        self.news_collector = NewsCollector()
        self.dart_collector = DartCollector()
        self.macro_collector = MacroCollector(market_client)
        self.config = config
        self._indicator_params = config.get("indicators", {})

    def collect_all_for_analysis(
        self, watchlist: list[str], held_tickers: list[str] | None = None,
    ) -> dict:
        """LLM 분석에 필요한 모든 데이터를 수집.

        Args:
            watchlist: 관심 종목 코드 리스트
            held_tickers: 보유 종목 코드 리스트

        Returns:
            LLM에게 전달할 통합 데이터
        """
        all_tickers = list(set(watchlist + (held_tickers or [])))
        result = {
            "market_data": {},
            "news_summary": "",
            "macro_data": {},
            "collected_at": datetime.now().isoformat(),
        }

        # 1. 종목별 현재가 + 기술적 지표
        for ticker in all_tickers:
            try:
                # 현재가
                current = self.price_collector.collect_current_price(ticker)

                # 일봉 + 지표
                daily_df = self.price_collector.collect_daily(ticker, days=120)
                if not daily_df.empty:
                    daily_df = calculate_all_indicators(daily_df, self._indicator_params)
                    indicator_summary = get_indicator_summary(daily_df, self._indicator_params)
                else:
                    indicator_summary = {}

                # 수급
                investor = self.price_collector.collect_investor_trends(ticker)

                result["market_data"][ticker] = {
                    "current": current,
                    "indicators": indicator_summary,
                    "investor_trends": investor,
                }

            except Exception as e:
                logger.error(f"Data collection failed for {ticker}: {e}")
                result["market_data"][ticker] = {"error": str(e)}

        # 2. 뉴스
        all_news = []
        for ticker in all_tickers[:5]:  # API 부하 제한
            news = self.news_collector.collect_stock_news(ticker, count=5)
            all_news.extend(news)
        market_news = self.news_collector.collect_market_news(count=10)
        all_news.extend(market_news)
        result["news_summary"] = self.news_collector.summarize_for_llm(all_news)

        # 3. 매크로
        macro = self.macro_collector.collect_all()
        result["macro_data"] = self.macro_collector.format_for_llm(macro)
        result["macro_raw"] = macro

        logger.info(f"Pipeline complete: {len(all_tickers)} tickers, {len(all_news)} news items")
        return result

    def collect_prices_only(self, tickers: list[str]) -> dict[str, float]:
        """현재가만 빠르게 수집 — 서킷브레이커/포트폴리오 업데이트용."""
        prices = {}
        for ticker in tickers:
            try:
                data = self.price_collector.collect_current_price(ticker)
                if data.get("price"):
                    prices[ticker] = data["price"]
            except Exception as e:
                logger.warning(f"Quick price collection failed for {ticker}: {e}")
        return prices

    def get_daily_df_with_indicators(self, ticker: str, days: int = 120) -> pd.DataFrame:
        """지표가 포함된 일봉 DataFrame 반환 — ML 피처용."""
        df = self.price_collector.get_stored_daily(ticker, days)
        if df.empty:
            df = self.price_collector.collect_daily(ticker, days)
        if not df.empty:
            df = calculate_all_indicators(df, self._indicator_params)
        return df
