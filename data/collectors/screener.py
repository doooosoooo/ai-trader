"""종목 자동 스크리닝 — Naver Finance API 기반 멀티팩터 스코어링.

3단계 파이프라인:
  Stage 1: 사전 필터 (시총, 거래대금, 가격, PER)
  Stage 2: 멀티팩터 스코어링 (모멘텀, 밸류, 거래량, 수급, 기술적)
  Stage 3: 최종 선택 (상위 N개 + 보유종목)
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

DB_DIR = Path(__file__).parent.parent / "storage"

NAVER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
NAVER_API_BASE = "https://m.stock.naver.com/api"


@dataclass
class ScreeningResult:
    """스크리닝 실행 결과."""

    timestamp: str
    candidates: list[dict]  # 점수 순 정렬된 후보 종목
    market_summary: dict  # KOSPI/KOSDAQ 시장 개요
    screening_stats: dict  # 필터 단계별 통과 수
    held_tickers_added: list[str] = field(default_factory=list)


class StockScreener:
    """KRX 전체 시장 데이터 기반 종목 스크리닝."""

    def __init__(self, config: dict, db_path: str | None = None):
        self.config = config.get("screening", config)
        self.db_path = db_path or str(DB_DIR / "trader.db")
        self._last_result: ScreeningResult | None = None
        self._session = requests.Session()
        self._session.headers.update(NAVER_HEADERS)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS screening_results (
                    date TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    market_summary_json TEXT NOT NULL,
                    stats_json TEXT NOT NULL
                )
            """)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_screening(
        self, held_tickers: list[str] | None = None,
    ) -> ScreeningResult:
        """전체 스크리닝 파이프라인 실행."""
        held_tickers = held_tickers or []
        stats = {"total_kospi": 0, "total_kosdaq": 0}

        try:
            # 1. Naver Finance에서 시장 전체 종목 데이터 수집
            logger.info("Screening: fetching market data from Naver Finance...")
            kospi_stocks = self._fetch_market_stocks("KOSPI")
            kosdaq_stocks = self._fetch_market_stocks("KOSDAQ")

            stats["total_kospi"] = len(kospi_stocks)
            stats["total_kosdaq"] = len(kosdaq_stocks)

            # KOSDAQ: 시총 상위 N개만
            kosdaq_top_n = self.config.get("kosdaq_top_n", 300)
            kosdaq_stocks = kosdaq_stocks[:kosdaq_top_n]
            stats["kosdaq_filtered"] = len(kosdaq_stocks)

            all_stocks = kospi_stocks + kosdaq_stocks
            stats["total_analyzed"] = len(all_stocks)
            logger.info(
                f"Screening: {stats['total_kospi']} KOSPI + "
                f"{stats['kosdaq_filtered']} KOSDAQ (top {kosdaq_top_n}) = "
                f"{stats['total_analyzed']} total"
            )

            # 2. Stage 1: 사전 필터
            filtered = self._stage1_filter(all_stocks)
            stats["after_filter"] = len(filtered)
            logger.info(f"Screening Stage 1: {len(filtered)} stocks passed filters")

            # 2.5. 상위 50종목 상세 정보 수집 (PER, 수급)
            logger.info("Screening: fetching detail info for top candidates...")
            self._enrich_with_details(filtered, max_stocks=50)

            # 3. Stage 2: 멀티팩터 스코어링
            scored = self._stage2_score(filtered)
            stats["after_scoring"] = len(scored)
            logger.info(f"Screening Stage 2: {len(scored)} stocks scored")

            # 4. Stage 3: 최종 선택
            max_size = self.config.get("max_watchlist_size", 8)
            candidates = scored[:max_size]

            # 보유종목 강제 포함
            candidate_tickers = {c["ticker"] for c in candidates}
            held_added = []
            for ht in held_tickers:
                if ht not in candidate_tickers:
                    held_added.append(ht)

            # 시장 요약
            market_summary = self._build_market_summary(kospi_stocks, kosdaq_stocks)

            result = ScreeningResult(
                timestamp=datetime.now().isoformat(),
                candidates=candidates,
                market_summary=market_summary,
                screening_stats=stats,
                held_tickers_added=held_added,
            )

            self._last_result = result
            self._save_to_db(result)

            logger.info(
                f"Screening complete: {len(candidates)} candidates selected"
            )
            return result

        except Exception as e:
            logger.error(f"Screening failed: {e}")
            # DB 캐시 시도
            cached = self._load_from_db()
            if cached:
                logger.info("Using cached screening result")
                return cached
            # 최종 폴백
            return ScreeningResult(
                timestamp=datetime.now().isoformat(),
                candidates=[],
                market_summary={},
                screening_stats={"error": str(e)},
                held_tickers_added=held_tickers,
            )

    def get_watchlist(
        self,
        held_tickers: list[str] | None = None,
        max_tickers: int = 8,
    ) -> list[str]:
        """최종 관심종목 리스트 반환."""
        held_tickers = held_tickers or []
        result = self._last_result or self._load_from_db()
        if not result or not result.candidates:
            return held_tickers[:max_tickers] if held_tickers else []

        # 스크리닝 상위 종목
        screened = [c["ticker"] for c in result.candidates]

        # 보유종목 우선 + 스크리닝 결과 합산
        watchlist = list(held_tickers)
        for t in screened:
            if t not in watchlist:
                watchlist.append(t)
            if len(watchlist) >= max_tickers:
                break

        return watchlist

    def get_last_result(self) -> ScreeningResult | None:
        if self._last_result:
            return self._last_result
        return self._load_from_db()

    # ------------------------------------------------------------------
    # Stage 1: 사전 필터
    # ------------------------------------------------------------------

    def _stage1_filter(self, stocks: list[dict]) -> list[dict]:
        """시총, 거래대금, 가격, PER 기준 필터."""
        filters = self.config.get("filters", {})
        min_cap = filters.get("min_market_cap", 3_000_000_000_000)
        min_trade_val = filters.get("min_trading_value_20d", 10_000_000_000)
        min_price = filters.get("min_price", 1000)
        max_per = filters.get("max_per", 100)
        min_per = filters.get("min_per", 0)
        exclude = set(self.config.get("exclude_tickers", []))

        passed = []
        for s in stocks:
            ticker = s["ticker"]
            if ticker in exclude:
                continue
            # 우선주 제외 (코드 마지막자리 5, 7, 8, 9 등)
            if not ticker[-1].isdigit() or ticker[-1] in ("5", "7", "8", "9"):
                if ticker[-1] != "0":
                    continue

            if s["market_cap"] < min_cap:
                continue
            if s["trading_value"] < min_trade_val:
                continue
            if s["close_price"] < min_price:
                continue
            if s["volume"] <= 0:
                continue
            # PER 필터 (0 이하 = 적자, 100 이상 = 극단적 고평가)
            per = s.get("per", 0)
            if per is not None and per != 0:
                if per <= min_per or per > max_per:
                    continue

            passed.append(s)

        return passed

    # ------------------------------------------------------------------
    # Stage 2: 멀티팩터 스코어링
    # ------------------------------------------------------------------

    def _stage2_score(self, stocks: list[dict]) -> list[dict]:
        """멀티팩터 스코어링 — 0~100점 가중합."""
        if not stocks:
            return []

        scoring = self.config.get("scoring", {})
        df = pd.DataFrame(stocks)

        # 각 팩터 점수 계산
        df["momentum_score"] = self._score_momentum(df, scoring)
        df["value_score"] = self._score_value(df, scoring)
        df["volume_score"] = self._score_volume(df, scoring)
        df["flow_score"] = self._score_investor_flow(df, scoring)
        df["technical_score"] = self._score_technical(df, scoring)

        # 가중합
        w_mom = scoring.get("momentum", {}).get("weight", 0.25)
        w_val = scoring.get("value", {}).get("weight", 0.20)
        w_vol = scoring.get("volume", {}).get("weight", 0.15)
        w_flow = scoring.get("investor_flow", {}).get("weight", 0.20)
        w_tech = scoring.get("technical", {}).get("weight", 0.20)

        df["composite_score"] = (
            df["momentum_score"] * w_mom
            + df["value_score"] * w_val
            + df["volume_score"] * w_vol
            + df["flow_score"] * w_flow
            + df["technical_score"] * w_tech
        )

        # 점수 순 정렬
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        # dict 리스트 변환 (상위 50개만 상세 정보 유지)
        result = []
        for _, row in df.head(50).iterrows():
            result.append({
                "ticker": row["ticker"],
                "name": row["name"],
                "market": row["market"],
                "close_price": int(row["close_price"]),
                "change_pct": float(row["change_pct"]),
                "market_cap": int(row["market_cap"]),
                "trading_value": int(row["trading_value"]),
                "volume": int(row["volume"]),
                "per": row.get("per"),
                "composite_score": round(float(row["composite_score"]), 1),
                "momentum_score": round(float(row["momentum_score"]), 1),
                "value_score": round(float(row["value_score"]), 1),
                "volume_score": round(float(row["volume_score"]), 1),
                "flow_score": round(float(row["flow_score"]), 1),
                "technical_score": round(float(row["technical_score"]), 1),
            })

        return result

    def _score_momentum(self, df: pd.DataFrame, scoring: dict) -> pd.Series:
        """모멘텀 점수: 등락률 백분위."""
        pct = df["change_pct"].fillna(0)
        # 극단치 제거 (5~95 백분위)
        lower = pct.quantile(0.05)
        upper = pct.quantile(0.95)
        clipped = pct.clip(lower, upper)
        return clipped.rank(pct=True) * 100

    def _score_value(self, df: pd.DataFrame, scoring: dict) -> pd.Series:
        """밸류 점수: PER 역순 백분위 (낮을수록 좋음)."""
        per = df["per"].fillna(0).replace(0, np.nan)
        # PER가 없는 종목은 50점 (중립)
        per_score = (1 - per.rank(pct=True)) * 100
        return per_score.fillna(50)

    def _score_volume(self, df: pd.DataFrame, scoring: dict) -> pd.Series:
        """거래대금 점수: 거래대금 백분위."""
        val = df["trading_value"].fillna(0)
        return val.rank(pct=True) * 100

    def _score_investor_flow(self, df: pd.DataFrame, scoring: dict) -> pd.Series:
        """수급 점수: 외국인/기관 순매수."""
        foreign = df["foreign_net"].fillna(0)
        institution = df["institution_net"].fillna(0)

        # 외국인 60%, 기관 40% 가중
        combined = foreign * 0.6 + institution * 0.4
        # 순매수 > 0 이면 가산
        score = combined.rank(pct=True) * 100
        return score.fillna(50)

    def _score_technical(self, df: pd.DataFrame, scoring: dict) -> pd.Series:
        """기술적 점수: 등락률 + 거래량 종합."""
        # 단순화: 양의 수익률 + 거래활성도 조합
        change = df["change_pct"].fillna(0)
        volume_rank = df["volume"].rank(pct=True)

        # 적당한 상승 (0~5%)이 가장 높은 점수
        sweet = scoring.get("technical", {}).get("rsi_sweet_spot", [30, 50])
        # 등락률 기반 점수: 0~3% 상승 = 고점수, 과도한 상승/하락 = 저점수
        change_score = pd.Series(np.zeros(len(df)), index=df.index)
        change_score = np.where(
            (change >= 0) & (change <= 5), 80 + change * 4,  # 0~5% → 80~100
            np.where(
                (change > 5) & (change <= 15), 70,  # 5~15% → 70
                np.where(
                    (change >= -3) & (change < 0), 60 + change * 5,  # -3~0% → 45~60
                    30  # 기타 → 30
                )
            )
        )
        tech_score = pd.Series(change_score, index=df.index) * 0.6 + volume_rank * 100 * 0.4
        return tech_score.clip(0, 100)

    # ------------------------------------------------------------------
    # Naver Finance API
    # ------------------------------------------------------------------

    def _fetch_market_stocks(self, market: str) -> list[dict]:
        """Naver Finance에서 시장 전체 종목 데이터 수집 (시총순)."""
        all_stocks = []
        page = 1
        page_size = 100

        while True:
            url = (
                f"{NAVER_API_BASE}/stocks/marketValue/{market}"
                f"?page={page}&pageSize={page_size}"
            )
            try:
                resp = self._session.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"Naver API failed (page {page}): {e}")
                break

            stocks = data.get("stocks", [])
            if not stocks:
                break

            for s in stocks:
                parsed = self._parse_naver_stock(s, market)
                if parsed:
                    all_stocks.append(parsed)

            total = data.get("totalCount", 0)
            if page * page_size >= total:
                break

            page += 1
            time.sleep(0.3)  # rate limit

        return all_stocks

    def _parse_naver_stock(self, s: dict, market: str) -> dict | None:
        """Naver API 응답을 내부 형식으로 변환."""
        try:
            ticker = s.get("itemCode", "")
            if not ticker or len(ticker) != 6:
                return None

            # ETF/ETN 제외
            stock_end_type = s.get("stockEndType", "")
            if stock_end_type in ("etf", "etn"):
                return None

            close = self._parse_number(s.get("closePrice", "0"))
            volume = self._parse_number(s.get("accumulatedTradingVolume", "0"))
            trading_value = self._parse_number(s.get("accumulatedTradingValue", "0"))
            # Naver의 accumulatedTradingValue는 백만원 단위
            trading_value_won = trading_value * 1_000_000
            market_cap = self._parse_number(s.get("marketValue", "0"))
            # Naver의 marketValue는 억원 단위
            market_cap_won = market_cap * 100_000_000
            change_pct = float(s.get("fluctuationsRatio", "0") or "0")

            return {
                "ticker": ticker,
                "name": s.get("stockName", ticker),
                "market": market,
                "close_price": close,
                "change_pct": change_pct,
                "volume": volume,
                "trading_value": trading_value_won,
                "market_cap": market_cap_won,
                "per": None,  # 상세 조회에서 채움
                "foreign_net": 0,
                "institution_net": 0,
            }
        except Exception as e:
            logger.debug(f"Failed to parse stock: {e}")
            return None

    def _fetch_stock_detail(self, ticker: str) -> dict:
        """개별 종목 상세 정보 (PER, PBR, 수급)."""
        url = f"{NAVER_API_BASE}/stock/{ticker}/integration"
        try:
            resp = self._session.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            info = {}
            for item in data.get("totalInfos", []):
                info[item.get("key", "")] = item.get("value", "")

            # 수급 정보
            trends = data.get("dealTrendInfos", [])
            foreign_net = 0
            institution_net = 0
            for t in trends[:5]:  # 최근 5일
                f_str = t.get("foreignerPureBuyQuant", "0")
                i_str = t.get("organPureBuyQuant", "0")
                foreign_net += self._parse_signed_number(f_str)
                institution_net += self._parse_signed_number(i_str)

            per_str = info.get("PER", "0")
            per = self._parse_float(per_str.replace("배", "").strip())

            return {
                "per": per,
                "foreign_net": foreign_net,
                "institution_net": institution_net,
            }
        except Exception as e:
            logger.debug(f"Detail fetch failed for {ticker}: {e}")
            return {"per": None, "foreign_net": 0, "institution_net": 0}

    def _enrich_with_details(self, stocks: list[dict], max_stocks: int = 50) -> list[dict]:
        """상위 N개 종목에 PER, 수급 상세 정보 추가."""
        for i, s in enumerate(stocks[:max_stocks]):
            detail = self._fetch_stock_detail(s["ticker"])
            s["per"] = detail.get("per")
            s["foreign_net"] = detail.get("foreign_net", 0)
            s["institution_net"] = detail.get("institution_net", 0)
            if (i + 1) % 10 == 0:
                logger.info(f"Screening detail: {i + 1}/{min(max_stocks, len(stocks))}")
            time.sleep(0.2)  # rate limit
        return stocks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_number(s: str | int | float) -> int:
        """콤마 포함 숫자 문자열 → int."""
        if isinstance(s, (int, float)):
            return int(s)
        cleaned = str(s).replace(",", "").replace(" ", "").strip()
        if not cleaned or cleaned == "-" or cleaned == "N/A":
            return 0
        return int(cleaned)

    @staticmethod
    def _parse_signed_number(s: str) -> int:
        """부호 포함 숫자 문자열 → int."""
        s = str(s).replace(",", "").replace("+", "").replace(" ", "")
        try:
            return int(s)
        except ValueError:
            return 0

    @staticmethod
    def _parse_float(s: str) -> float | None:
        """문자열 → float."""
        try:
            s = str(s).replace(",", "").replace(" ", "")
            return float(s) if s else None
        except ValueError:
            return None

    def _build_market_summary(
        self, kospi: list[dict], kosdaq: list[dict],
    ) -> dict:
        """시장 요약 통계."""
        def _summarize(stocks: list[dict]) -> dict:
            if not stocks:
                return {}
            changes = [s["change_pct"] for s in stocks if s.get("change_pct")]
            return {
                "total_stocks": len(stocks),
                "avg_change_pct": round(np.mean(changes), 2) if changes else 0,
                "advancing": sum(1 for c in changes if c > 0),
                "declining": sum(1 for c in changes if c < 0),
                "unchanged": sum(1 for c in changes if c == 0),
            }

        return {
            "kospi": _summarize(kospi),
            "kosdaq": _summarize(kosdaq),
        }

    # ------------------------------------------------------------------
    # DB 캐싱
    # ------------------------------------------------------------------

    def _save_to_db(self, result: ScreeningResult):
        today = datetime.now().strftime("%Y%m%d")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO screening_results "
                    "(date, timestamp, candidates_json, market_summary_json, stats_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        today,
                        result.timestamp,
                        json.dumps(result.candidates, ensure_ascii=False),
                        json.dumps(result.market_summary, ensure_ascii=False),
                        json.dumps(result.screening_stats, ensure_ascii=False),
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to cache screening result: {e}")

    def _load_from_db(self) -> ScreeningResult | None:
        """최근 캐시 로드 (당일 or 직전 거래일)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT timestamp, candidates_json, market_summary_json, stats_json "
                    "FROM screening_results ORDER BY date DESC LIMIT 1"
                ).fetchone()
            if not row:
                return None
            result = ScreeningResult(
                timestamp=row[0],
                candidates=json.loads(row[1]),
                market_summary=json.loads(row[2]),
                screening_stats=json.loads(row[3]),
            )
            self._last_result = result
            return result
        except Exception as e:
            logger.warning(f"Failed to load cached screening: {e}")
            return None
