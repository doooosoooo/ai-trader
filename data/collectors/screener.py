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
            # 0건이면 DB 캐시 보존 (이전 좋은 결과 유지)
            if candidates:
                self._save_to_db(result)
            else:
                logger.warning("Screening returned 0 candidates — keeping previous DB cache")

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

        reject_reasons = {"excluded": 0, "preferred": 0, "market_cap": 0,
                          "trading_value": 0, "price": 0, "volume": 0, "per": 0}
        passed = []
        for s in stocks:
            ticker = s["ticker"]
            if ticker in exclude:
                reject_reasons["excluded"] += 1
                continue
            # 우선주 제외 (코드 마지막자리 5, 7, 8, 9 등)
            if not ticker[-1].isdigit() or ticker[-1] in ("5", "7", "8", "9"):
                if ticker[-1] != "0":
                    reject_reasons["preferred"] += 1
                    continue

            if s["market_cap"] < min_cap:
                reject_reasons["market_cap"] += 1
                continue
            # 장 시작 전에는 당일 거래대금이 0이므로, 거래대금 > 0인 경우에만 필터 적용
            if s["trading_value"] > 0 and s["trading_value"] < min_trade_val:
                reject_reasons["trading_value"] += 1
                continue
            if s["close_price"] < min_price:
                reject_reasons["price"] += 1
                continue
            # 장 시작 전에는 당일 거래량이 0이므로, close_price > 0이면 유효 종목으로 간주
            if s["volume"] <= 0 and s["close_price"] <= 0:
                reject_reasons["volume"] += 1
                continue
            # PER 필터 (0 이하 = 적자, 100 이상 = 극단적 고평가)
            per = s.get("per", 0)
            if per is not None and per != 0:
                if per <= min_per or per > max_per:
                    reject_reasons["per"] += 1
                    continue

            passed.append(s)

        logger.info(
            f"Stage1 filter breakdown: {len(stocks)} total → {len(passed)} passed | "
            f"rejected: {dict((k, v) for k, v in reject_reasons.items() if v > 0)}"
        )
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
                "rsi_14": round(float(row.get("rsi_14", 50)), 1),
                "ma5_above_ma20": bool(row.get("ma5_above_ma20", False)),
                "volume_trend": round(float(row.get("volume_trend", 1.0)), 2),
                "week52_position": round(float(row.get("week52_position", 0.5)), 2),
                "composite_score": round(float(row["composite_score"]), 1),
                "momentum_score": round(float(row["momentum_score"]), 1),
                "value_score": round(float(row["value_score"]), 1),
                "volume_score": round(float(row["volume_score"]), 1),
                "flow_score": round(float(row["flow_score"]), 1),
                "technical_score": round(float(row["technical_score"]), 1),
            })

        return result

    def _score_momentum(self, df: pd.DataFrame, scoring: dict) -> pd.Series:
        """모멘텀 점수: 등락률 + 이동평균 추세 + 연속 상승일."""
        # 단일일 급등락에 과점수 주는 것 방지(추격매수 리스크 완화): ±5% 하드캡
        pct = df["change_pct"].fillna(0).clip(-5.0, 5.0)
        # 등락률 백분위 (40%)
        lower = pct.quantile(0.05)
        upper = pct.quantile(0.95)
        clipped = pct.clip(lower, upper)
        change_rank = clipped.rank(pct=True) * 100

        # 5일선 > 20일선 골든크로스 (30%)
        ma_cross = df.get("ma5_above_ma20", pd.Series([False] * len(df)))
        ma_score = pd.Series(np.where(ma_cross.fillna(False), 80, 30), index=df.index, dtype=float)

        # 연속 상승일 (30%) — 최대 5일까지 반영
        up_days = df.get("consecutive_up_days", pd.Series([0] * len(df))).fillna(0).clip(0, 5)
        up_score = up_days / 5 * 100

        return change_rank * 0.4 + ma_score * 0.3 + up_score * 0.3

    def _score_value(self, df: pd.DataFrame, scoring: dict) -> pd.Series:
        """밸류 점수: PER 역순 백분위 (낮을수록 좋음)."""
        per = df["per"].fillna(0).replace(0, np.nan)
        # PER가 없는 종목은 50점 (중립)
        per_score = (1 - per.rank(pct=True)) * 100
        return per_score.fillna(50)

    def _score_volume(self, df: pd.DataFrame, scoring: dict) -> pd.Series:
        """거래량 점수: 거래대금 백분위 + 거래량 추세."""
        val = df["trading_value"].fillna(0)
        val_rank = val.rank(pct=True) * 100

        # 거래량 추세: 최근 5일 / 20일 평균 (거래량 급증 감지)
        vol_trend = df.get("volume_trend", pd.Series([1.0] * len(df))).fillna(1.0)
        # 1.5배 이상이면 80점, 2배 이상이면 100점, 0.5배 이하면 20점
        trend_score = pd.Series(np.clip((vol_trend - 0.5) / 1.5 * 100, 10, 100), index=df.index)

        return val_rank * 0.5 + trend_score * 0.5

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
        """기술적 점수: RSI + 20일선 대비 위치 + 52주 위치.

        강세 추세 종목(RSI 65~80, 52주 신고가 근처, 20일선 위 멀리)을 페널티 주지 않도록
        설계. 과매수 극단(RSI 85+)에서만 약하게 감점.
        """
        sweet = scoring.get("technical", {}).get("rsi_sweet_spot", [40, 65])

        # RSI 점수 (40%): sweet spot 최고, 강세(65~80) 양호, 극과매수(85+)만 감점
        rsi = df.get("rsi_14", pd.Series([50.0] * len(df))).fillna(50.0)
        rsi_score = pd.Series(np.where(
            (rsi >= sweet[0]) & (rsi <= sweet[1]), 90,   # sweet spot
            np.where(
                (rsi > sweet[1]) & (rsi <= 75), 80,       # 강세 추세 = 양호
                np.where(
                    (rsi > 75) & (rsi <= 85), 65,         # 강한 과매수 = 약감점
                    np.where(
                        rsi > 85, 45,                      # 극단 과매수 = 감점 (이전 35→45)
                        np.where(
                            (rsi >= 30) & (rsi < sweet[0]), 60,  # 약세 회복 구간
                            40                             # RSI 30 미만 = 약세
                        )
                    )
                )
            )
        ), index=df.index, dtype=float)

        # 20일선 대비 위치 점수 (30%): 추세 상승 종목 보호
        price_ma = df.get("price_vs_ma20", pd.Series([0.0] * len(df))).fillna(0.0) * 100
        ma_pos_score = pd.Series(np.where(
            (price_ma >= 0) & (price_ma <= 5), 90,        # 20일선 바로 위 = 최고
            np.where(
                (price_ma > 5) & (price_ma <= 15), 80,     # 강한 추세 = 양호 (이전 70→80)
                np.where(
                    (price_ma > 15) & (price_ma <= 25), 65,  # 매우 강한 추세 (신규)
                    np.where(
                        price_ma > 25, 50,                  # 과열 (신규)
                        np.where(
                            (price_ma >= -5) & (price_ma < 0), 65,  # 눌림목
                            np.where(price_ma < -5, 40, 50)
                        )
                    )
                )
            )
        ), index=df.index, dtype=float)

        # 52주 위치 점수 (30%): 신고가 페널티 대폭 완화 — 추세 종목 허용
        w52 = df.get("week52_position", pd.Series([0.5] * len(df))).fillna(0.5)
        w52_score = pd.Series(np.where(
            (w52 >= 0.3) & (w52 <= 0.7), 75,              # 중간 = 양호
            np.where(
                (w52 > 0.7) & (w52 <= 0.85), 80,           # 고점 근처 = 추세 우수 (이전 60→80)
                np.where(
                    (w52 > 0.85) & (w52 <= 0.95), 75,      # 신고가 근처 (이전 35→75)
                    np.where(w52 > 0.95, 60,               # 52주 정점 직전 (이전 35→60)
                             55)                            # 저점권
                )
            )
        ), index=df.index, dtype=float)

        return (rsi_score * 0.4 + ma_pos_score * 0.3 + w52_score * 0.3).clip(0, 100)

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
        """개별 종목 상세 정보 (PER, PBR, 수급, 기술지표)."""
        result = {"per": None, "foreign_net": 0, "institution_net": 0,
                  "rsi_14": 50.0, "ma5_above_ma20": False, "price_vs_ma20": 0.0,
                  "volume_trend": 1.0, "week52_position": 0.5, "pbr": None,
                  "consecutive_up_days": 0, "prev_change_pct": None}

        # 1) 종합 정보 (PER, 수급, 52주 고저)
        try:
            url = f"{NAVER_API_BASE}/stock/{ticker}/integration"
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
            result["per"] = self._parse_float(per_str.replace("배", "").strip())
            pbr_str = info.get("PBR", "0")
            result["pbr"] = self._parse_float(pbr_str.replace("배", "").strip())
            result["foreign_net"] = foreign_net
            result["institution_net"] = institution_net

            # 52주 고저 대비 현재 위치
            w52_high = self._parse_number(info.get("52주 최고", "0"))
            w52_low = self._parse_number(info.get("52주 최저", "0"))
            if w52_high > w52_low > 0:
                close = self._parse_number(info.get("전일", "0"))
                result["week52_position"] = (close - w52_low) / (w52_high - w52_low)

        except Exception as e:
            logger.debug(f"Detail fetch failed for {ticker}: {e}")

        # 2) 일봉 데이터로 기술지표 계산 (RSI, MA, 거래량 추세)
        try:
            url = f"{NAVER_API_BASE}/stock/{ticker}/price?pageSize=30&page=1&chartType=day"
            resp = self._session.get(url, timeout=5)
            resp.raise_for_status()
            bars = resp.json()

            # 장 시작 전(또는 동시호가)이면 Naver가 오늘 날짜의 "빈/임시 봉"을 보냄.
            # closePrice = 어제 종가 그대로(fluctuationsRatio="0.00")이면 시간외/동시호가 임시 데이터 →
            # 이를 제거하지 않으면 closes[-1]=어제 종가, closes[-2]=어제 종가가 되어 모멘텀/등락률이 모두 0.
            today_str = datetime.now().strftime("%Y-%m-%d")
            if (bars and bars[0].get("localTradedAt") == today_str
                    and bars[0].get("fluctuationsRatio", "0") == "0.00"):
                bars = bars[1:]

            if bars and len(bars) >= 14:
                closes = [self._parse_number(b.get("closePrice", "0")) for b in bars]
                volumes = [int(b.get("accumulatedTradingVolume", 0)) for b in bars]
                closes.reverse()   # 오래된 순으로
                volumes.reverse()

                # RSI 14
                result["rsi_14"] = self._calc_rsi(closes, 14)

                # 5일선 vs 20일선
                if len(closes) >= 20:
                    ma5 = sum(closes[-5:]) / 5
                    ma20 = sum(closes[-20:]) / 20
                    result["ma5_above_ma20"] = ma5 > ma20
                    current = closes[-1]
                    result["price_vs_ma20"] = (current - ma20) / ma20 if ma20 > 0 else 0

                # 거래량 추세: 최근 5일 평균 / 20일 평균
                if len(volumes) >= 20:
                    vol_5 = sum(volumes[-5:]) / 5
                    vol_20 = sum(volumes[-20:]) / 20
                    result["volume_trend"] = vol_5 / vol_20 if vol_20 > 0 else 1.0

                # 연속 상승일
                up_days = 0
                for i in range(len(closes) - 1, 0, -1):
                    if closes[i] > closes[i - 1]:
                        up_days += 1
                    else:
                        break
                result["consecutive_up_days"] = up_days

                # 전일 등락률 (장 시작 전 스크리닝 시 오늘 change_pct=0인 문제 보완)
                if len(closes) >= 2 and closes[-2] > 0:
                    result["prev_change_pct"] = (closes[-1] - closes[-2]) / closes[-2] * 100

        except Exception as e:
            logger.debug(f"Price data fetch failed for {ticker}: {e}")

        return result

    @staticmethod
    def _calc_rsi(closes: list[float], period: int = 14) -> float:
        """RSI 계산."""
        if len(closes) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _enrich_with_details(self, stocks: list[dict], max_stocks: int = 50) -> list[dict]:
        """상위 N개 종목에 PER, 수급, 기술지표 상세 정보 추가."""
        for i, s in enumerate(stocks[:max_stocks]):
            detail = self._fetch_stock_detail(s["ticker"])
            s["per"] = detail.get("per")
            s["pbr"] = detail.get("pbr")
            s["foreign_net"] = detail.get("foreign_net", 0)
            s["institution_net"] = detail.get("institution_net", 0)
            s["rsi_14"] = detail.get("rsi_14", 50.0)
            s["ma5_above_ma20"] = detail.get("ma5_above_ma20", False)
            s["price_vs_ma20"] = detail.get("price_vs_ma20", 0.0)
            s["volume_trend"] = detail.get("volume_trend", 1.0)
            s["week52_position"] = detail.get("week52_position", 0.5)
            s["consecutive_up_days"] = detail.get("consecutive_up_days", 0)
            # 장 시작 전 스크리닝 시 change_pct=0이면 전일 등락률로 대체 (모멘텀 점수 정상화)
            prev_pct = detail.get("prev_change_pct")
            if prev_pct is not None and abs(s.get("change_pct", 0)) < 0.01:
                s["change_pct"] = prev_pct
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
