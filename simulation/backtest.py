"""백테스팅 엔진 — 과거 데이터 기반 전략 검증.

인메모리 포트폴리오로 라이브 DB와 완전 분리.
"""

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from core.executor import adjust_price_to_tick
from core.indicators import calculate_all_indicators
from simulation.metrics import calculate_buy_and_hold, calculate_metrics
from simulation.strategies import Strategy

# 수수료율 (core/executor.py 309-319행 기준)
BUY_FEE_RATE = 0.00015                # 매수: 0.015%
SELL_FEE_RATE = 0.00015 + 0.0018 + 0.0002  # 매도: 0.015% + 0.18% + 0.02%


@dataclass
class BacktestPosition:
    """인메모리 포지션."""

    ticker: str
    name: str
    quantity: int
    entry_price: float
    entry_date: str
    hold_days: int = 0
    peak_price: float = 0.0  # 트레일링 스톱용

    def __post_init__(self):
        if self.peak_price == 0:
            self.peak_price = self.entry_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price


@dataclass
class BacktestTrade:
    """매매 기록."""

    date: str
    ticker: str
    action: str  # "BUY" or "SELL"
    quantity: int
    price: float
    amount: float
    fee: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    hold_days: int = 0


@dataclass
class BacktestResult:
    """백테스트 실행 결과."""

    strategy_name: str
    params: dict
    tickers: list[str]
    period: str  # "20240301 ~ 20260227"
    trades: list[BacktestTrade]
    equity_curve: list[dict]
    metrics: dict

    def _calc_ticker_breakdown(self) -> dict:
        """종목별 성과 집계 — LLM 피드백에 사용."""
        breakdown: dict[str, dict] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total_pnl_pct": 0.0, "trades": 0}
        )
        for t in self.trades:
            if t.action != "SELL":
                continue
            entry = breakdown[t.ticker]
            entry["trades"] += 1
            entry["total_pnl_pct"] += t.pnl_pct
            if t.pnl > 0:
                entry["wins"] += 1
            elif t.pnl < 0:
                entry["losses"] += 1
        # 평균 수익률 계산
        result = {}
        for ticker, stats in breakdown.items():
            if stats["trades"] > 0:
                stats["avg_pnl_pct"] = stats["total_pnl_pct"] / stats["trades"]
            else:
                stats["avg_pnl_pct"] = 0.0
            result[ticker] = dict(stats)
        return result

    def save_to_db(self, db_path: str) -> int:
        """백테스트 결과를 DB에 저장. 반환: inserted row id."""
        period_parts = self.period.split("~")
        period_start = period_parts[0].strip() if len(period_parts) >= 1 else ""
        period_end = period_parts[1].strip() if len(period_parts) >= 2 else ""

        ticker_breakdown = self._calc_ticker_breakdown()
        m = self.metrics

        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    tickers TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    total_return REAL,
                    annualized_return REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    avg_hold_days REAL,
                    metrics_json TEXT NOT NULL,
                    ticker_breakdown_json TEXT NOT NULL
                )
            """)
            cursor = conn.execute(
                """INSERT INTO backtest_results
                (created_at, strategy_name, params_json, tickers, period_start, period_end,
                 total_return, annualized_return, max_drawdown, sharpe_ratio,
                 win_rate, profit_factor, total_trades, avg_hold_days,
                 metrics_json, ticker_breakdown_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    self.strategy_name,
                    json.dumps(self.params, ensure_ascii=False),
                    ",".join(self.tickers),
                    period_start, period_end,
                    m.get("total_return", 0),
                    m.get("annualized_return", 0),
                    m.get("max_drawdown", 0),
                    m.get("sharpe_ratio", 0),
                    m.get("win_rate", 0),
                    m.get("profit_factor", 0),
                    m.get("total_trades", 0),
                    m.get("avg_hold_days", 0),
                    json.dumps(m, ensure_ascii=False),
                    json.dumps(ticker_breakdown, ensure_ascii=False),
                ),
            )
            logger.info(
                f"Backtest result saved: {self.strategy_name} | "
                f"return={m.get('total_return', 0):.2%} | id={cursor.lastrowid}"
            )
            return cursor.lastrowid


class BacktestPortfolio:
    """인메모리 포트폴리오 — DB 접근 없음."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, BacktestPosition] = {}
        self.trades: list[BacktestTrade] = []
        self._last_market_value = 0.0

    @property
    def total_value(self) -> float:
        return self.cash + self._last_market_value

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """모든 포지션 시가 평가."""
        self._last_market_value = sum(
            pos.quantity * prices.get(pos.ticker, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.total_value

    def buy(
        self, ticker: str, name: str, price: float, date: str, size_pct: float,
    ) -> BacktestTrade | None:
        """매수. 자금 부족 시 None."""
        adjusted_price = adjust_price_to_tick(price, "down")
        if adjusted_price <= 0:
            return None

        order_amount = self.total_value * size_pct
        quantity = int(order_amount / adjusted_price)
        if quantity <= 0:
            return None

        cost = quantity * adjusted_price
        fee = cost * BUY_FEE_RATE
        if cost + fee > self.cash:
            # 자금에 맞게 수량 줄이기
            quantity = int(self.cash / (adjusted_price * (1 + BUY_FEE_RATE)))
            if quantity <= 0:
                return None
            cost = quantity * adjusted_price
            fee = cost * BUY_FEE_RATE

        self.cash -= (cost + fee)
        self.positions[ticker] = BacktestPosition(
            ticker=ticker, name=name, quantity=quantity,
            entry_price=adjusted_price, entry_date=date,
        )
        trade = BacktestTrade(
            date=date, ticker=ticker, action="BUY",
            quantity=quantity, price=adjusted_price, amount=cost, fee=fee,
        )
        self.trades.append(trade)
        return trade

    def sell(
        self, ticker: str, price: float, date: str, reason: str = "",
    ) -> BacktestTrade | None:
        """전량 매도. 포지션 없으면 None."""
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        adjusted_price = adjust_price_to_tick(price, "up")
        if adjusted_price <= 0:
            adjusted_price = price

        proceeds = pos.quantity * adjusted_price
        fee = proceeds * SELL_FEE_RATE
        pnl = proceeds - fee - pos.cost_basis - (pos.cost_basis * BUY_FEE_RATE)
        pnl_pct = pnl / pos.cost_basis if pos.cost_basis > 0 else 0

        self.cash += (proceeds - fee)
        hold_days = pos.hold_days

        del self.positions[ticker]

        trade = BacktestTrade(
            date=date, ticker=ticker, action="SELL",
            quantity=pos.quantity, price=adjusted_price,
            amount=proceeds, fee=fee,
            pnl=pnl, pnl_pct=pnl_pct,
            reason=reason, hold_days=hold_days,
        )
        self.trades.append(trade)
        return trade


class Backtester:
    """백테스팅 엔진.

    사용법:
        bt = Backtester(initial_capital=10_000_000)
        result = bt.run(
            tickers=["005930", "000660"],
            strategy=create_swing_strategy(),
            start_date="20240301",
            end_date="20260227",
        )
        print(result.metrics)
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000,
        db_path: str | None = None,
    ):
        self.initial_capital = initial_capital
        self.db_path = db_path or str(
            Path(__file__).parent.parent / "data" / "storage" / "trader.db"
        )

    def load_data(
        self, tickers: list[str], start_date: str, end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """DB에서 OHLCV 데이터 로드."""
        # 하이픈 제거 (20240301 형식 통일)
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        data = {}
        with sqlite3.connect(self.db_path) as conn:
            for ticker in tickers:
                df = pd.read_sql_query(
                    "SELECT * FROM daily_ohlcv "
                    "WHERE ticker = ? AND date >= ? AND date <= ? "
                    "ORDER BY date",
                    conn, params=(ticker, start, end),
                )
                if not df.empty:
                    data[ticker] = df
                    logger.debug(f"Loaded {len(df)} bars for {ticker} ({start}~{end})")
                else:
                    logger.warning(f"No data for {ticker} in range {start}~{end}")

        return data

    def run(
        self,
        tickers: list[str],
        strategy: Strategy,
        start_date: str = "",
        end_date: str = "",
        indicator_params: dict | None = None,
    ) -> BacktestResult:
        """백테스트 실행.

        1. DB에서 OHLCV 로드
        2. calculate_all_indicators() 적용
        3. 날짜별 순회: 청산 → 진입 → 자산곡선
        4. 잔여 포지션 강제 청산
        5. 성과지표 계산
        """
        # 데이터 로드
        raw_data = self.load_data(tickers, start_date, end_date)
        if not raw_data:
            logger.error("No data loaded for backtesting")
            return BacktestResult(
                strategy_name=strategy.name, params=strategy.to_dict(),
                tickers=tickers, period=f"{start_date}~{end_date}",
                trades=[], equity_curve=[], metrics={"error": "no_data"},
            )

        # 지표 계산
        ticker_data: dict[str, pd.DataFrame] = {}
        for ticker, df in raw_data.items():
            if len(df) < 30:
                logger.warning(f"Skipping {ticker}: only {len(df)} bars (need >=30)")
                continue
            df_ind = calculate_all_indicators(df.copy(), indicator_params or {})
            ticker_data[ticker] = df_ind.reset_index(drop=True)

        if not ticker_data:
            return BacktestResult(
                strategy_name=strategy.name, params=strategy.to_dict(),
                tickers=tickers, period=f"{start_date}~{end_date}",
                trades=[], equity_curve=[], metrics={"error": "insufficient_data"},
            )

        # 통합 거래일 인덱스 구축
        all_dates = sorted(set(
            date
            for df in ticker_data.values()
            for date in df["date"].tolist()
        ))

        # 종목별 날짜 인덱스 매핑
        ticker_date_idx: dict[str, dict[str, int]] = {}
        for ticker, df in ticker_data.items():
            ticker_date_idx[ticker] = {row["date"]: i for i, row in df.iterrows()}

        # 종목명 매핑
        ticker_names = {}
        for ticker, df in ticker_data.items():
            ticker_names[ticker] = ticker  # DB에 name 없으면 코드 사용

        # 포트폴리오 초기화
        portfolio = BacktestPortfolio(self.initial_capital)
        equity_curve = []

        # 지표 계산에 필요한 워밍업 기간 스킵 (60일)
        warmup = 60
        trading_dates = all_dates[warmup:] if len(all_dates) > warmup else all_dates[20:]

        for date in trading_dates:
            # 현재가 수집
            current_prices = {}
            for ticker, df in ticker_data.items():
                idx = ticker_date_idx[ticker].get(date)
                if idx is not None:
                    current_prices[ticker] = df.iloc[idx]["close"]

            # 포트폴리오 시가평가
            portfolio.mark_to_market(current_prices)

            # --- 1. 보유 포지션 청산 체크 ---
            tickers_to_sell = []
            for ticker, pos in list(portfolio.positions.items()):
                idx = ticker_date_idx.get(ticker, {}).get(date)
                if idx is None or idx < 1:
                    pos.hold_days += 1
                    continue

                df = ticker_data[ticker]
                row = df.iloc[idx]
                prev = df.iloc[idx - 1]

                # hold_days 업데이트
                pos.hold_days += 1

                # peak 가격 업데이트 (트레일링 스톱용)
                current_price = row["close"]
                if current_price > pos.peak_price:
                    pos.peak_price = current_price

                # 포지션 컨텍스트 구성
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                drawdown_from_peak = (pos.peak_price - current_price) / pos.peak_price if pos.peak_price > 0 else 0
                ctx = {
                    "entry_price": pos.entry_price,
                    "entry_date": pos.entry_date,
                    "hold_days": pos.hold_days,
                    "pnl_pct": pnl_pct,
                    "peak_price": pos.peak_price,
                    "drawdown_from_peak": drawdown_from_peak,
                }

                should_exit, reason = strategy.check_exit(row, prev, ctx)
                if should_exit and pos.hold_days >= strategy.min_hold_days:
                    tickers_to_sell.append((ticker, current_price, reason))

            # 매도 실행
            for ticker, price, reason in tickers_to_sell:
                portfolio.sell(ticker, price, date, reason)

            # --- 2. 신규 진입 체크 ---
            if len(portfolio.positions) < strategy.max_positions:
                for ticker, df in ticker_data.items():
                    if ticker in portfolio.positions:
                        continue
                    if len(portfolio.positions) >= strategy.max_positions:
                        break

                    idx = ticker_date_idx.get(ticker, {}).get(date)
                    if idx is None or idx < 1:
                        continue

                    row = df.iloc[idx]
                    prev = df.iloc[idx - 1]
                    ctx = {}  # 진입 시에는 포지션 컨텍스트 없음

                    if strategy.check_entry(row, prev, ctx):
                        portfolio.buy(
                            ticker, ticker_names.get(ticker, ticker),
                            row["close"], date, strategy.position_size_pct,
                        )

            # --- 3. 자산곡선 기록 ---
            portfolio.mark_to_market(current_prices)
            equity_curve.append({"date": date, "value": portfolio.total_value})

        # --- 4. 잔여 포지션 강제 청산 ---
        for ticker in list(portfolio.positions.keys()):
            last_price = current_prices.get(ticker, portfolio.positions[ticker].entry_price)
            portfolio.sell(ticker, last_price, trading_dates[-1] if trading_dates else "", "backtest_end")

        # --- 5. 벤치마크 (첫 종목 바이앤홀드) ---
        benchmark_equity = None
        if ticker_data:
            first_ticker = list(ticker_data.keys())[0]
            first_df = ticker_data[first_ticker]
            # 워밍업 이후 데이터만 사용
            bm_df = first_df[first_df["date"].isin(set(d["date"] for d in equity_curve))]
            if not bm_df.empty:
                benchmark_equity = calculate_buy_and_hold(bm_df, self.initial_capital)

        # --- 6. 성과지표 계산 ---
        metrics = calculate_metrics(
            equity_curve=equity_curve,
            trades=portfolio.trades,
            initial_capital=self.initial_capital,
            benchmark_equity=benchmark_equity,
        )

        period_str = f"{trading_dates[0]}~{trading_dates[-1]}" if trading_dates else ""
        logger.info(
            f"Backtest complete: {strategy.name} | "
            f"Return: {metrics.get('total_return', 0):.2%} | "
            f"MDD: {metrics.get('max_drawdown', 0):.2%} | "
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | "
            f"Trades: {metrics.get('total_trades', 0)}"
        )

        return BacktestResult(
            strategy_name=strategy.name,
            params=strategy.to_dict(),
            tickers=list(ticker_data.keys()),
            period=period_str,
            trades=portfolio.trades,
            equity_curve=equity_curve,
            metrics=metrics,
        )

    def run_multi_strategy(
        self,
        tickers: list[str],
        strategies: list[Strategy],
        start_date: str = "",
        end_date: str = "",
        indicator_params: dict | None = None,
    ) -> list[BacktestResult]:
        """여러 전략을 동일 데이터로 비교."""
        return [
            self.run(tickers, s, start_date, end_date, indicator_params)
            for s in strategies
        ]
