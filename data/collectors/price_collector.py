"""OHLCV 가격 데이터 수집 및 저장."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

DB_DIR = Path(__file__).parent.parent / "storage"


class PriceCollector:
    """KIS API에서 가격 데이터를 수집하여 DB에 저장."""

    def __init__(self, market_client, db_path: str | None = None):
        self.market = market_client
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(DB_DIR / "trader.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS daily_ohlcv (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume INTEGER, amount REAL,
                    PRIMARY KEY (ticker, date)
                );

                CREATE TABLE IF NOT EXISTS minute_ohlcv (
                    ticker TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, datetime)
                );

                CREATE TABLE IF NOT EXISTS current_prices (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    price REAL, change_pct REAL, volume INTEGER,
                    high REAL, low REAL, open REAL, prev_close REAL,
                    market_cap REAL, per REAL, pbr REAL,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS investor_trends (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    foreign_net INTEGER,
                    institution_net INTEGER,
                    individual_net INTEGER,
                    PRIMARY KEY (ticker, date)
                );
            """)

    def collect_daily(self, ticker: str, days: int = 120) -> pd.DataFrame:
        """일봉 데이터 수집 및 저장."""
        try:
            records = self.market.get_daily_ohlcv(ticker, period="D", count=days)
            if not records:
                logger.warning(f"No daily data for {ticker}")
                return pd.DataFrame()

            with sqlite3.connect(self.db_path) as conn:
                for r in records:
                    conn.execute("""
                        INSERT OR REPLACE INTO daily_ohlcv
                        (ticker, date, open, high, low, close, volume, amount)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (ticker, r["date"], r["open"], r["high"],
                          r["low"], r["close"], r["volume"], r.get("amount", 0)))

            logger.info(f"Collected {len(records)} daily bars for {ticker}")
            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Daily collection failed for {ticker}: {e}")
            return pd.DataFrame()

    def collect_current_price(self, ticker: str) -> dict:
        """현재가 수집 및 저장."""
        try:
            data = self.market.get_current_price(ticker)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO current_prices
                    (ticker, name, price, change_pct, volume, high, low, open,
                     prev_close, market_cap, per, pbr, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["ticker"], data["name"], data["price"],
                    data["change_pct"], data["volume"], data["high"],
                    data["low"], data["open"], data["prev_close"],
                    data["market_cap"], data["per"], data["pbr"],
                    datetime.now().isoformat(),
                ))
            return data
        except Exception as e:
            logger.error(f"Current price collection failed for {ticker}: {e}")
            return {}

    def collect_investor_trends(self, ticker: str) -> dict:
        """투자자별 매매동향 수집."""
        try:
            data = self.market.get_investor_trends(ticker)
            today = datetime.now().strftime("%Y%m%d")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO investor_trends
                    (ticker, date, foreign_net, institution_net, individual_net)
                    VALUES (?, ?, ?, ?, ?)
                """, (ticker, today, data.get("foreign_net", 0),
                      data.get("institution_net", 0),
                      data.get("individual_net", 0)))
            return data
        except Exception as e:
            logger.error(f"Investor trends failed for {ticker}: {e}")
            return {}

    def get_stored_daily(self, ticker: str, days: int = 120) -> pd.DataFrame:
        """DB에서 저장된 일봉 데이터 로드."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM daily_ohlcv WHERE ticker = ? ORDER BY date DESC LIMIT ?",
                conn, params=(ticker, days),
            )
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def get_stored_current_prices(self, tickers: list[str] | None = None) -> dict:
        """DB에서 저장된 현재가 로드."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if tickers:
                placeholders = ",".join("?" * len(tickers))
                rows = conn.execute(
                    f"SELECT * FROM current_prices WHERE ticker IN ({placeholders})",
                    tickers,
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM current_prices").fetchall()
        return {r["ticker"]: dict(r) for r in rows}
