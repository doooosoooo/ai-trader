"""포트폴리오 상태 관리 — 보유 종목, 수익률, 현금 추적."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

DB_DIR = Path(__file__).parent.parent / "data" / "storage"


class Position:
    """개별 보유 종목."""

    def __init__(
        self,
        ticker: str,
        name: str,
        quantity: int,
        avg_price: float,
        current_price: float = 0,
        bought_at: str = "",
    ):
        self.ticker = ticker
        self.name = name
        self.quantity = quantity
        self.avg_price = avg_price
        self.current_price = current_price or avg_price
        self.bought_at = bought_at or datetime.now().isoformat()

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    @property
    def pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.pnl / self.cost_basis

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "pnl": self.pnl,
            "pnl_pct": f"{self.pnl_pct:.2%}",
            "bought_at": self.bought_at,
        }


class Portfolio:
    """포트폴리오 상태 관리."""

    def __init__(self, db_path: str | None = None, mode: str = "simulation"):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(DB_DIR / "trader.db")
        self.mode = mode  # "simulation" or "live"
        self._init_db()
        self._migrate_db()
        self.positions: dict[str, Position] = {}
        self.cash: float = 0.0
        self.initial_capital: float = 0.0
        self._load_state()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL DEFAULT 0,
                    initial_capital REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    ticker TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    current_price REAL NOT NULL DEFAULT 0,
                    bought_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    fee REAL NOT NULL DEFAULT 0,
                    pnl REAL,
                    pnl_pct REAL,
                    reason TEXT,
                    signal_json TEXT,
                    mode TEXT NOT NULL DEFAULT 'simulation'
                );

                CREATE TABLE IF NOT EXISTS daily_snapshot (
                    date TEXT PRIMARY KEY,
                    total_asset REAL NOT NULL,
                    cash REAL NOT NULL,
                    invested REAL NOT NULL,
                    daily_pnl REAL,
                    daily_pnl_pct REAL,
                    cumulative_pnl_pct REAL,
                    positions_json TEXT
                );
            """)

    def _migrate_db(self) -> None:
        """기존 DB에 새 컬럼이 없으면 추가."""
        with sqlite3.connect(self.db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(trade_history)").fetchall()]
            if "mode" not in cols:
                conn.execute("ALTER TABLE trade_history ADD COLUMN mode TEXT NOT NULL DEFAULT 'simulation'")
                logger.info("Migrated trade_history: added mode column")

    def _load_state(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT cash, initial_capital FROM portfolio_state WHERE id = 1").fetchone()
            if row:
                self.cash = row[0]
                self.initial_capital = row[1]

            rows = conn.execute("SELECT ticker, name, quantity, avg_price, current_price, bought_at FROM positions").fetchall()
            for r in rows:
                self.positions[r[0]] = Position(
                    ticker=r[0], name=r[1], quantity=r[2],
                    avg_price=r[3], current_price=r[4], bought_at=r[5],
                )
        logger.info(f"Portfolio loaded: cash={self.cash:,.0f}, positions={len(self.positions)}")

    def _save_state(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                INSERT INTO portfolio_state (id, cash, initial_capital, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET cash=?, initial_capital=?, updated_at=?
            """, (
                self.cash, self.initial_capital, datetime.now().isoformat(),
                self.cash, self.initial_capital, datetime.now().isoformat(),
            ))

            conn.execute("DELETE FROM positions")
            for pos in self.positions.values():
                conn.execute("""
                    INSERT INTO positions (ticker, name, quantity, avg_price, current_price, bought_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (pos.ticker, pos.name, pos.quantity, pos.avg_price, pos.current_price, pos.bought_at))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Portfolio save failed, rolled back: {e}")
            raise
        finally:
            conn.close()

    def initialize(self, capital: float) -> None:
        """초기 자본금 설정."""
        self.cash = capital
        self.initial_capital = capital
        self._save_state()
        logger.info(f"Portfolio initialized: {capital:,.0f} KRW")

    @property
    def total_invested(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_asset(self) -> float:
        return self.cash + self.total_invested

    @property
    def total_pnl(self) -> float:
        return self.total_asset - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return self.total_pnl / self.initial_capital

    @property
    def cash_ratio(self) -> float:
        if self.total_asset == 0:
            return 1.0
        return self.cash / self.total_asset

    def update_prices(self, prices: dict[str, float]) -> None:
        """보유 종목 현재가 업데이트."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price
        self._save_state()

    def execute_buy(
        self,
        ticker: str,
        name: str,
        quantity: int,
        price: float,
        fee: float = 0,
        reason: str = "",
        signal_json: str = "",
    ) -> dict:
        """매수 실행 기록."""
        amount = quantity * price + fee
        if amount > self.cash:
            raise ValueError(f"Insufficient cash: need {amount:,.0f}, have {self.cash:,.0f}")

        self.cash -= amount

        if ticker in self.positions:
            pos = self.positions[ticker]
            total_qty = pos.quantity + quantity
            pos.avg_price = (pos.cost_basis + quantity * price) / total_qty
            pos.quantity = total_qty
            pos.current_price = price
        else:
            self.positions[ticker] = Position(
                ticker=ticker, name=name, quantity=quantity,
                avg_price=price, current_price=price,
            )

        trade = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "name": name,
            "action": "BUY",
            "quantity": quantity,
            "price": price,
            "amount": amount,
            "fee": fee,
            "reason": reason,
        }
        self._record_trade(trade, signal_json)
        self._save_state()
        logger.info(f"BUY {name}({ticker}) x{quantity} @{price:,.0f} = {amount:,.0f}")
        return trade

    def execute_sell(
        self,
        ticker: str,
        quantity: int,
        price: float,
        fee: float = 0,
        reason: str = "",
        signal_json: str = "",
    ) -> dict:
        """매도 실행 기록."""
        if ticker not in self.positions:
            raise ValueError(f"No position for {ticker}")

        pos = self.positions[ticker]
        if quantity > pos.quantity:
            raise ValueError(f"Insufficient quantity: have {pos.quantity}, sell {quantity}")

        amount = quantity * price - fee
        pnl = (price - pos.avg_price) * quantity - fee
        pnl_pct = (price - pos.avg_price) / pos.avg_price if pos.avg_price > 0 else 0

        self.cash += amount

        if quantity >= pos.quantity:
            del self.positions[ticker]
        else:
            pos.quantity -= quantity
            pos.current_price = price

        trade = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "name": pos.name,
            "action": "SELL",
            "quantity": quantity,
            "price": price,
            "amount": amount,
            "fee": fee,
            "pnl": pnl,
            "pnl_pct": f"{pnl_pct:.2%}",
            "reason": reason,
        }
        self._record_trade(trade, signal_json)
        self._save_state()
        logger.info(f"SELL {pos.name}({ticker}) x{quantity} @{price:,.0f} PnL={pnl:,.0f} ({pnl_pct:.2%})")
        return trade

    def _record_trade(self, trade: dict, signal_json: str = "") -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trade_history
                (timestamp, ticker, name, action, quantity, price, amount, fee, pnl, pnl_pct, reason, signal_json, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade["timestamp"], trade["ticker"], trade["name"], trade["action"],
                trade["quantity"], trade["price"], trade["amount"], trade.get("fee", 0),
                trade.get("pnl"), trade.get("pnl_pct"), trade.get("reason", ""),
                signal_json, self.mode,
            ))

    def save_daily_snapshot(self) -> None:
        """일일 스냅샷 저장."""
        today = datetime.now().strftime("%Y-%m-%d")
        positions_json = json.dumps(
            {t: p.to_dict() for t, p in self.positions.items()},
            ensure_ascii=False,
        )

        with sqlite3.connect(self.db_path) as conn:
            # 전일 스냅샷으로 일일 PnL 계산
            prev = conn.execute(
                "SELECT total_asset FROM daily_snapshot ORDER BY date DESC LIMIT 1"
            ).fetchone()
            prev_asset = prev[0] if prev else self.initial_capital
            daily_pnl = self.total_asset - prev_asset
            daily_pnl_pct = daily_pnl / prev_asset if prev_asset > 0 else 0

            conn.execute("""
                INSERT INTO daily_snapshot
                (date, total_asset, cash, invested, daily_pnl, daily_pnl_pct, cumulative_pnl_pct, positions_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                total_asset=?, cash=?, invested=?, daily_pnl=?, daily_pnl_pct=?, cumulative_pnl_pct=?, positions_json=?
            """, (
                today, self.total_asset, self.cash, self.total_invested,
                daily_pnl, daily_pnl_pct, self.total_pnl_pct, positions_json,
                self.total_asset, self.cash, self.total_invested,
                daily_pnl, daily_pnl_pct, self.total_pnl_pct, positions_json,
            ))

    def get_trade_history(self, ticker: str | None = None, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM trade_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?",
                    (ticker, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trade_history ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_daily_snapshots(self, days: int = 30) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM daily_snapshot ORDER BY date DESC LIMIT ?",
                (days,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_summary(self) -> dict:
        """포트폴리오 요약 — LLM/알림용."""
        return {
            "total_asset": self.total_asset,
            "cash": self.cash,
            "cash_ratio": f"{self.cash_ratio:.1%}",
            "invested": self.total_invested,
            "initial_capital": self.initial_capital,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": f"{self.total_pnl_pct:.2%}",
            "positions": {t: p.to_dict() for t, p in self.positions.items()},
            "num_positions": len(self.positions),
        }

    def get_today_trades(self) -> list[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trade_history WHERE timestamp LIKE ? ORDER BY timestamp",
                (f"{today}%",),
            ).fetchall()
            return [dict(r) for r in rows]
