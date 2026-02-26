"""시뮬레이션 모드 엔진 — 실제 주문 없이 가상 체결."""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class SimulationTracker:
    """시뮬레이션 모드 매매 추적 및 성과 분석."""

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self._session_start = None
        self._session_trades: list[dict] = []

    def start_session(self):
        """시뮬레이션 세션 시작."""
        self._session_start = datetime.now()
        self._session_trades = []
        logger.info("Simulation session started")

    def record_trade(self, trade: dict):
        """시뮬 매매 기록."""
        trade["simulated"] = True
        trade["session_start"] = self._session_start.isoformat() if self._session_start else None
        self._session_trades.append(trade)

    def get_session_report(self) -> dict:
        """현재 시뮬레이션 세션 리포트."""
        if not self._session_start:
            return {"status": "no_session"}

        total_buys = sum(1 for t in self._session_trades if t.get("action") == "BUY")
        total_sells = sum(1 for t in self._session_trades if t.get("action") == "SELL")
        total_pnl = sum(t.get("pnl", 0) or 0 for t in self._session_trades if t.get("pnl"))

        return {
            "session_start": self._session_start.isoformat(),
            "duration_hours": (datetime.now() - self._session_start).total_seconds() / 3600,
            "total_trades": len(self._session_trades),
            "buys": total_buys,
            "sells": total_sells,
            "realized_pnl": total_pnl,
            "portfolio": self.portfolio.get_summary(),
            "trades": self._session_trades[-20:],
        }

    def save_report(self) -> str:
        """세션 리포트 저장."""
        report = self.get_session_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"sim_{timestamp}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Simulation report saved: {path}")
        return str(path)
