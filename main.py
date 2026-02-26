"""AI Trader — 메인 엔트리포인트.

한국 주식 AI 자동매매 시스템.
LLM(Claude)을 전략 두뇌로, ML 모델을 예측 보조로 사용.
"""

import asyncio
import json
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# .env 로드
load_dotenv(Path(__file__).parent / "config" / ".env")

# 로깅 설정
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
(LOG_DIR / "decisions").mkdir(exist_ok=True)
(LOG_DIR / "trades").mkdir(exist_ok=True)
(LOG_DIR / "errors").mkdir(exist_ok=True)
(LOG_DIR / "reviews").mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
logger.add(LOG_DIR / "app_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days", level="DEBUG")
logger.add(LOG_DIR / "errors" / "error_{time:YYYY-MM-DD}.log", rotation="1 day", retention="60 days", level="ERROR")

from core.config_manager import ConfigManager
from core.market_data import KISAuth, MarketDataClient
from core.safety_guard import SafetyGuard
from core.circuit_breaker import CircuitBreaker
from core.llm_engine import LLMEngine
from core.ml_engine import MLEngine
from core.portfolio import Portfolio
from core.executor import OrderExecutor
from data.pipeline import DataPipeline
from interfaces.telegram_bot import TelegramBot
from review.daily_review import DailyReviewer
from review.strategy_evaluator import StrategyEvaluator
from simulation.simulator import SimulationTracker
from scheduler import TradingScheduler


class TradingSystem:
    """AI 트레이딩 시스템 — 전체 통합 관리."""

    def __init__(self):
        logger.info("Initializing AI Trading System...")

        # Config
        self.config_manager = ConfigManager()
        settings = self.config_manager.settings
        safety_rules = self.config_manager.safety_rules
        trading_params = self.config_manager.trading_params

        # KIS API
        self.kis_auth = KISAuth(settings)
        self.market_client = MarketDataClient(self.kis_auth)

        # Core
        self.safety_guard = SafetyGuard(safety_rules, trading_params)
        self.circuit_breaker = CircuitBreaker(
            safety_rules,
            notify_callback=lambda **kw: self._notify(**kw),
        )
        self.llm_engine = LLMEngine(settings)
        self.ml_engine = MLEngine(settings)
        self.portfolio = Portfolio()

        # 초기 자본금 설정 (포트폴리오가 비어있으면)
        if self.portfolio.total_asset == 0:
            self.portfolio.initialize(10_000_000)  # 기본 1000만원

        self.executor = OrderExecutor(
            self.kis_auth, settings, self.portfolio, self.safety_guard,
        )

        # Data
        self.data_pipeline = DataPipeline(
            self.market_client, trading_params,
        )

        # Telegram
        self.telegram = TelegramBot(settings, system_ref=self)

        # Review
        self.reviewer = DailyReviewer(self.llm_engine, self.portfolio)
        self.strategy_evaluator = StrategyEvaluator(self.portfolio)

        # Simulation
        self.sim_tracker = SimulationTracker(self.portfolio)

        # Scheduler
        self.trading_scheduler = TradingScheduler(self, settings)

        # State
        self._paused = False
        self._running = False
        self._watchlist = self._get_watchlist()

        logger.info(f"System initialized | Mode: {self.config_manager.get_mode()} | Watchlist: {self._watchlist}")

    def _get_watchlist(self) -> list[str]:
        """전략에서 관심 종목 추출."""
        strategy = self.llm_engine.load_strategy()
        # 6자리 종목코드 패턴 추출
        tickers = re.findall(r'\b(\d{6})\b', strategy)
        if not tickers:
            # 기본 관심종목
            tickers = ["005930", "000660", "373220", "006400"]
        return list(set(tickers))

    def _notify(self, level: str = "", message: str = ""):
        """Telegram 알림 (동기 래퍼)."""
        self.telegram.send_alert_sync(level, message)

    # --- 스케줄러 콜백 ---

    def cycle_data_collection(self):
        """데이터 수집 사이클."""
        if self._paused or not self.circuit_breaker.is_trading_allowed:
            return

        held_tickers = list(self.portfolio.positions.keys())
        prices = self.data_pipeline.collect_prices_only(held_tickers + self._watchlist)

        # 포트폴리오 가격 업데이트
        self.portfolio.update_prices(prices)

        logger.debug(f"Data collected: {len(prices)} tickers")

    def cycle_llm_analysis(self):
        """LLM 분석 사이클 — 핵심 매매 판단 루프."""
        if self._paused:
            logger.info("System paused, skipping LLM analysis")
            return

        if not self.circuit_breaker.is_trading_allowed:
            logger.info(f"Circuit breaker active ({self.circuit_breaker.state}), skipping analysis")
            return

        try:
            # 1. 데이터 수집
            held_tickers = list(self.portfolio.positions.keys())
            data = self.data_pipeline.collect_all_for_analysis(
                watchlist=self._watchlist,
                held_tickers=held_tickers,
            )
            self.circuit_breaker.record_api_success()

            # 2. ML 예측
            ml_predictions = {}
            if self.ml_engine.is_ready:
                for ticker in self._watchlist + held_tickers:
                    df = self.data_pipeline.get_daily_df_with_indicators(ticker)
                    if not df.empty:
                        ml_predictions[ticker] = self.ml_engine.predict(df)

            # 3. 포트폴리오 현황
            portfolio_summary = self.portfolio.get_summary()

            # 목표 달성률 추가
            targets = self.config_manager.trading_params.get("portfolio_targets", {})
            monthly_target = targets.get("monthly_target_pct", 5.0) / 100
            current_pnl = self.portfolio.total_pnl_pct
            portfolio_summary["target_progress"] = {
                "monthly_target": f"{monthly_target:.1%}",
                "current_pnl": f"{current_pnl:.2%}",
                "progress": f"{(current_pnl / monthly_target * 100):.0f}%" if monthly_target > 0 else "N/A",
            }

            # 4. LLM 분석
            signal = self.llm_engine.analyze_market(
                portfolio=portfolio_summary,
                market_data=data.get("market_data", {}),
                ml_predictions=ml_predictions,
                news_summary=data.get("news_summary", ""),
                macro_data=data.get("macro_data", {}),
            )
            self.circuit_breaker.record_llm_success()

            # 5. Config 자동 조정
            config_adjustments = signal.get("config_adjustments", [])
            if config_adjustments:
                results = self.config_manager.apply_adjustments(config_adjustments)
                for r in results:
                    if r["applied"]:
                        self._notify(
                            level="strategy_suggestion",
                            message=f"파라미터 조정: {r['param']} → {r['value']} ({r['reason']})",
                        )

            # 6. Safety Guard 필터링
            filtered = self.safety_guard.filter_actions(
                signal, self.portfolio.get_summary(), self.portfolio.total_asset,
            )

            if filtered.get("safety_filtered"):
                for sf in filtered["safety_filtered"]:
                    logger.warning(f"Action filtered: {sf['rule']} — {sf['message']}")

            # 7. 주문 실행
            actions = filtered.get("actions", [])
            if actions:
                current_prices = self.data_pipeline.collect_prices_only(
                    [a["ticker"] for a in actions if a.get("ticker")]
                )
                results = self.executor.execute_signal(filtered, current_prices)

                for result in results:
                    if result.get("status") in ("SIMULATED", "SUBMITTED"):
                        # 시뮬레이션 추적
                        if self.config_manager.get_mode() == "simulation":
                            self.sim_tracker.record_trade(result)

                        # Telegram 알림
                        asyncio.get_event_loop().create_task(
                            self.telegram.send_trade_alert(result)
                        ) if self.telegram.enabled else None

            logger.info(
                f"Analysis cycle complete | "
                f"Risk: {signal.get('risk_assessment', '?')} | "
                f"Actions: {len(actions)} | "
                f"Outlook: {signal.get('market_outlook', '')[:50]}"
            )

        except Exception as e:
            logger.error(f"LLM analysis cycle failed: {e}")
            self.circuit_breaker.record_llm_failure()

    def cycle_news_check(self):
        """뉴스/공시 체크 사이클."""
        if self._paused:
            return
        # 뉴스 수집은 data_pipeline에서 처리됨
        # 긴급 공시 감지 시 LLM 임시 분석 트리거 가능
        logger.debug("News check cycle")

    def cycle_circuit_check(self):
        """서킷브레이커 정기 체크."""
        try:
            # 코스피 체크
            kospi = self.market_client.get_kospi_index()
            self.circuit_breaker.check_kospi(kospi.get("change_pct", 0) / 100)

            # 일일 손실 체크
            snapshots = self.portfolio.get_daily_snapshots(days=1)
            if snapshots:
                daily_pnl_pct = snapshots[0].get("daily_pnl_pct", 0)
                self.circuit_breaker.check_daily_loss(daily_pnl_pct)

            # 총자산 비상 체크
            self.circuit_breaker.check_emergency_loss(self.portfolio.total_pnl_pct)

            # 시스템 리소스 체크
            self.circuit_breaker.check_system_resources()

        except Exception as e:
            logger.warning(f"Circuit check failed: {e}")
            self.circuit_breaker.record_api_failure()

    def on_market_open(self):
        """장 시작 전 초기화."""
        logger.info("Market opening — initializing")
        self.circuit_breaker.daily_reset()
        self.config_manager.reload()
        self._watchlist = self._get_watchlist()

        if self.config_manager.get_mode() == "simulation":
            self.sim_tracker.start_session()

    def on_market_close(self):
        """장 마감 후 정리."""
        logger.info("Market closed — running cleanup")
        self.portfolio.save_daily_snapshot()

        if self.config_manager.get_mode() == "simulation":
            self.sim_tracker.save_report()

    # --- 공개 API ---

    def run_daily_review(self) -> dict:
        """일일 복기 실행."""
        return self.reviewer.run_daily_review()

    def run_weekly_review(self) -> dict:
        return self.reviewer.run_weekly_review()

    def train_ml_models(self) -> dict:
        """ML 모델 수동 학습."""
        logger.info("Starting ML model training...")
        data = {}
        for ticker in self._watchlist:
            df = self.data_pipeline.get_daily_df_with_indicators(ticker)
            if not df.empty:
                data[ticker] = df

        if not data:
            return {"status": "failed", "reason": "no_data"}

        return self.ml_engine.train_all(data)

    def predict_ticker(self, ticker: str) -> dict:
        """특정 종목 ML 예측."""
        df = self.data_pipeline.get_daily_df_with_indicators(ticker)
        if df.empty:
            return {"error": "no_data"}
        return self.ml_engine.predict(df)

    def collect_data(self):
        """수동 데이터 수집."""
        held = list(self.portfolio.positions.keys())
        self.data_pipeline.collect_all_for_analysis(self._watchlist, held)

    def handle_natural_language(self, text: str) -> dict:
        """자연어 전략/파라미터 변경 처리."""
        current_params = self.config_manager.trading_params
        result = self.llm_engine.interpret_natural_language(text, current_params)

        adjustments = result.get("adjustments", [])
        if adjustments:
            applied = self.config_manager.apply_adjustments(adjustments)
            result["adjustments"] = applied

        if result.get("strategy_change_needed") and result.get("strategy_suggestion"):
            # 전략 자체 변경은 사용자 승인 후 적용
            logger.info(f"Strategy change suggested: {result['strategy_suggestion']}")

        return result

    def pause(self):
        self._paused = True
        self.trading_scheduler.pause_trading_jobs()
        logger.info("Trading paused")

    def resume(self):
        self._paused = False
        self.trading_scheduler.resume_trading_jobs()
        logger.info("Trading resumed")

    def get_status(self) -> dict:
        return {
            "mode": self.config_manager.get_mode(),
            "paused": self._paused,
            "circuit_state": self.circuit_breaker.state.value,
            "total_asset": self.portfolio.total_asset,
            "cash": self.portfolio.cash,
            "total_pnl_pct": f"{self.portfolio.total_pnl_pct:.2%}",
            "num_positions": len(self.portfolio.positions),
            "watchlist": self._watchlist,
            "llm_daily_cost": self.llm_engine.get_daily_cost(),
            "circuit_breaker": self.circuit_breaker.get_status(),
        }

    async def run_async(self):
        """비동기 메인 루프."""
        self._running = True

        # Telegram 봇 시작
        await self.telegram.start()

        # 스케줄러 시작
        self.trading_scheduler.start()

        # 시작 알림
        mode = self.config_manager.get_mode()
        await self.telegram.send_alert(
            "signal_generated",
            f"🚀 AI Trader 시작\n모드: {mode}\n관심종목: {', '.join(self._watchlist)}",
        )

        logger.info("AI Trading System is running")

        # 메인 루프
        try:
            while self._running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutdown signal received")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """안전한 종료."""
        logger.info("Shutting down...")
        self._running = False
        self.trading_scheduler.stop()
        await self.telegram.stop()
        self.portfolio.save_daily_snapshot()
        logger.info("AI Trading System stopped")

    def run(self):
        """동기 메인 실행."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 시그널 핸들러
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.shutdown()))

        try:
            loop.run_until_complete(self.run_async())
        finally:
            loop.close()


def create_system() -> TradingSystem:
    """TradingSystem 인스턴스 생성 (CLI/테스트용)."""
    return TradingSystem()


if __name__ == "__main__":
    system = TradingSystem()
    system.run()
