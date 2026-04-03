"""AI Trader — 메인 엔트리포인트.

한국 주식 AI 자동매매 시스템.
LLM(Claude)을 전략 두뇌로, ML 모델을 예측 보조로 사용.
"""

import asyncio
import json
import re
import signal
import sys
import time
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
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}", filter=lambda record: record["level"].no < 40)
logger.add(sys.stderr, level="ERROR", format="<red>{time:HH:mm:ss}</red> | <level>{level: <7}</level> | {message}")
logger.add(LOG_DIR / "app_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days", level="DEBUG")
logger.add(LOG_DIR / "errors" / "error_{time:YYYY-MM-DD}.log", rotation="1 day", retention="60 days", level="ERROR")

from core.config_manager import ConfigManager
from core.market_data import KISAuth, MarketDataClient
from core.safety_guard import SafetyGuard
from core.circuit_breaker import CircuitBreaker, CircuitState
from core.llm_engine import LLMEngine
from core.ml_engine import MLEngine
from core.portfolio import Portfolio
from core.executor import OrderExecutor
from core.analysis_store import AnalysisStore, TICKER_NAMES, ticker_display
from core.notification import NotificationService
from core.risk_manager import RiskManager
from core.account_manager import AccountManager
from data.pipeline import DataPipeline
from interfaces.telegram_bot import TelegramBot
from review.daily_review import DailyReviewer
from review.strategy_evaluator import StrategyEvaluator
from simulation.simulator import SimulationTracker
from data.collectors.screener import StockScreener
from scheduler import TradingScheduler


class TradingSystem:
    """AI 트레이딩 시스템 — 전체 통합 관리."""

    def __init__(self):
        logger.info("Initializing AI Trading System...")
        self._main_loop = None  # 메인 이벤트 루프 참조 (스레드에서 async 호출용)

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
        self.portfolio = Portfolio(mode=self.config_manager.get_mode())

        self.executor = OrderExecutor(
            self.kis_auth, settings, self.portfolio, self.safety_guard,
            market_client=self.market_client,
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

        # Sub-modules
        self.analysis_store = AnalysisStore()
        self.notifier = NotificationService(
            self.telegram, self.portfolio, self.config_manager,
            event_loop_fn=lambda: self._main_loop,
        )
        self.account_manager = AccountManager(
            self.market_client, self.portfolio, self.config_manager, self.executor,
        )
        self.risk_manager = RiskManager(
            portfolio=self.portfolio,
            executor=self.executor,
            config_manager=self.config_manager,
            safety_guard=self.safety_guard,
            telegram=self.telegram,
            sim_tracker=self.sim_tracker,
            sync_account_fn=lambda: self.account_manager.sync_account_from_broker(),
        )

        # live 모드면 실계좌 동기화, 시뮬레이션이면 초기 자본금 설정
        if self.config_manager.get_mode() == "live":
            try:
                self.account_manager.sync_account_from_broker()
                logger.info(f"Live account synced on startup")
            except Exception as e:
                logger.error(f"Live account sync failed on startup: {e}")
                if self.portfolio.total_asset == 0:
                    self.portfolio.initialize(10_000_000)
        elif self.portfolio.total_asset == 0:
            self.portfolio.initialize(10_000_000)  # 기본 1000만원

        # Screener
        screening_config = self.config_manager.load_yaml("screening-params.yaml")
        self.screener = StockScreener(screening_config) if screening_config.get("screening", {}).get("enabled", True) else None

        # Scheduler
        self.trading_scheduler = TradingScheduler(self, settings)

        # State
        self._paused = False
        self._running = False
        self._last_report_notify: datetime | None = None  # 마지막 정기 리포트 알림 시각
        self._cancelled_order_nos: set[str] = set()  # 취소 완료/실패한 주문번호 (반복 방지)
        self._last_analysis: dict | None = self.analysis_store.load_last_analysis()  # 마지막 LLM 분석 결과
        # 스크리닝 캐시가 없거나 오래됐으면 시작 시 즉시 실행
        if self.screener:
            cached = self.screener.get_last_result()
            if not cached or not cached.candidates:
                logger.info("No valid screening cache — running initial screening...")
                try:
                    self.cycle_screening()
                except Exception as e:
                    logger.warning(f"Initial screening failed: {e}")

        self._watchlist = self._get_watchlist()
        self._screened_tickers: list[str] | None = None

        # 시작 시 히스토리 데이터 사전 수집
        self._prefetch_historical_data()

        # 초기 분석은 run_async에서 봇 시작 후 실행 (알림 전송을 위해)
        self._need_initial_analysis = False
        if not self._last_analysis:
            self._need_initial_analysis = True
        elif self._last_analysis.get("timestamp"):
            last_ts = datetime.fromisoformat(self._last_analysis["timestamp"])
            if (datetime.now() - last_ts).total_seconds() > 7200:  # 2시간 이상 경과
                self._need_initial_analysis = True

        watchlist_display = [ticker_display(t) for t in self._watchlist]
        logger.info(f"System initialized | Mode: {self.config_manager.get_mode()} | Watchlist: {watchlist_display}")

    def _get_watchlist(self) -> list[str]:
        """스크리닝 결과 + 보유종목 + config 포함종목을 합산."""
        screening_max = self.screener.config.get("max_watchlist_size", 8) if self.screener else 8
        max_size = screening_max
        held = list(self.portfolio.positions.keys())
        watchlist: list[str] = []

        # 1. 보유종목 최우선
        for t in held:
            if t not in watchlist:
                watchlist.append(t)

        # 2. 스크리너 결과
        if self.screener:
            result = self.screener.get_last_result()
            if result and result.candidates:
                for c in result.candidates:
                    ticker = c["ticker"]
                    if ticker not in watchlist:
                        watchlist.append(ticker)
                    # TICKER_NAMES 동적 확장
                    if ticker not in TICKER_NAMES and c.get("name"):
                        TICKER_NAMES[ticker] = c["name"]

        # 3. config include_tickers
        screening_cfg = self.screener.config if self.screener else {}
        for t in screening_cfg.get("include_tickers", []):
            if t not in watchlist:
                watchlist.append(t)

        # 4. 전략 파일에서 추출 (위에서 부족할 때)
        if len(watchlist) < max_size:
            strategy = self.llm_engine.load_strategy()
            for t in re.findall(r'\b(\d{6})\b', strategy):
                if t not in watchlist:
                    watchlist.append(t)
                if len(watchlist) >= max_size:
                    break

        # 5. 최소 폴백
        if not watchlist:
            watchlist = ["005930", "000660", "005380", "006400",
                         "034020", "051910", "064350", "000720"]

        return watchlist[:max_size]

    def _prefetch_historical_data(self):
        """시작 시 관심종목 + 보유종목의 히스토리 데이터가 부족하면 사전 수집."""
        held = list(self.portfolio.positions.keys())
        all_tickers = list(set(self._watchlist + held))
        min_bars = 120  # 지표 계산에 필요한 최소 일수

        for ticker in all_tickers:
            stored = self.data_pipeline.price_collector.get_stored_daily(ticker, min_bars)
            if len(stored) < 60:
                logger.info(f"Prefetching historical data for {ticker_display(ticker)} ({len(stored)} bars → {min_bars})")
                try:
                    self.data_pipeline.price_collector.collect_daily(ticker, days=min_bars)
                    time.sleep(0.5)  # API 부하 방지
                except Exception as e:
                    logger.warning(f"Prefetch failed for {ticker}: {e}")
            else:
                logger.debug(f"{ticker_display(ticker)}: {len(stored)} bars in DB, skipping prefetch")

    def _notify(self, level: str = "", message: str = ""):
        """Telegram 알림 (동기 래퍼)."""
        self.notifier.notify(level, message)

    # --- 스케줄러 콜백 ---

    def _build_screening_context(self) -> str:
        """스크리닝 결과를 LLM 컨텍스트 문자열로 변환."""
        if not self.screener:
            return ""
        result = self.screener.get_last_result()
        if not result or not result.candidates:
            return ""

        lines = ["## 오늘의 스크리닝 결과 (자동 선별된 관심종목)"]
        lines.append(f"분석 시간: {result.timestamp[:16]}")
        lines.append(f"전체 {result.screening_stats.get('total_analyzed', 0)}종목 중 "
                      f"{result.screening_stats.get('after_filter', 0)}종목 필터 통과 → "
                      f"상위 {len(result.candidates)}종목 선정\n")
        lines.append("| 순위 | 종목 | 종합점수 | 모멘텀 | 밸류 | 거래량 | 수급 | 기술적 | 등락률 |")
        lines.append("|------|------|---------|--------|------|--------|------|--------|--------|")
        for i, c in enumerate(result.candidates, 1):
            lines.append(
                f"| {i} | {c.get('name', '')}({c['ticker']}) "
                f"| {c.get('composite_score', 0):.0f} "
                f"| {c.get('momentum_score', 0):.0f} "
                f"| {c.get('value_score', 0):.0f} "
                f"| {c.get('volume_score', 0):.0f} "
                f"| {c.get('flow_score', 0):.0f} "
                f"| {c.get('technical_score', 0):.0f} "
                f"| {c.get('change_pct', 0):+.1f}% |"
            )
        lines.append("")
        lines.append("위 종목들은 멀티팩터 스코어링(모멘텀·밸류·거래량·수급·기술적 지표)으로 자동 선별되었습니다.")
        lines.append("이 종목들을 중심으로 매매 판단을 내려주세요.")

        if result.held_tickers_added:
            held_names = [ticker_display(t) for t in result.held_tickers_added]
            lines.append(f"\n보유 중인 종목(스크리닝 외 추가): {', '.join(held_names)}")

        return "\n".join(lines)

    def cycle_data_collection(self):
        """데이터 수집 사이클."""
        if self._paused:
            logger.info("Data collection skipped: paused")
            return
        if not self.circuit_breaker.is_trading_allowed:
            logger.info(f"Data collection skipped: circuit={self.circuit_breaker.state.value}")
            return

        held_tickers = list(self.portfolio.positions.keys())
        all_tickers = list(set(held_tickers + self._watchlist))
        prices = self.data_pipeline.collect_prices_only(all_tickers)

        # 분봉 데이터 수집 (보유종목 + 워치리스트, 상위 10종목)
        try:
            for ticker in all_tickers[:10]:
                self.data_pipeline.price_collector.collect_minute(ticker, interval="5")
        except Exception as e:
            logger.debug(f"Minute bar collection skipped: {e}")

        # 포트폴리오 가격 업데이트
        self.portfolio.update_prices(prices)

        # 리스크 기반 자동 청산 체크 (손절/트레일링스탑/보유기간/스크리닝탈락)
        self.risk_manager._watchlist = getattr(self, '_screened_tickers', None)
        risk_exits = self.risk_manager.check_risk_exits()
        for r in risk_exits:
            self.risk_manager.process_trade_result(r)
            if self.telegram.enabled:
                name = r.get("name", r.get("ticker", ""))
                ticker = r.get("ticker", "")
                reason = r.get("reason", "")
                pnl = r.get("pnl", 0)
                self.telegram.send_alert_sync(
                    "risk_exit",
                    f"⚠️ <b>자동청산</b> {name}({ticker})\n"
                    f"사유: {reason}\n"
                    f"손익: {pnl:+,.0f}원",
                )

        logger.info(f"Data collected: {len(prices)} tickers")

    def cycle_llm_analysis(self, suppress_notification: bool = False):
        """LLM 분석 사이클 — 핵심 매매 판단 루프.

        Args:
            suppress_notification: True이면 텔레그램 알림을 보내지 않음
                (텔레그램 커맨드에서 직접 호출 시 커맨드 응답으로 대체)
        """
        if self._paused:
            logger.info("System paused, skipping LLM analysis")
            return

        if not self.circuit_breaker.is_trading_allowed:
            logger.info(f"Circuit breaker EMERGENCY ({self.circuit_breaker.state}), skipping analysis")
            return

        if self.circuit_breaker.state != CircuitState.NORMAL:
            logger.info(f"Circuit breaker {self.circuit_breaker.state.value} — analysis with restricted trading")

        try:
            # 0. 라이브 모드: 분석 전 계좌 동기화 (체결 반영)
            if self.config_manager.get_mode() == "live":
                try:
                    self.account_manager.sync_account_from_broker()
                except Exception as e:
                    logger.warning(f"Pre-analysis account sync failed: {e}")

            # 1. 데이터 수집
            held_tickers = list(self.portfolio.positions.keys())
            data = self.data_pipeline.collect_all_for_analysis(
                watchlist=self._watchlist,
                held_tickers=held_tickers,
            )
            self.circuit_breaker.record_api_success()

            # 1-b. 시세 데이터 유효성 검증
            market_data = data.get("market_data", {})
            valid_prices = {k: v for k, v in market_data.items()
                          if isinstance(v, dict) and v.get("current", {}).get("price", 0) > 0}
            if not valid_prices:
                logger.critical("All market data collection failed — skipping LLM analysis")
                return

            # 1-c. 이전 예측 평가 (현재 가격으로 이전 분석의 정확도 측정)
            all_tickers = list(set(self._watchlist + held_tickers))
            eval_prices = self.data_pipeline.collect_prices_only(all_tickers)
            self.analysis_store.evaluate_previous_predictions(eval_prices, self._last_analysis)

            # 2. ML 예측
            ml_predictions = {}
            if self.ml_engine.is_ready:
                for ticker in self._watchlist + held_tickers:
                    df = self.data_pipeline.get_daily_df_with_indicators(ticker)
                    if not df.empty:
                        ml_predictions[ticker] = self.ml_engine.predict(df)

            # 3. 포트폴리오 현황 + 포지션 리스크 컨텍스트
            portfolio_summary = self.portfolio.get_summary()

            # 포지션별 리스크 컨텍스트 추가 (보유일수, 손절거리, 트레일링, 비중)
            t_params = self.config_manager.trading_params
            sl_pct = t_params.get("stop_loss_pct", -0.07)
            ts_pct = t_params.get("trailing_stop_pct", 0.06)
            max_hold = t_params.get("hold_period_days", {}).get("max", 20)
            total_asset = self.portfolio.total_asset or 1

            for ticker, pos_dict in portfolio_summary.get("positions", {}).items():
                pos = self.portfolio.positions.get(ticker)
                if not pos:
                    continue
                bought = datetime.fromisoformat(pos.bought_at)
                days_held = (datetime.now().date() - bought.date()).days
                weight = pos.market_value / total_asset
                dist_to_sl = pos.pnl_pct - sl_pct  # 양수 = 손절까지 여유
                trailing_dd = (pos.peak_price - pos.current_price) / pos.peak_price if pos.peak_price > 0 else 0

                pos_dict["days_held"] = days_held
                pos_dict["max_hold_days"] = max_hold
                pos_dict["portfolio_weight"] = f"{weight:.1%}"
                pos_dict["peak_price"] = pos.peak_price
                pos_dict["dist_to_stop_loss"] = f"{dist_to_sl:+.1%}"
                pos_dict["trailing_drawdown"] = f"{trailing_dd:.1%}"
                pos_dict["trailing_stop_pct"] = f"{ts_pct:.0%}"
                # 스크리닝 탈락 여부 (관심종목에서 빠졌으면 모멘텀 이탈)
                pos_dict["in_watchlist"] = ticker in self._watchlist
                if not pos_dict["in_watchlist"]:
                    pos_dict["screening_note"] = "⚠️ 스크리닝 탈락 — 모멘텀 이탈, 매도 검토 대상"

            # 최근 매매 이력 추가 (쿨다운/재매수 금지 판단용)
            recent_trades = self.portfolio.get_trade_history(limit=30)
            from datetime import timedelta
            three_days_ago = (datetime.now() - timedelta(days=3)).isoformat()
            recent_sells = [
                {
                    "ticker": t["ticker"],
                    "name": t.get("name", ""),
                    "action": t["action"],
                    "price": t["price"],
                    "quantity": t["quantity"],
                    "timestamp": t["timestamp"],
                    "pnl_pct": t.get("pnl_pct"),
                }
                for t in recent_trades
                if t["timestamp"] >= three_days_ago
            ]
            if recent_sells:
                portfolio_summary["recent_trades_3days"] = recent_sells

            # 월간 목표 달성률 추가 (월초 자산 대비 당월 수익률)
            targets = self.config_manager.trading_params.get("portfolio_targets", {})
            monthly_target = targets.get("monthly_target_pct", 5.0) / 100
            monthly_pnl_pct = self.portfolio.get_monthly_pnl_pct()
            portfolio_summary["target_progress"] = {
                "monthly_target": f"{monthly_target:.1%}",
                "current_monthly_pnl": f"{monthly_pnl_pct:.2%}",
                "progress": f"{(monthly_pnl_pct / monthly_target * 100):.0f}%" if monthly_target > 0 else "N/A",
            }

            # 포트폴리오 드로다운 상태 추가 (LLM에게 리스크 인식 제공)
            drawdown = self.portfolio.portfolio_drawdown
            if drawdown > 0.02:  # 2% 이상 낙폭만 표시
                portfolio_summary["drawdown_status"] = {
                    "hwm": self.portfolio.high_water_mark,
                    "drawdown": f"{drawdown:.1%}",
                    "warning": "고점 대비 낙폭 발생 — 신규 매수 시 신중하게 판단하세요." if drawdown >= 0.05 else "",
                }

            # 4. LLM 분석 (스크리닝 결과 + 예측 피드백 + 백테스트 피드백 주입)
            screening_context = self._build_screening_context()
            prediction_feedback = self.analysis_store.build_prediction_feedback()
            backtest_feedback = self.analysis_store.build_backtest_feedback(
                self._watchlist, list(self.portfolio.positions.keys()),
            )
            signal = self.llm_engine.analyze_market(
                portfolio=portfolio_summary,
                market_data=data.get("market_data", {}),
                ml_predictions=ml_predictions,
                news_summary=data.get("news_summary", ""),
                macro_data=data.get("macro_data", {}),
                screening_context=screening_context,
                prediction_feedback=prediction_feedback,
                backtest_feedback=backtest_feedback,
            )
            self.circuit_breaker.record_llm_success()

            # 5. Config 자동 조정
            config_adjustments = signal.get("config_adjustments", [])
            # LLM이 문자열 리스트로 반환하는 경우 필터링 (dict만 유효)
            config_adjustments = [a for a in config_adjustments if isinstance(a, dict)]
            if config_adjustments:
                results = self.config_manager.apply_adjustments(config_adjustments)
                for r in results:
                    if r["applied"]:
                        self._notify(
                            level="strategy_suggestion",
                            message=f"파라미터 조정: {r['param']} → {r['value']} ({r['reason']})",
                        )

            # 5-b. LLM 출력 ticker 검증 — 관심종목 + 보유종목에 없는 ticker 제거
            valid_tickers = set(self._watchlist + held_tickers)
            validated_actions = []
            for action in signal.get("actions", []):
                action_type = action.get("type", "").upper()
                ticker = action.get("ticker", "")
                if action_type == "HOLD" or ticker in valid_tickers:
                    validated_actions.append(action)
                else:
                    logger.warning(f"LLM hallucinated ticker rejected: {ticker} ({action.get('name', '')})")
            signal["actions"] = validated_actions

            # 6. Safety Guard 필터링
            filtered = self.safety_guard.filter_actions(
                signal, self.portfolio.get_summary(), self.portfolio.total_asset,
            )

            if filtered.get("safety_filtered"):
                for sf in filtered["safety_filtered"]:
                    logger.warning(f"Action filtered: {sf['rule']} — {sf['message']}")

            # 7. 주문 실행 (익절 매도는 텔레그램 확인 후 실행)
            actions = filtered.get("actions", [])
            if actions:
                current_prices = self.data_pipeline.collect_prices_only(
                    [a["ticker"] for a in actions if a.get("ticker")]
                )

                # 서킷브레이커 상태에 따라 BUY/SELL 필터링
                if not self.circuit_breaker.is_buy_allowed:
                    blocked = [a for a in actions if a.get("type", "").upper() == "BUY"]
                    if blocked:
                        logger.warning(
                            f"Circuit breaker [{self.circuit_breaker.state.value}] blocked "
                            f"{len(blocked)} BUY action(s)"
                        )
                    actions = [a for a in actions if a.get("type", "").upper() != "BUY"]

                if not self.circuit_breaker.is_sell_allowed:
                    blocked = [a for a in actions if a.get("type", "").upper() == "SELL"]
                    if blocked:
                        logger.warning(
                            f"Circuit breaker [{self.circuit_breaker.state.value}] blocked "
                            f"{len(blocked)} SELL action(s)"
                        )
                    actions = [a for a in actions if a.get("type", "").upper() != "SELL"]

                if not actions:
                    logger.info("All actions blocked by circuit breaker")

                # 대형 주문만 확인 대기, 나머지는 즉시 실행
                auto_actions = []
                confirm_actions = []
                for action in actions:
                    ticker = action.get("ticker", "")
                    if action.get("type", "").upper() == "SELL" and ticker in self.portfolio.positions:
                        pos = self.portfolio.positions[ticker]
                        cp = current_prices.get(ticker, pos.current_price)
                        sell_amount = cp * pos.quantity * action.get("ratio", 1.0)
                        # 대형 매도만 확인 (safety_guard 임계값 기준)
                        if self.safety_guard.needs_confirmation(sell_amount):
                            confirm_actions.append(action)
                            continue
                    auto_actions.append(action)

                # 즉시 실행 (매수, 소형 매도, 손절 매도 모두)
                # 분석 사이클 내 매매는 llm_analysis 메시지에 통합되므로 개별 알림 억제
                if auto_actions:
                    auto_signal = {**filtered, "actions": auto_actions}
                    results = self.executor.execute_signal(auto_signal, current_prices)
                    for result in results:
                        self.risk_manager.process_trade_result(result, suppress_notification=True)

                # 대형 매도만 확인 요청
                for action in confirm_actions:
                    ticker = action["ticker"]
                    pos = self.portfolio.positions.get(ticker)
                    if not pos:
                        logger.warning(f"Position {ticker} already closed, skipping confirm")
                        continue
                    cp = current_prices.get(ticker, pos.current_price)
                    pnl = (cp - pos.avg_price) * pos.quantity
                    pnl_pct = (cp - pos.avg_price) / pos.avg_price * 100

                    if self.telegram.enabled:
                        self.notifier.request_sell_confirmation(
                            action=action,
                            signal=filtered,
                            current_price=cp,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )

            # 8. 분석 결과 저장 + 텔레그램 전송
            self._last_analysis = {
                "signal": signal,
                "actions": actions,
                "timestamp": datetime.now().isoformat(),
                "portfolio": portfolio_summary,
            }
            self.analysis_store.save_last_analysis(self._last_analysis)
            self.analysis_store.append_analysis_log(
                signal=signal,
                actions=actions,
                portfolio=portfolio_summary,
                ml_predictions=ml_predictions,
                prices=eval_prices,
                mode=self.config_manager.get_mode(),
            )
            if self.telegram.enabled and not suppress_notification:
                has_trades = any(a.get("type", "").upper() in ("BUY", "SELL") for a in actions)
                # 장중(09:00~15:30): 2시간 주기, 장외: 08:00·16:30 각 1회만
                from datetime import time as dtime
                now_time = datetime.now().time()
                is_market_hours = dtime(9, 0) <= now_time <= dtime(15, 30)
                if is_market_hours:
                    due_for_report = (
                        self._last_report_notify is None
                        or (datetime.now() - self._last_report_notify).total_seconds() >= 7200
                    )
                else:
                    # 장전 08:00~09:00, 장후 16:00~17:00 구간에서만 1회
                    is_pre_market = dtime(8, 0) <= now_time < dtime(9, 0)
                    is_post_market = dtime(16, 0) <= now_time < dtime(17, 0)
                    due_for_report = (
                        (is_pre_market or is_post_market)
                        and (
                            self._last_report_notify is None
                            or (datetime.now() - self._last_report_notify).total_seconds() >= 3600
                        )
                    )
                if has_trades or due_for_report:
                    try:
                        analysis_msg = self.notifier.format_analysis_msg(signal, actions)
                        self.telegram.send_alert_sync("llm_analysis", analysis_msg)
                        self._last_report_notify = datetime.now()
                    except Exception as e:
                        logger.warning(f"Failed to send analysis to Telegram: {e}")

            # 9. LLM vs 규칙전략 알파 추적
            try:
                self.analysis_store.log_alpha_comparison(
                    actions, eval_prices,
                    self._watchlist, set(self.portfolio.positions.keys()),
                    self.data_pipeline, self.analysis_store.load_optimized_params,
                )
            except Exception as e:
                logger.debug(f"Alpha tracking failed (non-critical): {e}")

            logger.info(
                f"Analysis cycle complete | "
                f"Risk: {signal.get('risk_assessment', '?')} | "
                f"Actions: {len(actions)} | "
                f"Outlook: {signal.get('market_outlook', '')[:50]}"
            )

        except Exception as e:
            logger.exception(f"LLM analysis cycle failed: {e}")
            self.circuit_breaker.record_llm_failure()

    _NEWS_LOG_PATH = Path(__file__).parent / "data" / "news_history.jsonl"

    def cycle_news_check(self):
        """뉴스/공시 체크 사이클 — 주말/공휴일 포함 상시 수집."""
        if self._paused:
            return

        try:
            all_news = []
            # 관심종목 뉴스
            for ticker in self._watchlist[:5]:
                news = self.data_pipeline.news_collector.collect_stock_news(ticker, count=5)
                all_news.extend(news)
            # 보유종목 뉴스 (관심종목과 겹치지 않는 것만)
            held = [t for t in self.portfolio.positions.keys() if t not in self._watchlist[:5]]
            for ticker in held[:3]:
                news = self.data_pipeline.news_collector.collect_stock_news(ticker, count=5)
                all_news.extend(news)
            # 시장 뉴스
            market_news = self.data_pipeline.news_collector.collect_market_news(count=10)
            all_news.extend(market_news)

            if all_news:
                # JSONL에 누적 저장
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "count": len(all_news),
                    "news": all_news,
                }
                with open(self._NEWS_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(f"News check complete: {len(all_news)} items collected")

        except Exception as e:
            logger.warning(f"News check failed: {e}")

    def cycle_circuit_check(self):
        """서킷브레이커 정기 체크."""
        # 코스피 체크 — API 실패는 스킵 (서킷 체크 자체 실패로 서킷 발동하면 안 됨)
        try:
            kospi = self.market_client.get_kospi_index()
            self.circuit_breaker.check_kospi(kospi.get("change_pct", 0) / 100)
        except Exception as e:
            logger.debug(f"KOSPI check skipped (API unavailable): {e}")

        # 일일 손실 체크 — 전일(거래일) 종가 기준 vs 현재 자산
        snapshots = self.portfolio.get_daily_snapshots(days=7)  # 주말/공휴일 고려
        today_str = datetime.now().strftime("%Y-%m-%d")
        # 가장 최근 전일 스냅샷 찾기 (오늘 것 제외)
        prev_asset = None
        for snap in snapshots:
            if snap.get("date") != today_str:
                prev_asset = snap.get("total_asset", 0)
                break
        # 전일 스냅샷이 없으면 일일 손실 체크 스킵 (initial_capital과 비교하면 오탐)
        if prev_asset is None or prev_asset <= 0:
            prev_asset = self.portfolio.total_asset  # 손실 0%로 처리
        if prev_asset > 0:
            daily_pnl_pct = (self.portfolio.total_asset - prev_asset) / prev_asset
            self.circuit_breaker.check_daily_loss(daily_pnl_pct)

        # 총자산 비상 체크
        self.circuit_breaker.check_emergency_loss(self.portfolio.total_pnl_pct)

        # 시스템 리소스 체크
        self.circuit_breaker.check_system_resources()

    def cycle_unfilled_order_check(self):
        """미체결 주문 관리 사이클.

        - 매수: 현재가 > 주문가 + threshold% → 취소
        - 매도: 현재가 < 주문가 - threshold% → 취소
        - 15:20 이후: 모든 미체결 일괄 취소
        """
        if self._paused:
            return
        if self.config_manager.get_mode() == "simulation":
            return

        from datetime import time as dtime
        now = datetime.now()
        is_near_close = now.time() >= dtime(15, 20)

        try:
            orders = self.market_client.get_today_orders()
        except Exception as e:
            logger.warning(f"Unfilled order check — query failed: {e}")
            return

        unfilled = [o for o in orders if not o["filled"]]
        if not unfilled:
            return

        t_params = self.config_manager.trading_params
        threshold = t_params.get("unfilled_cancel_threshold_pct", 0.01)
        resubmit = t_params.get("unfilled_resubmit", False)

        cancelled = 0
        resubmitted = 0

        for order in unfilled:
            remaining = order["ord_qty"] - order["ccld_qty"]
            if remaining <= 0:
                continue

            order_no = order["order_no"]
            ticker = order["ticker"]
            ord_price = order["ord_price"]

            # 이미 취소 시도 실패한 주문은 스킵 (이번 사이클 내에서)
            if order_no in self._cancelled_order_nos:
                continue

            if is_near_close:
                result = self.executor.cancel_unfilled_order(order, current_price=0, resubmit=False)
                if result["action"] == "cancelled":
                    cancelled += 1
                    self._cancelled_order_nos.add(order_no)
                    logger.info(f"EOD cancel: {order['side']} {order['name']}({ticker}) "
                                f"#{order_no} {remaining}주 @{ord_price:,}")
                elif result["action"] == "failed":
                    self._cancelled_order_nos.add(order_no)
                continue

            try:
                price_data = self.market_client.get_current_price(ticker)
                current_price = price_data["price"]
            except Exception:
                continue

            if current_price <= 0:
                continue

            should_cancel = False
            if order["side"] == "매수" and current_price > ord_price * (1 + threshold):
                should_cancel = True
                logger.info(f"BUY drift: {order['name']}({ticker}) ord@{ord_price:,} → now@{current_price:,} "
                            f"(+{(current_price / ord_price - 1):.2%})")
            elif order["side"] == "매도" and current_price < ord_price * (1 - threshold):
                should_cancel = True
                logger.info(f"SELL drift: {order['name']}({ticker}) ord@{ord_price:,} → now@{current_price:,} "
                            f"({(current_price / ord_price - 1):.2%})")

            if should_cancel:
                result = self.executor.cancel_unfilled_order(order, current_price, resubmit=resubmit)
                if result["action"] == "cancelled":
                    cancelled += 1
                    self._cancelled_order_nos.add(order_no)
                elif result["action"] == "resubmitted":
                    cancelled += 1
                    resubmitted += 1
                    self._cancelled_order_nos.add(order_no)
                elif result["action"] == "failed":
                    self._cancelled_order_nos.add(order_no)

        if cancelled > 0:
            parts = [f"🔄 미체결 관리: {cancelled}건 취소"]
            if resubmitted > 0:
                parts.append(f", {resubmitted}건 재주문")
            if is_near_close:
                parts.append(" (장마감 전)")
            msg = "".join(parts)
            logger.info(msg)
            if self.telegram.enabled:
                self.telegram.send_alert_sync("unfilled_order", msg)

            try:
                import time as _time
                _time.sleep(1)
                self.account_manager.sync_account_from_broker()
            except Exception as e:
                logger.warning(f"Post-cancel sync failed: {e}")

    def cycle_screening(self, suppress_notification: bool = False):
        """종목 스크리닝 사이클.

        Args:
            suppress_notification: True이면 텔레그램 알림을 보내지 않음
                (텔레그램 커맨드에서 직접 호출 시 커맨드 응답으로 대체)
        """
        if not self.screener:
            logger.info("Screener disabled, skipping")
            return

        try:
            held = list(self.portfolio.positions.keys())
            result = self.screener.run_screening(held_tickers=held)

            # 관심종목 갱신
            self._watchlist = self._get_watchlist()
            # 순수 스크리닝 후보 (보유종목 강제 추가 전) — 스크리닝 탈락 판단용
            self._screened_tickers = [c["ticker"] for c in result.candidates if c["ticker"] not in (result.held_tickers_added or [])]

            # 새 종목 히스토리 프리페치
            self._prefetch_historical_data()

            # 텔레그램 알림 (수동 호출 시 억제)
            if result.candidates and not suppress_notification:
                msg = self.notifier.format_screening_msg(result)
                self.telegram.send_alert_sync("screening_result", msg)

            watchlist_display = [ticker_display(t) for t in self._watchlist]
            logger.info(f"Screening complete | Watchlist: {watchlist_display}")

        except Exception as e:
            logger.error(f"Screening failed: {e}")

    def cycle_backtest(self):
        """주간 자동 백테스트 + 최적화 — watchlist 종목 대상 × 3개 전략."""
        from simulation.backtest import Backtester
        from simulation.strategies import STRATEGY_REGISTRY
        from simulation.optimizer import train_test_split

        tickers = list(set(self._watchlist + list(self.portfolio.positions.keys())))
        if not tickers:
            logger.info("No tickers for backtest, skipping")
            return

        db_path = str(Path(__file__).parent / "data" / "storage" / "trader.db")
        end_date = datetime.now().strftime("%Y%m%d")
        from dateutil.relativedelta import relativedelta
        start_date = (datetime.now() - relativedelta(months=6)).strftime("%Y%m%d")

        param_grid = {
            "rsi_oversold": [35, 40, 45],
            "take_profit_pct": [0.07, 0.10, 0.15],
            "stop_loss_pct": [0.03, 0.05, 0.07],
            "position_size_pct": [0.08, 0.10],
        }

        results_summary = []
        for strategy_name, factory in STRATEGY_REGISTRY.items():
            try:
                bt = Backtester(initial_capital=10_000_000, db_path=db_path)

                # 1. 파라미터 최적화 (train/test split)
                logger.info(f"Optimizing {strategy_name}...")
                split_result = train_test_split(
                    bt, tickers, factory, param_grid, start_date, end_date,
                )

                if "error" not in split_result:
                    best_params = split_result["best_params"]
                    test_m = split_result["test_metrics"]
                    train_m = split_result["train_metrics"]
                    overfit = split_result["overfit_ratio"]

                    # 과적합 비율 검증: 0.3~5.0 범위 + test 수익률 양수일 때만 저장
                    test_return = test_m.get("total_return", 0)
                    if 0.3 <= overfit <= 5.0 and test_return > 0:
                        self.analysis_store.save_optimized_params(strategy_name, best_params, {
                            "train_return": train_m.get("total_return", 0),
                            "test_return": test_return,
                            "train_sharpe": train_m.get("sharpe_ratio", 0),
                            "test_sharpe": test_m.get("sharpe_ratio", 0),
                            "overfit_ratio": overfit,
                        })
                        logger.info(f"Optimized {strategy_name}: {best_params} (overfit={overfit:.2f}, test_return={test_return:.2%})")
                    else:
                        logger.warning(
                            f"Skipping optimization for {strategy_name}: "
                            f"overfit={overfit:.2f}, test_return={test_return:.2%}"
                        )

                # 2. 최적화된 파라미터로 전체 기간 백테스트
                opt_params = self.analysis_store.load_optimized_params(strategy_name)
                strategy = factory(opt_params)
                result = bt.run(tickers, strategy, start_date, end_date)
                if "error" not in result.metrics:
                    result.save_to_db(db_path)
                    results_summary.append({
                        "name": strategy_name,
                        "return": result.metrics.get("total_return", 0),
                        "win_rate": result.metrics.get("win_rate", 0),
                        "sharpe": result.metrics.get("sharpe_ratio", 0),
                        "mdd": result.metrics.get("max_drawdown", 0),
                        "trades": result.metrics.get("total_trades", 0),
                        "optimized": opt_params is not None,
                    })
            except Exception as e:
                logger.error(f"Backtest/optimize failed for {strategy_name}: {e}")

        if results_summary and self.telegram.enabled:
            msg = NotificationService.format_backtest_summary(results_summary, start_date, end_date)
            self.telegram.send_alert_sync("backtest_result", msg)

        logger.info(f"Weekly backtest+optimize complete | {len(results_summary)} strategies")

    def sync_account_from_broker(self) -> dict | None:
        """KIS API에서 실제 계좌 잔고/보유종목을 가져와 포트폴리오 동기화."""
        return self.account_manager.sync_account_from_broker()

    def switch_mode(self, mode: str) -> str:
        """모드 전환 + 필요 시 계좌 동기화."""
        was_paused = self._paused
        self._paused = True
        msg = self.account_manager.switch_mode(mode)
        if not was_paused:
            self._paused = False
        return msg

    def on_market_open(self):
        """장 시작 전 초기화."""
        logger.info("Market opening — initializing")
        self.circuit_breaker.daily_reset()
        self._cancelled_order_nos.clear()
        self.config_manager.reload()
        self._watchlist = self._get_watchlist()

        if self.config_manager.get_mode() == "live":
            try:
                self.account_manager.sync_account_from_broker()
            except Exception as e:
                logger.error(f"Market open account sync failed: {e}")
        else:
            self.sim_tracker.start_session()

    def on_market_close(self):
        """장 마감 후 정리 + 일일 리포트 발송."""
        logger.info("Market closed — running cleanup")
        self.portfolio.save_daily_snapshot()

        if self.config_manager.get_mode() == "simulation":
            self.sim_tracker.save_report()

        try:
            self.notifier.send_daily_report()
        except Exception as e:
            logger.error(f"Daily report failed: {e}")

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

    def run_backtest(
        self,
        mode: str = "single",
        strategy_name: str = "swing",
        start_date: str = "",
        end_date: str = "",
    ) -> list[str]:
        """백테스트/비교/최적화 실행 → 텔레그램 메시지 리스트 반환.

        mode: "single" | "compare" | "optimize"
        """
        from simulation.backtest import Backtester
        from simulation.strategies import STRATEGY_REGISTRY
        from simulation.report import format_telegram_report, format_optimization_report

        tickers = list(self._watchlist)
        if not tickers:
            return ["관심종목이 없어 백테스트를 실행할 수 없습니다."]

        if not end_date:
            from datetime import datetime as dt
            end_date = dt.now().strftime("%Y%m%d")
        if not start_date:
            start_date = "20240301"

        capital = self.portfolio.initial_capital or 10_000_000
        bt = Backtester(initial_capital=capital)

        # 데이터 부족 종목 자동 수집 (최소 400일)
        min_bars = 400
        for ticker in tickers:
            stored = self.data_pipeline.price_collector.get_stored_daily(ticker, min_bars)
            if len(stored) < min_bars:
                logger.info(f"Backtest: collecting historical data for {ticker_display(ticker)} ({len(stored)}/{min_bars})")
                try:
                    self.data_pipeline.price_collector.collect_historical(ticker, start_date, end_date)
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Backtest data collection failed for {ticker}: {e}")

        messages = []

        if mode == "single":
            factory = STRATEGY_REGISTRY.get(strategy_name)
            if not factory:
                return [f"지원 전략: {list(STRATEGY_REGISTRY.keys())}"]
            opt_params = self.analysis_store.load_optimized_params(strategy_name)
            strategy = factory(opt_params)
            if opt_params:
                messages.append(f"⚙️ 최적화된 파라미터 적용: {opt_params}")
            result = bt.run(tickers, strategy, start_date, end_date)
            if "error" not in result.metrics:
                result.save_to_db(bt.db_path)
            messages.append(format_telegram_report(result))

        elif mode == "compare":
            for name, factory in STRATEGY_REGISTRY.items():
                opt_params = self.analysis_store.load_optimized_params(name)
                strategy = factory(opt_params)
                result = bt.run(tickers, strategy, start_date, end_date)
                if "error" not in result.metrics:
                    result.save_to_db(bt.db_path)
                messages.append(format_telegram_report(result))

        elif mode == "optimize":
            from simulation.optimizer import train_test_split

            factory = STRATEGY_REGISTRY.get(strategy_name)
            if not factory:
                return [f"지원 전략: {list(STRATEGY_REGISTRY.keys())}"]

            param_grid = {
                "rsi_oversold": [35, 40, 45],
                "take_profit_pct": [0.07, 0.10, 0.15],
                "stop_loss_pct": [0.03, 0.05, 0.07],
                "position_size_pct": [0.08, 0.10],
            }

            split_result = train_test_split(
                bt, tickers, factory, param_grid, start_date, end_date,
            )

            if "error" in split_result:
                return [f"최적화 실패: {split_result['error']}"]

            bp = split_result["best_params"]
            train_m = split_result["train_metrics"]
            test_m = split_result["test_metrics"]
            overfit = split_result["overfit_ratio"]

            msg = (
                f"🔧 <b>파라미터 최적화 결과: {strategy_name}</b>\n"
                f"━━━━━━━━━━━━━━\n\n"
                f"<b>최적 파라미터:</b>\n"
            )
            for k, v in bp.items():
                if isinstance(v, float):
                    msg += f"  {k}: {v:.2%}\n" if v < 1 else f"  {k}: {v}\n"
                else:
                    msg += f"  {k}: {v}\n"

            msg += (
                f"\n<b>Train 기간</b> ({split_result['train_period']})\n"
                f"  수익률: {train_m.get('total_return', 0):+.2%}\n"
                f"  샤프: {train_m.get('sharpe_ratio', 0):.2f}\n"
                f"  MDD: -{train_m.get('max_drawdown', 0):.2%}\n"
                f"\n<b>Test 기간</b> ({split_result['test_period']})\n"
                f"  수익률: {test_m.get('total_return', 0):+.2%}\n"
                f"  샤프: {test_m.get('sharpe_ratio', 0):.2f}\n"
                f"  MDD: -{test_m.get('max_drawdown', 0):.2%}\n"
                f"\n과적합 비율: {overfit:.2f} (1.0에 가까울수록 좋음)"
            )
            messages.append(msg)

            # 최적 파라미터로 전체 기간 백테스트
            best_strategy = factory(bp)
            full_result = bt.run(tickers, best_strategy, start_date, end_date)
            if "error" not in full_result.metrics:
                full_result.save_to_db(bt.db_path)
            messages.append("📊 <b>최적 파라미터 전체 기간 결과</b>\n" + format_telegram_report(full_result))

            # 최적 파라미터 저장 — overfit 가드 적용
            test_return = test_m.get("total_return", 0)
            if 0.3 <= overfit <= 5.0 and test_return > 0:
                self.analysis_store.save_optimized_params(strategy_name, bp, {
                    "train_return": train_m.get("total_return", 0),
                    "test_return": test_return,
                    "train_sharpe": train_m.get("sharpe_ratio", 0),
                    "test_sharpe": test_m.get("sharpe_ratio", 0),
                    "overfit_ratio": overfit,
                })
                messages.append(
                    f"✅ 최적 파라미터가 저장되었습니다.\n"
                    f"다음 /backtest 실행 시 최적화된 파라미터로 실행됩니다."
                )
            else:
                messages.append(
                    f"⚠️ 과적합 비율({overfit:.2f}) 또는 test 수익률({test_return:+.2%})이 "
                    f"기준 미달로 파라미터를 저장하지 않았습니다."
                )

        return messages or ["결과 없음"]

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

    def get_last_analysis(self) -> dict | None:
        """마지막 LLM 분석 결과 반환."""
        return self._last_analysis

    def get_status(self) -> dict:
        return {
            "mode": self.config_manager.get_mode(),
            "paused": self._paused,
            "circuit_state": self.circuit_breaker.state.value,
            "total_asset": self.portfolio.total_asset,
            "cash": self.portfolio.cash,
            "invested": self.portfolio.total_invested,
            "initial_capital": self.portfolio.initial_capital,
            "total_pnl": self.portfolio.total_pnl,
            "total_pnl_pct": f"{self.portfolio.total_pnl_pct:.2%}",
            "num_positions": len(self.portfolio.positions),
            "positions": {t: p.to_dict() for t, p in self.portfolio.positions.items()},
            "watchlist": [ticker_display(t) for t in self._watchlist],
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
        watchlist_str = "\n".join(f"  • {ticker_display(t)}" for t in self._watchlist)
        await self.telegram.send_alert(
            "signal_generated",
            f"🚀 AI Trader 시작\n모드: {mode}\n\n관심종목:\n{watchlist_str}",
        )

        logger.info("AI Trading System is running")

        # 봇 시작 후 초기 분석 실행 (알림 전송 가능 상태)
        if self._need_initial_analysis:
            logger.info("Running initial LLM analysis (no recent result)...")
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.cycle_llm_analysis)
            except Exception as e:
                logger.warning(f"Initial LLM analysis failed: {e}")

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

    def _get_event_loop(self):
        """메인 이벤트 루프 반환 (스레드에서 async 호출용)."""
        return self._main_loop

    def run(self):
        """동기 메인 실행."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._main_loop = loop  # 스레드에서 참조할 수 있도록 저장

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
