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
from core.portfolio import Portfolio, Position
from core.executor import OrderExecutor
from data.pipeline import DataPipeline
from interfaces.telegram_bot import TelegramBot
from review.daily_review import DailyReviewer
from review.strategy_evaluator import StrategyEvaluator
from simulation.simulator import SimulationTracker
from data.collectors.screener import StockScreener
from scheduler import TradingScheduler


# 종목코드 → 회사명 매핑
TICKER_NAMES: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "373220": "LG에너지솔루션",
    "006400": "삼성SDI",
    "035420": "NAVER",
    "035720": "카카오",
    "051910": "LG화학",
    "005490": "POSCO홀딩스",
    "105560": "KB금융",
    "055550": "신한지주",
    "003670": "포스코퓨처엠",
    "247540": "에코프로비엠",
    "068270": "셀트리온",
    "207940": "삼성바이오로직스",
    "000270": "기아",
    "005380": "현대차",
    "012330": "현대모비스",
    "066570": "LG전자",
    "028260": "삼성물산",
    "003550": "LG",
}


def ticker_display(ticker: str) -> str:
    """종목코드를 '회사명(코드)' 형태로 변환."""
    name = TICKER_NAMES.get(ticker, "")
    return f"{name}({ticker})" if name else ticker


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

        # live 모드면 실계좌 동기화, 시뮬레이션이면 초기 자본금 설정
        if self.config_manager.get_mode() == "live":
            try:
                account = self.market_client.get_account_balance()
                # API 실패로 빈 결과가 돌아온 경우 동기화 스킵
                if account["total_asset"] == 0 and not account["positions"]:
                    raise RuntimeError("Empty account data returned (API may be unavailable)")
                self.portfolio.cash = account["cash"]
                # initial_capital은 최초 1회만 자동 설정 — 이후 변경은 /setcapital로
                if self.portfolio.initial_capital == 0 and account["total_asset"] > 0:
                    self.portfolio.initial_capital = account["total_asset"]
                    logger.info(f"Initial capital set: {account['total_asset']:,.0f}")
                self.portfolio.positions.clear()
                for pos_data in account["positions"]:
                    self.portfolio.positions[pos_data["ticker"]] = Position(
                        ticker=pos_data["ticker"],
                        name=pos_data["name"],
                        quantity=pos_data["quantity"],
                        avg_price=pos_data["avg_price"],
                        current_price=pos_data["current_price"],
                    )
                    if pos_data["ticker"] not in TICKER_NAMES and pos_data["name"]:
                        TICKER_NAMES[pos_data["ticker"]] = pos_data["name"]
                self.portfolio._save_state()
                logger.info(f"Live account synced: cash={account['cash']:,}, positions={len(account['positions'])}")
            except Exception as e:
                logger.error(f"Live account sync failed on startup: {e}")
                if self.portfolio.total_asset == 0:
                    self.portfolio.initialize(10_000_000)
        elif self.portfolio.total_asset == 0:
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

        # Screener
        screening_config = self.config_manager.load_yaml("screening-params.yaml")
        self.screener = StockScreener(screening_config) if screening_config.get("screening", {}).get("enabled", True) else None

        # Scheduler
        self.trading_scheduler = TradingScheduler(self, settings)

        # State
        self._paused = False
        self._running = False
        self._last_analysis: dict | None = self._load_last_analysis()  # 마지막 LLM 분석 결과
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

        # 시작 시 히스토리 데이터 사전 수집
        self._prefetch_historical_data()

        # 시작 시 분석 결과가 없거나 오래됐으면 자동 실행
        run_initial_analysis = False
        if not self._last_analysis:
            run_initial_analysis = True
        elif self._last_analysis.get("timestamp"):
            last_ts = datetime.fromisoformat(self._last_analysis["timestamp"])
            if (datetime.now() - last_ts).total_seconds() > 7200:  # 2시간 이상 경과
                run_initial_analysis = True
        if run_initial_analysis:
            logger.info("Running initial LLM analysis (no recent result)...")
            try:
                self.cycle_llm_analysis()
            except Exception as e:
                logger.warning(f"Initial LLM analysis failed: {e}")

        watchlist_display = [ticker_display(t) for t in self._watchlist]
        logger.info(f"System initialized | Mode: {self.config_manager.get_mode()} | Watchlist: {watchlist_display}")

    def _get_watchlist(self) -> list[str]:
        """스크리닝 결과 + 보유종목 + config 포함종목을 합산."""
        max_size = 8
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
        self.telegram.send_alert_sync(level, message)

    def _format_analysis_msg(self, signal: dict, actions: list) -> str:
        """LLM 분석 결과를 텔레그램 메시지로 포맷."""
        now = datetime.now().strftime("%H:%M")
        mode = self.config_manager.get_mode()
        mode_tag = "[SIM]" if mode == "simulation" else "[LIVE]"

        risk = signal.get("risk_assessment", "?")
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")
        outlook = signal.get("market_outlook", "없음")
        reasoning = signal.get("reasoning", "없음")

        msg = (
            f"📊 LLM 분석 결과 {mode_tag} [{now}]\n"
            f"━━━━━━━━━━━━━━\n"
            f"{risk_emoji} 리스크: {risk}\n"
            f"🔍 시장 전망: {outlook}\n"
        )

        # 종목별 판단
        all_actions = signal.get("actions", [])
        if all_actions:
            msg += "\n📋 종목 판단:\n"
            for a in all_actions:
                action_type = a.get("type", "HOLD")
                ticker = a.get("ticker", "")
                name = a.get("name", ticker_display(ticker) if ticker else "")
                reason = a.get("reason", "")
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action_type, "⚪")
                ratio_str = ""
                if a.get("ratio"):
                    ratio_str = f" ({a['ratio']:.0%})"
                msg += f"  {emoji} {action_type} {name}{ratio_str}"
                if reason:
                    msg += f"\n     └ {reason}"
                msg += "\n"
        else:
            msg += "\n📋 판단: 전종목 HOLD\n"

        # 실행된 매매 수 (HOLD 제외)
        executed = sum(1 for a in actions if a.get("type", "").upper() in ("BUY", "SELL"))
        if executed > 0:
            msg += f"\n⚡ 매매 실행: {executed}건"

        # 판단 근거
        if reasoning and reasoning != "없음":
            # 너무 길면 자르기
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            msg += f"\n\n💬 근거:\n{reasoning}"

        return msg

    @staticmethod
    def _format_trade_msg(trade: dict) -> str:
        """매매 결과를 텔레그램 메시지로 포맷."""
        action = trade.get("action", "")
        emoji = "🟢" if action == "BUY" else "🔴"
        name = trade.get("name", trade.get("ticker", ""))
        ticker = trade.get("ticker", "")
        msg = (
            f"{emoji} {action} {name}({ticker})\n"
            f"수량: {trade.get('quantity', 0):,}주 @ {trade.get('price', 0):,.0f}원\n"
            f"금액: {trade.get('amount', 0):,.0f}원"
        )
        if trade.get("pnl") is not None:
            pnl = trade["pnl"]
            pnl_emoji = "📈" if pnl > 0 else "📉"
            msg += f"\n손익: {pnl_emoji} {pnl:,.0f}원 ({trade.get('pnl_pct', '')})"
        msg += f"\n상태: {trade.get('status', '')}"
        if trade.get("reason"):
            msg += f"\n사유: {trade['reason']}"
        return msg

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
        prices = self.data_pipeline.collect_prices_only(list(set(held_tickers + self._watchlist)))

        # 포트폴리오 가격 업데이트
        self.portfolio.update_prices(prices)

        # 리스크 기반 자동 청산 체크 (손절/트레일링스탑/보유기간)
        risk_exits = self.check_risk_exits()
        for r in risk_exits:
            self._process_trade_result(r)
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

    def cycle_llm_analysis(self):
        """LLM 분석 사이클 — 핵심 매매 판단 루프."""
        if self._paused:
            logger.info("System paused, skipping LLM analysis")
            return

        if not self.circuit_breaker.is_trading_allowed:
            logger.info(f"Circuit breaker EMERGENCY ({self.circuit_breaker.state}), skipping analysis")
            return

        if self.circuit_breaker.state != CircuitState.NORMAL:
            logger.info(f"Circuit breaker {self.circuit_breaker.state.value} — analysis with restricted trading")

        try:
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
            self._evaluate_previous_predictions(eval_prices)

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
                days_held = (datetime.now() - bought).days
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

            # 목표 달성률 추가
            targets = self.config_manager.trading_params.get("portfolio_targets", {})
            monthly_target = targets.get("monthly_target_pct", 5.0) / 100
            current_pnl = self.portfolio.total_pnl_pct
            portfolio_summary["target_progress"] = {
                "monthly_target": f"{monthly_target:.1%}",
                "current_pnl": f"{current_pnl:.2%}",
                "progress": f"{(current_pnl / monthly_target * 100):.0f}%" if monthly_target > 0 else "N/A",
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
            prediction_feedback = self._build_prediction_feedback()
            backtest_feedback = self._build_backtest_feedback()
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
                if auto_actions:
                    auto_signal = {**filtered, "actions": auto_actions}
                    results = self.executor.execute_signal(auto_signal, current_prices)
                    for result in results:
                        self._process_trade_result(result)

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
                        self._request_sell_confirmation(
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
            self._save_last_analysis()
            self._append_analysis_log(
                signal=signal,
                actions=actions,
                portfolio=portfolio_summary,
                ml_predictions=ml_predictions,
                prices=eval_prices,
            )
            if self.telegram.enabled:
                try:
                    analysis_msg = self._format_analysis_msg(signal, actions)
                    self.telegram.send_alert_sync("llm_analysis", analysis_msg)
                except Exception as e:
                    logger.warning(f"Failed to send analysis to Telegram: {e}")

            # 9. LLM vs 규칙전략 알파 추적
            try:
                self._log_alpha_comparison(actions, eval_prices)
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

    def check_risk_exits(self) -> list[dict]:
        """리스크 기반 자동 청산 — 손절/트레일링스탑/보유기간 초과.

        cycle_data_collection()에서 가격 업데이트 직후 호출.
        해당 조건 발생 시 텔레그램 확인 없이 즉시 매도.
        """
        params = self.config_manager.trading_params
        stop_loss_pct = params.get("stop_loss_pct", -0.07)
        trailing_stop_pct = params.get("trailing_stop_pct", 0.06)
        max_hold_days = params.get("hold_period_days", {}).get("max", 20)

        results = []
        for ticker, pos in list(self.portfolio.positions.items()):
            reason = None

            # 1. 하드 손절
            if pos.pnl_pct <= stop_loss_pct:
                reason = f"하드손절 ({pos.pnl_pct:.1%} ≤ {stop_loss_pct:.0%})"

            # 2. 트레일링 스탑 (수익 구간에서만 적용)
            if reason is None and pos.peak_price > pos.avg_price:
                drawdown = (pos.peak_price - pos.current_price) / pos.peak_price
                if drawdown >= trailing_stop_pct:
                    reason = (
                        f"트레일링스탑 (고점{pos.peak_price:,.0f}"
                        f"→현재{pos.current_price:,.0f}, -{drawdown:.1%})"
                    )

            # 3. 보유기간 초과
            if reason is None:
                bought = datetime.fromisoformat(pos.bought_at)
                days_held = (datetime.now() - bought).days
                if days_held >= max_hold_days:
                    reason = f"보유기간초과 ({days_held}일 ≥ {max_hold_days}일)"

            if reason:
                logger.warning(f"Risk exit triggered: {pos.name}({ticker}) — {reason}")
                result = self._execute_risk_exit(ticker, pos, reason)
                if result:
                    results.append(result)

        # 4. 집중도 경고 (자동 매도는 아니지만 텔레그램 알림)
        # 2종목 이하면 비중 초과는 불가피하므로 경고 스킵
        num_positions = len(self.portfolio.positions)
        if num_positions >= 3:
            max_weight = self.safety_guard.rules.get("max_position_ratio", 0.15) + 0.05
            total_asset = self.portfolio.total_asset or 1
            if not hasattr(self, "_concentration_alerted"):
                self._concentration_alerted = {}
            today_str = datetime.now().strftime("%Y-%m-%d")
            for ticker, pos in self.portfolio.positions.items():
                weight = pos.market_value / total_asset
                if weight > max_weight:
                    # 하루 1회만 알림 (종목별 쿨다운)
                    if self._concentration_alerted.get(ticker) == today_str:
                        continue
                    self._concentration_alerted[ticker] = today_str
                    logger.warning(
                        f"Concentration alert: {pos.name}({ticker}) "
                        f"weight {weight:.1%} > {max_weight:.1%}"
                    )
                    if self.telegram.enabled:
                        self.telegram.send_alert_sync(
                            "concentration_alert",
                            f"⚠️ <b>집중도 경고</b> {pos.name}({ticker})\n"
                            f"포트폴리오 비중: {weight:.1%} (한도 {max_weight:.1%})\n"
                            f"일부 매도를 검토하세요.",
                        )

        # 5. 포트폴리오 낙폭 경고 (HWM 대비 드로다운)
        drawdown = self.portfolio.portfolio_drawdown
        dd_alert_pct = params.get("portfolio_drawdown_alert_pct", 0.05)  # 기본 5%
        if drawdown >= dd_alert_pct:
            logger.warning(
                f"Portfolio drawdown alert: {drawdown:.1%} from HWM "
                f"{self.portfolio.high_water_mark:,.0f}"
            )
            if self.telegram.enabled and not getattr(self, "_dd_alerted", False):
                self.telegram.send_alert_sync(
                    "drawdown_alert",
                    f"⚠️ <b>포트폴리오 낙폭 경고</b>\n"
                    f"고점(HWM): {self.portfolio.high_water_mark:,.0f}원\n"
                    f"현재 총자산: {self.portfolio.total_asset:,.0f}원\n"
                    f"낙폭: {drawdown:.1%}\n"
                    f"신규 매수를 자제하고 리스크를 점검하세요.",
                )
                self._dd_alerted = True  # 중복 알림 방지
        else:
            self._dd_alerted = False  # 드로다운 회복 시 알림 리셋

        return results

    def _execute_risk_exit(self, ticker: str, pos, reason: str) -> dict | None:
        """리스크 청산 즉시 실행 — 쿨다운/확인 우회."""
        try:
            # PnL을 매도 전에 계산 (live 매도 결과에는 pnl이 없으므로)
            pre_pnl = pos.pnl
            pre_pnl_pct = pos.pnl_pct
            result = self.executor._execute_sell(
                ticker=ticker,
                name=pos.name,
                ratio=1.0,
                urgency="high",
                limit_price=None,
                current_price=pos.current_price,
                reason=f"[AUTO] {reason}",
                signal_json="",
            )
            if result:
                result["reason"] = reason
                if "pnl" not in result:
                    result["pnl"] = pre_pnl
                    result["pnl_pct"] = f"{pre_pnl_pct:.2%}"
            return result
        except Exception as e:
            logger.error(f"Risk exit execution failed for {ticker}: {e}")
            return None

    def _process_trade_result(self, result: dict) -> None:
        """매매 결과 처리 — 알림 + 시뮬레이션 추적 + live 계좌 동기화."""
        status = result.get("status", "")

        if status == "SIMULATED":
            self.sim_tracker.record_trade(result)
            if self.telegram.enabled:
                self.telegram.send_alert_sync(
                    "trade_executed",
                    f"[SIM] {self._format_trade_msg(result)}",
                )

        elif status == "SUBMITTED":
            # Live 주문 접수됨 — trade_history에 기록 + 계좌 동기화
            self.portfolio._record_trade(result, result.get("signal_json", ""))
            if self.telegram.enabled:
                mode_tag = "[LIVE]"
                order_no = result.get("order_no", "")
                self.telegram.send_alert_sync(
                    "trade_executed",
                    f"{mode_tag} {self._format_trade_msg(result)}\n주문번호: {order_no}",
                )
            # 주문 후 계좌 동기화 (체결 반영) — 최대 3회 재시도
            synced = False
            for attempt in range(3):
                try:
                    if attempt > 0:
                        time.sleep(2 * attempt)  # 2초, 4초 대기
                    self.sync_account_from_broker()
                    synced = True
                    break
                except Exception as e:
                    logger.warning(f"Post-trade sync attempt {attempt + 1}/3 failed: {e}")
            if not synced:
                logger.error("Post-trade account sync failed after 3 attempts")
                if self.telegram.enabled:
                    self.telegram.send_alert_sync(
                        "error",
                        "⚠️ 주문 후 계좌 동기화 실패 (3회 재시도)\n"
                        "포트폴리오가 실제 계좌와 다를 수 있습니다.\n"
                        "/status 로 확인해주세요.",
                    )

        elif status == "PENDING_CONFIRMATION":
            # 대규모 주문 확인 필요 — 텔레그램으로 확인 요청
            if self.telegram.enabled:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(
                            self.telegram.request_confirmation(result)
                        )
                    else:
                        loop.run_until_complete(
                            self.telegram.request_confirmation(result)
                        )
                except Exception as e:
                    logger.warning(f"Failed to send confirmation request: {e}")

        elif status == "FAILED":
            if self.telegram.enabled:
                error = result.get("error", "unknown")
                self.telegram.send_alert_sync(
                    "error",
                    f"주문 실패: {result.get('name', result.get('ticker', ''))} "
                    f"{result.get('action', '')} — {error}",
                )

    def _request_sell_confirmation(
        self, action: dict, signal: dict, current_price: float,
        pnl: float, pnl_pct: float,
    ) -> None:
        """익절 매도 확인 요청을 텔레그램으로 전송."""
        ticker = action["ticker"]
        name = action.get("name", ticker)
        pos = self.portfolio.positions.get(ticker)
        if not pos:
            return

        order_info = {
            "ticker": ticker,
            "name": name,
            "action": "SELL",
            "quantity": int(pos.quantity * action.get("ratio", 1.0)) or pos.quantity,
            "price": current_price,
            "amount": int((int(pos.quantity * action.get("ratio", 1.0)) or pos.quantity) * current_price),
            "reason": action.get("reason", ""),
            "pnl": f"{pnl:+,.0f}원 ({pnl_pct:+.1f}%)",
            "signal": signal,
            "action_data": action,
        }

        # 텔레그램 확인 요청 (스레드에서도 동작하도록 run_coroutine_threadsafe 사용)
        try:
            loop = self._get_event_loop()
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.telegram.request_sell_confirmation(order_info),
                    loop,
                )
            else:
                asyncio.run(self.telegram.request_sell_confirmation(order_info))
            logger.info(f"Sell confirmation requested: {name}({ticker}) PnL: {pnl:+,.0f}원")
        except Exception as e:
            logger.warning(f"Failed to request sell confirmation: {e}")
            # 확인 실패 시 자동 실행하지 않음 (안전)

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

    def cycle_screening(self):
        """종목 스크리닝 사이클."""
        if not self.screener:
            logger.info("Screener disabled, skipping")
            return

        try:
            held = list(self.portfolio.positions.keys())
            result = self.screener.run_screening(held_tickers=held)

            # 관심종목 갱신
            self._watchlist = self._get_watchlist()

            # 새 종목 히스토리 프리페치
            self._prefetch_historical_data()

            # 텔레그램 알림
            if result.candidates:
                msg = self._format_screening_msg(result)
                self.telegram.send_alert_sync("screening_result", msg)

            watchlist_display = [ticker_display(t) for t in self._watchlist]
            logger.info(f"Screening complete | Watchlist: {watchlist_display}")

        except Exception as e:
            logger.error(f"Screening failed: {e}")

    def _format_screening_msg(self, result) -> str:
        """스크리닝 결과 텔레그램 메시지 포맷."""
        ts = result.timestamp[:16].replace("T", " ")
        stats = result.screening_stats
        mode = self.config_manager.get_mode()
        mode_tag = "[SIM]" if mode == "simulation" else "[LIVE]"

        msg = (
            f"🔍 <b>종목 스크리닝 결과</b> {mode_tag}\n"
            f"⏰ {ts}\n"
            f"━━━━━━━━━━━━━━\n"
            f"분석: {stats.get('total_analyzed', 0):,}종목 → "
            f"필터: {stats.get('after_filter', 0)}종목 → "
            f"선정: {len(result.candidates)}종목\n\n"
        )

        for i, c in enumerate(result.candidates[:8], 1):
            name = c.get("name", c["ticker"])
            score = c.get("composite_score", 0)
            change = c.get("change_pct", 0)
            ch_emoji = "📈" if change > 0 else ("📉" if change < 0 else "➡️")
            msg += (
                f"{i}. <b>{name}</b>({c['ticker']}) "
                f"점수: {score:.0f} {ch_emoji}{change:+.1f}%\n"
                f"   M:{c.get('momentum_score', 0):.0f} "
                f"V:{c.get('value_score', 0):.0f} "
                f"거:{c.get('volume_score', 0):.0f} "
                f"수:{c.get('flow_score', 0):.0f} "
                f"기:{c.get('technical_score', 0):.0f}\n"
            )

        if result.held_tickers_added:
            held_names = [ticker_display(t) for t in result.held_tickers_added]
            msg += f"\n📌 보유종목 추가: {', '.join(held_names)}"

        # 시장 요약
        kospi = result.market_summary.get("kospi", {})
        if kospi:
            msg += (
                f"\n\n📊 KOSPI: 상승 {kospi.get('advancing', 0)} / "
                f"하락 {kospi.get('declining', 0)} "
                f"(평균 {kospi.get('avg_change_pct', 0):+.2f}%)"
            )

        return msg

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
                        self._save_optimized_params(strategy_name, best_params, {
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
                opt_params = self._load_optimized_params(strategy_name)
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
            msg = self._format_backtest_summary(results_summary, start_date, end_date)
            self.telegram.send_alert_sync("backtest_result", msg)

        logger.info(f"Weekly backtest+optimize complete | {len(results_summary)} strategies")

    def _format_backtest_summary(self, results: list[dict], start: str, end: str) -> str:
        """백테스트 요약 텔레그램 메시지."""
        msg = (
            f"📊 <b>주간 백테스트 결과</b>\n"
            f"기간: {start[:4]}-{start[4:6]}-{start[6:]} ~ {end[:4]}-{end[4:6]}-{end[6:]}\n"
            f"━━━━━━━━━━━━━━\n"
        )
        for r in results:
            ret_emoji = "🟢" if r["return"] > 0 else "🔴"
            opt_tag = " ⚙️" if r.get("optimized") else ""
            msg += (
                f"\n{ret_emoji} <b>{r['name']}</b>{opt_tag}\n"
                f"  수익률: {r['return']:+.1%} | 승률: {r['win_rate']:.0%}\n"
                f"  샤프: {r['sharpe']:.2f} | MDD: {r['mdd']:.1%}\n"
                f"  거래: {r['trades']}건\n"
            )
        # 최고 전략 추천
        if results:
            best = max(results, key=lambda x: x["sharpe"])
            msg += f"\n💡 최적 전략: <b>{best['name']}</b> (샤프 {best['sharpe']:.2f})"
        return msg

    def sync_account_from_broker(self) -> dict | None:
        """KIS API에서 실제 계좌 잔고/보유종목을 가져와 포트폴리오 동기화.

        Returns:
            동기화 결과 dict 또는 실패 시 None
        """
        try:
            account = self.market_client.get_account_balance()
            # API 실패로 빈 결과가 돌아온 경우 기존 포트폴리오 유지
            if account["total_asset"] == 0 and not account["positions"]:
                raise RuntimeError("Empty account data returned (API may be unavailable)")
            logger.info(f"Account sync: cash={account['cash']:,}, "
                       f"positions={len(account['positions'])}, "
                       f"total={account['total_asset']:,}")

            # 포트폴리오 현금 동기화
            self.portfolio.cash = account["cash"]
            # initial_capital은 최초 1회만 설정 (이후 동기화에서 덮어쓰면 PnL 추적이 깨짐)
            if self.portfolio.initial_capital == 0 and account["total_asset"] > 0:
                self.portfolio.initial_capital = account["total_asset"]

            # 보유종목 동기화 — 기존 포지션의 peak_price 보존
            old_peaks = {t: p.peak_price for t, p in self.portfolio.positions.items()}
            self.portfolio.positions.clear()
            for pos_data in account["positions"]:
                ticker = pos_data["ticker"]
                cur_price = pos_data["current_price"]
                # 기존 peak_price 유지, 없으면 현재가로 초기화
                prev_peak = old_peaks.get(ticker, cur_price)
                self.portfolio.positions[ticker] = Position(
                    ticker=ticker,
                    name=pos_data["name"],
                    quantity=pos_data["quantity"],
                    avg_price=pos_data["avg_price"],
                    current_price=cur_price,
                    peak_price=max(prev_peak, cur_price),
                )
                # TICKER_NAMES에 추가
                if pos_data["ticker"] not in TICKER_NAMES and pos_data["name"]:
                    TICKER_NAMES[pos_data["ticker"]] = pos_data["name"]

            # HWM 갱신
            if self.portfolio.total_asset > self.portfolio.high_water_mark:
                self.portfolio.high_water_mark = self.portfolio.total_asset
            self.portfolio._save_state()
            logger.info(f"Portfolio synced from broker: {len(account['positions'])} positions")
            return account

        except Exception as e:
            logger.error(f"Account sync failed: {e}")
            raise

    def switch_mode(self, mode: str) -> str:
        """모드 전환 + 필요 시 계좌 동기화.

        거래 일시 중지 → 모드 전환 → 재개 순서로 안전하게 전환.

        Returns:
            결과 메시지
        """
        old_mode = self.config_manager.get_mode()
        # 전환 중 매매 방지
        was_paused = self._paused
        self._paused = True

        self.config_manager.set_mode(mode)
        self.portfolio.mode = mode
        self.executor.mode = mode

        if mode == "live":
            # 실계좌 동기화
            try:
                account = self.sync_account_from_broker()
                pos_count = len(account["positions"])
                msg = (
                    f"🔴 LIVE 모드 전환 완료\n"
                    f"계좌 동기화 성공\n"
                    f"  예수금: {account['cash']:,.0f}원\n"
                    f"  보유종목: {pos_count}개\n"
                    f"  총평가: {account['total_asset']:,.0f}원"
                )
            except Exception as e:
                logger.error(f"Mode switch account sync failed: {e}")
                msg = (
                    f"🔴 LIVE 모드 전환 완료\n"
                    f"⚠️ 계좌 동기화 실패 — 수동 확인 필요"
                )
        else:
            msg = f"🔵 시뮬레이션 모드 전환 완료 (이전: {old_mode})"

        # 거래 재개
        if not was_paused:
            self._paused = False

        logger.info(f"Mode switched: {old_mode} → {mode}")
        return msg

    def on_market_open(self):
        """장 시작 전 초기화."""
        logger.info("Market opening — initializing")
        self.circuit_breaker.daily_reset()
        self.config_manager.reload()
        self._watchlist = self._get_watchlist()

        # live 모드면 장 시작 전 계좌 동기화
        if self.config_manager.get_mode() == "live":
            try:
                self.sync_account_from_broker()
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

        # 일일 리포트 텔레그램 발송
        try:
            self._send_daily_report()
        except Exception as e:
            logger.error(f"Daily report failed: {e}")

    def _send_daily_report(self):
        """장 마감 일일 손익 리포트 생성 및 발송."""
        summary = self.portfolio.get_summary()
        trades = self.portfolio.get_today_trades()
        mode = self.config_manager.get_mode()
        mode_label = "🔵시뮬레이션" if mode == "simulation" else "🔴실거래"

        # 오늘 실현 손익
        realized_pnl = sum(t.get("pnl", 0) or 0 for t in trades)
        buy_count = sum(1 for t in trades if t["action"] == "BUY")
        sell_count = sum(1 for t in trades if t["action"] == "SELL")

        # 보유 종목 평가 손익
        unrealized_pnl = sum(
            p.pnl for p in self.portfolio.positions.values()
        )

        msg = (
            f"📊 <b>일일 리포트</b> ({mode_label})\n"
            f"{'─' * 26}\n\n"
            f"<b>자산 현황</b>\n"
            f"  총자산: {summary['total_asset']:,.0f}원\n"
            f"  현금: {summary['cash']:,.0f}원 ({summary['cash_ratio']})\n"
            f"  평가금: {summary['invested']:,.0f}원\n\n"
            f"<b>오늘 매매</b>\n"
            f"  매수 {buy_count}건 / 매도 {sell_count}건\n"
            f"  실현손익: {realized_pnl:+,.0f}원\n\n"
            f"<b>보유 종목 평가</b>\n"
        )

        for pos in self.portfolio.positions.values():
            pnl_emoji = "📈" if pos.pnl > 0 else "📉" if pos.pnl < 0 else "➡️"
            msg += (
                f"  {pnl_emoji} {pos.name}({pos.ticker})\n"
                f"    {pos.quantity}주 | 매입 {pos.avg_price:,.0f} → 현재 {pos.current_price:,.0f}\n"
                f"    평가손익: {pos.pnl:+,.0f}원 ({pos.pnl_pct:+.2%})\n"
            )

        msg += (
            f"\n{'─' * 26}\n"
            f"<b>미실현손익: {unrealized_pnl:+,.0f}원</b>\n"
            f"<b>총손익(누적): {summary['total_pnl']:,.0f}원 ({summary['total_pnl_pct']})</b>"
        )

        self.telegram.send_alert_sync("daily_report", msg)

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
            opt_params = self._load_optimized_params(strategy_name)
            strategy = factory(opt_params)
            if opt_params:
                messages.append(f"⚙️ 최적화된 파라미터 적용: {opt_params}")
            result = bt.run(tickers, strategy, start_date, end_date)
            if "error" not in result.metrics:
                result.save_to_db(bt.db_path)
            messages.append(format_telegram_report(result))

        elif mode == "compare":
            for name, factory in STRATEGY_REGISTRY.items():
                opt_params = self._load_optimized_params(name)
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
                self._save_optimized_params(strategy_name, bp, {
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

    _ANALYSIS_PATH = Path(__file__).parent / "data" / "last_analysis.json"
    _ANALYSIS_LOG_PATH = Path(__file__).parent / "data" / "analysis_history.jsonl"
    _OUTCOMES_PATH = Path(__file__).parent / "data" / "prediction_outcomes.jsonl"
    _OPTIMIZED_PARAMS_PATH = Path(__file__).parent / "data" / "optimized_params.json"

    def _load_last_analysis(self) -> dict | None:
        """파일에서 마지막 분석 결과 로드."""
        try:
            if self._ANALYSIS_PATH.exists():
                with open(self._ANALYSIS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load last analysis: {e}")
        return None

    def _save_last_analysis(self) -> None:
        """마지막 분석 결과를 파일에 저장."""
        try:
            with open(self._ANALYSIS_PATH, "w", encoding="utf-8") as f:
                json.dump(self._last_analysis, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save last analysis: {e}")

    def _save_optimized_params(self, strategy_name: str, params: dict, metrics: dict):
        """최적화된 전략 파라미터를 파일에 저장."""
        try:
            # 기존 데이터 로드
            all_params = {}
            if self._OPTIMIZED_PARAMS_PATH.exists():
                with open(self._OPTIMIZED_PARAMS_PATH, "r", encoding="utf-8") as f:
                    all_params = json.load(f)

            all_params[strategy_name] = {
                "params": params,
                "metrics": metrics,
                "updated_at": datetime.now().isoformat(),
            }

            with open(self._OPTIMIZED_PARAMS_PATH, "w", encoding="utf-8") as f:
                json.dump(all_params, f, ensure_ascii=False, indent=2)
            logger.info(f"Optimized params saved for {strategy_name}: {params}")
        except Exception as e:
            logger.warning(f"Failed to save optimized params: {e}")

    def _load_optimized_params(self, strategy_name: str) -> dict | None:
        """저장된 최적화 파라미터 로드. 없으면 None."""
        try:
            if not self._OPTIMIZED_PARAMS_PATH.exists():
                return None
            with open(self._OPTIMIZED_PARAMS_PATH, "r", encoding="utf-8") as f:
                all_params = json.load(f)
            entry = all_params.get(strategy_name)
            return entry.get("params") if entry else None
        except Exception:
            return None

    _ALPHA_LOG_PATH = Path(__file__).parent / "data" / "alpha_comparison.jsonl"

    def _log_alpha_comparison(self, llm_actions: list[dict], prices: dict) -> None:
        """LLM 판단 vs 규칙전략 신호 비교 기록 — 알파 추적용."""
        from simulation.strategies import STRATEGY_REGISTRY

        held_tickers = set(self.portfolio.positions.keys())
        watch_tickers = set(self._watchlist)
        all_tickers = held_tickers | watch_tickers

        # LLM 판단 요약
        llm_signals = {}
        for action in llm_actions:
            ticker = action.get("ticker", "")
            llm_signals[ticker] = action.get("type", "HOLD").upper()

        # 규칙전략 신호 생성
        rule_signals = {}
        for ticker in all_tickers:
            df = self.data_pipeline.get_daily_df_with_indicators(ticker)
            if df.empty or len(df) < 2:
                continue
            row = df.iloc[-1]
            prev = df.iloc[-2]
            ctx = {}

            for name, factory in STRATEGY_REGISTRY.items():
                strategy = factory(self._load_optimized_params(name))
                entry = strategy.check_entry(row, prev, ctx)
                exit_signal, exit_reason = strategy.check_exit(row, prev, ctx)

                if ticker not in rule_signals:
                    rule_signals[ticker] = {}
                if entry:
                    rule_signals[ticker][name] = "BUY"
                elif ticker in held_tickers and exit_signal:
                    rule_signals[ticker][name] = f"SELL({exit_reason})"

        # 비교 기록
        record = {
            "timestamp": datetime.now().isoformat(),
            "llm": llm_signals,
            "rules": rule_signals,
            "prices": {t: prices.get(t, 0) for t in all_tickers},
        }

        with open(self._ALPHA_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 불일치 로깅
        for ticker in all_tickers:
            llm_act = llm_signals.get(ticker, "HOLD")
            rule_acts = rule_signals.get(ticker, {})
            if rule_acts:
                rule_summary = ", ".join(f"{k}:{v}" for k, v in rule_acts.items())
                if llm_act != "HOLD" or any("BUY" in v or "SELL" in v for v in rule_acts.values()):
                    logger.info(f"Alpha compare {ticker}: LLM={llm_act} vs Rules=[{rule_summary}]")

    def _build_backtest_feedback(self) -> str:
        """최근 백테스트 결과를 LLM 프롬프트용 피드백 텍스트로 생성."""
        try:
            import sqlite3 as _sqlite3
            db_path = str(Path(__file__).parent / "data" / "storage" / "trader.db")
            with _sqlite3.connect(db_path) as conn:
                conn.row_factory = _sqlite3.Row
                # 최근 백테스트 결과 조회 (전략별 최신 1건씩)
                rows = conn.execute("""
                    SELECT * FROM backtest_results
                    WHERE id IN (
                        SELECT MAX(id) FROM backtest_results
                        GROUP BY strategy_name
                    )
                    ORDER BY created_at DESC
                """).fetchall()

            if not rows:
                return ""

            results = [dict(r) for r in rows]
            created = results[0].get("created_at", "")[:10]

            parts = [f"최근 백테스트 실행일: {created}", ""]

            # 전략별 성과 요약
            parts.append("전략별 성과:")
            for r in results:
                total_ret = r.get("total_return", 0)
                win_rate = r.get("win_rate", 0)
                mdd = r.get("max_drawdown", 0)
                sharpe = r.get("sharpe_ratio", 0)
                pf = r.get("profit_factor", 0)
                trades = r.get("total_trades", 0)
                period = f"{r.get('period_start', '')}~{r.get('period_end', '')}"
                parts.append(
                    f"- {r['strategy_name']}: 수익률 {total_ret:+.1%}, "
                    f"승률 {win_rate:.0%}, MDD {mdd:.1%}, "
                    f"샤프 {sharpe:.2f}, 손익비 {pf:.2f}, "
                    f"거래 {trades}건 ({period})"
                )

            # 종목별 성과 (현재 watchlist 기준)
            watchlist_tickers = set(self._watchlist + list(self.portfolio.positions.keys()))
            ticker_summary: dict[str, list[str]] = {}
            for r in results:
                try:
                    breakdown = json.loads(r.get("ticker_breakdown_json", "{}"))
                except Exception:
                    continue
                for ticker, stats in breakdown.items():
                    if ticker not in watchlist_tickers:
                        continue
                    wins = stats.get("wins", 0)
                    losses = stats.get("losses", 0)
                    avg_pnl = stats.get("avg_pnl_pct", 0)
                    name = TICKER_NAMES.get(ticker, ticker)
                    label = "양호" if avg_pnl > 0 else "주의"
                    if ticker not in ticker_summary:
                        ticker_summary[ticker] = []
                    ticker_summary[ticker].append(
                        f"{r['strategy_name']} {avg_pnl:+.1%}({wins}승{losses}패)"
                    )

            if ticker_summary:
                parts.extend(["", "종목별 백테스트 성과 (관심종목):"])
                for ticker, summaries in sorted(ticker_summary.items()):
                    name = TICKER_NAMES.get(ticker, ticker)
                    parts.append(f"- {name}({ticker}): {', '.join(summaries)}")

            # 최적화된 파라미터 정보
            if self._OPTIMIZED_PARAMS_PATH.exists():
                try:
                    with open(self._OPTIMIZED_PARAMS_PATH, "r", encoding="utf-8") as f:
                        opt_data = json.load(f)
                    if opt_data:
                        parts.extend(["", "최적화된 전략 파라미터:"])
                        for sname, entry in opt_data.items():
                            p = entry.get("params", {})
                            m = entry.get("metrics", {})
                            param_str = ", ".join(f"{k}={v}" for k, v in p.items())
                            parts.append(
                                f"- {sname}: {param_str} "
                                f"(test 수익률 {m.get('test_return', 0):+.1%}, "
                                f"test 샤프 {m.get('test_sharpe', 0):.2f})"
                            )
                except Exception:
                    pass

            return "\n".join(parts)

        except Exception as e:
            logger.debug(f"Backtest feedback unavailable: {e}")
            return ""

    def _build_prediction_feedback(self, lookback: int = 20) -> str:
        """최근 예측 결과를 LLM 프롬프트용 피드백 텍스트로 생성."""
        if not self._OUTCOMES_PATH.exists():
            return ""

        try:
            with open(self._OUTCOMES_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return ""

        if not lines:
            return ""

        # 최근 N건 로드
        recent = []
        for line in lines[-lookback:]:
            try:
                recent.append(json.loads(line.strip()))
            except Exception:
                continue

        if not recent:
            return ""

        # 전체 통계
        total_correct = sum(r.get("correct_count", 0) for r in recent)
        total_evaluated = sum(r.get("total_evaluated", 0) for r in recent)
        if total_evaluated == 0:
            return ""

        overall_accuracy = total_correct / total_evaluated

        # 액션 타입별 통계
        type_stats: dict[str, dict] = {}
        ticker_stats: dict[str, list] = {}
        for record in recent:
            for pred in record.get("predictions", []):
                action = pred.get("predicted_action", "HOLD")
                correct = pred.get("correct", False)
                ret = pred.get("actual_return_pct", 0)
                ticker = pred.get("ticker", "")
                name = pred.get("name", ticker)

                if action not in type_stats:
                    type_stats[action] = {"correct": 0, "total": 0, "returns": []}
                type_stats[action]["total"] += 1
                type_stats[action]["returns"].append(ret)
                if correct:
                    type_stats[action]["correct"] += 1

                key = f"{name}({ticker})"
                if key not in ticker_stats:
                    ticker_stats[key] = []
                ticker_stats[key].append({
                    "action": action, "return": ret, "correct": correct
                })

        # 피드백 텍스트 생성
        parts = [
            f"최근 {len(recent)}회 분석 예측 정확도: {overall_accuracy:.0%} ({total_correct}/{total_evaluated})",
            "",
        ]

        # 액션 타입별
        for action_type in ["BUY", "SELL", "HOLD"]:
            stats = type_stats.get(action_type)
            if not stats or stats["total"] == 0:
                continue
            acc = stats["correct"] / stats["total"]
            avg_ret = sum(stats["returns"]) / len(stats["returns"])
            parts.append(
                f"- {action_type}: 정확도 {acc:.0%} ({stats['correct']}/{stats['total']}), "
                f"평균 실제수익률 {avg_ret:+.1f}%"
            )

        # 반복 오류 종목 (3회 이상 예측, 정확도 50% 미만)
        bad_tickers = []
        for name_ticker, preds in ticker_stats.items():
            if len(preds) >= 3:
                correct_cnt = sum(1 for p in preds if p["correct"])
                if correct_cnt / len(preds) < 0.5:
                    avg_ret = sum(p["return"] for p in preds) / len(preds)
                    bad_tickers.append((name_ticker, correct_cnt, len(preds), avg_ret))

        if bad_tickers:
            parts.append("")
            parts.append("주의 종목 (반복 오류):")
            for name_ticker, correct, total, avg_ret in bad_tickers[:5]:
                parts.append(
                    f"- {name_ticker}: 정확도 {correct}/{total}, 평균수익률 {avg_ret:+.1f}%"
                )

        return "\n".join(parts)

    def _evaluate_previous_predictions(self, current_prices: dict[str, float]) -> None:
        """이전 분석의 예측을 현재 가격과 비교하여 결과를 기록 (학습 데이터용).

        각 예측(BUY/SELL/HOLD)에 대해 실제 수익률을 계산하고
        prediction_outcomes.jsonl에 누적 기록한다.
        """
        prev = self._last_analysis
        if not prev or not prev.get("signal"):
            return

        prev_actions = prev["signal"].get("actions", [])
        if not prev_actions:
            return

        prev_ts = prev.get("timestamp", "")
        now = datetime.now()

        # 이전 분석 시점 파싱
        try:
            prev_dt = datetime.fromisoformat(prev_ts)
            hours_elapsed = (now - prev_dt).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_elapsed = 0

        # 이전 분석 시점의 가격 (portfolio positions + market_data에서 추출)
        prev_portfolio = prev.get("portfolio", {}) if "portfolio" in prev else {}

        outcomes = []
        correct_count = 0
        total_evaluated = 0

        for action in prev_actions:
            ticker = action.get("ticker", "")
            if not ticker or ticker not in current_prices:
                continue

            current_price = current_prices[ticker]
            action_type = action.get("type", "HOLD")

            # 이전 가격: portfolio에 보유 중이었으면 그 가격, 아니면 analysis_history에서
            prev_price = None
            if prev_portfolio and isinstance(prev_portfolio, dict):
                positions = prev_portfolio.get("positions", {})
                if ticker in positions:
                    prev_price = positions[ticker].get("current_price")

            # 보유종목이 아니었으면 analysis_history에서 찾기
            if not prev_price:
                prev_price = self._get_price_at_analysis(ticker, prev_ts)

            if not prev_price or prev_price == 0:
                continue

            # 실제 수익률 계산
            actual_return = (current_price - prev_price) / prev_price
            total_evaluated += 1

            # 예측 정확도 판단
            if action_type == "BUY":
                is_correct = actual_return > 0  # BUY 했는데 올랐으면 정답
            elif action_type == "SELL":
                is_correct = actual_return < 0  # SELL 했는데 내렸으면 정답
            else:  # HOLD
                is_correct = abs(actual_return) < 0.03  # 3% 미만 변동이면 HOLD 정답

            if is_correct:
                correct_count += 1

            outcomes.append({
                "ticker": ticker,
                "name": action.get("name", ""),
                "predicted_action": action_type,
                "predicted_reason": action.get("reason", ""),
                "price_at_prediction": prev_price,
                "price_at_evaluation": current_price,
                "actual_return_pct": round(actual_return * 100, 2),
                "correct": is_correct,
            })

        if not outcomes:
            return

        record = {
            "analysis_timestamp": prev_ts,
            "evaluation_timestamp": now.isoformat(),
            "hours_elapsed": round(hours_elapsed, 1),
            "predictions": outcomes,
            "accuracy": round(correct_count / total_evaluated, 2) if total_evaluated else 0,
            "total_evaluated": total_evaluated,
            "correct_count": correct_count,
        }

        try:
            with open(self._OUTCOMES_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                f"Prediction evaluation: {correct_count}/{total_evaluated} correct "
                f"({record['accuracy']:.0%}) over {hours_elapsed:.1f}h"
            )
        except Exception as e:
            logger.warning(f"Failed to write prediction outcomes: {e}")

    def _get_price_at_analysis(self, ticker: str, analysis_ts: str) -> float | None:
        """analysis_history.jsonl에서 특정 분석 시점의 종목 가격을 찾는다."""
        try:
            if not self._ANALYSIS_LOG_PATH.exists():
                return None
            with open(self._ANALYSIS_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines[-20:]):  # 최근 20건만
                record = json.loads(line.strip())
                if record.get("timestamp", "")[:16] == analysis_ts[:16]:
                    # 1순위: prices_snapshot
                    snapshot = record.get("prices_snapshot", {})
                    if ticker in snapshot:
                        return snapshot[ticker]
                    # 2순위: portfolio positions
                    positions = record.get("portfolio", {}).get("positions", {})
                    if ticker in positions:
                        return positions[ticker].get("current_price")
                    break
        except Exception:
            pass
        return None

    def _append_analysis_log(self, signal: dict, actions: list,
                              portfolio: dict, ml_predictions: dict,
                              prices: dict[str, float] | None = None) -> None:
        """분석 결과를 JSONL 히스토리에 누적 (학습 데이터용)."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.config_manager.get_mode(),
            "signal": signal,
            "actions_executed": actions,
            "portfolio": {
                "total_asset": portfolio.get("total_asset"),
                "cash": portfolio.get("cash"),
                "cash_ratio": portfolio.get("cash_ratio"),
                "total_pnl_pct": portfolio.get("total_pnl_pct"),
                "positions": portfolio.get("positions", {}),
            },
            "prices_snapshot": prices or {},
            "ml_predictions": {
                t: {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in pred.items()}
                for t, pred in ml_predictions.items()
            } if ml_predictions else {},
        }
        try:
            with open(self._ANALYSIS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append analysis log: {e}")

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
