"""리스크 기반 자동 청산 + 매매 결과 처리."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from core.notification import NotificationService

_ALERT_STATE_FILE = Path(__file__).parent.parent / "data" / "storage" / "alert_state.json"


class RiskManager:
    """손절/트레일링스탑/보유기간 초과 자동 청산 및 매매 결과 후처리."""

    def __init__(
        self,
        portfolio,
        executor,
        config_manager,
        safety_guard,
        telegram,
        sim_tracker,
        sync_account_fn=None,
        rca_callback=None,
        circuit_breaker=None,
    ):
        """
        Args:
            portfolio: Portfolio 인스턴스
            executor: OrderExecutor 인스턴스
            config_manager: ConfigManager 인스턴스
            safety_guard: SafetyGuard 인스턴스
            telegram: TelegramBot 인스턴스
            sim_tracker: SimulationTracker 인스턴스
            sync_account_fn: 계좌 동기화 콜백 (account_manager.sync_account_from_broker)
            rca_callback: IncidentAnalyzer.trigger 콜백 (선택)
        """
        self.portfolio = portfolio
        self.executor = executor
        self.config_manager = config_manager
        self.safety_guard = safety_guard
        self.telegram = telegram
        self.sim_tracker = sim_tracker
        self.sync_account_fn = sync_account_fn
        self._rca_callback = rca_callback
        self.circuit_breaker = circuit_breaker  # 폭락장 가드 다중 신호용
        self._watchlist: list[str] | None = None  # 스크리닝 관심종목 (외부에서 설정)
        self._gap_sold_today: dict[str, str] = {}  # {ticker: date_str} 갭상승 익절 중복 방지
        # 자동 매도 연속 실패 추적 — 동일 종목 N회 실패 시 일시 차단 + 알림 (KIS 장애 등 무한 재시도 방지)
        self._sell_failures: dict[str, dict] = {}  # {ticker: {"count": n, "first": ts, "alerted": bool}}
        self._SELL_FAIL_THRESHOLD = 3      # 3회 연속 실패 시 차단
        self._SELL_FAIL_BACKOFF_MIN = 15   # 15분간 자동 매도 시도 중지

        # 집중도/드로다운 알림 상태 — 파일 영속화 (PM2 재시작 시 스팸 방지)
        self._concentration_alerted: dict[str, str] = {}
        self._dd_alerted = False
        # 분할 추가매수 (물타기) 일별 dedupe — {ticker: {tier: date_str}}
        self._avg_down_fired: dict[str, dict[int, str]] = {}
        # 강세장 사이즈업 일별 dedupe — {ticker: date_str}, 하루 1회만 추가매수
        self._fill_up_fired: dict[str, str] = {}
        self._load_alert_state()

    def set_rca_callback(self, cb) -> None:
        """RCA 콜백 늦은 wiring."""
        self._rca_callback = cb

    def _safe_rca(self, event_type: str, ticker: str | None = None, event_detail: str = "") -> None:
        """RCA 콜백 안전 호출 — 실패해도 트레이딩 중단되지 않음."""
        if not self._rca_callback:
            return
        try:
            self._rca_callback(event_type=event_type, ticker=ticker, event_detail=event_detail)
        except Exception as e:
            logger.warning(f"RCA callback failed [{event_type}/{ticker}]: {e}")

    def _load_alert_state(self) -> None:
        """영속화된 알림 상태 복원."""
        try:
            if _ALERT_STATE_FILE.exists():
                data = json.loads(_ALERT_STATE_FILE.read_text())
                self._concentration_alerted = data.get("concentration_alerted", {})
                self._dd_alerted = data.get("dd_alerted", False)
        except Exception as e:
            logger.warning(f"Alert state load failed: {e}")

    def _save_alert_state(self) -> None:
        """알림 상태 저장."""
        try:
            _ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "concentration_alerted": self._concentration_alerted,
                "dd_alerted": self._dd_alerted,
                "updated_at": datetime.now().isoformat(),
            }
            _ALERT_STATE_FILE.write_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"Alert state save failed: {e}")

    # 폭락장 손절 일시정지 임계값 (코스피 일중 등락률)
    _CRASH_PAUSE_THRESHOLD = -0.02
    # 폭락장 일시정지가 작동하더라도 평가손실이 이 절대선을 넘으면 무조건 매도
    _ABSOLUTE_FLOOR_PNL = -0.15
    # 우량주 하드손절 면제 — 시총 임계값 (10조원). include_tickers는 시총 무관 면제.
    # 우량주는 단기 변동성으로 -5% 터치 후 반등하는 경우 다수 — 손절-재매수 사이클로 자본 잠식 방지.
    # 트레일링 스탑(8~10%)과 절대선 -15%는 그대로 적용.
    # current_prices.market_cap 단위는 억원(100M won) — KIS hts_avls 원본 단위.
    _PREMIUM_MARKET_CAP_MIN_OK = 100_000  # 10조원 = 100,000억원
    # 분할 추가매수 (물타기) 트리거 — 폭락장 + 현금 ≥ 40% 시 활성화
    _AVG_DOWN_CASH_RATIO_MIN = 0.40
    _AVG_DOWN_TIER_1_PNL = -0.05  # 1차 추가매수 트리거 평단 대비 손실률
    _AVG_DOWN_TIER_2_PNL = -0.10  # 2차 추가매수 트리거
    _AVG_DOWN_RATIO_PER_TIER = 1 / 3  # 매 차수마다 보유 수량의 1/3 추가매수
    # 강세장 자동 추가매수 (포지션 사이즈업) — 토막 진입 종목 키우기
    # 폭락장(crash_pause)에서는 비활성, avg_down이 담당.
    _FILL_UP_CASH_RATIO_MIN = 0.30   # 현금 30% 이상일 때만 작동
    _FILL_UP_WEIGHT_TARGET = 0.05    # 비중 5% 미만 종목이 대상
    _FILL_UP_PNL_MIN = -0.02         # 평단 대비 -2% 이하(=손실 큰)는 제외 (avg_down 영역)
    _FILL_UP_RATIO_STEP = 0.02       # 1회 추가 매수 ≒ 총자산의 2%

    def _is_premium_stock(self, ticker: str) -> tuple[bool, str]:
        """우량주 여부 — include_tickers 또는 시총 10조+ 종목.

        Returns:
            (is_premium, reason)
        """
        try:
            if self.safety_guard and ticker in getattr(self.safety_guard, "_include_tickers", set()):
                return True, "include_tickers"
        except Exception:
            pass
        try:
            import sqlite3
            db_path = Path(__file__).parent.parent / "data" / "storage" / "trader.db"
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    "SELECT market_cap FROM current_prices WHERE ticker=?",
                    (ticker,),
                ).fetchone()
            if row and row[0] and row[0] >= self._PREMIUM_MARKET_CAP_MIN_OK:
                return True, f"시총 {row[0]/10000:.1f}조"
        except Exception:
            pass
        return False, ""

    def check_risk_exits(self) -> list[dict]:
        """리스크 기반 자동 청산 — 손절/트레일링스탑/보유기간 초과.

        cycle_data_collection()에서 가격 업데이트 직후 호출.
        해당 조건 발생 시 텔레그램 확인 없이 즉시 매도.

        폭락장 가드: 코스피 일중 -2% 이상 하락 중에는 하드손절/트레일링/조기익절을 일시정지.
        단, 평가손실 -15% 도달 종목은 절대선으로 무조건 매도.
        같은 조건 충족 시 분할 추가매수(물타기) 자동 트리거.
        """
        params = self.config_manager.trading_params
        default_trailing_stop_pct = params.get("trailing_stop_pct", 0.06)
        sector_trailing_stop = params.get("trailing_stop_by_sector", {}) or {}
        max_hold_days = params.get("holding_period_days", {}).get("max", 20)

        # 폭락장 여부 — 다중 신호로 판단 (KOSPI API 실패 시초가 우회 방지)
        # (a) KOSPI 명시적 폭락 (-2% 이하)
        # (b) Circuit breaker 이미 HALTED/EMERGENCY (서킷이 폭락 인식)
        # (c) KOSPI 데이터 없을 때 — 포트폴리오 일일 손실 -2% 이상 폴백
        # (d) 시초가 30분 (09:00~09:30) + KOSPI 데이터 없음 — 갭다운 가능성 보수적 처리
        kospi_change = self._get_kospi_change_pct()
        crash_by_kospi = kospi_change is not None and kospi_change <= self._CRASH_PAUSE_THRESHOLD

        crash_by_circuit = False
        try:
            if self.circuit_breaker:
                from core.circuit_breaker import CircuitState
                crash_by_circuit = self.circuit_breaker.state in (
                    CircuitState.HALTED, CircuitState.EMERGENCY,
                )
        except Exception:
            pass

        crash_by_portfolio = False
        try:
            snapshots = self.portfolio.get_daily_snapshots(days=3)
            if snapshots:
                today_str = datetime.now().strftime("%Y-%m-%d")
                prev_asset = None
                for snap in snapshots:
                    if str(snap.get("date", ""))[:10] != today_str:
                        prev_asset = snap.get("total_asset", 0)
                        break
                cur_asset = self.portfolio.total_asset or 0
                if prev_asset and prev_asset > 0 and cur_asset > 0:
                    daily_pct = (cur_asset - prev_asset) / prev_asset
                    crash_by_portfolio = daily_pct <= -0.02
        except Exception:
            pass

        crash_by_opening_unknown = False
        try:
            from datetime import time as dtime
            now_t = datetime.now().time()
            if kospi_change is None and dtime(9, 0) <= now_t < dtime(9, 30):
                crash_by_opening_unknown = True
        except Exception:
            pass

        crash_pause = crash_by_kospi or crash_by_circuit or crash_by_portfolio or crash_by_opening_unknown

        if crash_pause:
            kospi_str = f"{kospi_change:.2%}" if kospi_change is not None else "N/A"
            signals = []
            if crash_by_kospi: signals.append(f"KOSPI {kospi_str}")
            if crash_by_circuit: signals.append(f"circuit {self.circuit_breaker.state.value}")
            if crash_by_portfolio: signals.append("포트폴리오 일일손실 -2%↓")
            if crash_by_opening_unknown: signals.append("시초가30분 KOSPI 데이터 없음")
            logger.warning(
                f"Crash market pause active [{', '.join(signals)}] — "
                f"하드손절/트레일링/조기익절 일시정지, "
                f"절대선 {self._ABSOLUTE_FLOOR_PNL:.0%}만 유지"
            )

        results = []
        for ticker, pos in list(self.portfolio.positions.items()):
            reason = None
            rules = pos.rules  # strategy_type별 규칙 (value/swing/daytrading)
            pos_sl = rules["stop_loss"]
            pos_tp = rules["take_profit"]

            # 종목 섹터 → 섹터별 트레일링 스탑 (변동성 차등). 매핑 없으면 default.
            sector = self.safety_guard.get_sector(ticker) if hasattr(self.safety_guard, "get_sector") else None
            trailing_stop_pct = sector_trailing_stop.get(sector, default_trailing_stop_pct) if sector else default_trailing_stop_pct

            # 0. 절대선 — 폭락장 일시정지와 무관하게 평가손실 -15% 도달 시 무조건 매도
            if pos.pnl_pct <= self._ABSOLUTE_FLOOR_PNL:
                reason = (
                    f"절대선손절 ({pos.pnl_pct:.1%} ≤ {self._ABSOLUTE_FLOOR_PNL:.0%}) [{pos.label}]"
                )

            # 1. 하드 손절 (strategy_type별) — 우량주 면제 + 폭락장 일시정지 적용
            if reason is None and pos.pnl_pct <= pos_sl:
                is_premium, premium_reason = self._is_premium_stock(ticker)
                if is_premium:
                    # 우량주는 하드손절 면제 — 트레일링 스탑/절대선 -15%로만 보호
                    logger.info(
                        f"[premium_exempt] 하드손절 면제 {pos.name}({ticker}) "
                        f"pnl={pos.pnl_pct:.1%} ({premium_reason}) — 트레일링/절대선만 적용"
                    )
                elif crash_pause:
                    kospi_str = f"{kospi_change:.2%}" if kospi_change is not None else "N/A"
                    logger.info(
                        f"[crash_pause] 하드손절 보류 {pos.name}({ticker}) "
                        f"pnl={pos.pnl_pct:.1%} (코스피 {kospi_str})"
                    )
                else:
                    reason = f"하드손절 ({pos.pnl_pct:.1%} ≤ {pos_sl:.0%}) [{pos.label}]"

            # 2. 트레일링 스탑 — 폭락장 일시정지 적용
            if reason is None and pos.avg_price > 0 and pos.peak_price > pos.avg_price:
                drawdown = (pos.peak_price - pos.current_price) / pos.peak_price
                if drawdown >= trailing_stop_pct:
                    if crash_pause:
                        logger.info(
                            f"[crash_pause] 트레일링스탑 보류 {pos.name}({ticker}) "
                            f"drawdown={drawdown:.1%}"
                        )
                    else:
                        sector_label = f" [{sector} {trailing_stop_pct:.0%}]" if sector and sector in sector_trailing_stop else ""
                        reason = (
                            f"트레일링스탑 (고점{pos.peak_price:,.0f}"
                            f"→현재{pos.current_price:,.0f}, -{drawdown:.1%}){sector_label} [{pos.label}]"
                        )

            # 3. 조기 익절 (수익 +5% 이상이면서 고점 대비 -5% 하락 시) — 폭락장 일시정지 적용
            if reason is None and pos.pnl_pct >= 0.05:
                if pos.peak_price > 0 and pos.current_price < pos.peak_price * 0.95:
                    if crash_pause:
                        logger.info(
                            f"[crash_pause] 조기익절 보류 {pos.name}({ticker}) pnl={pos.pnl_pct:.1%}"
                        )
                    else:
                        reason = (
                            f"조기익절 (수익 {pos.pnl_pct:.1%}, "
                            f"고점{pos.peak_price:,.0f}→현재{pos.current_price:,.0f} 하락반전) [{pos.label}]"
                        )

            # 3-B. 갭상승 부분 익절 (장시작 30분 + 코스피 +1.5% + 수익 +3% 이상)
            # value는 장기 보유이므로 제외, 당일 1회만
            today_str = datetime.now().strftime("%Y-%m-%d")
            gap_sold_today = self._gap_sold_today.get(ticker) == today_str
            if reason is None and pos.strategy_type != "value" and pos.pnl_pct >= 0.03 and not gap_sold_today:
                now_t = datetime.now().time()
                from datetime import time as dtime
                if dtime(9, 0) <= now_t < dtime(9, 30):
                    kospi_change = self._get_kospi_change_pct()
                    if kospi_change is not None and kospi_change >= 0.015:
                        # 부분 매도 (50%) — 갭상승 되돌림 대비
                        reason = (
                            f"갭상승익절 (수익 {pos.pnl_pct:.1%}, 코스피 +{kospi_change:.1%} 갭상승) [{pos.label}][PARTIAL:50%]"
                        )
                        self._gap_sold_today[ticker] = today_str

            # 4. 스크리닝 탈락 + 마이너스 → 즉시 매도 (value 제외)
            if reason is None and self._watchlist is not None:
                if rules["screening_dropout_sell"] and ticker not in self._watchlist and pos.pnl_pct < 0:
                    reason = f"스크리닝탈락+손실 ({pos.pnl_pct:.1%}, 모멘텀 이탈) [{pos.label}]"

            # 5. 보유기간 초과
            if reason is None:
                bought = datetime.fromisoformat(pos.bought_at)
                days_held = (datetime.now().date() - bought.date()).days
                if days_held >= max_hold_days:
                    reason = f"보유기간초과 ({days_held}일 ≥ {max_hold_days}일) [{pos.label}]"

            if reason:
                logger.warning(f"Risk exit triggered: {pos.name}({ticker}) — {reason}")
                result = self._execute_risk_exit(ticker, pos, reason)
                if result:
                    results.append(result)

        # 3.5 폭락장 분할 추가매수 (물타기) — 손절 일시정지와 짝으로 활성화
        if crash_pause:
            avg_down_results = self._trigger_avg_down_orders(kospi_change)
            results.extend(avg_down_results)
        else:
            # 3.6 강세/정상장 자동 사이즈업 — 토막 진입한 종목 비중 키우기
            fill_up_results = self._trigger_fill_up_orders()
            results.extend(fill_up_results)

        # 4. 집중도 경고 (자동 매도는 아니지만 텔레그램 알림)
        # 2종목 이하면 비중 초과는 불가피하므로 경고 스킵
        num_positions = len(self.portfolio.positions)
        if num_positions >= 3:
            max_weight = self.safety_guard.rules.get("max_position_ratio", 0.15) + 0.05
            total_asset = self.portfolio.total_asset or 1
            today_str = datetime.now().strftime("%Y-%m-%d")
            for ticker, pos in self.portfolio.positions.items():
                weight = pos.market_value / total_asset
                if weight > max_weight:
                    # 하루 1회만 알림 (종목별 쿨다운)
                    if self._concentration_alerted.get(ticker) == today_str:
                        continue
                    self._concentration_alerted[ticker] = today_str
                    self._save_alert_state()
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
            if self.telegram.enabled and not self._dd_alerted:
                self.telegram.send_alert_sync(
                    "drawdown_alert",
                    f"⚠️ <b>포트폴리오 낙폭 경고</b>\n"
                    f"고점(HWM): {self.portfolio.high_water_mark:,.0f}원\n"
                    f"현재 총자산: {self.portfolio.total_asset:,.0f}원\n"
                    f"낙폭: {drawdown:.1%}\n"
                    f"신규 매수를 자제하고 리스크를 점검하세요.",
                )
                self._dd_alerted = True  # 중복 알림 방지
                self._save_alert_state()
        else:
            if self._dd_alerted:
                self._dd_alerted = False  # 드로다운 회복 시 알림 리셋
                self._save_alert_state()

        return results

    def _trigger_avg_down_orders(self, kospi_change: float | None) -> list[dict]:
        """폭락장 + 현금 ≥ 40% 시 평단 -5%/-10% 도달 종목에 자동 분할 추가매수.

        - swing/value 종목만 (daytrading 제외 — 짧은 손익 구조에 맞지 않음)
        - 1차(평단 -5%) / 2차(평단 -10%) 각각 보유 비중의 1/3 추가매수
        - 동일 종목 동일 차수는 일별 1회만 (`_avg_down_fired` dedupe)
        - 종목당 max_position_ratio (15%) 초과 금지 (캡)
        - 절대선 -15% 종목은 위 check_risk_exits에서 이미 청산되므로 진입 안 함
        """
        if not self.portfolio.positions:
            return []

        total_asset = self.portfolio.total_asset or 1
        cash = getattr(self.portfolio, "cash", 0) or 0
        cash_ratio = cash / total_asset
        if cash_ratio < self._AVG_DOWN_CASH_RATIO_MIN:
            logger.info(
                f"[avg_down] 현금비율 {cash_ratio:.1%} < {self._AVG_DOWN_CASH_RATIO_MIN:.0%} — skip"
            )
            return []

        today_str = datetime.now().strftime("%Y-%m-%d")
        max_position_ratio = self.safety_guard.rules.get("max_position_ratio", 0.15)
        actions: list[dict] = []
        fired_log: list[tuple[str, int]] = []

        for ticker, pos in self.portfolio.positions.items():
            if pos.avg_price <= 0:
                continue
            if (pos.strategy_type or "swing") == "daytrading":
                continue

            # 차수 결정 — 2차가 우선 (절대선 -15% 직전 마지막 분할)
            if pos.pnl_pct <= self._AVG_DOWN_TIER_2_PNL:
                tier = 2
            elif pos.pnl_pct <= self._AVG_DOWN_TIER_1_PNL:
                tier = 1
            else:
                continue

            # 일별 dedupe (같은 종목 같은 차수는 하루 1회)
            fired = self._avg_down_fired.get(ticker, {})
            if fired.get(tier) == today_str:
                continue

            # 매수 비중 = 보유 비중의 1/3
            current_weight = (pos.market_value or 0) / total_asset
            buy_ratio = current_weight * self._AVG_DOWN_RATIO_PER_TIER

            # 종목당 한도 캡
            if current_weight + buy_ratio > max_position_ratio:
                buy_ratio = max(0.0, max_position_ratio - current_weight)

            # 너무 작은 주문은 의미 없음 (수수료/슬리피지 고려, 0.5% 미만 컷)
            if buy_ratio < 0.005:
                logger.info(
                    f"[avg_down] {pos.name}({ticker}) {tier}차 skip — "
                    f"종목 비중 {current_weight:.1%} 한도 근접 (buy_ratio={buy_ratio:.2%})"
                )
                continue

            # 현금 잔액 캡 — 동시 여러 종목 추가매수 시 현금 부족 방지
            order_amount = total_asset * buy_ratio
            if order_amount > cash:
                buy_ratio = cash / total_asset
                if buy_ratio < 0.005:
                    continue

            limit_price = int(pos.current_price * 0.99)  # 현재가 -1% 보수적 지정가
            actions.append({
                "type": "BUY",
                "ticker": ticker,
                "name": pos.name,
                "ratio": buy_ratio,
                "urgency": "limit",
                "limit_price": limit_price,
                "reason": (
                    f"폭락장 분할추가매수 {tier}차 (평단대비 {pos.pnl_pct:.1%}, "
                    f"비중 {current_weight:.1%}+{buy_ratio:.1%}, 코스피 {f'{kospi_change:.2%}' if kospi_change is not None else 'N/A'})"
                ),
                "strategy_type": pos.strategy_type or "swing",
            })
            fired_log.append((ticker, tier))
            cash -= order_amount  # 다음 종목 계산 시 차감

        if not actions:
            return []

        pseudo_signal = {
            "reasoning": f"RiskManager 자동 분할 추가매수 (폭락장 + 현금 {cash_ratio:.0%})",
            "actions": actions,
            "risk_assessment": "MEDIUM",
            "market_outlook": f"코스피 {f'{kospi_change:.2%}' if kospi_change is not None else 'N/A'} 폭락장 — 분할 평단조정",
            "config_adjustments": [],
        }

        current_prices = {tk: p.current_price for tk, p in self.portfolio.positions.items()}
        logger.warning(f"Avg-down triggered: {len(actions)} orders")

        try:
            results = self.executor.execute_signal(pseudo_signal, current_prices)
        except Exception as e:
            logger.error(f"Avg-down execution failed: {e}")
            return []

        # 성공 시 dedupe 마킹 (실패 시 다음 사이클에서 재시도 가능)
        for r in results or []:
            if r and r.get("status") not in ("FAILED",):
                tk = r.get("ticker")
                for (etk, tier) in fired_log:
                    if etk == tk:
                        f = self._avg_down_fired.get(tk, {})
                        f[tier] = today_str
                        self._avg_down_fired[tk] = f

        # 텔레그램 알림
        if self.telegram and getattr(self.telegram, "enabled", False):
            try:
                lines = [
                    f"📥 <b>폭락장 분할 추가매수 (자동)</b>",
                    f"코스피 {f'{kospi_change:.2%}' if kospi_change is not None else 'N/A'} / 현금비율 {cash_ratio:.0%}",
                    "",
                ]
                for a in actions:
                    lines.append(
                        f"• {a['name']}({a['ticker']}) "
                        f"비중 +{a['ratio']:.1%} @ {a['limit_price']:,}원"
                    )
                    lines.append(f"  └ {a['reason']}")
                self.telegram.send_alert_sync("trade_executed", "\n".join(lines))
            except Exception as e:
                logger.warning(f"Avg-down telegram send failed: {e}")

        return results or []

    def _trigger_fill_up_orders(self) -> list[dict]:
        """강세/정상장에서 토막 진입한 종목을 자동 사이즈업.

        Why: LLM은 신규 BUY만 하고 기존 작은 포지션을 키우지 않음 →
             1차 진입(3%) 후 추가 매수 안 되어 현금 누적. 5/21 현금 55.7%,
             종목 7개 비중 <3% 사태.

        Trigger:
        - NOT crash_pause (avg_down이 담당하는 영역과 충돌 회피)
        - 현금 ≥ 30%
        - 종목 비중 < 5%
        - 평단 대비 손실 -2% 이하는 제외 (avg_down 영역)
        - daytrading 제외 (짧은 손익)

        Action:
        - 종목당 +2% 추가 매수 (총자산 기준)
        - 종목당 max_position_ratio 한도(15%) 초과 금지
        - 종목당 하루 1회만 (dedupe)
        - limit_price = 현재가 -1%
        """
        if not self.portfolio.positions:
            return []

        total_asset = self.portfolio.total_asset or 1
        cash = getattr(self.portfolio, "cash", 0) or 0
        cash_ratio = cash / total_asset
        if cash_ratio < self._FILL_UP_CASH_RATIO_MIN:
            return []

        today_str = datetime.now().strftime("%Y-%m-%d")
        max_position_ratio = self.safety_guard.rules.get("max_position_ratio", 0.15)
        actions: list[dict] = []
        fired_log: list[str] = []

        # 비중 작은 순으로 정렬 — 가장 토막난 종목부터 채움
        candidates = sorted(
            self.portfolio.positions.items(),
            key=lambda kv: (kv[1].market_value or 0) / total_asset,
        )

        for ticker, pos in candidates:
            if pos.avg_price <= 0 or pos.current_price <= 0:
                continue
            if (pos.strategy_type or "swing") == "daytrading":
                continue
            if pos.pnl_pct < self._FILL_UP_PNL_MIN:
                continue  # 평단 -2% 이하 손실 종목은 avg_down 영역

            current_weight = (pos.market_value or 0) / total_asset
            if current_weight >= self._FILL_UP_WEIGHT_TARGET:
                continue  # 이미 충분히 큰 종목

            # 일별 dedupe
            if self._fill_up_fired.get(ticker) == today_str:
                continue

            buy_ratio = self._FILL_UP_RATIO_STEP

            # 종목당 한도 캡
            if current_weight + buy_ratio > max_position_ratio:
                buy_ratio = max(0.0, max_position_ratio - current_weight)
            if buy_ratio < 0.005:
                continue

            # 현금 잔액 캡
            order_amount = total_asset * buy_ratio
            if order_amount > cash:
                buy_ratio = cash / total_asset
                if buy_ratio < 0.005:
                    continue
                order_amount = total_asset * buy_ratio

            limit_price = int(pos.current_price * 0.99)
            actions.append({
                "type": "BUY",
                "ticker": ticker,
                "name": pos.name,
                "ratio": buy_ratio,
                "urgency": "limit",
                "limit_price": limit_price,
                "reason": (
                    f"강세장 자동 사이즈업 (비중 {current_weight:.1%}+{buy_ratio:.1%}, "
                    f"손익 {pos.pnl_pct:+.1%}, 현금 {cash_ratio:.0%})"
                ),
                "strategy_type": pos.strategy_type or "swing",
            })
            fired_log.append(ticker)
            cash -= order_amount

            # 한 사이클에 너무 많이 풀지 않음 — 최대 3종목
            if len(actions) >= 3:
                break

        if not actions:
            return []

        pseudo_signal = {
            "reasoning": f"RiskManager 자동 사이즈업 (강세/정상장, 현금 {cash_ratio:.0%})",
            "actions": actions,
            "risk_assessment": "LOW",
            "market_outlook": "정상/강세장 — 토막 진입 종목 사이즈업",
            "config_adjustments": [],
        }

        current_prices = {tk: p.current_price for tk, p in self.portfolio.positions.items()}
        logger.warning(f"Fill-up triggered: {len(actions)} orders")

        try:
            results = self.executor.execute_signal(pseudo_signal, current_prices)
        except Exception as e:
            logger.error(f"Fill-up execution failed: {e}")
            return []

        # 성공 시 dedupe 마킹
        for r in results or []:
            if r and r.get("status") not in ("FAILED",):
                tk = r.get("ticker")
                if tk in fired_log:
                    self._fill_up_fired[tk] = today_str

        # 텔레그램 알림
        if self.telegram and getattr(self.telegram, "enabled", False):
            try:
                lines = [
                    f"📥 <b>강세장 자동 사이즈업</b>",
                    f"현금비율 {cash_ratio:.0%} → 토막 종목 비중 키움",
                    "",
                ]
                for a in actions:
                    lines.append(
                        f"• {a['name']}({a['ticker']}) "
                        f"비중 +{a['ratio']:.1%} @ {a['limit_price']:,}원"
                    )
                    lines.append(f"  └ {a['reason']}")
                self.telegram.send_alert_sync("trade_executed", "\n".join(lines))
            except Exception as e:
                logger.warning(f"Fill-up telegram send failed: {e}")

        return results or []

    def _get_kospi_change_pct(self) -> float | None:
        """현재 코스피 당일 등락률 조회."""
        try:
            import requests
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(
                'https://m.stock.naver.com/api/index/KOSPI/basic',
                headers=headers, timeout=3,
            )
            data = resp.json()
            return float(data.get('fluctuationsRatio', 0)) / 100
        except Exception:
            return None

    def _execute_risk_exit(self, ticker: str, pos, reason: str) -> dict | None:
        """리스크 청산 즉시 실행 — 쿨다운/확인 우회. 연속 실패 시 backoff."""
        # 연속 실패 backoff 체크 — KIS 장애 등으로 같은 종목 무한 재시도 방지
        fail_info = self._sell_failures.get(ticker)
        if fail_info and fail_info["count"] >= self._SELL_FAIL_THRESHOLD:
            from datetime import timedelta
            backoff_end = fail_info["first"] + timedelta(minutes=self._SELL_FAIL_BACKOFF_MIN)
            if datetime.now() < backoff_end:
                # 차단 중 — 첫 차단 시점에만 알림 (스팸 방지)
                if not fail_info.get("alerted"):
                    fail_info["alerted"] = True
                    logger.error(
                        f"Auto SELL blocked for {pos.name}({ticker}): "
                        f"{fail_info['count']}회 연속 실패 — {self._SELL_FAIL_BACKOFF_MIN}분 차단. "
                        f"수동 매도 필요."
                    )
                    if self.telegram and self.telegram.enabled:
                        try:
                            self.telegram.send_alert_sync(
                                "error",
                                f"⚠️ <b>자동 매도 차단</b>\n"
                                f"{pos.name}({ticker}) — {fail_info['count']}회 연속 KIS API 실패\n"
                                f"사유: {reason}\n"
                                f"→ {self._SELL_FAIL_BACKOFF_MIN}분 자동 시도 중지. KIS 앱에서 수동 매도 필요.",
                            )
                        except Exception:
                            pass
                    # 첫 차단 시점에 RCA 1회 트리거 (60분 dedupe로 반복 호출 자동 차단)
                    self._safe_rca(
                        event_type="sell_backoff_blocked",
                        ticker=ticker,
                        event_detail=(
                            f"{pos.name}({ticker}) — {fail_info['count']}회 연속 매도 실패, "
                            f"{self._SELL_FAIL_BACKOFF_MIN}분 차단. 사유: {reason}"
                        ),
                    )
                return None
            else:
                # backoff 만료 — 카운터 리셋하고 재시도
                self._sell_failures.pop(ticker, None)

        try:
            # [PARTIAL:X%] 태그로 부분 매도 비율 파싱
            import re
            partial_match = re.search(r"\[PARTIAL:(\d+)%\]", reason)
            ratio = float(partial_match.group(1)) / 100 if partial_match else 1.0

            # PnL을 매도 전에 계산 (live 매도 결과에는 pnl이 없으므로)
            pre_pnl = pos.pnl
            pre_pnl_pct = pos.pnl_pct
            result = self.executor._execute_sell(
                ticker=ticker,
                name=pos.name,
                ratio=ratio,
                urgency="high",
                limit_price=None,
                current_price=pos.current_price,
                reason=f"[AUTO] {reason}",
                signal_json="",
            )
            # 실패 카운터 갱신
            if result and result.get("status") == "FAILED":
                info = self._sell_failures.setdefault(ticker, {"count": 0, "first": datetime.now(), "alerted": False})
                info["count"] += 1
            elif result and result.get("status") in ("SUBMITTED", "SIMULATED"):
                self._sell_failures.pop(ticker, None)  # 성공 시 카운터 리셋

            if result:
                result["reason"] = reason
                if "pnl" not in result:
                    result["pnl"] = pre_pnl
                    result["pnl_pct"] = f"{pre_pnl_pct:.2%}"

            # 하드손절 매도 성공 시 RCA 트리거 (pnl ≤ -5%)
            if (
                result
                and result.get("status") in ("SUBMITTED", "SIMULATED")
                and "하드손절" in reason
                and pre_pnl_pct <= -0.05
            ):
                self._safe_rca(
                    event_type="hard_stop_loss",
                    ticker=ticker,
                    event_detail=(
                        f"{pos.name}({ticker}) 하드손절 매도 — "
                        f"pnl {pre_pnl_pct:.2%} (avg {pos.avg_price:,.0f} → "
                        f"{pos.current_price:,.0f}). {reason}"
                    ),
                )
            return result
        except Exception as e:
            logger.error(f"Risk exit execution failed for {ticker}: {e}")
            info = self._sell_failures.setdefault(ticker, {"count": 0, "first": datetime.now(), "alerted": False})
            info["count"] += 1
            return None

    def process_trade_result(self, result: dict, suppress_notification: bool = False) -> None:
        """매매 결과 처리 — 알림 + 시뮬레이션 추적 + live 계좌 동기화.

        Args:
            suppress_notification: True이면 trade_executed 알림을 보내지 않음
                (분석 사이클에서 호출 시 llm_analysis 메시지에 통합되므로 중복 방지)
        """
        status = result.get("status", "")

        if status == "SIMULATED":
            self.sim_tracker.record_trade(result)
            if self.telegram.enabled and not suppress_notification:
                self.telegram.send_alert_sync(
                    "trade_executed",
                    f"[SIM] {NotificationService.format_trade_msg(result)}",
                )

        elif status == "SUBMITTED":
            # Live 주문 접수됨 — trade_history에 기록 + 계좌 동기화
            self.portfolio._record_trade(result, result.get("signal_json", ""))
            if self.telegram.enabled and not suppress_notification:
                mode_tag = "[LIVE]"
                order_no = result.get("order_no", "")
                self.telegram.send_alert_sync(
                    "trade_executed",
                    f"{mode_tag} {NotificationService.format_trade_msg(result)}\n주문번호: {order_no}",
                )
            # 주문 후 계좌 동기화 (체결 반영) — 최대 3회 재시도
            if self.sync_account_fn:
                synced = False
                for attempt in range(3):
                    try:
                        if attempt > 0:
                            time.sleep(2 * attempt)  # 2초, 4초 대기
                        self.sync_account_fn()
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
