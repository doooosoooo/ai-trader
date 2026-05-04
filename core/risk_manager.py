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
        """
        self.portfolio = portfolio
        self.executor = executor
        self.config_manager = config_manager
        self.safety_guard = safety_guard
        self.telegram = telegram
        self.sim_tracker = sim_tracker
        self.sync_account_fn = sync_account_fn
        self._watchlist: list[str] | None = None  # 스크리닝 관심종목 (외부에서 설정)
        self._gap_sold_today: dict[str, str] = {}  # {ticker: date_str} 갭상승 익절 중복 방지

        # 집중도/드로다운 알림 상태 — 파일 영속화 (PM2 재시작 시 스팸 방지)
        self._concentration_alerted: dict[str, str] = {}
        self._dd_alerted = False
        self._load_alert_state()

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

    def check_risk_exits(self) -> list[dict]:
        """리스크 기반 자동 청산 — 손절/트레일링스탑/보유기간 초과.

        cycle_data_collection()에서 가격 업데이트 직후 호출.
        해당 조건 발생 시 텔레그램 확인 없이 즉시 매도.
        """
        params = self.config_manager.trading_params
        default_trailing_stop_pct = params.get("trailing_stop_pct", 0.06)
        sector_trailing_stop = params.get("trailing_stop_by_sector", {}) or {}
        max_hold_days = params.get("holding_period_days", {}).get("max", 20)

        results = []
        for ticker, pos in list(self.portfolio.positions.items()):
            reason = None
            rules = pos.rules  # strategy_type별 규칙 (value/swing/daytrading)
            pos_sl = rules["stop_loss"]
            pos_tp = rules["take_profit"]

            # 종목 섹터 → 섹터별 트레일링 스탑 (변동성 차등). 매핑 없으면 default.
            sector = self.safety_guard.get_sector(ticker) if hasattr(self.safety_guard, "get_sector") else None
            trailing_stop_pct = sector_trailing_stop.get(sector, default_trailing_stop_pct) if sector else default_trailing_stop_pct

            # 1. 하드 손절 (strategy_type별 다른 손절선)
            if pos.pnl_pct <= pos_sl:
                reason = f"하드손절 ({pos.pnl_pct:.1%} ≤ {pos_sl:.0%}) [{pos.label}]"

            # 2. 트레일링 스탑 (수익 구간에서만 적용, avg_price=0 포지션은 스킵)
            #    섹터별 변동성 기반 스탑 적용 (반도체/2차전지=10%, 금융=5% 등)
            if reason is None and pos.avg_price > 0 and pos.peak_price > pos.avg_price:
                drawdown = (pos.peak_price - pos.current_price) / pos.peak_price
                if drawdown >= trailing_stop_pct:
                    sector_label = f" [{sector} {trailing_stop_pct:.0%}]" if sector and sector in sector_trailing_stop else ""
                    reason = (
                        f"트레일링스탑 (고점{pos.peak_price:,.0f}"
                        f"→현재{pos.current_price:,.0f}, -{drawdown:.1%}){sector_label} [{pos.label}]"
                    )

            # 3. 조기 익절 (수익 +5% 이상이면서 고점 대비 -5% 하락 시)
            if reason is None and pos.pnl_pct >= 0.05:
                if pos.peak_price > 0 and pos.current_price < pos.peak_price * 0.95:
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
        """리스크 청산 즉시 실행 — 쿨다운/확인 우회."""
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
            if result:
                result["reason"] = reason
                if "pnl" not in result:
                    result["pnl"] = pre_pnl
                    result["pnl_pct"] = f"{pre_pnl_pct:.2%}"
            return result
        except Exception as e:
            logger.error(f"Risk exit execution failed for {ticker}: {e}")
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
