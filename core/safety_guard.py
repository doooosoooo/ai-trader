"""Safety Guard — LLM 출력과 무관하게 반드시 지켜야 할 하드코딩 규칙."""

from datetime import datetime, timedelta
from typing import Any

from loguru import logger


class SafetyViolation:
    """안전 규칙 위반 상세."""

    def __init__(self, rule: str, message: str, action_index: int = -1):
        self.rule = rule
        self.message = message
        self.action_index = action_index
        self.timestamp = datetime.now()

    def __repr__(self):
        return f"SafetyViolation({self.rule}: {self.message})"


class SafetyGuard:
    """모든 매매 시그널을 주문 실행 전에 반드시 검증."""

    def __init__(self, safety_rules: dict, trading_params: dict):
        self.rules = safety_rules
        self.trading_params = trading_params
        self._last_trade_time: dict[str, datetime] = {}  # ticker -> last trade time
        self._daily_pnl: float = 0.0
        self._daily_pnl_date: str = ""

    def validate_signal(
        self,
        signal: dict,
        portfolio: dict,
        total_asset: float,
    ) -> tuple[bool, list[SafetyViolation]]:
        """매매 시그널 전체를 검증.

        Args:
            signal: LLM이 출력한 전체 시그널 (actions 리스트 포함)
            portfolio: 현재 포트폴리오 상태
            total_asset: 현재 총자산

        Returns:
            (통과 여부, 위반 목록)
        """
        violations = []
        actions = signal.get("actions", [])

        for i, action in enumerate(actions):
            action_violations = self._validate_action(action, portfolio, total_asset, i)
            violations.extend(action_violations)

        passed = len(violations) == 0
        if not passed:
            for v in violations:
                logger.warning(f"Safety violation: {v}")

        return passed, violations

    def _validate_action(
        self,
        action: dict,
        portfolio: dict,
        total_asset: float,
        index: int,
    ) -> list[SafetyViolation]:
        violations = []
        action_type = action.get("type", "").upper()
        ticker = action.get("ticker", "")
        ratio = action.get("ratio", 0)

        if action_type == "HOLD":
            return []

        # 1. 거래 시간 확인
        if self.rules.get("trading_hours_only", True):
            now = datetime.now()
            hour = now.hour
            minute = now.minute
            market_open = hour > 9 or (hour == 9 and minute >= 0)
            market_close = hour < 15 or (hour == 15 and minute <= 30)
            if not (market_open and market_close):
                violations.append(SafetyViolation(
                    "trading_hours_only",
                    f"장 시간 외 거래 시도: {now.strftime('%H:%M')}",
                    index,
                ))

        if action_type == "BUY":
            # 2. 단일 종목 최대 비중
            max_ratio = self.rules.get("max_position_ratio", 0.15)
            if ratio > max_ratio:
                violations.append(SafetyViolation(
                    "max_position_ratio",
                    f"종목 비중 {ratio:.1%} > 최대 {max_ratio:.1%}",
                    index,
                ))

            # 3. 최대 동시 보유 종목 수
            max_positions = self.rules.get("max_positions", 5)
            current_positions = len(portfolio.get("positions", {}))
            is_existing = ticker in portfolio.get("positions", {})
            if not is_existing and current_positions >= max_positions:
                violations.append(SafetyViolation(
                    "max_positions",
                    f"보유 종목 수 {current_positions} >= 최대 {max_positions}",
                    index,
                ))

            # 4. 단일 주문 최대 금액
            max_order = self.rules.get("max_single_order_amount", 5_000_000)
            order_amount = total_asset * ratio
            if order_amount > max_order:
                violations.append(SafetyViolation(
                    "max_single_order_amount",
                    f"주문 금액 {order_amount:,.0f}원 > 최대 {max_order:,.0f}원",
                    index,
                ))

        # 5. 쿨다운 (같은 종목 연속 매매 간격)
        cool_down = self.rules.get("cool_down_minutes", 5)
        last_trade = self._last_trade_time.get(ticker)
        if last_trade:
            elapsed = (datetime.now() - last_trade).total_seconds() / 60
            if elapsed < cool_down:
                violations.append(SafetyViolation(
                    "cool_down_minutes",
                    f"{ticker} 마지막 거래 후 {elapsed:.1f}분 < 최소 {cool_down}분",
                    index,
                ))

        return violations

    def check_daily_loss(self, daily_pnl_pct: float) -> tuple[bool, str]:
        """일일 손실 한도 체크.

        Returns:
            (거래 가능 여부, 메시지)
        """
        max_loss = self.rules.get("max_daily_loss_pct", -0.03)
        if daily_pnl_pct <= max_loss:
            msg = f"일일 손실 한도 도달: {daily_pnl_pct:.2%} <= {max_loss:.2%}. 당일 거래 중단."
            logger.critical(msg)
            return False, msg
        return True, "OK"

    def check_emergency_stop(self, total_pnl_pct: float) -> tuple[bool, str]:
        """총자산 비상 손실 한도 체크.

        Returns:
            (거래 가능 여부, 메시지)
        """
        emergency = self.rules.get("emergency_stop_loss_pct", -0.10)
        if total_pnl_pct <= emergency:
            msg = f"비상 손실 한도 도달: {total_pnl_pct:.2%} <= {emergency:.2%}. 전체 시스템 중단."
            logger.critical(msg)
            return False, msg
        return True, "OK"

    def needs_confirmation(self, order_amount: float) -> bool:
        """Telegram 확인 필요 여부."""
        threshold = self.rules.get("require_confirmation_above", 3_000_000)
        return order_amount >= threshold

    def should_use_limit_order(self, order_amount: float) -> bool:
        """시장가 주문 한도 초과 → 지정가 강제."""
        max_market = self.rules.get("market_order_max_amount", 1_000_000)
        return order_amount > max_market

    def record_trade(self, ticker: str) -> None:
        """매매 실행 시각 기록 (쿨다운용)."""
        self._last_trade_time[ticker] = datetime.now()

    def filter_actions(
        self,
        signal: dict,
        portfolio: dict,
        total_asset: float,
    ) -> dict:
        """위반 액션을 제거하고 안전한 액션만 남긴 시그널을 반환."""
        passed, violations = self.validate_signal(signal, portfolio, total_asset)
        if passed:
            return signal

        violated_indices = {v.action_index for v in violations}
        filtered_actions = [
            action for i, action in enumerate(signal.get("actions", []))
            if i not in violated_indices
        ]

        filtered_signal = {**signal}
        filtered_signal["actions"] = filtered_actions
        filtered_signal["safety_filtered"] = [
            {"index": v.action_index, "rule": v.rule, "message": v.message}
            for v in violations
        ]

        return filtered_signal
