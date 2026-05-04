"""Safety Guard — LLM 출력과 무관하게 반드시 지켜야 할 하드코딩 규칙."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import holidays
import yaml
from loguru import logger

_KR_HOLIDAYS = holidays.KR(years=range(2025, 2030))


def _trading_days_between(start: datetime, end: datetime) -> int:
    """start ~ end 사이의 거래일 수 (start 익일부터 end 까지, end 포함)."""
    if end < start:
        return 0
    days = 0
    cur = start.date() + timedelta(days=1)
    end_d = end.date()
    while cur <= end_d:
        if cur.weekday() < 5 and cur not in _KR_HOLIDAYS:
            days += 1
        cur += timedelta(days=1)
    return days


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

    def __init__(self, safety_rules: dict, trading_params: dict, db_path: str | None = None):
        self.rules = safety_rules
        self.trading_params = trading_params
        self._last_trade_time: dict[str, datetime] = {}  # ticker -> last trade time
        self._daily_pnl: float = 0.0
        self._daily_pnl_date: str = ""
        # 재매수 쿨다운 검사용 DB 경로 (trade_history 조회)
        self.db_path = db_path or str(
            Path(__file__).parent.parent / "data" / "storage" / "trader.db"
        )
        # 섹터 매핑 (ticker -> sector). 매핑 없는 종목은 섹터 캡 미적용
        self._sector_map: dict[str, str] = self._load_sector_map()

    def _load_sector_map(self) -> dict[str, str]:
        """config/sector-mapping.yaml 로드하여 ticker → sector dict 생성."""
        mapping_path = Path(__file__).parent.parent / "config" / "sector-mapping.yaml"
        if not mapping_path.exists():
            logger.warning(f"sector-mapping.yaml not found at {mapping_path}; sector cap disabled")
            return {}
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            sectors = data.get("sectors", {})
            ticker_to_sector: dict[str, str] = {}
            for sector_name, tickers in sectors.items():
                for t in tickers or []:
                    ticker_to_sector[str(t)] = sector_name
            logger.info(f"Loaded sector mapping: {len(ticker_to_sector)} tickers across {len(sectors)} sectors")
            return ticker_to_sector
        except Exception as e:
            logger.error(f"Failed to load sector-mapping.yaml: {e}")
            return {}

    def get_sector(self, ticker: str) -> str | None:
        """종목코드의 섹터 반환. 매핑 없으면 None."""
        return self._sector_map.get(ticker)

    def _last_sell_for(self, ticker: str) -> tuple[datetime, float] | None:
        """해당 종목의 가장 최근 SELL 거래 (timestamp, price) 반환. 없으면 None."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT timestamp, price FROM trade_history "
                    "WHERE ticker = ? AND action = 'SELL' "
                    "ORDER BY timestamp DESC LIMIT 1",
                    (ticker,),
                ).fetchone()
            if not row:
                return None
            ts = datetime.fromisoformat(row[0])
            return ts, float(row[1])
        except Exception as e:
            logger.warning(f"_last_sell_for failed for {ticker}: {e}")
            return None

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

        # 1. 거래 시간 확인 (09:00 ~ 15:30)
        if self.rules.get("trading_hours_only", True):
            now = datetime.now()
            hour = now.hour
            minute = now.minute
            # 09:00 이상
            after_open = hour > 9 or (hour == 9 and minute >= 0)
            # 15:30 이하
            before_close = hour < 15 or (hour == 15 and minute <= 30)
            in_market_hours = after_open and before_close
            if not in_market_hours:
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

            # 3.5. 동일 섹터 최대 보유 종목 수 (섹터 쏠림 방지)
            #     매핑에 등록된 종목만 적용. 신규 매수일 때만 카운트 증가하여 검사.
            max_sector = self.rules.get("max_sector_positions")
            if max_sector and not is_existing:
                target_sector = self._sector_map.get(ticker)
                if target_sector:
                    held_in_sector = sum(
                        1 for held_ticker in portfolio.get("positions", {})
                        if self._sector_map.get(held_ticker) == target_sector
                    )
                    if held_in_sector >= max_sector:
                        violations.append(SafetyViolation(
                            "max_sector_positions",
                            f"{target_sector} 섹터 보유 {held_in_sector}종목 >= 최대 {max_sector}종목",
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

            # 4.5. 재매수 쿨다운 — 매도 후 N거래일 내 동일 종목 재매수 차단
            #     LLM 프롬프트(active.md) 규칙을 코드 레벨에서 강제
            cooldown_days = self.rules.get("rebuy_cooldown_trading_days", 2)
            last_sell = self._last_sell_for(ticker)
            if last_sell is not None:
                sell_ts, sell_price = last_sell
                elapsed_days = _trading_days_between(sell_ts, datetime.now())
                if elapsed_days < cooldown_days:
                    violations.append(SafetyViolation(
                        "rebuy_cooldown",
                        f"{ticker} 매도({sell_ts.strftime('%m-%d %H:%M')}) "
                        f"후 {elapsed_days}거래일 < 쿨다운 {cooldown_days}거래일",
                        index,
                    ))
                # 매도가보다 높은 가격에 같은 날 재매수 금지 (active.md 규칙 D)
                same_day = sell_ts.date() == datetime.now().date()
                action_price = action.get("limit_price")
                # limit_price가 없으면 reference로 매도가 자체 비교는 못 하지만,
                # 같은 날 재매수 자체가 위험하므로 차단
                if same_day:
                    violations.append(SafetyViolation(
                        "same_day_rebuy",
                        f"{ticker} 당일 매도({sell_ts.strftime('%H:%M')} "
                        f"@{sell_price:,.0f}) 후 재매수 차단",
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
