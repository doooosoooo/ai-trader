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
        # 재매수 쿨다운 면제 대상 (include_tickers) — main.py에서 set_include_tickers로 주입
        self._include_tickers: set[str] = set()
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

    def set_include_tickers(self, tickers: list[str]) -> None:
        """재매수 쿨다운 면제 종목 등록. include_tickers (screening-params.yaml)는
        강제 후보화 종목이라 손절-재매수-손절 사이클 위험이 낮고, 폭락장 회복
        기회를 놓치지 않기 위해 같은 종목/같은 섹터 쿨다운에서 면제."""
        self._include_tickers = set(tickers or [])

    def get_sector(self, ticker: str) -> str | None:
        """종목코드의 섹터 반환. 매핑 없으면 None."""
        return self._sector_map.get(ticker)

    def _last_sell_for(self, ticker: str) -> tuple[datetime, float] | None:
        """해당 종목의 가장 최근 '자발적' SELL 거래 (timestamp, price) 반환. 없으면 None.

        손절/트레일링/보유기간초과 같은 자동 매도는 제외 — 의도와 무관한 매도까지
        rebuy_cooldown으로 차단하면 좋은 재매수 기회를 놓침.
        """
        AUTO_SELL_KEYWORDS = ("하드손절", "트레일링스탑", "조기익절", "갭상승익절", "보유기간초과", "스크리닝탈락")
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT timestamp, price, reason FROM trade_history "
                    "WHERE ticker = ? AND action = 'SELL' "
                    "ORDER BY timestamp DESC LIMIT 20",
                    (ticker,),
                ).fetchall()
            for ts_str, price, reason in rows:
                reason_text = reason or ""
                if any(kw in reason_text for kw in AUTO_SELL_KEYWORDS):
                    continue
                return datetime.fromisoformat(ts_str), float(price)
            return None
        except Exception as e:
            logger.warning(f"_last_sell_for failed for {ticker}: {e}")
            return None

    def _last_sell_in_sector(self, sector: str) -> tuple[str, datetime] | None:
        """해당 섹터에서 가장 최근 '자발적' SELL 거래 (ticker, timestamp) 반환. 없으면 None.

        '자발적' 매도만 카운트 — 손절/트레일링/보유기간초과 같은 자동 매도는 제외.
        의도가 아닌 매도까지 cooldown 적용하면 적극 매매 차단 부작용 발생.
        """
        sector_tickers = [t for t, s in self._sector_map.items() if s == sector]
        if not sector_tickers:
            return None
        # 자동 매도 사유 키워드 — 이 키워드가 reason에 포함되면 cooldown 제외
        AUTO_SELL_KEYWORDS = ("하드손절", "트레일링스탑", "조기익절", "갭상승익절", "보유기간초과", "스크리닝탈락")
        try:
            placeholders = ",".join("?" * len(sector_tickers))
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    f"SELECT ticker, timestamp, reason FROM trade_history "
                    f"WHERE action = 'SELL' AND ticker IN ({placeholders}) "
                    f"ORDER BY timestamp DESC LIMIT 20",
                    sector_tickers,
                ).fetchall()
            for ticker, ts_str, reason in rows:
                reason_text = reason or ""
                if any(kw in reason_text for kw in AUTO_SELL_KEYWORDS):
                    continue  # 자동 매도는 제외
                return ticker, datetime.fromisoformat(ts_str)
            return None
        except Exception as e:
            logger.warning(f"_last_sell_in_sector failed for {sector}: {e}")
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

        # 사이클 레벨 검증: 동일 섹터 BUY 다중 진입 차단 (섹터 집중 방지)
        # 단, include_tickers 종목은 한도 계산에서 제외 (강제 후보 + 폭락장 분산 진입 허용)
        max_same_sector_buys = self.rules.get("max_same_sector_buys_per_cycle")
        if max_same_sector_buys and max_same_sector_buys > 0:
            sector_buys: dict[str, list[int]] = {}
            for i, action in enumerate(actions):
                if action.get("type", "").upper() != "BUY":
                    continue
                ticker = action.get("ticker", "")
                if ticker in self._include_tickers:
                    continue  # include_tickers는 섹터 한도 면제
                sector = self._sector_map.get(ticker)
                if not sector:
                    continue
                sector_buys.setdefault(sector, []).append(i)
            for sector, indices in sector_buys.items():
                if len(indices) <= max_same_sector_buys:
                    continue
                # ratio 큰 순서대로 우선순위 부여, 상위 max_same_sector_buys개만 통과
                sorted_idx = sorted(indices, key=lambda j: -actions[j].get("ratio", 0))
                for j in sorted_idx[max_same_sector_buys:]:
                    a = actions[j]
                    violations.append(SafetyViolation(
                        "max_same_sector_buys_per_cycle",
                        f"{sector} 섹터 동일 사이클 BUY {len(indices)}건 > 최대 {max_same_sector_buys}건 — {a.get('name', a.get('ticker', ''))} 차단 (낮은 비중 우선 차단)",
                        j,
                    ))

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
            # 1.5. 매수가 명시 필수 (buy-high 차단). swing/value는 limit_price 필수, daytrading은 예외
            strategy_type = action.get("strategy_type", "swing")
            if strategy_type in ("swing", "value"):
                limit_price = action.get("limit_price")
                if limit_price is None or (isinstance(limit_price, (int, float)) and limit_price <= 0):
                    violations.append(SafetyViolation(
                        "limit_price_required",
                        f"{strategy_type} 매수는 limit_price 필수 (시장가 추격매수 차단)",
                        index,
                    ))

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

            # 3.6. 동일 섹터 매도 후 재매수 쿨다운 (섹터 즉시 회전 차단)
            #     같은 섹터에서 최근 SELL이 있으면 N거래일간 신규 매수 금지.
            #     단, include_tickers 종목은 면제 (강제 후보 + 폭락장 회복 기회).
            sector_cooldown = self.rules.get("same_sector_rebuy_cooldown_days")
            if sector_cooldown and not is_existing and ticker not in self._include_tickers:
                target_sector = self._sector_map.get(ticker)
                if target_sector:
                    last_sell = self._last_sell_in_sector(target_sector)
                    if last_sell:
                        sold_ticker, sold_ts = last_sell
                        days_since = _trading_days_between(sold_ts, datetime.now())
                        if days_since < sector_cooldown:
                            violations.append(SafetyViolation(
                                "same_sector_rebuy_cooldown",
                                f"{target_sector} 섹터 {sold_ticker} 매도 후 {days_since}거래일 < {sector_cooldown}일 쿨다운",
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
            #     LLM 프롬프트(active.md) 규칙을 코드 레벨에서 강제.
            cooldown_days = self.rules.get("rebuy_cooldown_trading_days", 2)
            last_sell = self._last_sell_for(ticker)
            if last_sell is not None:
                sell_ts, sell_price = last_sell
                elapsed_days = _trading_days_between(sell_ts, datetime.now())
                is_include = ticker in self._include_tickers
                # 거래일 쿨다운 — include_tickers는 면제 (강제 후보, 폭락장 회복 기회)
                if not is_include and elapsed_days < cooldown_days:
                    violations.append(SafetyViolation(
                        "rebuy_cooldown",
                        f"{ticker} 매도({sell_ts.strftime('%m-%d %H:%M')}) "
                        f"후 {elapsed_days}거래일 < 쿨다운 {cooldown_days}거래일",
                        index,
                    ))
                # 같은 날 재매수 차단 — include_tickers에도 적용 (단타 추격 방지)
                same_day = sell_ts.date() == datetime.now().date()
                if same_day:
                    violations.append(SafetyViolation(
                        "same_day_rebuy",
                        f"{ticker} 당일 매도({sell_ts.strftime('%H:%M')} "
                        f"@{sell_price:,.0f}) 후 재매수 차단",
                        index,
                    ))

        # 4.4. 보유하지 않은 종목 SELL 차단 — LLM 환각 방지.
        #      Why: LLM이 워치리스트의 약세 종목을 "rotation 대상"으로 잘못 분류해 SELL 액션 생성.
        #      executor가 "No position to sell"로 차단하지만 그 전에 텔레그램 알림이 발송돼 사용자 혼란.
        if action_type == "SELL":
            held = portfolio.get("positions", {})
            if ticker not in held:
                violations.append(SafetyViolation(
                    "sell_without_position",
                    f"{ticker} 미보유 종목 SELL 시도 — LLM 환각 또는 sync 지연 (보유 종목만 매도 가능)",
                    index,
                ))

        # 4.5. 최소 보유 30분 룰: 매수 직후 LLM 자가매도 차단.
        #      Why: swing_pullback으로 산 종목을 다음 사이클이 swing 표준 잣대로 평가해 청산하는 모순 사례 발생(2026-05-13 NAVER 12분 매도).
        #      자동 매도(손절/트레일링/보유기간초과)는 reason 키워드로 식별해 통과.
        if action_type == "SELL":
            pos_data = portfolio.get("positions", {}).get(ticker, {})
            if pos_data:
                reason_text = action.get("reason", "")
                AUTO_SELL_KEYWORDS = ("하드손절", "트레일링스탑", "조기익절", "갭상승익절", "보유기간초과", "스크리닝탈락")
                is_auto_sell = any(kw in reason_text for kw in AUTO_SELL_KEYWORDS)
                if not is_auto_sell:
                    bought_at_str = pos_data.get("bought_at", "")
                    if bought_at_str:
                        try:
                            bought_at = datetime.fromisoformat(bought_at_str)
                            held_minutes = (datetime.now() - bought_at).total_seconds() / 60
                            min_hold_min = self.rules.get("min_hold_minutes_after_buy", 30)
                            if held_minutes < min_hold_min:
                                violations.append(SafetyViolation(
                                    "min_hold_minutes",
                                    f"{ticker} 매수 후 {held_minutes:.1f}분 < 최소 {min_hold_min}분 — LLM 자가매도 차단",
                                    index,
                                ))
                        except (ValueError, TypeError):
                            pass

        # 4.6. LLM 선제 매도 차단: 손실 중인데 손절선 미도달인 SELL은 거부.
        #      Why: LLM이 dist_to_stop_loss를 "임박"으로 해석해 -3.5~-4%에서 선제 매도하던 패턴 차단.
        #      익절(수익 중)/실제 손절 도달은 통과.
        if action_type == "SELL":
            pos_data = portfolio.get("positions", {}).get(ticker, {})
            if pos_data:
                pnl_str = str(pos_data.get("pnl_pct", "0%")).strip().rstrip("%")
                try:
                    pnl_pct_val = float(pnl_str) / 100
                except (ValueError, TypeError):
                    pnl_pct_val = 0.0
                from core.portfolio import Position
                strategy_type = pos_data.get("strategy_type", "swing")
                stop_loss = Position.STRATEGY_RULES.get(
                    strategy_type, Position.STRATEGY_RULES["swing"]
                )["stop_loss"]
                if pnl_pct_val < 0 and pnl_pct_val > stop_loss:
                    violations.append(SafetyViolation(
                        "premature_sell_block",
                        f"{ticker} 손익 {pnl_pct_val:.2%} > 손절선 {stop_loss:.0%} "
                        f"— 손절선 미도달 선제 매도 차단",
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
