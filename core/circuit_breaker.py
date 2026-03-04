"""서킷브레이커 — LLM/ML과 무관한 시스템 레벨 비상 정지."""

import json
import os
import psutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from loguru import logger

_STATE_FILE = Path(__file__).parent.parent / "data" / "storage" / "circuit_state.json"


class CircuitState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    HALTED = "halted"          # 신규 매수 중단
    EMERGENCY = "emergency"    # 전체 거래 중단


class CircuitTrigger(str, Enum):
    KOSPI_CRASH = "kospi_crash"
    API_FAILURE = "api_failure"
    LLM_FAILURE = "llm_failure"
    DAILY_LOSS = "daily_loss"
    EMERGENCY_LOSS = "emergency_loss"
    RESOURCE_ALERT = "resource_alert"
    MANUAL = "manual"


class CircuitBreaker:
    """시스템 레벨 비상 정지 관리."""

    def __init__(self, safety_rules: dict, notify_callback: Callable | None = None):
        self.safety_rules = safety_rules
        self.state = CircuitState.NORMAL
        self.triggers: list[dict] = []
        self._notify = notify_callback  # Telegram 알림 콜백
        self._api_failure_count = 0
        self._llm_failure_count = 0
        self._halted_at: datetime | None = None
        self._load_state()

    def _load_state(self) -> None:
        """영속화된 상태 복원 (PM2 재시작 시 알림 스팸 방지)."""
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text())
                self.state = CircuitState(data.get("state", "normal"))
                if data.get("halted_at"):
                    self._halted_at = datetime.fromisoformat(data["halted_at"])
                self.triggers = data.get("triggers", [])[-10:]
                logger.info(f"Circuit breaker state restored: {self.state.value}")
        except Exception as e:
            logger.warning(f"Circuit breaker state load failed, starting NORMAL: {e}")

    def _save_state(self) -> None:
        """현재 상태를 파일에 저장."""
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "state": self.state.value,
                "halted_at": self._halted_at.isoformat() if self._halted_at else None,
                "triggers": self.triggers[-10:],
                "updated_at": datetime.now().isoformat(),
            }
            _STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning(f"Circuit breaker state save failed: {e}")

    @property
    def is_trading_allowed(self) -> bool:
        """분석 + 매매 가능 여부. EMERGENCY에서만 전체 중단."""
        return self.state != CircuitState.EMERGENCY

    @property
    def is_buy_allowed(self) -> bool:
        return self.state in (CircuitState.NORMAL,)

    @property
    def is_sell_allowed(self) -> bool:
        # HALTED에서 매도 차단 — 급락장에서 LLM이 패닉 매도하는 것 방지
        return self.state in (CircuitState.NORMAL, CircuitState.WARNING)

    def check_kospi(self, kospi_change_pct: float) -> CircuitState:
        """코스피 급락 체크."""
        threshold = self.safety_rules.get("market_conditions", {}).get(
            "kospi_drop_threshold", -0.03
        )
        # safety_rules에서 더 엄격한 기준 적용
        if kospi_change_pct <= -0.03:
            self._trigger(
                CircuitTrigger.KOSPI_CRASH,
                CircuitState.HALTED,
                f"코스피 {kospi_change_pct:.2%} 급락 (기준: -3%)",
            )
        elif kospi_change_pct <= threshold:
            self._trigger(
                CircuitTrigger.KOSPI_CRASH,
                CircuitState.WARNING,
                f"코스피 {kospi_change_pct:.2%} 하락 (기준: {threshold:.1%})",
            )
        return self.state

    def record_api_failure(self) -> CircuitState:
        """KIS API 실패 기록."""
        self._api_failure_count += 1
        if self._api_failure_count >= 3:
            self._trigger(
                CircuitTrigger.API_FAILURE,
                CircuitState.EMERGENCY,
                f"KIS API 연속 실패 {self._api_failure_count}회",
            )
        return self.state

    def record_api_success(self) -> None:
        self._api_failure_count = 0

    def record_llm_failure(self) -> CircuitState:
        """LLM 파싱 실패 기록."""
        self._llm_failure_count += 1
        max_retry = self.safety_rules.get("max_llm_retry", 3)
        if self._llm_failure_count >= max_retry:
            self._trigger(
                CircuitTrigger.LLM_FAILURE,
                CircuitState.WARNING,
                f"LLM 파싱 연속 실패 {self._llm_failure_count}회 — 사이클 스킵",
            )
        return self.state

    def record_llm_success(self) -> None:
        self._llm_failure_count = 0

    def check_daily_loss(self, daily_pnl_pct: float) -> CircuitState:
        """일일 손실 한도 체크."""
        max_loss = self.safety_rules.get("max_daily_loss_pct", -0.03)
        if daily_pnl_pct <= max_loss:
            self._trigger(
                CircuitTrigger.DAILY_LOSS,
                CircuitState.HALTED,
                f"일일 손실 {daily_pnl_pct:.2%} (한도: {max_loss:.2%})",
            )
        return self.state

    def check_emergency_loss(self, total_pnl_pct: float) -> CircuitState:
        """총자산 비상 손실 한도 체크."""
        emergency = self.safety_rules.get("emergency_stop_loss_pct", -0.10)
        if total_pnl_pct <= emergency:
            self._trigger(
                CircuitTrigger.EMERGENCY_LOSS,
                CircuitState.EMERGENCY,
                f"총자산 손실 {total_pnl_pct:.2%} (비상 한도: {emergency:.2%})",
            )
        return self.state

    def check_system_resources(self) -> CircuitState:
        """서버 리소스 상태 체크."""
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            if mem.percent > 95:
                self._trigger(
                    CircuitTrigger.RESOURCE_ALERT,
                    CircuitState.EMERGENCY,
                    f"메모리 사용률 {mem.percent}% — 위험",
                )
            elif mem.percent > 85:
                self._trigger(
                    CircuitTrigger.RESOURCE_ALERT,
                    CircuitState.WARNING,
                    f"메모리 사용률 {mem.percent}% — 주의",
                )

            if disk.percent > 95:
                self._trigger(
                    CircuitTrigger.RESOURCE_ALERT,
                    CircuitState.EMERGENCY,
                    f"디스크 사용률 {disk.percent}% — 위험",
                )
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")

        return self.state

    def _trigger(self, trigger: CircuitTrigger, new_state: CircuitState, message: str) -> None:
        """서킷브레이커 발동."""
        # 더 심각한 상태로만 전환
        state_priority = {
            CircuitState.NORMAL: 0,
            CircuitState.WARNING: 1,
            CircuitState.HALTED: 2,
            CircuitState.EMERGENCY: 3,
        }

        state_changed = state_priority[new_state] > state_priority[self.state]
        if state_changed:
            old_state = self.state
            self.state = new_state
            self._halted_at = datetime.now()
            logger.critical(f"Circuit breaker: {old_state} → {new_state} | {message}")

        trigger_record = {
            "trigger": trigger.value,
            "state": new_state.value,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.triggers.append(trigger_record)

        # 상태 영속화
        if state_changed:
            self._save_state()

        # Telegram 알림 — state 변경 시에만 (반복 스팸 방지)
        if state_changed and self._notify:
            try:
                self._notify(
                    level="circuit_breaker",
                    message=f"🚨 서킷브레이커 발동\n상태: {new_state.value}\n원인: {message}",
                )
            except Exception as e:
                logger.error(f"Circuit breaker notification failed: {e}")

    def manual_reset(self) -> str:
        """사용자 수동 리셋."""
        old_state = self.state
        self.state = CircuitState.NORMAL
        self._api_failure_count = 0
        self._llm_failure_count = 0
        self._halted_at = None
        self._save_state()
        msg = f"서킷브레이커 수동 리셋: {old_state} → NORMAL"
        logger.info(msg)
        return msg

    def daily_reset(self) -> None:
        """일일 리셋 (장 시작 전). EMERGENCY는 수동 리셋만 가능."""
        if self.state != CircuitState.EMERGENCY:
            self.state = CircuitState.NORMAL
            self._api_failure_count = 0
            self._llm_failure_count = 0
            self._save_state()
            logger.info("Circuit breaker daily reset")

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "is_trading_allowed": self.is_trading_allowed,
            "is_buy_allowed": self.is_buy_allowed,
            "is_sell_allowed": self.is_sell_allowed,
            "api_failure_count": self._api_failure_count,
            "llm_failure_count": self._llm_failure_count,
            "halted_at": self._halted_at.isoformat() if self._halted_at else None,
            "recent_triggers": self.triggers[-5:] if self.triggers else [],
        }
