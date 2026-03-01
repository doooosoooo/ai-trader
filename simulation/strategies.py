"""규칙 기반 전략 정의 — 백테스팅용 조건 조합 시스템."""

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass
class Condition:
    """단일 진입/청산 조건."""

    name: str
    evaluate: Callable[[pd.Series, pd.Series, dict], bool]


@dataclass
class Strategy:
    """조합 가능한 규칙 기반 전략.

    진입: 모든 entry_conditions가 True (AND)
    청산: 하나라도 exit_condition이 True (OR)
    """

    name: str
    entry_conditions: list[Condition] = field(default_factory=list)
    exit_conditions: list[Condition] = field(default_factory=list)

    # 포지션 설정
    position_size_pct: float = 0.10
    max_positions: int = 5

    # 익절/손절
    take_profit_pct: float = 0.15
    stop_loss_pct: float = 0.05  # 양수 (적용 시 음수로 비교)
    trailing_stop_pct: float | None = None

    # 보유 기간
    min_hold_days: int = 1
    max_hold_days: int = 14

    def check_entry(self, row: pd.Series, prev: pd.Series, ctx: dict) -> bool:
        """모든 진입 조건이 충족되면 True."""
        if not self.entry_conditions:
            return False
        return all(c.evaluate(row, prev, ctx) for c in self.entry_conditions)

    def check_exit(self, row: pd.Series, prev: pd.Series, ctx: dict) -> tuple[bool, str]:
        """하나라도 청산 조건이 충족되면 (True, 사유) 반환."""
        for c in self.exit_conditions:
            if c.evaluate(row, prev, ctx):
                return True, c.name
        return False, ""

    def to_dict(self) -> dict:
        """전략 파라미터를 dict로 직렬화 (callable 제외)."""
        return {
            "name": self.name,
            "position_size_pct": self.position_size_pct,
            "max_positions": self.max_positions,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "min_hold_days": self.min_hold_days,
            "max_hold_days": self.max_hold_days,
            "entry_conditions": [c.name for c in self.entry_conditions],
            "exit_conditions": [c.name for c in self.exit_conditions],
        }


# ──────────────────────────────────────────────
# 조건 팩토리 함수 — 진입 조건
# ──────────────────────────────────────────────

def rsi_below(threshold: float) -> Condition:
    """RSI가 임계값 미만."""
    return Condition(
        name=f"rsi<{threshold}",
        evaluate=lambda row, prev, ctx: row.get("rsi", 50) < threshold,
    )


def rsi_above(threshold: float) -> Condition:
    """RSI가 임계값 초과."""
    return Condition(
        name=f"rsi>{threshold}",
        evaluate=lambda row, prev, ctx: row.get("rsi", 50) > threshold,
    )


def volume_surge(multiplier: float) -> Condition:
    """거래량이 20일 평균의 multiplier배 이상."""
    return Condition(
        name=f"vol_surge>={multiplier}x",
        evaluate=lambda row, prev, ctx: row.get("volume_ratio", 1) >= multiplier,
    )


def golden_cross() -> Condition:
    """골든크로스 발생 (MA5 > MA20 전환)."""
    return Condition(
        name="golden_cross",
        evaluate=lambda row, prev, ctx: bool(row.get("golden_cross", False)),
    )


def dead_cross() -> Condition:
    """데드크로스 발생 (MA5 < MA20 전환)."""
    return Condition(
        name="dead_cross",
        evaluate=lambda row, prev, ctx: bool(row.get("dead_cross", False)),
    )


def macd_bullish() -> Condition:
    """MACD 히스토그램 양전환 (매수 전환)."""
    return Condition(
        name="macd_bullish",
        evaluate=lambda row, prev, ctx: (
            row.get("macd_hist", 0) > 0 and prev.get("macd_hist", 0) <= 0
        ),
    )


def macd_bearish() -> Condition:
    """MACD 히스토그램 음전환 (매도 전환)."""
    return Condition(
        name="macd_bearish",
        evaluate=lambda row, prev, ctx: (
            row.get("macd_hist", 0) < 0 and prev.get("macd_hist", 0) >= 0
        ),
    )


def bb_below_lower() -> Condition:
    """볼린저밴드 하단 돌파."""
    return Condition(
        name="bb_below_lower",
        evaluate=lambda row, prev, ctx: row.get("close", 0) < row.get("bb_lower", float("inf")),
    )


def bb_above_upper() -> Condition:
    """볼린저밴드 상단 돌파."""
    return Condition(
        name="bb_above_upper",
        evaluate=lambda row, prev, ctx: row.get("close", 0) > row.get("bb_upper", float("inf")),
    )


def adx_trending(threshold: float = 25) -> Condition:
    """ADX가 임계값 이상 (추세 존재)."""
    return Condition(
        name=f"adx>={threshold}",
        evaluate=lambda row, prev, ctx: row.get("adx", 0) >= threshold,
    )


def price_above_ma(period: int) -> Condition:
    """현재가가 이동평균 위에 위치."""
    col = f"ma{period}"
    return Condition(
        name=f"price>ma{period}",
        evaluate=lambda row, prev, ctx: row.get("close", 0) > row.get(col, float("inf")),
    )


def price_below_ma(period: int) -> Condition:
    """현재가가 이동평균 아래에 위치."""
    col = f"ma{period}"
    return Condition(
        name=f"price<ma{period}",
        evaluate=lambda row, prev, ctx: row.get("close", 0) < row.get(col, 0),
    )


# ──────────────────────────────────────────────
# 조건 팩토리 함수 — 청산 조건
# ──────────────────────────────────────────────

def take_profit_reached(pct: float) -> Condition:
    """수익률이 목표에 도달."""
    return Condition(
        name=f"take_profit>={pct:.0%}",
        evaluate=lambda row, prev, ctx: ctx.get("pnl_pct", 0) >= pct,
    )


def stop_loss_reached(pct: float) -> Condition:
    """손실률이 한계에 도달 (pct는 양수, e.g. 0.05 = -5%)."""
    return Condition(
        name=f"stop_loss<=-{pct:.0%}",
        evaluate=lambda row, prev, ctx: ctx.get("pnl_pct", 0) <= -pct,
    )


def trailing_stop_hit(pct: float) -> Condition:
    """고점 대비 pct 이상 하락 (트레일링 스톱)."""
    return Condition(
        name=f"trailing_stop>={pct:.0%}",
        evaluate=lambda row, prev, ctx: ctx.get("drawdown_from_peak", 0) >= pct,
    )


def max_hold_exceeded(days: int) -> Condition:
    """최대 보유 기간 초과."""
    return Condition(
        name=f"hold>{days}days",
        evaluate=lambda row, prev, ctx: ctx.get("hold_days", 0) >= days,
    )


# ──────────────────────────────────────────────
# 프리셋 전략 생성
# ──────────────────────────────────────────────

def create_swing_strategy(params: dict | None = None) -> Strategy:
    """스윙 전략: RSI 과매도 + 거래량 급증 + 골든크로스.

    config/strategies/swing.md 기반.
    """
    p = params or {}
    tp = p.get("take_profit_pct", 0.15)
    sl = p.get("stop_loss_pct", 0.05)
    max_days = p.get("max_hold_days", 14)

    return Strategy(
        name="swing",
        entry_conditions=[
            rsi_below(p.get("rsi_oversold", 35)),
            volume_surge(p.get("volume_surge_multiplier", 2.0)),
        ],
        exit_conditions=[
            take_profit_reached(tp),
            stop_loss_reached(sl),
            dead_cross(),
            max_hold_exceeded(max_days),
        ],
        position_size_pct=p.get("position_size_pct", 0.10),
        max_positions=p.get("max_positions", 5),
        take_profit_pct=tp,
        stop_loss_pct=sl,
        min_hold_days=p.get("min_hold_days", 3),
        max_hold_days=max_days,
    )


def create_daytrading_strategy(params: dict | None = None) -> Strategy:
    """단타(모멘텀 돌파) 전략: 상승추세 + 거래량 급증 진입.

    config/strategies/daytrading.md 기반.
    MA20 위 + Vol>=1.5x: 상승추세에서 거래량 돌파 시 진입.
    tp/sl = 4%/1.5% (손익비 2.7:1) — 승률보다 기대값 극대화.
    RSI 과매수(>70)는 청산 조건으로 활용.
    """
    p = params or {}
    tp = p.get("take_profit_pct", 0.04)
    sl = p.get("stop_loss_pct", 0.015)
    max_days = p.get("max_hold_days", 3)

    return Strategy(
        name="daytrading",
        entry_conditions=[
            price_above_ma(p.get("trend_ma_period", 20)),
            volume_surge(p.get("volume_surge_multiplier", 1.5)),
        ],
        exit_conditions=[
            take_profit_reached(tp),
            stop_loss_reached(sl),
            rsi_above(p.get("rsi_overbought", 70)),
            max_hold_exceeded(max_days),
        ],
        position_size_pct=p.get("position_size_pct", 0.05),
        max_positions=p.get("max_positions", 5),
        take_profit_pct=tp,
        stop_loss_pct=sl,
        min_hold_days=0,
        max_hold_days=max_days,
    )


def create_defensive_strategy(params: dict | None = None) -> Strategy:
    """방어적 전략: 극단적 과매도 + 볼린저밴드 하단.

    config/strategies/defensive.md 기반.
    """
    p = params or {}
    tp = p.get("take_profit_pct", 0.08)
    sl = p.get("stop_loss_pct", 0.03)
    max_days = p.get("max_hold_days", 20)

    return Strategy(
        name="defensive",
        entry_conditions=[
            rsi_below(p.get("rsi_oversold", 25)),
            bb_below_lower(),
        ],
        exit_conditions=[
            take_profit_reached(tp),
            stop_loss_reached(sl),
            rsi_above(p.get("rsi_overbought", 70)),
            max_hold_exceeded(max_days),
        ],
        position_size_pct=p.get("position_size_pct", 0.05),
        max_positions=p.get("max_positions", 3),
        take_profit_pct=tp,
        stop_loss_pct=sl,
        min_hold_days=p.get("min_hold_days", 1),
        max_hold_days=max_days,
    )


# 전략 이름 → 생성 함수 매핑
STRATEGY_REGISTRY = {
    "swing": create_swing_strategy,
    "daytrading": create_daytrading_strategy,
}
