"""백테스팅 엔진 — 과거 데이터 기반 전략 검증."""

import numpy as np
import pandas as pd
from loguru import logger

from core.indicators import calculate_all_indicators


class Backtester:
    """규칙 기반 + ML 시그널 백테스팅."""

    def __init__(self, initial_capital: float = 10_000_000):
        self.initial_capital = initial_capital

    def run(
        self,
        df: pd.DataFrame,
        strategy_params: dict,
        ml_predictions: pd.Series | None = None,
    ) -> dict:
        """백테스트 실행.

        Args:
            df: 지표가 포함된 OHLCV DataFrame
            strategy_params: 전략 파라미터
            ml_predictions: ML 예측 라벨 Series (optional)

        Returns:
            백테스트 결과 딕셔너리
        """
        if df.empty or len(df) < 30:
            return {"error": "insufficient_data"}

        df = df.copy().reset_index(drop=True)

        capital = self.initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity_curve = []

        take_profit = strategy_params.get("take_profit_pct", 0.15)
        stop_loss = strategy_params.get("stop_loss_pct", -0.05)
        position_size = strategy_params.get("position_size_pct", 0.10)
        rsi_oversold = strategy_params.get("indicators", {}).get("rsi_oversold", 35)
        rsi_overbought = strategy_params.get("indicators", {}).get("rsi_overbought", 70)

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            price = row["close"]

            # 현재 포트폴리오 가치
            portfolio_value = capital + position * price
            equity_curve.append({
                "date": row.get("date", str(i)),
                "value": portfolio_value,
            })

            if position > 0:
                # 보유 중 — 매도 시그널 체크
                pnl_pct = (price - entry_price) / entry_price

                # 익절
                if pnl_pct >= take_profit:
                    capital += position * price
                    trades.append({
                        "date": row.get("date", ""),
                        "action": "SELL",
                        "price": price,
                        "quantity": position,
                        "pnl_pct": pnl_pct,
                        "reason": "take_profit",
                    })
                    position = 0
                    continue

                # 손절
                if pnl_pct <= stop_loss:
                    capital += position * price
                    trades.append({
                        "date": row.get("date", ""),
                        "action": "SELL",
                        "price": price,
                        "quantity": position,
                        "pnl_pct": pnl_pct,
                        "reason": "stop_loss",
                    })
                    position = 0
                    continue

                # 데드크로스 매도
                if row.get("dead_cross", False):
                    capital += position * price
                    trades.append({
                        "date": row.get("date", ""),
                        "action": "SELL",
                        "price": price,
                        "quantity": position,
                        "pnl_pct": pnl_pct,
                        "reason": "dead_cross",
                    })
                    position = 0
                    continue

            else:
                # 미보유 — 매수 시그널 체크
                buy_signal = False

                # RSI 과매도 + 거래량 급증
                rsi = row.get("rsi", 50)
                vol_ratio = row.get("volume_ratio", 1)
                if rsi <= rsi_oversold and vol_ratio >= 2.0:
                    buy_signal = True

                # 골든크로스
                if row.get("golden_cross", False):
                    buy_signal = True

                # ML 예측이 있으면 참고
                if ml_predictions is not None and i < len(ml_predictions):
                    if ml_predictions.iloc[i] == 2:  # 상승 예측
                        buy_signal = True

                if buy_signal:
                    invest_amount = portfolio_value * position_size
                    quantity = int(invest_amount / price)
                    if quantity > 0:
                        cost = quantity * price
                        capital -= cost
                        position = quantity
                        entry_price = price
                        trades.append({
                            "date": row.get("date", ""),
                            "action": "BUY",
                            "price": price,
                            "quantity": quantity,
                            "reason": f"rsi={rsi:.0f}, vol_ratio={vol_ratio:.1f}",
                        })

        # 최종 결산
        final_value = capital + position * df.iloc[-1]["close"]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 성과 지표 계산
        equity_values = [e["value"] for e in equity_curve]
        returns = pd.Series(equity_values).pct_change().dropna()

        winning_trades = [t for t in trades if t.get("pnl_pct", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl_pct", 0) < 0]
        win_rate = len(winning_trades) / len([t for t in trades if "pnl_pct" in t]) if trades else 0

        max_drawdown = 0
        peak = self.initial_capital
        for v in equity_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0

        return {
            "initial_capital": self.initial_capital,
            "final_value": round(final_value),
            "total_return": round(total_return, 4),
            "total_trades": len(trades),
            "win_rate": round(win_rate, 4),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe_ratio": round(sharpe, 4),
            "trades": trades,
            "equity_curve": equity_curve[-30:],  # 마지막 30개만
        }
