"""백테스트 성과지표 계산."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_metrics(
    equity_curve: list[dict],
    trades: list,
    initial_capital: float,
    benchmark_equity: list[dict] | None = None,
) -> dict:
    """종합 백테스트 성과지표 계산.

    Args:
        equity_curve: [{"date": ..., "value": ...}, ...]
        trades: BacktestTrade 리스트 (pnl, pnl_pct, hold_days 등 포함)
        initial_capital: 초기 자본
        benchmark_equity: 벤치마크 자산곡선 (선택)
    """
    if not equity_curve:
        return {"error": "no_data"}

    values = pd.Series([e["value"] for e in equity_curve])
    dates = [e["date"] for e in equity_curve]
    trading_days = len(values)

    final_value = values.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # 연환산 수익률
    if trading_days > 1:
        annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / trading_days) - 1
    else:
        annualized_return = 0.0

    # 일간 수익률
    daily_returns = values.pct_change().dropna()

    # MDD
    mdd, mdd_peak_date, mdd_trough_date = calculate_max_drawdown(values, dates)

    # 샤프/소르티노
    sharpe = calculate_sharpe(daily_returns)
    sortino = calculate_sortino(daily_returns)

    # 칼마
    calmar = annualized_return / mdd if mdd > 0 else 0.0

    # 매매 통계
    sell_trades = [t for t in trades if getattr(t, "action", "") == "SELL"]
    winning = [t for t in sell_trades if getattr(t, "pnl", 0) > 0]
    losing = [t for t in sell_trades if getattr(t, "pnl", 0) < 0]
    total_sell = len(sell_trades)

    win_rate = len(winning) / total_sell if total_sell > 0 else 0.0
    profit_factor = calculate_profit_factor(sell_trades)

    avg_hold = np.mean([getattr(t, "hold_days", 0) for t in sell_trades]) if sell_trades else 0.0
    avg_win_pct = np.mean([getattr(t, "pnl_pct", 0) for t in winning]) if winning else 0.0
    avg_loss_pct = np.mean([getattr(t, "pnl_pct", 0) for t in losing]) if losing else 0.0

    max_wins, max_losses = calculate_win_streak(sell_trades)

    result = {
        "initial_capital": initial_capital,
        "final_value": round(final_value),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": mdd,
        "mdd_peak_date": mdd_peak_date,
        "mdd_trough_date": mdd_trough_date,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "total_trades": len(trades),
        "total_sells": total_sell,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_hold_days": avg_hold,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "trading_days": trading_days,
        "daily_return_mean": float(daily_returns.mean()) if len(daily_returns) > 0 else 0,
        "daily_return_std": float(daily_returns.std()) if len(daily_returns) > 0 else 0,
    }

    # 벤치마크 비교
    if benchmark_equity:
        bm_values = pd.Series([e["value"] for e in benchmark_equity])
        bm_return = (bm_values.iloc[-1] - bm_values.iloc[0]) / bm_values.iloc[0]
        result["benchmark_return"] = bm_return
        result["excess_return"] = total_return - bm_return

    return result


def calculate_max_drawdown(
    values: pd.Series, dates: list | None = None,
) -> tuple[float, str, str]:
    """최대낙폭(MDD) 계산.

    Returns: (mdd_pct, peak_date, trough_date)
    """
    peak = values.expanding().max()
    drawdown = (values - peak) / peak
    mdd = abs(drawdown.min())

    if dates and len(dates) == len(values):
        trough_idx = drawdown.idxmin()
        peak_idx = values.iloc[:trough_idx + 1].idxmax()
        peak_date = dates[peak_idx] if peak_idx < len(dates) else ""
        trough_date = dates[trough_idx] if trough_idx < len(dates) else ""
    else:
        peak_date, trough_date = "", ""

    return float(mdd), peak_date, trough_date


def calculate_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.035) -> float:
    """연환산 샤프비율. 무위험이자율 기본값: 한국 10년국채 ~3.5%."""
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_sortino(daily_returns: pd.Series, risk_free_rate: float = 0.035) -> float:
    """소르티노비율 — 하방편차만 사용."""
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 0
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_profit_factor(sell_trades: list) -> float:
    """수익팩터 = 총이익 / 총손실."""
    gross_profit = sum(getattr(t, "pnl", 0) for t in sell_trades if getattr(t, "pnl", 0) > 0)
    gross_loss = abs(sum(getattr(t, "pnl", 0) for t in sell_trades if getattr(t, "pnl", 0) < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calculate_win_streak(sell_trades: list) -> tuple[int, int]:
    """최대 연승/연패 횟수."""
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0

    for t in sell_trades:
        if getattr(t, "pnl", 0) > 0:
            cur_wins += 1
            cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        elif getattr(t, "pnl", 0) < 0:
            cur_losses += 1
            cur_wins = 0
            max_losses = max(max_losses, cur_losses)
        else:
            cur_wins = cur_losses = 0

    return max_wins, max_losses


def calculate_buy_and_hold(
    df: pd.DataFrame, initial_capital: float,
) -> list[dict]:
    """단일 종목 바이앤홀드 벤치마크 자산곡선 생성."""
    if df.empty:
        return []

    first_price = df.iloc[0]["close"]
    shares = int(initial_capital / first_price)
    remaining_cash = initial_capital - shares * first_price

    equity = []
    for _, row in df.iterrows():
        value = shares * row["close"] + remaining_cash
        equity.append({"date": row["date"], "value": value})

    return equity
