"""파라미터 그리드 서치 + Walk-Forward 최적화."""

import itertools
from typing import Callable

from loguru import logger

from simulation.backtest import Backtester, BacktestResult


def grid_search(
    backtester: Backtester,
    tickers: list[str],
    strategy_factory: Callable[[dict], any],
    param_grid: dict[str, list],
    start_date: str,
    end_date: str,
    indicator_params: dict | None = None,
    sort_by: str = "sharpe_ratio",
) -> list[dict]:
    """파라미터 조합 전수 테스트.

    Args:
        strategy_factory: params dict를 받아 Strategy를 반환하는 함수
        param_grid: {"rsi_oversold": [25,30,35], "take_profit_pct": [0.05,0.10]}
        sort_by: 정렬 기준 지표 (sharpe_ratio, total_return, etc.)

    Returns:
        정렬된 결과 리스트 [{"params": {...}, "metrics": {...}, "result": BacktestResult}, ...]
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total_combos = 1
    for v in values:
        total_combos *= len(v)

    logger.info(f"Grid search: {total_combos} combinations")
    results = []

    for i, combo in enumerate(itertools.product(*values), 1):
        params = dict(zip(keys, combo))
        strategy = strategy_factory(params)

        bt_result = backtester.run(
            tickers, strategy, start_date, end_date, indicator_params,
        )

        m = bt_result.metrics
        sharpe = m.get("sharpe_ratio", 0)
        ret = m.get("total_return", 0)

        results.append({
            "params": params,
            "metrics": m,
            "result": bt_result,
        })

        if i % 10 == 0 or i == total_combos:
            logger.info(f"Grid search progress: {i}/{total_combos}")

    # 정렬 (내림차순)
    results.sort(key=lambda r: r["metrics"].get(sort_by, 0), reverse=True)

    if results:
        best = results[0]
        logger.info(
            f"Best params: {best['params']} | "
            f"{sort_by}={best['metrics'].get(sort_by, 0):.4f}"
        )

    return results


def walk_forward(
    backtester: Backtester,
    tickers: list[str],
    strategy_factory: Callable[[dict], any],
    param_grid: dict[str, list],
    start_date: str,
    end_date: str,
    in_sample_days: int = 252,
    out_of_sample_days: int = 63,
    step_days: int = 63,
    indicator_params: dict | None = None,
    sort_by: str = "sharpe_ratio",
) -> dict:
    """Walk-Forward 최적화.

    1. in_sample 기간에서 최적 파라미터 탐색
    2. out_of_sample 기간에서 검증
    3. step_days만큼 윈도우 이동 후 반복

    Returns:
        {"windows": [...], "combined_oos_metrics": {...}, "overfit_ratio": float}
    """
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(start_date.replace("-", ""), "%Y%m%d")
    end_dt = datetime.strptime(end_date.replace("-", ""), "%Y%m%d")

    windows = []
    current_start = start_dt

    while True:
        is_end = current_start + timedelta(days=int(in_sample_days * 1.5))
        oos_start = is_end + timedelta(days=1)
        oos_end = oos_start + timedelta(days=int(out_of_sample_days * 1.5))

        if oos_end > end_dt:
            break

        is_start_str = current_start.strftime("%Y%m%d")
        is_end_str = is_end.strftime("%Y%m%d")
        oos_start_str = oos_start.strftime("%Y%m%d")
        oos_end_str = oos_end.strftime("%Y%m%d")

        logger.info(f"Walk-forward window: IS={is_start_str}~{is_end_str}, OOS={oos_start_str}~{oos_end_str}")

        # In-sample 최적화
        is_results = grid_search(
            backtester, tickers, strategy_factory, param_grid,
            is_start_str, is_end_str, indicator_params, sort_by,
        )

        if not is_results:
            current_start += timedelta(days=int(step_days * 1.5))
            continue

        best_params = is_results[0]["params"]
        is_metrics = is_results[0]["metrics"]

        # Out-of-sample 검증
        oos_strategy = strategy_factory(best_params)
        oos_result = backtester.run(
            tickers, oos_strategy, oos_start_str, oos_end_str, indicator_params,
        )

        windows.append({
            "is_period": f"{is_start_str}~{is_end_str}",
            "oos_period": f"{oos_start_str}~{oos_end_str}",
            "best_params": best_params,
            "is_sharpe": is_metrics.get("sharpe_ratio", 0),
            "oos_sharpe": oos_result.metrics.get("sharpe_ratio", 0),
            "is_return": is_metrics.get("total_return", 0),
            "oos_return": oos_result.metrics.get("total_return", 0),
        })

        current_start += timedelta(days=int(step_days * 1.5))

    # 종합 결과
    if not windows:
        return {"windows": [], "combined_oos_metrics": {}, "overfit_ratio": 0}

    avg_is_sharpe = sum(w["is_sharpe"] for w in windows) / len(windows)
    avg_oos_sharpe = sum(w["oos_sharpe"] for w in windows) / len(windows)
    overfit_ratio = avg_is_sharpe / avg_oos_sharpe if avg_oos_sharpe != 0 else float("inf")

    return {
        "windows": windows,
        "avg_is_sharpe": avg_is_sharpe,
        "avg_oos_sharpe": avg_oos_sharpe,
        "overfit_ratio": overfit_ratio,
        "window_count": len(windows),
    }


def train_test_split(
    backtester: Backtester,
    tickers: list[str],
    strategy_factory: Callable[[dict], any],
    param_grid: dict[str, list],
    start_date: str,
    end_date: str,
    train_ratio: float = 0.7,
    indicator_params: dict | None = None,
    sort_by: str = "sharpe_ratio",
) -> dict:
    """단순 70:30 학습/검증 분할.

    Returns:
        {"best_params": dict, "train_metrics": dict, "test_metrics": dict, "overfit_ratio": float}
    """
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(start_date.replace("-", ""), "%Y%m%d")
    end_dt = datetime.strptime(end_date.replace("-", ""), "%Y%m%d")
    total_days = (end_dt - start_dt).days
    split_dt = start_dt + timedelta(days=int(total_days * train_ratio))

    train_end = split_dt.strftime("%Y%m%d")
    test_start = (split_dt + timedelta(days=1)).strftime("%Y%m%d")

    logger.info(f"Train/Test split: Train={start_date}~{train_end}, Test={test_start}~{end_date}")

    # Train 구간 최적화
    train_results = grid_search(
        backtester, tickers, strategy_factory, param_grid,
        start_date, train_end, indicator_params, sort_by,
    )

    if not train_results:
        return {"error": "no_train_results"}

    best_params = train_results[0]["params"]
    train_metrics = train_results[0]["metrics"]

    # Test 구간 검증
    test_strategy = strategy_factory(best_params)
    test_result = backtester.run(
        tickers, test_strategy, test_start, end_date, indicator_params,
    )
    test_metrics = test_result.metrics

    train_sharpe = train_metrics.get("sharpe_ratio", 0)
    test_sharpe = test_metrics.get("sharpe_ratio", 0)
    overfit_ratio = train_sharpe / test_sharpe if test_sharpe != 0 else float("inf")

    return {
        "best_params": best_params,
        "train_period": f"{start_date}~{train_end}",
        "test_period": f"{test_start}~{end_date}",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "overfit_ratio": overfit_ratio,
    }
