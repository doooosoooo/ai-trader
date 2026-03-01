#!/usr/bin/env python3
"""백테스팅 CLI.

사용법:
    # 단일 전략 백테스트
    python run_backtest.py --strategy swing --start 20240301

    # 전략 비교
    python run_backtest.py --compare --start 20240301

    # 파라미터 최적화
    python run_backtest.py --optimize --strategy swing --start 20240301

    # 과거 데이터 수집
    python run_backtest.py --collect-data --start 20240301

    # 데이터 현황 확인
    python run_backtest.py --data-status
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# 로깅 설정
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")

DB_PATH = str(Path(__file__).parent / "data" / "storage" / "trader.db")
DEFAULT_TICKERS = ["005930", "000660", "373220", "006400"]


def cmd_data_status():
    """DB 데이터 현황 확인."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT ticker, COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date "
            "FROM daily_ohlcv GROUP BY ticker ORDER BY ticker"
        ).fetchall()

    if not rows:
        print("데이터 없음")
        return

    print(f"\n{'종목':>8} | {'건수':>5} | {'시작일':>10} | {'종료일':>10}")
    print("-" * 45)
    for ticker, cnt, min_d, max_d in rows:
        print(f"{ticker:>8} | {cnt:>5} | {min_d:>10} | {max_d:>10}")
    print()


def cmd_collect_data(tickers: list[str], start_date: str, end_date: str):
    """과거 데이터 대량 수집."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / "config" / ".env")

    from core.market_data import KISAuth, MarketDataClient
    from core.config_manager import ConfigManager
    from data.collectors.price_collector import PriceCollector

    config = ConfigManager()
    auth = KISAuth(config.settings)
    market = MarketDataClient(auth)
    collector = PriceCollector(market)

    print(f"\n수집 대상: {tickers}")
    print(f"기간: {start_date} ~ {end_date}\n")

    for ticker in tickers:
        print(f"수집 중: {ticker}...", end=" ", flush=True)
        df = collector.collect_historical(ticker, start_date, end_date)
        print(f"완료 ({len(df)}건)")

    print("\n수집 완료!")
    cmd_data_status()


def cmd_backtest(
    strategy_name: str, tickers: list[str],
    start_date: str, end_date: str, capital: float,
):
    """단일 전략 백테스트 실행."""
    from simulation.backtest import Backtester
    from simulation.strategies import STRATEGY_REGISTRY
    from simulation.report import print_console_report

    if strategy_name not in STRATEGY_REGISTRY:
        print(f"지원 전략: {list(STRATEGY_REGISTRY.keys())}")
        return

    strategy = STRATEGY_REGISTRY[strategy_name]()
    bt = Backtester(initial_capital=capital, db_path=DB_PATH)
    result = bt.run(tickers, strategy, start_date, end_date)
    print_console_report(result)
    result.save_to_db(DB_PATH)


def cmd_compare(tickers: list[str], start_date: str, end_date: str, capital: float):
    """3개 전략 비교."""
    from simulation.backtest import Backtester
    from simulation.strategies import STRATEGY_REGISTRY
    from simulation.report import print_console_report, print_comparison_report

    bt = Backtester(initial_capital=capital, db_path=DB_PATH)
    results = []

    for name, factory in STRATEGY_REGISTRY.items():
        strategy = factory()
        result = bt.run(tickers, strategy, start_date, end_date)
        results.append(result)
        result.save_to_db(DB_PATH)

    print_comparison_report(results)

    # 각 전략 상세 결과
    for r in results:
        print_console_report(r)


def cmd_optimize(
    strategy_name: str, tickers: list[str],
    start_date: str, end_date: str, capital: float,
):
    """파라미터 최적화."""
    from simulation.backtest import Backtester
    from simulation.strategies import STRATEGY_REGISTRY
    from simulation.optimizer import grid_search, train_test_split
    from simulation.report import print_console_report

    if strategy_name not in STRATEGY_REGISTRY:
        print(f"지원 전략: {list(STRATEGY_REGISTRY.keys())}")
        return

    factory = STRATEGY_REGISTRY[strategy_name]

    # 기본 그리드
    param_grid = {
        "rsi_oversold": [25, 30, 35, 40],
        "take_profit_pct": [0.03, 0.05, 0.08, 0.12, 0.15],
        "stop_loss_pct": [0.02, 0.03, 0.05, 0.07],
        "position_size_pct": [0.05, 0.08, 0.10, 0.15],
    }

    bt = Backtester(initial_capital=capital, db_path=DB_PATH)

    # Train/Test 분할
    print("\n=== Train/Test 분할 최적화 (70:30) ===\n")
    split_result = train_test_split(
        bt, tickers, factory, param_grid, start_date, end_date,
    )

    if "error" in split_result:
        print(f"최적화 실패: {split_result['error']}")
        return

    print(f"최적 파라미터: {split_result['best_params']}")
    print(f"Train 기간: {split_result['train_period']}")
    print(f"  수익률: {split_result['train_metrics'].get('total_return', 0):+.2%}")
    print(f"  샤프: {split_result['train_metrics'].get('sharpe_ratio', 0):.2f}")
    print(f"Test 기간: {split_result['test_period']}")
    print(f"  수익률: {split_result['test_metrics'].get('total_return', 0):+.2%}")
    print(f"  샤프: {split_result['test_metrics'].get('sharpe_ratio', 0):.2f}")
    print(f"과적합 비율: {split_result['overfit_ratio']:.2f} (1.0에 가까울수록 좋음)")

    # 최적 파라미터로 전체 기간 백테스트
    print("\n=== 최적 파라미터로 전체 기간 백테스트 ===\n")
    best_strategy = factory(split_result["best_params"])
    best_result = bt.run(tickers, best_strategy, start_date, end_date)
    print_console_report(best_result)


def main():
    parser = argparse.ArgumentParser(description="AI Trader 백테스팅 도구")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backtest", action="store_true", help="단일 전략 백테스트")
    group.add_argument("--compare", action="store_true", help="3개 전략 비교")
    group.add_argument("--optimize", action="store_true", help="파라미터 최적화")
    group.add_argument("--collect-data", action="store_true", help="과거 데이터 수집")
    group.add_argument("--data-status", action="store_true", help="데이터 현황")

    parser.add_argument("--strategy", type=str, default="swing", help="전략 (swing/daytrading)")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS), help="종목코드 (콤마 구분)")
    parser.add_argument("--start", type=str, default="20240301", help="시작일 (YYYYMMDD)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y%m%d"), help="종료일 (YYYYMMDD)")
    parser.add_argument("--capital", type=float, default=10_000_000, help="초기자본 (원)")

    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]

    if args.data_status:
        cmd_data_status()
    elif args.collect_data:
        cmd_collect_data(tickers, args.start, args.end)
    elif args.backtest:
        cmd_backtest(args.strategy, tickers, args.start, args.end, args.capital)
    elif args.compare:
        cmd_compare(tickers, args.start, args.end, args.capital)
    elif args.optimize:
        cmd_optimize(args.strategy, tickers, args.start, args.end, args.capital)


if __name__ == "__main__":
    main()
