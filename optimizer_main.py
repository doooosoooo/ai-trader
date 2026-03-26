#!/usr/bin/env python3
"""AI Optimizer — 파라미터 최적화 전용 프로세스.

장 마감 후 자동으로 백테스트 기반 파라미터 최적화를 실행하고,
현재 라이브 파라미터 대비 개선된 조합을 발견하면 텔레그램으로 알림.

- 라이브 DB를 snapshot 복사하여 읽기 전용으로 사용
- KIS API 호출 없음 (데이터는 라이브 프로세스가 수집)
- 텔레그램 send-only (polling 없음, 라이브 프로세스와 충돌 방지)
"""

import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

# 로깅 설정
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
logger.add(
    "logs/optimizer.log",
    level="DEBUG",
    rotation="10 MB",
    retention="14 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
)

BASE_DIR = Path(__file__).parent
LIVE_DB = BASE_DIR / "data" / "storage" / "trader.db"
OPTIMIZER_DB = BASE_DIR / "data" / "storage" / "optimizer.db"
CONFIG_PATH = BASE_DIR / "config" / "optimizer-params.yaml"
TRADING_PARAMS_PATH = BASE_DIR / "config" / "trading-params.yaml"
RESULTS_PATH = BASE_DIR / "data" / "storage" / "optimizer_suggestions.json"


def load_config() -> dict:
    """optimizer-params.yaml 로드."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_trading_params() -> dict:
    """현재 라이브 trading-params.yaml 로드 (비교 기준)."""
    with open(TRADING_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def refresh_db_snapshot():
    """라이브 DB를 optimizer DB로 안전하게 복사 (SQLite backup API)."""
    if not LIVE_DB.exists():
        logger.error(f"Live DB not found: {LIVE_DB}")
        return False

    try:
        src = sqlite3.connect(str(LIVE_DB))
        dst = sqlite3.connect(str(OPTIMIZER_DB))
        src.backup(dst)
        dst.close()
        src.close()
        logger.info(f"DB snapshot refreshed: {OPTIMIZER_DB}")
        return True
    except Exception as e:
        logger.error(f"DB snapshot failed: {e}")
        return False


def get_available_tickers(config: dict) -> list[str]:
    """설정 파일에서 대상 종목 목록 가져오기."""
    tickers = config.get("tickers", [])
    if tickers:
        return tickers

    # 설정에 없으면 DB에서 데이터가 충분한 종목 조회
    with sqlite3.connect(str(OPTIMIZER_DB)) as conn:
        rows = conn.execute(
            "SELECT ticker, COUNT(*) as cnt FROM daily_ohlcv "
            "GROUP BY ticker HAVING cnt >= 60 ORDER BY cnt DESC"
        ).fetchall()
    return [r[0] for r in rows]


def build_current_params() -> dict:
    """현재 라이브 파라미터를 백테스트 파라미터 형식으로 변환."""
    tp = load_trading_params()
    return {
        "rsi_oversold": tp.get("indicators", {}).get("rsi_oversold", 35),
        "take_profit_pct": tp.get("take_profit_pct", 0.25),
        "stop_loss_pct": abs(tp.get("stop_loss_pct", -0.05)),
        "position_size_pct": tp.get("position_size_pct", 0.13),
        "max_hold_days": tp.get("holding_period_days", {}).get("max", 40),
    }


def send_telegram_suggestion(message: str, config: dict):
    """텔레그램 메시지 직접 전송 (send-only, polling 없음)."""
    try:
        import os
        import telegram

        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        if not token or not chat_id:
            logger.warning("Telegram not configured, skipping notification")
            return

        bot = telegram.Bot(token=token)
        import asyncio
        asyncio.run(bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="HTML",
        ))
        logger.info("Optimization suggestion sent via Telegram")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


def format_suggestion_message(
    strategy_name: str,
    current_params: dict,
    best_params: dict,
    current_metrics: dict,
    best_metrics: dict,
    overfit_ratio: float,
    period: str,
) -> str:
    """최적화 제안 텔레그램 메시지 포맷."""
    cur_ret = current_metrics.get("total_return", 0)
    best_ret = best_metrics.get("total_return", 0)
    cur_sharpe = current_metrics.get("sharpe_ratio", 0)
    best_sharpe = best_metrics.get("sharpe_ratio", 0)
    cur_mdd = current_metrics.get("max_drawdown", 0)
    best_mdd = best_metrics.get("max_drawdown", 0)

    msg = (
        f"📈 <b>파라미터 최적화 제안</b> [ai-optimizer]\n"
        f"━━━━━━━━━━━━━━\n"
        f"전략: {strategy_name}\n"
        f"기간: {period}\n\n"
        f"<b>현재 파라미터:</b>\n"
    )
    for k, v in current_params.items():
        if isinstance(v, float):
            msg += f"  {k}: {v:.2%}\n" if v < 1 else f"  {k}: {v}\n"
        else:
            msg += f"  {k}: {v}\n"

    msg += f"\n<b>제안 파라미터:</b>\n"
    for k, v in best_params.items():
        changed = " ⬅️" if current_params.get(k) != v else ""
        if isinstance(v, float):
            msg += f"  {k}: {v:.2%}{changed}\n" if v < 1 else f"  {k}: {v}{changed}\n"
        else:
            msg += f"  {k}: {v}{changed}\n"

    ret_emoji = "🟢" if best_ret > cur_ret else "🔴"
    sharpe_emoji = "🟢" if best_sharpe > cur_sharpe else "🔴"
    mdd_emoji = "🟢" if abs(best_mdd) < abs(cur_mdd) else "🔴"

    msg += (
        f"\n<b>성과 비교:</b>\n"
        f"  {ret_emoji} 수익률: {cur_ret:+.1%} → {best_ret:+.1%} ({best_ret - cur_ret:+.1%}p)\n"
        f"  {sharpe_emoji} 샤프: {cur_sharpe:.2f} → {best_sharpe:.2f} ({best_sharpe - cur_sharpe:+.2f})\n"
        f"  {mdd_emoji} MDD: {cur_mdd:.1%} → {best_mdd:.1%}\n"
        f"  과적합비율: {overfit_ratio:.2f}\n"
        f"\n⚠️ 적용하려면 trading-params.yaml을 수동 변경하세요."
    )
    return msg


def run_optimization(mode: str = "daily"):
    """최적화 사이클 실행.

    Args:
        mode: "daily" (빠른 탐색) 또는 "deep" (토요일 심층 탐색)
    """
    config = load_config()
    opt_config = config.get("optimization", {})
    thresholds = opt_config.get("thresholds", {})

    logger.info(f"=== Optimization cycle started (mode={mode}) ===")

    # 1. DB 스냅샷 갱신
    if not refresh_db_snapshot():
        return

    # 2. 종목 및 기간 설정
    tickers = get_available_tickers(config)
    if not tickers:
        logger.warning("No tickers available for optimization")
        return

    capital = opt_config.get("default_capital", 60_000_000)
    start_date = opt_config.get("default_start", "20240901")
    end_date = datetime.now().strftime("%Y%m%d")
    sort_by = opt_config.get("sort_by", "sharpe_ratio")
    period = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]} ~ {end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    logger.info(f"Tickers: {tickers[:5]}{'...' if len(tickers) > 5 else ''} | Period: {start_date}~{end_date}")

    # 3. 현재 파라미터 기준 백테스트
    from simulation.backtest import Backtester
    from simulation.strategies import STRATEGY_REGISTRY
    from simulation.optimizer import grid_search, train_test_split

    bt = Backtester(initial_capital=capital, db_path=str(OPTIMIZER_DB))
    current_params = build_current_params()

    # 4. 전략별 최적화
    grids = config.get("grids_deep" if mode == "deep" else "grids", {})
    strategies_to_test = ["swing"]
    if mode == "deep":
        strategies_to_test = list(grids.keys())

    for strategy_name in strategies_to_test:
        if strategy_name not in STRATEGY_REGISTRY:
            continue

        factory = STRATEGY_REGISTRY[strategy_name]
        param_grid = grids.get(strategy_name, {})
        if not param_grid:
            continue

        total_combos = 1
        for v in param_grid.values():
            total_combos *= len(v)
        logger.info(f"Strategy: {strategy_name} | Grid: {total_combos} combinations")

        # 현재 파라미터로 기준선 백테스트
        try:
            current_strategy = factory(current_params)
            current_result = bt.run(tickers, current_strategy, start_date, end_date)
            current_metrics = current_result.metrics
            logger.info(
                f"Baseline ({strategy_name}): return={current_metrics.get('total_return', 0):+.2%}, "
                f"sharpe={current_metrics.get('sharpe_ratio', 0):.2f}"
            )
        except Exception as e:
            logger.error(f"Baseline backtest failed: {e}")
            continue

        # Train/Test 분할 최적화
        try:
            split_result = train_test_split(
                bt, tickers, factory, param_grid,
                start_date, end_date, sort_by=sort_by,
            )

            if "error" in split_result:
                logger.warning(f"Optimization failed for {strategy_name}: {split_result['error']}")
                continue

            best_params = split_result["best_params"]
            test_metrics = split_result["test_metrics"]
            train_metrics = split_result["train_metrics"]
            overfit_ratio = split_result["overfit_ratio"]

            logger.info(
                f"Best params: {best_params} | "
                f"Train: {train_metrics.get('total_return', 0):+.2%} (sharpe {train_metrics.get('sharpe_ratio', 0):.2f}) | "
                f"Test: {test_metrics.get('total_return', 0):+.2%} (sharpe {test_metrics.get('sharpe_ratio', 0):.2f}) | "
                f"Overfit: {overfit_ratio:.2f}"
            )

        except Exception as e:
            logger.error(f"Optimization error for {strategy_name}: {e}")
            continue

        # 5. 개선 여부 판단
        min_sharpe_imp = thresholds.get("min_sharpe_improvement", 0.2)
        min_return_imp = thresholds.get("min_return_improvement", 0.03)
        max_overfit = thresholds.get("max_overfit_ratio", 3.0)
        min_test_ret = thresholds.get("min_test_return", 0.0)

        cur_sharpe = current_metrics.get("sharpe_ratio", 0)
        test_sharpe = test_metrics.get("sharpe_ratio", 0)
        test_return = test_metrics.get("total_return", 0)
        cur_return = current_metrics.get("total_return", 0)

        sharpe_improved = (test_sharpe - cur_sharpe) >= min_sharpe_imp
        return_improved = (test_return - cur_return) >= min_return_imp
        overfit_ok = 0 < overfit_ratio <= max_overfit
        test_positive = test_return > min_test_ret

        if (sharpe_improved or return_improved) and overfit_ok and test_positive:
            logger.info(f"✅ Improvement found for {strategy_name}! Sending suggestion...")

            msg = format_suggestion_message(
                strategy_name, current_params, best_params,
                current_metrics, test_metrics, overfit_ratio, period,
            )
            send_telegram_suggestion(msg, config)

            # 결과 저장
            import json
            suggestion = {
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy_name,
                "current_params": current_params,
                "suggested_params": best_params,
                "current_metrics": {k: v for k, v in current_metrics.items() if isinstance(v, (int, float))},
                "test_metrics": {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))},
                "overfit_ratio": overfit_ratio,
            }
            with open(RESULTS_PATH, "w") as f:
                json.dump(suggestion, f, indent=2, ensure_ascii=False)
        else:
            logger.info(
                f"No significant improvement for {strategy_name}. "
                f"Sharpe: {cur_sharpe:.2f}→{test_sharpe:.2f} (need +{min_sharpe_imp}), "
                f"Return: {cur_return:+.2%}→{test_return:+.2%} (need +{min_return_imp:.0%})"
            )

    logger.info("=== Optimization cycle complete ===")


def get_minute_bar_count() -> int:
    """optimizer DB의 분봉 데이터 수 확인."""
    if not OPTIMIZER_DB.exists():
        return 0
    try:
        with sqlite3.connect(str(OPTIMIZER_DB)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM minute_ohlcv").fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def main():
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / "config" / ".env")
    logger.info("AI Optimizer starting (continuous mode)...")

    config = load_config()
    interval = config.get("schedule", {}).get("interval_minutes", 30)
    last_minute_count = 0

    cycle = 0
    while True:
        cycle += 1
        try:
            # DB 스냅샷 갱신 (분봉 데이터 변경 감지)
            refresh_db_snapshot()
            new_minute_count = get_minute_bar_count()

            hour = datetime.now().hour
            is_market_hours = 9 <= hour < 16

            if is_market_hours and new_minute_count > last_minute_count:
                # 장중: 새 분봉 데이터가 쌓였을 때만 실행
                logger.info(f"--- Cycle {cycle} (minute bars: {last_minute_count}→{new_minute_count}) ---")
                last_minute_count = new_minute_count
                run_optimization("daily")
            elif not is_market_hours:
                # 장외: 일봉 기반 심층 최적화
                logger.info(f"--- Cycle {cycle} (deep, off-hours) ---")
                run_optimization("deep")
            else:
                logger.info(f"Cycle {cycle}: no new minute data, skipping")

        except Exception as e:
            logger.error(f"Optimization cycle {cycle} failed: {e}")

        logger.info(f"Next cycle in {interval} minutes...")
        time.sleep(interval * 60)


if __name__ == "__main__":
    main()
