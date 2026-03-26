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
    """DB에서 데이터가 충분한 전체 종목 목록 가져오기 (풀마켓 모드)."""
    # DB에서 충분한 데이터(최소 200봉)가 있는 종목 전부 사용
    min_bars = 200
    with sqlite3.connect(str(OPTIMIZER_DB)) as conn:
        rows = conn.execute(
            "SELECT ticker, COUNT(*) as cnt FROM daily_ohlcv "
            "GROUP BY ticker HAVING cnt >= ? ORDER BY cnt DESC",
            (min_bars,),
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


def send_telegram_suggestion(message: str, config: dict, suggestion_data: dict | None = None):
    """텔레그램 메시지 + 적용/거부 버튼 전송."""
    try:
        import os
        import json
        import telegram

        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        if not token or not chat_id:
            logger.warning("Telegram not configured, skipping notification")
            return

        bot = telegram.Bot(token=token)
        import asyncio

        # 제안 데이터를 파일에 저장 (텔레그램 콜백에서 읽기 위해)
        if suggestion_data:
            pending_path = BASE_DIR / "data" / "storage" / "pending_optimizer_suggestion.json"
            with open(pending_path, "w") as f:
                json.dump(suggestion_data, f, indent=2, ensure_ascii=False)

            # 인라인 버튼 추가
            keyboard = telegram.InlineKeyboardMarkup([
                [
                    telegram.InlineKeyboardButton("✅ 적용", callback_data="opt_apply"),
                    telegram.InlineKeyboardButton("❌ 무시", callback_data="opt_reject"),
                ]
            ])
            asyncio.run(bot.send_message(
                chat_id=chat_id, text=message,
                parse_mode="HTML", reply_markup=keyboard,
            ))
        else:
            asyncio.run(bot.send_message(
                chat_id=chat_id, text=message, parse_mode="HTML",
            ))

        logger.info("Optimization suggestion sent via Telegram")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


PARAM_LABELS = {
    "rsi_oversold": ("매수 RSI 기준", "", "이하일 때 매수 신호"),
    "take_profit_pct": ("익절 기준", "%", "수익 시 매도"),
    "stop_loss_pct": ("손절 기준", "%", "손실 시 매도"),
    "position_size_pct": ("1종목 투자 비중", "%", "총 자산 대비"),
    "max_hold_days": ("최대 보유일", "일", "이후 자동 매도"),
}


def format_suggestion_message(
    strategy_name: str,
    current_params: dict,
    best_params: dict,
    current_metrics: dict,
    best_metrics: dict,
    overfit_ratio: float,
    period: str,
) -> str:
    """최적화 제안을 쉬운 한국어로 포맷."""
    cur_ret = current_metrics.get("total_return", 0)
    best_ret = best_metrics.get("total_return", 0)
    best_mdd = best_metrics.get("max_drawdown", 0)

    msg = (
        f"📈 <b>더 나은 전략을 찾았습니다!</b>\n"
        f"━━━━━━━━━━━━━━\n\n"
    )

    # 변경된 파라미터만 쉬운 한국어로
    changes = []
    for k, v in best_params.items():
        cur_v = current_params.get(k)
        if cur_v == v:
            continue
        label, unit, desc = PARAM_LABELS.get(k, (k, "", ""))
        if unit == "%":
            changes.append(f"  • {label}: {cur_v:.0%} → <b>{v:.0%}</b> ({desc})")
        elif unit == "일":
            changes.append(f"  • {label}: {cur_v}일 → <b>{v}일</b> ({desc})")
        else:
            changes.append(f"  • {label}: {cur_v} → <b>{v}</b> ({desc})")

    if changes:
        msg += "<b>변경 내용:</b>\n"
        msg += "\n".join(changes)
        msg += "\n\n"

    msg += (
        f"<b>같은 기간 비교 ({period}):</b>\n"
        f"  현재 전략: <b>{cur_ret:+.1%}</b>\n"
        f"  제안 전략: <b>{best_ret:+.1%}</b> ({best_ret - cur_ret:+.1%}p)\n"
        f"  📉 최대 손실: {best_mdd:.1%}\n\n"
        f"⚠️ 과거 데이터 기반이라 실전과 다를 수 있습니다.\n"
        f"아래 버튼으로 적용 여부를 선택하세요."
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
    grids = config.get("grids", {})
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
            test_period = split_result["test_period"]  # "YYYYMMDD~YYYYMMDD"

            logger.info(
                f"Best params: {best_params} | "
                f"Train: {train_metrics.get('total_return', 0):+.2%} (sharpe {train_metrics.get('sharpe_ratio', 0):.2f}) | "
                f"Test: {test_metrics.get('total_return', 0):+.2%} (sharpe {test_metrics.get('sharpe_ratio', 0):.2f}) | "
                f"Overfit: {overfit_ratio:.2f}"
            )

        except Exception as e:
            logger.error(f"Optimization error for {strategy_name}: {e}")
            continue

        # 같은 테스트 기간에서 현재 파라미터 기준선 백테스트
        try:
            test_start, test_end = test_period.split("~")
            current_strategy = factory(current_params)
            current_result = bt.run(tickers, current_strategy, test_start, test_end)
            current_metrics = current_result.metrics
            logger.info(
                f"Baseline on test period ({strategy_name}): "
                f"return={current_metrics.get('total_return', 0):+.2%}, "
                f"sharpe={current_metrics.get('sharpe_ratio', 0):.2f}"
            )
        except Exception as e:
            logger.error(f"Baseline backtest failed: {e}")
            continue

        # 5. 개선 여부 판단 (같은 테스트 기간 기준)
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

            # 이전 제안과 동일하면 중복 알림 스킵
            prev_params = None
            if RESULTS_PATH.exists():
                try:
                    with open(RESULTS_PATH) as f:
                        prev = json.load(f)
                    prev_params = prev.get("suggested_params")
                except Exception:
                    pass

            if prev_params == best_params:
                logger.info(f"Same suggestion as before for {strategy_name}, skipping notification")
            else:
                logger.info(f"✅ Improvement found for {strategy_name}! Sending suggestion...")
                # 테스트 기간을 읽기 좋은 형식으로 변환
                ts, te = test_period.split("~")
                test_period_display = f"{ts[:4]}.{ts[4:6]}.{ts[6:]}~{te[:4]}.{te[4:6]}.{te[6:]}"
                msg = format_suggestion_message(
                    strategy_name, current_params, best_params,
                    current_metrics, test_metrics, overfit_ratio, test_period_display,
                )
                send_telegram_suggestion(msg, config, suggestion_data=suggestion)

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
