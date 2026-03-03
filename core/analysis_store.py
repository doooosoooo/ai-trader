"""분석 결과 저장/로드 + LLM 피드백 생성."""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger


_DATA_DIR = Path(__file__).parent.parent / "data"
_ANALYSIS_PATH = _DATA_DIR / "last_analysis.json"
_ANALYSIS_LOG_PATH = _DATA_DIR / "analysis_history.jsonl"
_OUTCOMES_PATH = _DATA_DIR / "prediction_outcomes.jsonl"
_OPTIMIZED_PARAMS_PATH = _DATA_DIR / "optimized_params.json"
_ALPHA_LOG_PATH = _DATA_DIR / "alpha_comparison.jsonl"
_NEWS_LOG_PATH = Path(__file__).parent.parent / "logs" / "news_check.jsonl"
_DB_PATH = str(Path(__file__).parent.parent / "data" / "storage" / "trader.db")

# 종목코드 → 회사명 매핑 (main.py에서도 사용)
TICKER_NAMES: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "373220": "LG에너지솔루션",
    "006400": "삼성SDI",
    "035420": "NAVER",
    "035720": "카카오",
    "051910": "LG화학",
    "005490": "POSCO홀딩스",
    "105560": "KB금융",
    "055550": "신한지주",
    "003670": "포스코퓨처엠",
    "247540": "에코프로비엠",
    "068270": "셀트리온",
    "207940": "삼성바이오로직스",
    "000270": "기아",
    "005380": "현대차",
    "012330": "현대모비스",
    "066570": "LG전자",
    "028260": "삼성물산",
    "003550": "LG",
}


def ticker_display(ticker: str) -> str:
    """종목코드를 '회사명(코드)' 형태로 변환."""
    name = TICKER_NAMES.get(ticker, "")
    return f"{name}({ticker})" if name else ticker


class AnalysisStore:
    """분석 결과 영속화 + 예측 평가 + LLM 피드백 생성."""

    def load_last_analysis(self) -> dict | None:
        """파일에서 마지막 분석 결과 로드."""
        try:
            if _ANALYSIS_PATH.exists():
                with open(_ANALYSIS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load last analysis: {e}")
        return None

    def save_last_analysis(self, analysis: dict) -> None:
        """마지막 분석 결과를 파일에 저장."""
        try:
            with open(_ANALYSIS_PATH, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save last analysis: {e}")

    def save_optimized_params(self, strategy_name: str, params: dict, metrics: dict):
        """최적화된 전략 파라미터를 파일에 저장."""
        try:
            all_params = {}
            if _OPTIMIZED_PARAMS_PATH.exists():
                with open(_OPTIMIZED_PARAMS_PATH, "r", encoding="utf-8") as f:
                    all_params = json.load(f)

            all_params[strategy_name] = {
                "params": params,
                "metrics": metrics,
                "updated_at": datetime.now().isoformat(),
            }

            with open(_OPTIMIZED_PARAMS_PATH, "w", encoding="utf-8") as f:
                json.dump(all_params, f, ensure_ascii=False, indent=2)
            logger.info(f"Optimized params saved for {strategy_name}: {params}")
        except Exception as e:
            logger.warning(f"Failed to save optimized params: {e}")

    def load_optimized_params(self, strategy_name: str) -> dict | None:
        """저장된 최적화 파라미터 로드. 없으면 None."""
        try:
            if not _OPTIMIZED_PARAMS_PATH.exists():
                return None
            with open(_OPTIMIZED_PARAMS_PATH, "r", encoding="utf-8") as f:
                all_params = json.load(f)
            entry = all_params.get(strategy_name)
            return entry.get("params") if entry else None
        except Exception:
            return None

    def log_alpha_comparison(
        self, llm_actions: list[dict], prices: dict,
        watchlist: list[str], held_tickers: set[str],
        data_pipeline, load_optimized_params_fn,
    ) -> None:
        """LLM 판단 vs 규칙전략 신호 비교 기록 — 알파 추적용."""
        from simulation.strategies import STRATEGY_REGISTRY

        all_tickers = held_tickers | set(watchlist)

        # LLM 판단 요약
        llm_signals = {}
        for action in llm_actions:
            ticker = action.get("ticker", "")
            llm_signals[ticker] = action.get("type", "HOLD").upper()

        # 규칙전략 신호 생성
        rule_signals = {}
        for ticker in all_tickers:
            df = data_pipeline.get_daily_df_with_indicators(ticker)
            if df.empty or len(df) < 2:
                continue
            row = df.iloc[-1]
            prev = df.iloc[-2]
            ctx = {}

            for name, factory in STRATEGY_REGISTRY.items():
                strategy = factory(load_optimized_params_fn(name))
                entry = strategy.check_entry(row, prev, ctx)
                exit_signal, exit_reason = strategy.check_exit(row, prev, ctx)

                if ticker not in rule_signals:
                    rule_signals[ticker] = {}
                if entry:
                    rule_signals[ticker][name] = "BUY"
                elif ticker in held_tickers and exit_signal:
                    rule_signals[ticker][name] = f"SELL({exit_reason})"

        # 비교 기록
        record = {
            "timestamp": datetime.now().isoformat(),
            "llm": llm_signals,
            "rules": rule_signals,
            "prices": {t: prices.get(t, 0) for t in all_tickers},
        }

        with open(_ALPHA_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 불일치 로깅
        for ticker in all_tickers:
            llm_act = llm_signals.get(ticker, "HOLD")
            rule_acts = rule_signals.get(ticker, {})
            if rule_acts:
                rule_summary = ", ".join(f"{k}:{v}" for k, v in rule_acts.items())
                if llm_act != "HOLD" or any("BUY" in v or "SELL" in v for v in rule_acts.values()):
                    logger.info(f"Alpha compare {ticker}: LLM={llm_act} vs Rules=[{rule_summary}]")

    def build_backtest_feedback(self, watchlist: list[str], held_tickers: list[str]) -> str:
        """최근 백테스트 결과를 LLM 프롬프트용 피드백 텍스트로 생성."""
        try:
            import sqlite3 as _sqlite3
            with _sqlite3.connect(_DB_PATH) as conn:
                conn.row_factory = _sqlite3.Row
                rows = conn.execute("""
                    SELECT * FROM backtest_results
                    WHERE id IN (
                        SELECT MAX(id) FROM backtest_results
                        GROUP BY strategy_name
                    )
                    ORDER BY created_at DESC
                """).fetchall()

            if not rows:
                return ""

            results = [dict(r) for r in rows]
            created = results[0].get("created_at", "")[:10]

            parts = [f"최근 백테스트 실행일: {created}", ""]

            # 전략별 성과 요약
            parts.append("전략별 성과:")
            for r in results:
                total_ret = r.get("total_return", 0)
                win_rate = r.get("win_rate", 0)
                mdd = r.get("max_drawdown", 0)
                sharpe = r.get("sharpe_ratio", 0)
                pf = r.get("profit_factor", 0)
                trades = r.get("total_trades", 0)
                period = f"{r.get('period_start', '')}~{r.get('period_end', '')}"
                parts.append(
                    f"- {r['strategy_name']}: 수익률 {total_ret:+.1%}, "
                    f"승률 {win_rate:.0%}, MDD {mdd:.1%}, "
                    f"샤프 {sharpe:.2f}, 손익비 {pf:.2f}, "
                    f"거래 {trades}건 ({period})"
                )

            # 종목별 성과
            watchlist_tickers = set(watchlist + held_tickers)
            ticker_summary: dict[str, list[str]] = {}
            for r in results:
                try:
                    breakdown = json.loads(r.get("ticker_breakdown_json", "{}"))
                except Exception:
                    continue
                for ticker, stats in breakdown.items():
                    if ticker not in watchlist_tickers:
                        continue
                    wins = stats.get("wins", 0)
                    losses = stats.get("losses", 0)
                    avg_pnl = stats.get("avg_pnl_pct", 0)
                    name = TICKER_NAMES.get(ticker, ticker)
                    if ticker not in ticker_summary:
                        ticker_summary[ticker] = []
                    ticker_summary[ticker].append(
                        f"{r['strategy_name']} {avg_pnl:+.1%}({wins}승{losses}패)"
                    )

            if ticker_summary:
                parts.extend(["", "종목별 백테스트 성과 (관심종목):"])
                for ticker, summaries in sorted(ticker_summary.items()):
                    name = TICKER_NAMES.get(ticker, ticker)
                    parts.append(f"- {name}({ticker}): {', '.join(summaries)}")

            # 최적화된 파라미터 정보
            if _OPTIMIZED_PARAMS_PATH.exists():
                try:
                    with open(_OPTIMIZED_PARAMS_PATH, "r", encoding="utf-8") as f:
                        opt_data = json.load(f)
                    if opt_data:
                        parts.extend(["", "최적화된 전략 파라미터:"])
                        for sname, entry in opt_data.items():
                            p = entry.get("params", {})
                            m = entry.get("metrics", {})
                            param_str = ", ".join(f"{k}={v}" for k, v in p.items())
                            parts.append(
                                f"- {sname}: {param_str} "
                                f"(test 수익률 {m.get('test_return', 0):+.1%}, "
                                f"test 샤프 {m.get('test_sharpe', 0):.2f})"
                            )
                except Exception:
                    pass

            return "\n".join(parts)

        except Exception as e:
            logger.debug(f"Backtest feedback unavailable: {e}")
            return ""

    def build_prediction_feedback(self, lookback: int = 20) -> str:
        """최근 예측 결과를 LLM 프롬프트용 피드백 텍스트로 생성."""
        if not _OUTCOMES_PATH.exists():
            return ""

        try:
            with open(_OUTCOMES_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return ""

        if not lines:
            return ""

        # 최근 N건 로드
        recent = []
        for line in lines[-lookback:]:
            try:
                recent.append(json.loads(line.strip()))
            except Exception:
                continue

        if not recent:
            return ""

        # 전체 통계
        total_correct = sum(r.get("correct_count", 0) for r in recent)
        total_evaluated = sum(r.get("total_evaluated", 0) for r in recent)
        if total_evaluated == 0:
            return ""

        overall_accuracy = total_correct / total_evaluated

        # 액션 타입별 통계
        type_stats: dict[str, dict] = {}
        ticker_stats: dict[str, list] = {}
        for record in recent:
            for pred in record.get("predictions", []):
                action = pred.get("predicted_action", "HOLD")
                correct = pred.get("correct", False)
                ret = pred.get("actual_return_pct", 0)
                ticker = pred.get("ticker", "")
                name = pred.get("name", ticker)

                if action not in type_stats:
                    type_stats[action] = {"correct": 0, "total": 0, "returns": []}
                type_stats[action]["total"] += 1
                type_stats[action]["returns"].append(ret)
                if correct:
                    type_stats[action]["correct"] += 1

                key = f"{name}({ticker})"
                if key not in ticker_stats:
                    ticker_stats[key] = []
                ticker_stats[key].append({
                    "action": action, "return": ret, "correct": correct
                })

        # 피드백 텍스트 생성
        parts = [
            f"최근 {len(recent)}회 분석 예측 정확도: {overall_accuracy:.0%} ({total_correct}/{total_evaluated})",
            "",
        ]

        for action_type in ["BUY", "SELL", "HOLD"]:
            stats = type_stats.get(action_type)
            if not stats or stats["total"] == 0:
                continue
            acc = stats["correct"] / stats["total"]
            avg_ret = sum(stats["returns"]) / len(stats["returns"])
            parts.append(
                f"- {action_type}: 정확도 {acc:.0%} ({stats['correct']}/{stats['total']}), "
                f"평균 실제수익률 {avg_ret:+.1f}%"
            )

        # 반복 오류 종목
        bad_tickers = []
        for name_ticker, preds in ticker_stats.items():
            if len(preds) >= 3:
                correct_cnt = sum(1 for p in preds if p["correct"])
                if correct_cnt / len(preds) < 0.5:
                    avg_ret = sum(p["return"] for p in preds) / len(preds)
                    bad_tickers.append((name_ticker, correct_cnt, len(preds), avg_ret))

        if bad_tickers:
            parts.append("")
            parts.append("주의 종목 (반복 오류):")
            for name_ticker, correct, total, avg_ret in bad_tickers[:5]:
                parts.append(
                    f"- {name_ticker}: 정확도 {correct}/{total}, 평균수익률 {avg_ret:+.1f}%"
                )

        return "\n".join(parts)

    def evaluate_previous_predictions(
        self, current_prices: dict[str, float], last_analysis: dict | None
    ) -> None:
        """이전 분석의 예측을 현재 가격과 비교하여 결과를 기록."""
        prev = last_analysis
        if not prev or not prev.get("signal"):
            return

        prev_actions = prev["signal"].get("actions", [])
        if not prev_actions:
            return

        prev_ts = prev.get("timestamp", "")
        now = datetime.now()

        try:
            prev_dt = datetime.fromisoformat(prev_ts)
            hours_elapsed = (now - prev_dt).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_elapsed = 0

        prev_portfolio = prev.get("portfolio", {}) if "portfolio" in prev else {}

        outcomes = []
        correct_count = 0
        total_evaluated = 0

        for action in prev_actions:
            ticker = action.get("ticker", "")
            if not ticker or ticker not in current_prices:
                continue

            current_price = current_prices[ticker]
            action_type = action.get("type", "HOLD")

            prev_price = None
            if prev_portfolio and isinstance(prev_portfolio, dict):
                positions = prev_portfolio.get("positions", {})
                if ticker in positions:
                    prev_price = positions[ticker].get("current_price")

            if not prev_price:
                prev_price = self._get_price_at_analysis(ticker, prev_ts)

            if not prev_price or prev_price == 0:
                continue

            actual_return = (current_price - prev_price) / prev_price
            total_evaluated += 1

            if action_type == "BUY":
                is_correct = actual_return > 0
            elif action_type == "SELL":
                is_correct = actual_return < 0
            else:
                is_correct = abs(actual_return) < 0.03

            if is_correct:
                correct_count += 1

            outcomes.append({
                "ticker": ticker,
                "name": action.get("name", ""),
                "predicted_action": action_type,
                "predicted_reason": action.get("reason", ""),
                "price_at_prediction": prev_price,
                "price_at_evaluation": current_price,
                "actual_return_pct": round(actual_return * 100, 2),
                "correct": is_correct,
            })

        if not outcomes:
            return

        record = {
            "analysis_timestamp": prev_ts,
            "evaluation_timestamp": now.isoformat(),
            "hours_elapsed": round(hours_elapsed, 1),
            "predictions": outcomes,
            "accuracy": round(correct_count / total_evaluated, 2) if total_evaluated else 0,
            "total_evaluated": total_evaluated,
            "correct_count": correct_count,
        }

        try:
            with open(_OUTCOMES_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                f"Prediction evaluation: {correct_count}/{total_evaluated} correct "
                f"({record['accuracy']:.0%}) over {hours_elapsed:.1f}h"
            )
        except Exception as e:
            logger.warning(f"Failed to write prediction outcomes: {e}")

    def _get_price_at_analysis(self, ticker: str, analysis_ts: str) -> float | None:
        """analysis_history.jsonl에서 특정 분석 시점의 종목 가격을 찾는다."""
        try:
            if not _ANALYSIS_LOG_PATH.exists():
                return None
            with open(_ANALYSIS_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines[-20:]):
                record = json.loads(line.strip())
                if record.get("timestamp", "")[:16] == analysis_ts[:16]:
                    snapshot = record.get("prices_snapshot", {})
                    if ticker in snapshot:
                        return snapshot[ticker]
                    positions = record.get("portfolio", {}).get("positions", {})
                    if ticker in positions:
                        return positions[ticker].get("current_price")
                    break
        except Exception:
            pass
        return None

    def append_analysis_log(
        self, signal: dict, actions: list, portfolio: dict,
        ml_predictions: dict, prices: dict[str, float] | None = None,
        mode: str = "simulation",
    ) -> None:
        """분석 결과를 JSONL 히스토리에 누적."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "signal": signal,
            "actions_executed": actions,
            "portfolio": {
                "total_asset": portfolio.get("total_asset"),
                "cash": portfolio.get("cash"),
                "cash_ratio": portfolio.get("cash_ratio"),
                "total_pnl_pct": portfolio.get("total_pnl_pct"),
                "positions": portfolio.get("positions", {}),
            },
            "prices_snapshot": prices or {},
            "ml_predictions": {
                t: {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in pred.items()}
                for t, pred in ml_predictions.items()
            } if ml_predictions else {},
        }
        try:
            with open(_ANALYSIS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append analysis log: {e}")
