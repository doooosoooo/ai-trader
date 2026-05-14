"""Incident RCA — 사고 발생 시 Opus 4.7로 근본 원인 분석.

트리거 이벤트 (4종):
1. daily_loss_2pct — 일일 손실 -2% 이상
2. circuit_breaker_halted — 서킷 HALTED 진입
3. sell_backoff_blocked — 자동 매도 backoff 차단 발동
4. hard_stop_loss — 개별 종목 pnl_pct ≤ -5% 하드손절 매도

가드레일:
- 60분 dedupe per (event_type, ticker)
- 일일 최대 8회 호출 hard cap
- 비동기 threading.Thread(daemon=True) — 트레이딩 사이클 블로킹 방지
- 입력 컨텍스트 char cap (≈50k 토큰)
- secret 마스킹 (KIS_APP_KEY=, Bearer ...)
"""

import json
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

INCIDENT_DIR = Path(__file__).parent.parent / "logs" / "reviews"
INCIDENT_DIR.mkdir(parents=True, exist_ok=True)

DECISIONS_DIR = Path(__file__).parent.parent / "logs" / "decisions"
TRADES_DIR = Path(__file__).parent.parent / "logs" / "trades"

_DEDUPE_WINDOW_MIN = 60
_DAILY_MAX_RUNS = 8
_CONTEXT_CHAR_CAP = 200_000  # ≈ 50k 토큰

_SECRET_PATTERNS = [
    re.compile(r"(KIS_APP_KEY\s*[:=]\s*)([^\s\"',}]+)", re.IGNORECASE),
    re.compile(r"(KIS_APP_SECRET\s*[:=]\s*)([^\s\"',}]+)", re.IGNORECASE),
    re.compile(r"(ANTHROPIC_API_KEY\s*[:=]\s*)([^\s\"',}]+)", re.IGNORECASE),
    re.compile(r"(TELEGRAM_BOT_TOKEN\s*[:=]\s*)([^\s\"',}]+)", re.IGNORECASE),
    re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]+)"),
    re.compile(r"(sk-ant-[A-Za-z0-9._\-]+)"),
]


def _mask_secrets(text: str) -> str:
    """API 키/토큰 마스킹."""
    if not text:
        return text
    masked = text
    for pat in _SECRET_PATTERNS:
        masked = pat.sub(lambda m: m.group(1) + "***MASKED***" if m.lastindex == 2 else "***MASKED***", masked)
    return masked


class IncidentAnalyzer:
    """이벤트 트리거 기반 근본 원인 분석 — Opus 4.7."""

    def __init__(self, llm_engine, portfolio, circuit_breaker, telegram, db_path: str = "data/storage/trader.db"):
        self.llm = llm_engine
        self.portfolio = portfolio
        self.circuit_breaker = circuit_breaker
        self.telegram = telegram
        self.db_path = db_path

        # dedupe: {(event_type, ticker): datetime}
        self._last_fired: dict[tuple[str, str], datetime] = {}
        # daily 카운터
        self._daily_count = 0
        self._count_date = ""
        self._lock = threading.Lock()

    def _reset_daily_if_new_day(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if self._count_date != today:
            self._daily_count = 0
            self._count_date = today
            # dedupe 윈도우는 60분이므로 어제 키도 자연스럽게 만료되지만, 명시 정리
            self._last_fired.clear()

    def trigger(self, event_type: str, ticker: str | None = None, event_detail: str = "") -> bool:
        """RCA 트리거 — 가드 통과 시 백그라운드 스레드로 실행. 반환값: 실제 실행 여부."""
        with self._lock:
            self._reset_daily_if_new_day()

            # 1. 일일 한도
            if self._daily_count >= _DAILY_MAX_RUNS:
                logger.info(f"RCA skipped (daily cap {_DAILY_MAX_RUNS} reached): {event_type}/{ticker}")
                return False

            # 2. dedupe 60분
            key = (event_type, ticker or "")
            last = self._last_fired.get(key)
            if last and (datetime.now() - last) < timedelta(minutes=_DEDUPE_WINDOW_MIN):
                logger.debug(f"RCA dedupe skip: {event_type}/{ticker} (last {last:%H:%M})")
                return False

            # 가드 통과 — 카운트 선점하고 컨텍스트 수집 후 스레드 fire
            self._last_fired[key] = datetime.now()
            self._daily_count += 1

        # 컨텍스트 수집은 sync (사고 시점 스냅샷 보장). LLM 호출은 thread.
        try:
            context = self._gather_context(event_type, ticker)
        except Exception as e:
            logger.error(f"RCA context gather failed for {event_type}/{ticker}: {e}")
            context = {"gather_error": str(e)}

        t = threading.Thread(
            target=self._run_rca,
            args=(event_type, ticker, event_detail, context),
            daemon=True,
            name=f"rca-{event_type}",
        )
        t.start()
        logger.warning(f"RCA triggered: event={event_type} ticker={ticker} count={self._daily_count}/{_DAILY_MAX_RUNS}")
        return True

    def _run_rca(self, event_type: str, ticker: str | None, event_detail: str, context: dict) -> None:
        """백그라운드 스레드에서 LLM 호출 + 저장 + 알림."""
        try:
            rca = self.llm.generate_incident_rca(
                event_type=event_type,
                ticker=ticker,
                event_detail=event_detail,
                context=context,
            )
            if not rca:
                logger.error(f"RCA returned empty for {event_type}/{ticker}")
                return

            rca["event_type"] = event_type
            rca["ticker"] = ticker
            rca["timestamp"] = datetime.now().isoformat()

            self._save_incident(event_type, ticker, rca)
            self._send_telegram(event_type, ticker, rca)
            logger.info(f"RCA complete: {event_type}/{ticker} severity={rca.get('severity', '?')}")

        except Exception as e:
            logger.error(f"RCA execution failed for {event_type}/{ticker}: {e}")
            # 텔레그램으로 실패 알림 (조용히 사라지지 않도록)
            try:
                if self.telegram and self.telegram.enabled:
                    self.telegram.send_alert_sync(
                        "error",
                        f"⚠️ <b>Incident RCA 실패</b>\n이벤트: {event_type} {ticker or ''}\n오류: {str(e)[:200]}",
                    )
            except Exception:
                pass

    def _gather_context(self, event_type: str, ticker: str | None) -> dict:
        """사고 시점 컨텍스트 수집 — 최근 결정, 매매, 포지션, 시장 상태."""
        ctx: dict = {}

        # 1. 최근 결정 로그 (오늘 + 어제 최대 20개)
        try:
            files = sorted(DECISIONS_DIR.glob("*.json"))[-20:]
            decisions = []
            for f in files:
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    sig = d.get("signal", {})
                    decisions.append({
                        "ts": d.get("timestamp", "")[:16],
                        "risk": sig.get("risk_assessment", ""),
                        "outlook": (sig.get("market_outlook", "") or "")[:200],
                        "reasoning": (sig.get("reasoning", "") or "")[:400],
                        "actions": [
                            {
                                "type": a.get("type"), "ticker": a.get("ticker"),
                                "ratio": a.get("ratio"),
                                "reason": (a.get("reason", "") or "")[:150],
                            }
                            for a in sig.get("actions", [])
                        ],
                        "safety_filtered": sig.get("safety_filtered", []) or [],
                    })
                except Exception:
                    continue
            ctx["recent_decisions"] = decisions
        except Exception as e:
            ctx["recent_decisions_error"] = str(e)

        # 2. 최근 매매 (오늘 + 어제 최대 30건)
        try:
            trades = self.portfolio.get_trade_history(limit=30) if hasattr(self.portfolio, "get_trade_history") else []
            ctx["recent_trades"] = trades
        except Exception as e:
            ctx["recent_trades_error"] = str(e)

        # 3. 현재 포지션 + 자산 요약
        try:
            positions = []
            total = getattr(self.portfolio, "total_asset", 0) or 1
            for tk, pos in self.portfolio.positions.items():
                positions.append({
                    "ticker": tk,
                    "name": getattr(pos, "name", tk),
                    "strategy_type": getattr(pos, "strategy_type", "swing"),
                    "avg_price": getattr(pos, "avg_price", 0),
                    "current_price": getattr(pos, "current_price", 0),
                    "peak_price": getattr(pos, "peak_price", 0),
                    "quantity": getattr(pos, "quantity", 0),
                    "pnl_pct": round(getattr(pos, "pnl_pct", 0), 4),
                    "weight": round((getattr(pos, "market_value", 0) or 0) / total, 4),
                    "bought_at": getattr(pos, "bought_at", ""),
                })
            ctx["portfolio"] = {
                "positions": positions,
                "cash": getattr(self.portfolio, "cash", 0),
                "total_asset": total,
                "total_pnl_pct": round(getattr(self.portfolio, "total_pnl_pct", 0), 4),
                "drawdown": round(getattr(self.portfolio, "portfolio_drawdown", 0), 4),
                "high_water_mark": getattr(self.portfolio, "high_water_mark", 0),
            }
        except Exception as e:
            ctx["portfolio_error"] = str(e)

        # 4. 서킷브레이커 상태
        try:
            ctx["circuit_breaker"] = self.circuit_breaker.get_status() if self.circuit_breaker else {}
        except Exception as e:
            ctx["circuit_breaker_error"] = str(e)

        # 5. 사고 종목 상세 (있을 때)
        if ticker:
            try:
                pos = self.portfolio.positions.get(ticker) if hasattr(self.portfolio, "positions") else None
                if pos:
                    ctx["focus_position"] = {
                        "name": getattr(pos, "name", ticker),
                        "strategy_type": getattr(pos, "strategy_type", "swing"),
                        "avg_price": getattr(pos, "avg_price", 0),
                        "current_price": getattr(pos, "current_price", 0),
                        "peak_price": getattr(pos, "peak_price", 0),
                        "pnl_pct": round(getattr(pos, "pnl_pct", 0), 4),
                        "bought_at": getattr(pos, "bought_at", ""),
                    }
            except Exception:
                pass

            # 해당 종목 최근 매매만 추출
            try:
                ticker_trades = [
                    t for t in (ctx.get("recent_trades") or [])
                    if t.get("ticker") == ticker
                ][-10:]
                ctx["focus_ticker_trades"] = ticker_trades
            except Exception:
                pass

        # 6. 최근 스크리닝 결과 (참고)
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT date, candidates_json FROM screening_results ORDER BY date DESC LIMIT 1"
                ).fetchone()
            if row:
                cands = json.loads(row[1])[:10]
                ctx["latest_screening"] = {
                    "date": row[0],
                    "top_candidates": [
                        {"ticker": c.get("ticker"), "name": c.get("name"), "score": c.get("composite_score")}
                        for c in cands
                    ],
                }
        except Exception:
            pass

        # 7. 컨텍스트 직렬화 + 시크릿 마스킹 + 길이 cap
        try:
            serialized = json.dumps(ctx, ensure_ascii=False, default=str)
            masked = _mask_secrets(serialized)
            if len(masked) > _CONTEXT_CHAR_CAP:
                masked = masked[:_CONTEXT_CHAR_CAP] + "...[truncated]"
                ctx = {"_truncated": True, "raw": masked}
            else:
                ctx = json.loads(masked)
        except Exception as e:
            logger.warning(f"RCA context masking failed: {e}")

        return ctx

    def _save_incident(self, event_type: str, ticker: str | None, rca: dict) -> None:
        """logs/reviews/incident_YYYYMMDD_HHMM_{type}.md + .json 저장."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        suffix = f"{event_type}" + (f"_{ticker}" if ticker else "")
        base = INCIDENT_DIR / f"incident_{ts}_{suffix}"

        try:
            base.with_suffix(".json").write_text(
                json.dumps(rca, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Incident JSON save failed: {e}")

        try:
            md = self._format_incident_md(event_type, ticker, rca)
            base.with_suffix(".md").write_text(md, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Incident MD save failed: {e}")

    def _format_incident_md(self, event_type: str, ticker: str | None, rca: dict) -> str:
        lines = [
            f"# Incident RCA — {event_type}" + (f" / {ticker}" if ticker else ""),
            "",
            f"- 발생 시각: {rca.get('timestamp', '')}",
            f"- 심각도: {rca.get('severity', '?')}",
            f"- 헤드라인: {rca.get('headline', '')}",
            "",
            "## 근본 원인",
            rca.get("root_cause", ""),
            "",
        ]
        factors = rca.get("contributing_factors", []) or []
        if factors:
            lines.append("## 기여 요인")
            for f in factors:
                lines.append(f"- {f}")
            lines.append("")

        timeline = rca.get("timeline", []) or []
        if timeline:
            lines.append("## 타임라인")
            for ev in timeline:
                lines.append(f"- {ev.get('when', '')} — {ev.get('what', '')} [{ev.get('verdict', '')}]")
            lines.append("")

        actions = rca.get("preventive_actions", []) or []
        if actions:
            lines.append("## 재발 방지 조치")
            for a in actions:
                lines.append(f"- [{a.get('scope', '')}] {a.get('action', '')}")
            lines.append("")

        suggest = rca.get("rule_change_suggestion", {}) or {}
        if suggest.get("needed"):
            lines.extend([
                "## 룰 수정 제안",
                f"- 대상: {suggest.get('rule_id', '')}",
                f"- 현재: {suggest.get('current', '')}",
                f"- 제안: {suggest.get('suggestion', '')}",
            ])

        return "\n".join(lines)

    def _send_telegram(self, event_type: str, ticker: str | None, rca: dict) -> None:
        """텔레그램 error 레벨 알림."""
        if not self.telegram or not getattr(self.telegram, "enabled", False):
            return

        headline = rca.get("headline", "사고 분석 결과")
        severity = rca.get("severity", "?")
        emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(severity, "⚪")

        lines = [
            f"{emoji} <b>Incident RCA</b> [{severity}]",
            f"이벤트: <code>{event_type}</code>" + (f" / <code>{ticker}</code>" if ticker else ""),
            f"<i>{headline}</i>",
            "",
            f"<b>근본 원인</b>",
            rca.get("root_cause", "")[:400],
        ]

        factors = rca.get("contributing_factors", []) or []
        if factors:
            lines.append("")
            lines.append("<b>기여 요인</b>")
            for f in factors[:4]:
                lines.append(f"• {str(f)[:120]}")

        actions = rca.get("preventive_actions", []) or []
        if actions:
            lines.append("")
            lines.append("<b>재발 방지</b>")
            for a in actions[:4]:
                scope = a.get("scope", "")
                lines.append(f"• [{scope}] {str(a.get('action', ''))[:120]}")

        suggest = rca.get("rule_change_suggestion", {}) or {}
        if suggest.get("needed"):
            lines.append("")
            lines.append("<b>룰 수정 제안</b>")
            lines.append(f"대상: {suggest.get('rule_id', '')[:80]}")
            lines.append(f"제안: {str(suggest.get('suggestion', ''))[:200]}")

        try:
            self.telegram.send_alert_sync("error", "\n".join(lines))
        except Exception as e:
            logger.warning(f"RCA telegram send failed: {e}")
