"""일일 자기 복기 — LLM 기반 매매 판단 평가."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

REVIEWS_DIR = Path(__file__).parent / "reports"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

LOG_REVIEWS_DIR = Path(__file__).parent.parent / "logs" / "reviews"
LOG_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

DECISIONS_DIR = Path(__file__).parent.parent / "logs" / "decisions"


class DailyReviewer:
    """일일/주간/월간 자기 복기."""

    def __init__(self, llm_engine, portfolio, db_path: str = "data/storage/trader.db"):
        self.llm = llm_engine
        self.portfolio = portfolio
        self.db_path = db_path

    def run_daily_review(self) -> dict:
        """당일 매매 복기 실행."""
        today = datetime.now().strftime("%Y-%m-%d")
        trades = self.portfolio.get_today_trades()

        if not trades:
            logger.info("No trades today, skipping review")
            return {"summary": "오늘은 매매가 없었습니다.", "overall_score": 0}

        # 포트폴리오 변동 계산
        snapshots = self.portfolio.get_daily_snapshots(days=2)
        if len(snapshots) >= 2:
            portfolio_change = {
                "today_asset": snapshots[0]["total_asset"],
                "yesterday_asset": snapshots[1]["total_asset"],
                "daily_pnl": snapshots[0].get("daily_pnl", 0),
                "daily_pnl_pct": snapshots[0].get("daily_pnl_pct", 0),
            }
        else:
            portfolio_change = {
                "total_asset": self.portfolio.total_asset,
                "total_pnl": self.portfolio.total_pnl,
            }

        market_summary = {
            "date": today,
            "num_trades": len(trades),
            "buys": len([t for t in trades if t["action"] == "BUY"]),
            "sells": len([t for t in trades if t["action"] == "SELL"]),
        }

        try:
            review = self.llm.generate_review(
                trades=trades,
                portfolio_change=portfolio_change,
                market_summary=market_summary,
            )

            # 리포트 저장
            self._save_review(today, review)
            logger.info(f"Daily review complete: score={review.get('overall_score', '?')}")
            return review

        except Exception as e:
            logger.error(f"Daily review failed: {e}")
            return {"summary": f"복기 실패: {e}", "overall_score": 0}

    def run_weekly_review(self) -> dict:
        """주간 종합 리포트."""
        snapshots = self.portfolio.get_daily_snapshots(days=7)
        trades = self.portfolio.get_trade_history(limit=100)

        # 최근 7일 거래만 필터
        from datetime import timedelta
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        weekly_trades = [t for t in trades if t.get("timestamp", "") >= week_ago]

        portfolio_change = {}
        if snapshots:
            first = snapshots[-1] if snapshots else {}
            last = snapshots[0] if snapshots else {}
            portfolio_change = {
                "start_asset": first.get("total_asset", 0),
                "end_asset": last.get("total_asset", 0),
                "weekly_return": (last.get("total_asset", 0) - first.get("total_asset", 0)) / first.get("total_asset", 1) if first.get("total_asset") else 0,
            }

        market_summary = {
            "period": "weekly",
            "num_trades": len(weekly_trades),
        }

        try:
            review = self.llm.generate_review(
                trades=weekly_trades,
                portfolio_change=portfolio_change,
                market_summary=market_summary,
            )
            date_str = datetime.now().strftime("%Y-%m-%d")
            self._save_review(f"weekly_{date_str}", review)
            return review
        except Exception as e:
            logger.error(f"Weekly review failed: {e}")
            return {"summary": f"주간 복기 실패: {e}"}

    def _save_review(self, name: str, review: dict) -> None:
        # JSON 저장
        json_path = LOG_REVIEWS_DIR / f"{name}.json"
        json_path.write_text(
            json.dumps(review, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 마크다운 리포트 생성
        md_path = REVIEWS_DIR / f"{name}.md"
        md_content = self._format_review_md(name, review)
        md_path.write_text(md_content, encoding="utf-8")

    def _format_review_md(self, name: str, review: dict) -> str:
        lines = [
            f"# 매매 복기 — {name}",
            "",
            f"## 종합 평가 (점수: {review.get('overall_score', '?')}/10)",
            review.get("summary", ""),
            "",
        ]

        trade_reviews = review.get("trade_reviews", [])
        if trade_reviews:
            lines.append("## 개별 매매 평가")
            for tr in trade_reviews:
                lines.extend([
                    f"### {tr.get('ticker', '')} — {tr.get('action', '')}",
                    f"- 평가: {tr.get('evaluation', '')}",
                    f"- 타이밍: {tr.get('timing_score', '?')}/10",
                    f"- 비중: {tr.get('size_score', '?')}/10",
                    f"- 종목선택: {tr.get('selection_score', '?')}/10",
                    f"- 코멘트: {tr.get('comment', '')}",
                    "",
                ])

        improvements = review.get("improvements", [])
        if improvements:
            lines.append("## 개선점")
            for imp in improvements:
                lines.append(f"- {imp}")
            lines.append("")

        strategy_mod = review.get("strategy_modification", {})
        if strategy_mod.get("needed"):
            lines.extend([
                "## 전략 수정 제안",
                f"- 제안: {strategy_mod.get('suggestion', '')}",
                f"- 이유: {strategy_mod.get('reason', '')}",
            ])

        if review.get("market_insight"):
            lines.extend(["", f"## 시장 인사이트", review["market_insight"]])

        return "\n".join(lines)

    # ============================================================
    # Deep Daily Review (Opus 4.7) — 장 마감 후 1회
    # ============================================================

    def run_deep_daily_review(self, strategy_excerpt: str = "") -> dict:
        """장 마감 후 Opus 1회 호출로 심층 복기.

        오늘 매매 + 결정 로그 + 룰 적용 결과 + 놓친 기회 종합 분석.
        실패 시 Sonnet fallback.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        trades = self.portfolio.get_today_trades()
        decisions = self._gather_today_decisions()

        # 매매도 결정도 없으면 skip
        if not trades and not decisions:
            logger.info("No trades or decisions today — skipping deep review")
            return {"summary": "오늘은 매매/분석이 없었습니다.", "overall_score": 0}

        safety_stats = self._gather_safety_filtered_stats(decisions)
        missed = self._gather_missed_opportunities(decisions, trades)
        portfolio_change = self._gather_portfolio_change()
        portfolio_summary = self._gather_portfolio_summary()
        market_summary = {
            "date": today,
            "num_trades": len(trades),
            "buys": len([t for t in trades if t.get("action") == "BUY"]),
            "sells": len([t for t in trades if t.get("action") == "SELL"]),
            "num_cycles": len(decisions),
        }

        # Opus 호출 (실패 시 Sonnet fallback)
        review = None
        used_model = ""
        try:
            review = self.llm.generate_deep_review(
                trades=trades,
                decisions=decisions,
                safety_stats=safety_stats,
                missed_opportunities=missed,
                portfolio_change=portfolio_change,
                portfolio_summary=portfolio_summary,
                market_summary=market_summary,
                strategy_excerpt=strategy_excerpt[:3000],
            )
            used_model = "opus-4.7"
        except Exception as e:
            logger.warning(f"Deep review (Opus) failed, falling back to Sonnet generate_review: {e}")
            try:
                review = self.llm.generate_review(
                    trades=trades,
                    portfolio_change=portfolio_change,
                    market_summary=market_summary,
                )
                used_model = "sonnet-4.6 (fallback)"
                review["fallback_note"] = "Opus 실패로 Sonnet으로 대체됨"
            except Exception as e2:
                logger.error(f"Daily review fallback also failed: {e2}")
                return {"summary": f"복기 실패: {e2}", "overall_score": 0}

        review["model_used"] = used_model
        review["date"] = today
        self._save_review(today + "_deep", review)
        logger.info(f"Deep daily review complete [{used_model}]: score={review.get('overall_score', '?')}")
        return review

    def _gather_today_decisions(self) -> list[dict]:
        """오늘자 logs/decisions/*.json 로드 (최근 30개 cap, raw_response 2000자 truncate)."""
        today_prefix = datetime.now().strftime("%Y%m%d")
        files = sorted(DECISIONS_DIR.glob(f"{today_prefix}_*.json"))[-30:]
        out = []
        for f in files:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                # raw_response가 길어 토큰 폭주 가능 — truncate
                if "raw_response" in d and isinstance(d["raw_response"], str) and len(d["raw_response"]) > 2000:
                    d["raw_response"] = d["raw_response"][:2000] + "...[truncated]"
                out.append(d)
            except Exception as e:
                logger.warning(f"Failed to load decision log {f.name}: {e}")
        return out

    def _gather_safety_filtered_stats(self, decisions: list[dict]) -> dict:
        """결정 로그에서 룰 차단 통계 추출."""
        rule_counts: dict[str, int] = {}
        circuit_blocks = 0
        for d in decisions:
            sig = d.get("signal", {})
            for sf in sig.get("safety_filtered", []) or []:
                rule = sf.get("rule", "unknown")
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
            reasoning = sig.get("reasoning", "")
            if "circuit" in reasoning.lower() or "halted" in reasoning.lower():
                circuit_blocks += 1
        return {
            "rule_block_counts": rule_counts,
            "circuit_blocks": circuit_blocks,
            "total_cycles": len(decisions),
        }

    def _gather_missed_opportunities(self, decisions: list[dict], trades: list[dict]) -> list[dict]:
        """스크리닝 상위지만 매수 안 한 종목 추출 (top 5).

        오늘 BUY 액션에 등장한 ticker는 제외. 결정 로그에 candidate로 자주 등장한 종목 우선.
        """
        bought_tickers = {t.get("ticker") for t in trades if t.get("action") == "BUY"}
        held_tickers = set(self.portfolio.positions.keys()) if hasattr(self.portfolio, "positions") else set()

        candidate_freq: dict[str, dict] = {}
        for d in decisions:
            raw = d.get("raw_response", "")
            # candidates 리스트는 raw_response 안에 있음. 간단히 ticker 패턴 카운트.
            import re
            for ticker in re.findall(r'"(\d{6})"', raw):
                if ticker in bought_tickers or ticker in held_tickers:
                    continue
                info = candidate_freq.setdefault(ticker, {"ticker": ticker, "appearance": 0})
                info["appearance"] += 1

        # screening_results에서 오늘 상위 점수 후보 추가 (보조)
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT candidates_json FROM screening_results ORDER BY date DESC LIMIT 1"
                ).fetchone()
            if row:
                cands = json.loads(row[0])[:10]
                for c in cands:
                    t = c.get("ticker")
                    if not t or t in bought_tickers or t in held_tickers:
                        continue
                    info = candidate_freq.setdefault(t, {"ticker": t, "appearance": 0})
                    info["name"] = c.get("name", "")
                    info["score"] = c.get("composite_score")
        except Exception as e:
            logger.debug(f"Screening lookup failed: {e}")

        sorted_missed = sorted(candidate_freq.values(), key=lambda x: -x.get("appearance", 0))[:5]
        return sorted_missed

    def _gather_portfolio_change(self) -> dict:
        try:
            snapshots = self.portfolio.get_daily_snapshots(days=2)
        except Exception:
            snapshots = []
        if len(snapshots) >= 2:
            return {
                "today_asset": snapshots[0].get("total_asset"),
                "yesterday_asset": snapshots[1].get("total_asset"),
                "daily_pnl": snapshots[0].get("daily_pnl", 0),
                "daily_pnl_pct": snapshots[0].get("daily_pnl_pct", 0),
            }
        return {
            "total_asset": getattr(self.portfolio, "total_asset", 0),
            "total_pnl": getattr(self.portfolio, "total_pnl", 0),
        }

    def _gather_portfolio_summary(self) -> dict:
        """현재 포지션 + 현금 비율 + 섹터 분포 요약."""
        try:
            positions = self.portfolio.positions
        except Exception:
            return {}
        total = getattr(self.portfolio, "total_asset", 0) or 1
        pos_list = []
        for ticker, pos in positions.items():
            pos_list.append({
                "ticker": ticker,
                "name": getattr(pos, "name", ticker),
                "strategy_type": getattr(pos, "strategy_type", "swing"),
                "pnl_pct": round(getattr(pos, "pnl_pct", 0), 4),
                "weight": round((getattr(pos, "market_value", 0) or 0) / total, 4),
                "days_held": (datetime.now().date() - datetime.fromisoformat(getattr(pos, "bought_at", datetime.now().isoformat())).date()).days,
            })
        return {
            "positions": pos_list,
            "cash_ratio": round((getattr(self.portfolio, "cash", 0) or 0) / total, 4),
            "total_asset": total,
        }

    def format_deep_review_telegram(self, review: dict) -> str:
        """심층 복기 결과를 텔레그램 메시지로 포맷."""
        date = review.get("date", "")
        model = review.get("model_used", "")
        headline = review.get("headline", "오늘의 복기")
        score = review.get("overall_score", "?")

        lines = [f"📊 <b>일일 심층 복기</b> ({date})"]
        if review.get("fallback_note"):
            lines.append(f"⚠️ {review['fallback_note']}")
        lines.append(f"종합 점수: {score}/10")
        lines.append(f"<i>{headline}</i>")
        lines.append("")

        findings = review.get("key_findings", []) or []
        if findings:
            lines.append("📍 <b>핵심 발견</b>")
            for f in findings[:5]:
                lines.append(f"• {f}")
            lines.append("")

        rc = review.get("rule_consistency", {}) or {}
        if rc:
            lines.append(f"⚖️ <b>룰 일관성</b> {rc.get('score', '?')}/10")
            if rc.get("notes"):
                lines.append(f"{rc['notes']}")
            lines.append("")

        missed = review.get("missed_opportunities", []) or []
        if missed:
            lines.append(f"🎯 <b>놓친 기회</b> ({len(missed)}건)")
            for m in missed[:5]:
                lines.append(f"• {m.get('ticker', '')} {m.get('name', '')} — {m.get('reason_not_bought', '')[:50]} → {m.get('evaluation', '')}")
            lines.append("")

        conflicts = review.get("rule_conflicts", []) or []
        if conflicts:
            lines.append(f"🚧 <b>룰 충돌</b> ({len(conflicts)}건)")
            for c in conflicts[:5]:
                lines.append(f"• {c.get('rule', '')}: {c.get('blocked', '')[:60]} → {c.get('verdict', '')}")
            lines.append("")

        if review.get("portfolio_balance"):
            lines.append(f"💼 <b>포트폴리오</b>")
            lines.append(review["portfolio_balance"])
            lines.append("")

        improvements = review.get("improvements", []) or []
        if improvements:
            lines.append("✅ <b>내일 개선점</b>")
            for imp in improvements[:5]:
                lines.append(f"• {imp}")
            lines.append("")

        sh = review.get("strategy_modification_hint", {}) or {}
        if sh.get("needed"):
            lines.append("🔧 <b>전략 수정 힌트</b>")
            lines.append(f"룰: {sh.get('rule_id', '')}")
            lines.append(f"현재: {sh.get('current', '')[:80]}")
            lines.append(f"제안: {sh.get('suggestion', '')[:120]}")

        lines.append(f"\n<i>[{model}]</i>")
        return "\n".join(lines)
