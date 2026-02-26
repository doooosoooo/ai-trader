"""일일 자기 복기 — LLM 기반 매매 판단 평가."""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

REVIEWS_DIR = Path(__file__).parent / "reports"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

LOG_REVIEWS_DIR = Path(__file__).parent.parent / "logs" / "reviews"
LOG_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)


class DailyReviewer:
    """일일/주간/월간 자기 복기."""

    def __init__(self, llm_engine, portfolio):
        self.llm = llm_engine
        self.portfolio = portfolio

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
