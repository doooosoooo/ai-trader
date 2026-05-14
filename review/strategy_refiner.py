"""주간 전략 다듬기 (Strategy Refinement) — Opus 4.7로 active.md를 데이터 기반 미세 조정.

스케줄: 매주 토요일 10:00 (scheduler.py weekly_review cron 활용)
승인 흐름: LLM 제안 생성 → pending 저장 → 텔레그램 알림 → 사용자가 /refine_approve {id} 시에만 적용
자동 적용 절대 없음.

저장:
- data/storage/pending_refinements.json — pending dict
- logs/reviews/refinement_YYYYMMDD_HHMM.{json,md} — 제안 본문
- strategies/archive/active_pre_refine_{id}.md — 적용 전 백업
- strategies/active.md — 새 전략 (승인 후)
"""

import json
import sqlite3
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
STRATEGIES_DIR = PROJECT_ROOT / "strategies"
ARCHIVE_DIR = STRATEGIES_DIR / "archive"
ACTIVE_PATH = STRATEGIES_DIR / "active.md"
DECISIONS_DIR = PROJECT_ROOT / "logs" / "decisions"
REVIEWS_DIR = PROJECT_ROOT / "logs" / "reviews"
PENDING_PATH = PROJECT_ROOT / "data" / "storage" / "pending_refinements.json"

REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# 가드레일 — 적용 전 검증
_MIN_ACTIVE_MD_LEN = 2000
_REQUIRED_KEYWORDS = [
    "트레이딩 전략",  # 헤더
    "손절",
    "value",
    "swing",
    "daytrading",
    "strategy_type",
]
_PENDING_TTL_DAYS = 14  # 2주 지난 pending 자동 만료


class StrategyRefiner:
    """주간 전략 다듬기 — 데이터 수집 → Opus 제안 → 텔레그램 승인 게이트 → 적용."""

    def __init__(self, llm_engine, portfolio, telegram, db_path: str = "data/storage/trader.db"):
        self.llm = llm_engine
        self.portfolio = portfolio
        self.telegram = telegram
        self.db_path = db_path

    # ============================================================
    # 메인 진입점
    # ============================================================

    def run_weekly_refinement(self) -> dict:
        """주간 전략 다듬기 실행 — 토 10:00 cron에서 호출.

        반환: pending에 저장된 refinement dict (id 포함). 실패 시 빈 dict.
        """
        logger.info("Weekly strategy refinement starting...")

        trades_week = self._gather_week_trades()
        decisions_week = self._gather_week_decisions()

        # 매매도 결정도 없으면 skip
        if not trades_week and not decisions_week:
            logger.info("No trades/decisions in past 7 days — skipping weekly refinement")
            return {}

        safety_stats = self._gather_safety_stats(decisions_week)
        portfolio_summary = self._gather_portfolio_summary()
        weekly_pnl = self._gather_weekly_pnl()
        backtest_summary = self._gather_backtest_summary()
        current_active_md = ACTIVE_PATH.read_text(encoding="utf-8") if ACTIVE_PATH.exists() else ""

        try:
            refinement = self.llm.generate_weekly_refinement(
                trades_week=trades_week,
                decisions_week=decisions_week,
                backtest_summary=backtest_summary,
                safety_stats=safety_stats,
                portfolio_summary=portfolio_summary,
                weekly_pnl_summary=weekly_pnl,
                current_active_md=current_active_md,
            )
        except Exception as e:
            logger.error(f"Weekly refinement LLM call failed: {e}")
            self._notify_failure(str(e))
            return {}

        if not refinement or not refinement.get("new_active_md"):
            logger.warning("Weekly refinement returned empty / no new_active_md")
            self._notify_failure("LLM이 new_active_md를 생성하지 못함")
            return {}

        # pending 저장
        refinement_id = self._save_pending(refinement)
        refinement["id"] = refinement_id

        # 파일 아카이브 (제안 본문)
        self._save_proposal_files(refinement_id, refinement)

        # 텔레그램 승인 게이트 알림
        self._notify_pending(refinement)
        logger.info(f"Weekly refinement pending: id={refinement_id}")
        return refinement

    # ============================================================
    # 데이터 수집
    # ============================================================

    def _gather_week_trades(self) -> list[dict]:
        """최근 7거래일 매매."""
        try:
            all_trades = self.portfolio.get_trade_history(limit=300)
        except Exception as e:
            logger.warning(f"Failed to load trade history: {e}")
            return []
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        return [t for t in all_trades if t.get("timestamp", "") >= week_ago]

    def _gather_week_decisions(self) -> list[dict]:
        """최근 7거래일 logs/decisions/*.json. raw_response는 잘라냄."""
        if not DECISIONS_DIR.exists():
            return []
        week_ago = (datetime.now() - timedelta(days=7))
        out = []
        for f in sorted(DECISIONS_DIR.glob("*.json")):
            try:
                stem_date = datetime.strptime(f.stem.split("_")[0], "%Y%m%d")
            except ValueError:
                continue
            if stem_date < week_ago:
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                if "raw_response" in d:
                    # raw_response는 너무 길어 cap
                    rr = d["raw_response"]
                    if isinstance(rr, str) and len(rr) > 1000:
                        d["raw_response"] = rr[:1000] + "...[truncated]"
                out.append(d)
            except Exception as e:
                logger.debug(f"Failed to load decision {f.name}: {e}")
        return out

    def _gather_safety_stats(self, decisions: list[dict]) -> dict:
        """결정 로그에서 룰 차단/충돌 통계 추출."""
        rule_counts: dict[str, int] = {}
        circuit_blocks = 0
        action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        strategy_type_counts: dict[str, int] = {}

        for d in decisions:
            sig = d.get("signal", {})
            for sf in sig.get("safety_filtered", []) or []:
                rule = sf.get("rule", "unknown")
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
            reasoning = (sig.get("reasoning", "") or "").lower()
            if "circuit" in reasoning or "halted" in reasoning:
                circuit_blocks += 1
            for a in sig.get("actions", []) or []:
                at = (a.get("type") or "").upper()
                if at in action_counts:
                    action_counts[at] += 1
                st = a.get("strategy_type")
                if st:
                    strategy_type_counts[st] = strategy_type_counts.get(st, 0) + 1

        return {
            "rule_block_counts": rule_counts,
            "circuit_blocks": circuit_blocks,
            "total_cycles": len(decisions),
            "action_counts": action_counts,
            "strategy_type_counts": strategy_type_counts,
        }

    def _gather_portfolio_summary(self) -> dict:
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
            })
        return {
            "positions": pos_list,
            "cash_ratio": round((getattr(self.portfolio, "cash", 0) or 0) / total, 4),
            "total_asset": total,
        }

    def _gather_weekly_pnl(self) -> dict:
        try:
            snapshots = self.portfolio.get_daily_snapshots(days=10)
        except Exception:
            snapshots = []
        if not snapshots:
            return {}
        first = snapshots[-1]
        last = snapshots[0]
        start_asset = first.get("total_asset", 0) or 0
        end_asset = last.get("total_asset", 0) or 0
        weekly_return = (end_asset - start_asset) / start_asset if start_asset else 0
        daily_pnls = [s.get("daily_pnl_pct", 0) for s in snapshots]
        return {
            "start_asset": start_asset,
            "end_asset": end_asset,
            "weekly_return_pct": round(weekly_return * 100, 3),
            "num_snapshots": len(snapshots),
            "daily_pnl_pcts": daily_pnls,
        }

    def _gather_backtest_summary(self) -> dict:
        """backtest.db에서 최근 백테스트 결과 요약. 실패 시 빈 dict."""
        bt_db = PROJECT_ROOT / "data" / "storage" / "backtest.db"
        if not bt_db.exists():
            return {}
        try:
            with sqlite3.connect(str(bt_db)) as conn:
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                if "backtest_results" not in tables:
                    return {"tables": tables}
                cols = [r[1] for r in conn.execute("PRAGMA table_info(backtest_results)").fetchall()]
                rows = conn.execute(
                    "SELECT * FROM backtest_results ORDER BY rowid DESC LIMIT 10"
                ).fetchall()
            return {
                "recent_results": [dict(zip(cols, r)) for r in rows],
            }
        except Exception as e:
            logger.debug(f"Backtest summary skipped: {e}")
            return {}

    # ============================================================
    # Pending 저장/로드
    # ============================================================

    def _load_pending(self) -> dict:
        if not PENDING_PATH.exists():
            return {}
        try:
            data = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"pending_refinements.json corrupt — starting fresh: {e}")
            return {}
        # TTL 만료 정리
        cutoff = (datetime.now() - timedelta(days=_PENDING_TTL_DAYS)).isoformat()
        return {
            rid: item for rid, item in data.items()
            if item.get("created_at", "") >= cutoff and item.get("status") == "pending"
        }

    def _save_pending_dict(self, data: dict) -> None:
        PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        PENDING_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_pending(self, refinement: dict) -> str:
        """pending dict에 저장. 반환: refinement_id."""
        refinement_id = "refine_" + datetime.now().strftime("%Y%m%d_%H%M")
        data = self._load_pending()
        data[refinement_id] = {
            "id": refinement_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "refinement": refinement,
        }
        self._save_pending_dict(data)
        return refinement_id

    def list_pending(self) -> list[dict]:
        """승인 대기 중인 refinement 목록."""
        data = self._load_pending()
        return sorted(data.values(), key=lambda x: x.get("created_at", ""), reverse=True)

    def get_pending(self, refinement_id: str) -> dict | None:
        return self._load_pending().get(refinement_id)

    # ============================================================
    # 승인 / 거부
    # ============================================================

    def approve(self, refinement_id: str) -> tuple[bool, str]:
        """승인 — archive 백업 → active.md 교체 → git commit. 반환: (성공여부, 메시지)."""
        data = self._load_pending()
        item = data.get(refinement_id)
        if not item:
            return False, f"Refinement {refinement_id} 없음 (이미 처리되었거나 만료)"

        refinement = item.get("refinement", {})
        new_md = refinement.get("new_active_md", "")
        ok, reason = self._validate_new_active_md(new_md)
        if not ok:
            return False, f"검증 실패: {reason}"

        # 1. 현재 active.md 백업
        try:
            if ACTIVE_PATH.exists():
                backup_path = ARCHIVE_DIR / f"active_pre_{refinement_id}.md"
                shutil.copy2(ACTIVE_PATH, backup_path)
                logger.info(f"Active strategy backed up: {backup_path.name}")
        except Exception as e:
            return False, f"백업 실패: {e}"

        # 2. active.md 교체
        try:
            ACTIVE_PATH.write_text(new_md, encoding="utf-8")
        except Exception as e:
            return False, f"active.md 쓰기 실패: {e}"

        # 3. git commit (실패해도 적용은 성공)
        commit_msg = (
            f"전략 다듬기 {refinement_id} — {refinement.get('headline', '주간 refinement')[:80]}\n\n"
            f"승인자: doooosoooo\n승인 시각: {datetime.now().isoformat()}"
        )
        git_ok = self._git_commit(commit_msg)

        # 4. pending status 업데이트
        item["status"] = "approved"
        item["approved_at"] = datetime.now().isoformat()
        item["git_committed"] = git_ok
        data[refinement_id] = item
        self._save_pending_dict(data)

        msg = f"✅ Refinement {refinement_id} 적용 완료"
        if not git_ok:
            msg += " (git commit 실패 — 수동 커밋 필요)"
        return True, msg

    def reject(self, refinement_id: str) -> tuple[bool, str]:
        data = self._load_pending()
        item = data.get(refinement_id)
        if not item:
            return False, f"Refinement {refinement_id} 없음"
        item["status"] = "rejected"
        item["rejected_at"] = datetime.now().isoformat()
        data[refinement_id] = item
        self._save_pending_dict(data)
        return True, f"❌ Refinement {refinement_id} 거부됨"

    def _validate_new_active_md(self, text: str) -> tuple[bool, str]:
        """가드레일 — min_length + 필수 키워드."""
        if not text or not isinstance(text, str):
            return False, "new_active_md 비어있음"
        if len(text) < _MIN_ACTIVE_MD_LEN:
            return False, f"길이 부족 ({len(text)} < {_MIN_ACTIVE_MD_LEN})"
        missing = [kw for kw in _REQUIRED_KEYWORDS if kw not in text]
        if missing:
            return False, f"필수 키워드 누락: {missing}"
        return True, "ok"

    def _git_commit(self, message: str) -> bool:
        """strategies/active.md를 git에 commit. 실패해도 예외 안 던짐."""
        try:
            subprocess.run(
                ["git", "add", "strategies/active.md"],
                cwd=str(PROJECT_ROOT), check=True, capture_output=True, timeout=10,
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(PROJECT_ROOT), check=True, capture_output=True, timeout=15,
            )
            return True
        except Exception as e:
            logger.warning(f"git commit failed (manual commit needed): {e}")
            return False

    # ============================================================
    # 저장 / 알림 / 포맷
    # ============================================================

    def _save_proposal_files(self, refinement_id: str, refinement: dict) -> None:
        """logs/reviews/refinement_YYYYMMDD_HHMM.{json,md} 저장."""
        base = REVIEWS_DIR / f"{refinement_id}"
        try:
            base.with_suffix(".json").write_text(
                json.dumps(refinement, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Refinement JSON save failed: {e}")
        try:
            base.with_suffix(".md").write_text(
                self.format_refinement_md(refinement_id, refinement),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Refinement MD save failed: {e}")

    def _notify_pending(self, refinement: dict) -> None:
        """텔레그램으로 승인 요청 알림."""
        if not self.telegram or not getattr(self.telegram, "enabled", False):
            return
        try:
            self.telegram.send_alert_sync(
                "strategy_suggestion",
                self.format_refinement_telegram(refinement),
            )
        except Exception as e:
            logger.warning(f"Refinement telegram send failed: {e}")

    def _notify_failure(self, reason: str) -> None:
        if not self.telegram or not getattr(self.telegram, "enabled", False):
            return
        try:
            self.telegram.send_alert_sync(
                "error",
                f"⚠️ <b>Weekly Refinement 실패</b>\n사유: {reason[:300]}",
            )
        except Exception:
            pass

    def format_refinement_telegram(self, refinement: dict) -> str:
        rid = refinement.get("id", "?")
        headline = refinement.get("headline", "주간 전략 다듬기 제안")
        lines = [
            "🔧 <b>주간 전략 다듬기 (승인 대기)</b>",
            f"ID: <code>{rid}</code>",
            f"<i>{headline}</i>",
            "",
        ]

        obs = refinement.get("key_observations", []) or []
        if obs:
            lines.append("<b>이번 주 관찰</b>")
            for o in obs[:5]:
                lines.append(f"• {str(o)[:140]}")
            lines.append("")

        worked = refinement.get("what_worked", []) or []
        if worked:
            lines.append("<b>✅ 잘 된 점</b>")
            for w in worked[:4]:
                lines.append(f"• {str(w)[:120]}")
            lines.append("")

        failed = refinement.get("what_failed", []) or []
        if failed:
            lines.append("<b>❌ 아쉬운 점</b>")
            for w in failed[:4]:
                lines.append(f"• {str(w)[:120]}")
            lines.append("")

        changes = refinement.get("proposed_changes", []) or []
        if changes:
            lines.append(f"<b>📝 제안 변경 ({len(changes)}건)</b>")
            for c in changes[:6]:
                section = c.get("section", "")
                new = c.get("new", "")
                rationale = c.get("rationale", "")
                lines.append(f"• [{section[:30]}] {new[:80]}")
                if rationale:
                    lines.append(f"  └ 근거: {rationale[:100]}")
            lines.append("")

        risk = refinement.get("risk_notes", "")
        if risk:
            lines.append(f"<b>⚠️ 주의</b>")
            lines.append(str(risk)[:300])
            lines.append("")

        rid_safe = rid.replace("refine_", "")
        lines.extend([
            "<b>승인하려면</b>:",
            f"/refine_show {rid}  — 전문 확인",
            f"/refine_approve {rid}  — 적용",
            f"/refine_reject {rid}  — 거부",
        ])
        return "\n".join(lines)

    def format_refinement_md(self, refinement_id: str, refinement: dict) -> str:
        lines = [
            f"# 주간 전략 다듬기 — {refinement_id}",
            "",
            f"- 생성 시각: {datetime.now().isoformat()}",
            f"- 헤드라인: {refinement.get('headline', '')}",
            "",
            "## 이번 주 관찰",
        ]
        for o in refinement.get("key_observations", []) or []:
            lines.append(f"- {o}")

        worked = refinement.get("what_worked", []) or []
        if worked:
            lines.extend(["", "## 잘 된 점"])
            for w in worked:
                lines.append(f"- {w}")

        failed = refinement.get("what_failed", []) or []
        if failed:
            lines.extend(["", "## 아쉬운 점"])
            for w in failed:
                lines.append(f"- {w}")

        changes = refinement.get("proposed_changes", []) or []
        if changes:
            lines.extend(["", "## 제안 변경"])
            for c in changes:
                lines.extend([
                    f"### {c.get('section', '')}",
                    f"- 현재: {c.get('current', '')}",
                    f"- 수정안: {c.get('new', '')}",
                    f"- 근거: {c.get('rationale', '')}",
                    f"- 기대 효과: {c.get('expected_impact', '')}",
                ])

        risk = refinement.get("risk_notes", "")
        if risk:
            lines.extend(["", "## 모니터링 포인트", risk])

        new_md = refinement.get("new_active_md", "")
        if new_md:
            lines.extend(["", "## 새 active.md (전문)", "```markdown", new_md, "```"])

        return "\n".join(lines)
