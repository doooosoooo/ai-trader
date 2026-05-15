"""텔레그램 인라인 버튼으로 트리거되는 룰 수정 제안 적용 엔진.

- apply_param(): 자연어 suggestion → Opus가 trading-params {param, value} 추출
  → config_manager.apply_adjustments로 화이트리스트 검증 + 즉시 적용
- convert_to_refinement(): active.md 자연어 수정 제안 → Opus가 새 active.md 전문 생성
  → strategy_refiner pending 큐에 저장 (사용자가 추가로 /refine_approve 필요)
"""

from datetime import datetime

from loguru import logger


class SuggestionApplier:
    """텔레그램 버튼 콜백 → 룰 수정 제안 적용."""

    def __init__(self, llm_engine, config_manager, strategy_refiner, pending_store):
        self.llm = llm_engine
        self.config_manager = config_manager
        self.strategy_refiner = strategy_refiner
        self.pending_store = pending_store

    def apply_param(self, suggestion_id: str) -> tuple[bool, str]:
        """파라미터 즉시 적용. 반환: (성공 여부, 사용자에게 보여줄 메시지)."""
        item = self.pending_store.get(suggestion_id)
        if not item:
            return False, "제안을 찾을 수 없음 (만료되었거나 이미 처리됨)"
        if item.get("status") != "pending":
            return False, f"이미 처리됨 (status: {item.get('status')})"

        sugg = item.get("suggestion", {}) or {}
        rule_id = sugg.get("rule_id", "")
        suggestion_text = sugg.get("suggestion", "") or sugg.get("current", "")

        if not suggestion_text:
            self.pending_store.mark(suggestion_id, "failed", {"error": "suggestion 본문 비어있음"})
            return False, "제안 본문이 비어있어 적용 불가"

        # 1) Opus/Sonnet으로 자연어 → {param, value, reason} 추출
        try:
            current_params = self.config_manager.get_all_params_flat()
            limits = self.config_manager.safety_rules.get("adjustable_limits", {})
            extracted = self.llm.extract_param_adjustment(
                suggestion_text=suggestion_text,
                rule_id=rule_id,
                current_params_flat=current_params,
                adjustable_limits=limits,
            )
        except Exception as e:
            logger.error(f"extract_param_adjustment failed: {e}")
            self.pending_store.mark(suggestion_id, "failed", {"error": str(e)})
            return False, f"LLM 추출 실패: {str(e)[:100]}"

        if not extracted or extracted.get("error"):
            err = (extracted or {}).get("error", "unknown")
            self.pending_store.mark(suggestion_id, "failed", {"error": err})
            return False, f"파라미터 추출 불가: {err[:120]}\n→ active.md 전략 수정이 필요한 제안일 수 있음. [📝 검토용 변환] 사용"

        param = extracted.get("param")
        value = extracted.get("value")
        reason = extracted.get("reason", "")[:200]
        if not param or value is None:
            self.pending_store.mark(suggestion_id, "failed", {"error": "param/value 누락"})
            return False, "추출 결과에 param/value 누락"

        # 2) config_manager.apply_adjustments — 화이트리스트 + 범위 검증 + 적용
        try:
            results = self.config_manager.apply_adjustments([{
                "param": param,
                "value": value,
                "reason": f"{reason} [via Telegram button, {item.get('source', '?')}]",
            }])
        except Exception as e:
            logger.error(f"apply_adjustments failed: {e}")
            self.pending_store.mark(suggestion_id, "failed", {"error": str(e)})
            return False, f"적용 실패: {str(e)[:100]}"

        r = results[0] if results else {}
        if r.get("applied"):
            self.pending_store.mark(suggestion_id, "applied", r)
            old = r.get("old_value")
            return True, f"✅ 적용: <code>{param}</code> {old} → {value}\n사유: {reason[:100]}"
        else:
            self.pending_store.mark(suggestion_id, "failed", r)
            return False, f"적용 차단: {r.get('reason', '')[:150]}"

    def convert_to_refinement(self, suggestion_id: str) -> tuple[bool, str]:
        """active.md 검토용 변환 — Opus가 새 active.md 전문 생성 → pending_refinement에 저장.

        반환: (성공 여부, 메시지). 성공 시 메시지에 refinement_id와 다음 명령 안내.
        """
        item = self.pending_store.get(suggestion_id)
        if not item:
            return False, "제안을 찾을 수 없음 (만료/처리됨)"
        if item.get("status") != "pending":
            return False, f"이미 처리됨 (status: {item.get('status')})"

        sugg = item.get("suggestion", {}) or {}
        rule_id = sugg.get("rule_id", "")
        suggestion_text = sugg.get("suggestion", "")

        if not suggestion_text:
            self.pending_store.mark(suggestion_id, "failed", {"error": "suggestion 본문 비어있음"})
            return False, "제안 본문이 비어있어 변환 불가"

        # 1) Opus로 active.md 변경본 생성
        try:
            from pathlib import Path
            active_path = Path(__file__).parent.parent / "strategies" / "active.md"
            current_active_md = active_path.read_text(encoding="utf-8") if active_path.exists() else ""

            refinement = self.llm.apply_suggestion_to_strategy(
                suggestion_text=suggestion_text,
                rule_id=rule_id,
                current_active_md=current_active_md,
            )
        except Exception as e:
            logger.error(f"apply_suggestion_to_strategy failed: {e}")
            self.pending_store.mark(suggestion_id, "failed", {"error": str(e)})
            return False, f"Opus 변환 실패: {str(e)[:120]}"

        if not refinement or not refinement.get("new_active_md"):
            self.pending_store.mark(suggestion_id, "failed", {"error": "new_active_md 비어있음"})
            return False, "Opus가 변경본을 생성하지 못함"

        # 2) strategy_refiner pending 큐에 저장 — 기존 /refine_approve {id} 흐름 재사용
        try:
            # weekly_refinement와 동일 구조로 보강
            refinement.setdefault("key_observations", [
                f"[{item.get('source', '?')}] {rule_id} — 텔레그램 버튼 변환"
            ])
            refinement.setdefault("what_worked", [])
            refinement.setdefault("what_failed", [])
            refinement_id = self.strategy_refiner._save_pending(refinement)
            refinement["id"] = refinement_id
            self.strategy_refiner._save_proposal_files(refinement_id, refinement)
        except Exception as e:
            logger.error(f"strategy_refiner save failed: {e}")
            self.pending_store.mark(suggestion_id, "failed", {"error": str(e)})
            return False, f"저장 실패: {str(e)[:120]}"

        self.pending_store.mark(suggestion_id, "converted", {"refinement_id": refinement_id})
        headline = refinement.get("headline", "변환 완료")
        return True, (
            f"📝 변환 완료\n"
            f"ID: <code>{refinement_id}</code>\n"
            f"<i>{headline[:120]}</i>\n\n"
            f"다음 단계:\n"
            f"/refine_show {refinement_id} — 전문 확인\n"
            f"/refine_approve {refinement_id} — 적용\n"
            f"/refine_reject {refinement_id} — 거부"
        )

    def dismiss(self, suggestion_id: str) -> tuple[bool, str]:
        item = self.pending_store.get(suggestion_id)
        if not item:
            return False, "제안을 찾을 수 없음"
        if item.get("status") != "pending":
            return False, f"이미 처리됨 (status: {item.get('status')})"
        self.pending_store.mark(suggestion_id, "dismissed")
        return True, "❌ 무시 처리됨"
