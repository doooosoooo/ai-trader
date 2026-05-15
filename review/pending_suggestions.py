"""Daily Review / Incident RCA의 룰 수정 제안을 텔레그램 인라인 버튼으로 처리.

흐름:
1. daily_review/RCA가 strategy_modification_hint 또는 rule_change_suggestion 생성
2. main.py/incident_analyzer가 PendingSuggestionStore.save()로 저장 → suggestion_id 반환
3. 텔레그램 알림에 inline keyboard ([⚙️ 파라미터 즉시 적용][📝 검토용 변환][❌ 무시]) 추가
4. 사용자가 버튼 누르면 telegram_bot._handle_callback이 라우팅
5. 적용 결과를 callback 답변으로 표시

저장 위치: data/storage/pending_suggestions.json
TTL: 7일 (조회 시 자동 정리)
"""

import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

_STORE_PATH = Path(__file__).parent.parent / "data" / "storage" / "pending_suggestions.json"
_TTL_DAYS = 7


def _rand_suffix(n: int = 4) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


class PendingSuggestionStore:
    """룰 수정 제안 저장/조회/상태 갱신 — 텔레그램 버튼 콜백용."""

    def __init__(self, store_path: Path = _STORE_PATH):
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self.store_path.exists():
            return {}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"pending_suggestions.json corrupt — reset: {e}")
            return {}
        # TTL 만료 자동 정리
        cutoff = (datetime.now() - timedelta(days=_TTL_DAYS)).isoformat()
        return {
            sid: item for sid, item in data.items()
            if item.get("created_at", "") >= cutoff
        }

    def _save(self, data: dict) -> None:
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save(
        self,
        source: str,                  # "daily_review" | "incident_rca"
        suggestion: dict,             # {rule_id, current, suggestion, ...}
        context: dict | None = None,  # 추가 컨텍스트 (severity, event_type, ticker 등)
    ) -> str:
        """제안 저장. 반환: suggestion_id (텔레그램 callback_data에 들어감)."""
        sid = f"sugg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_rand_suffix()}"
        data = self._load()
        data[sid] = {
            "id": sid,
            "source": source,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "suggestion": suggestion,
            "context": context or {},
        }
        self._save(data)
        return sid

    def get(self, sid: str) -> dict | None:
        return self._load().get(sid)

    def mark(self, sid: str, status: str, result: dict | None = None) -> None:
        """상태 갱신 — applied / converted / dismissed / failed."""
        data = self._load()
        item = data.get(sid)
        if not item:
            return
        item["status"] = status
        item["resolved_at"] = datetime.now().isoformat()
        if result:
            item["result"] = result
        data[sid] = item
        self._save(data)

    def list_pending(self) -> list[dict]:
        data = self._load()
        return sorted(
            [v for v in data.values() if v.get("status") == "pending"],
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )
