"""전략 성과 추적 및 버전 비교."""

import json
import shutil
from datetime import datetime
from pathlib import Path

from loguru import logger

STRATEGIES_DIR = Path(__file__).parent.parent / "strategies"


class StrategyEvaluator:
    """전략 버전 관리 및 성과 추적."""

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self._meta_path = STRATEGIES_DIR / "strategy_meta.json"
        self._meta = self._load_meta()

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        return {"versions": [], "active_since": None}

    def _save_meta(self):
        self._meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def archive_current_strategy(self, reason: str = "") -> str:
        """현재 전략을 아카이브로 저장."""
        active = STRATEGIES_DIR / "active.md"
        if not active.exists():
            return ""

        # 버전명 생성
        version_num = len(self._meta.get("versions", [])) + 1
        timestamp = datetime.now().strftime("%Y%m%d")
        version_name = f"v{version_num}_{timestamp}"

        archive_dir = STRATEGIES_DIR / "archive"
        archive_dir.mkdir(exist_ok=True)
        archive_path = archive_dir / f"{version_name}.md"

        shutil.copy2(active, archive_path)

        # 메타데이터 갱신
        version_meta = {
            "name": version_name,
            "archived_at": datetime.now().isoformat(),
            "active_since": self._meta.get("active_since"),
            "reason": reason,
            "portfolio_snapshot": {
                "total_asset": self.portfolio.total_asset,
                "total_pnl_pct": self.portfolio.total_pnl_pct,
            },
        }
        self._meta.setdefault("versions", []).append(version_meta)
        self._save_meta()

        logger.info(f"Strategy archived: {version_name}")
        return version_name

    def activate_strategy(self, content: str | None = None, path: Path | None = None) -> None:
        """새 전략 활성화."""
        active = STRATEGIES_DIR / "active.md"

        if path:
            shutil.copy2(path, active)
        elif content:
            active.write_text(content, encoding="utf-8")

        self._meta["active_since"] = datetime.now().isoformat()
        self._save_meta()
        logger.info("New strategy activated")

    def rollback(self, version_name: str) -> bool:
        """특정 버전으로 롤백."""
        # 현재 전략 아카이브
        self.archive_current_strategy(reason=f"rollback to {version_name}")

        # 대상 버전 찾기
        archive_dir = STRATEGIES_DIR / "archive"
        matches = list(archive_dir.glob(f"*{version_name}*"))

        if not matches:
            logger.warning(f"Strategy version not found: {version_name}")
            return False

        target = matches[0]
        self.activate_strategy(path=target)
        logger.info(f"Strategy rolled back to: {target.name}")
        return True

    def get_version_history(self) -> list[dict]:
        """전략 버전 이력 반환."""
        return self._meta.get("versions", [])

    def compare_versions(self, v1: str, v2: str) -> dict:
        """두 버전의 성과 비교."""
        versions = {v["name"]: v for v in self._meta.get("versions", [])}

        if v1 not in versions or v2 not in versions:
            return {"error": "Version not found"}

        snap1 = versions[v1].get("portfolio_snapshot", {})
        snap2 = versions[v2].get("portfolio_snapshot", {})

        return {
            v1: snap1,
            v2: snap2,
            "pnl_diff": snap2.get("total_pnl_pct", 0) - snap1.get("total_pnl_pct", 0),
        }
