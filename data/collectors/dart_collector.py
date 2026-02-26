"""DART 공시 수집 — DART OpenAPI 기반."""

import os
from datetime import datetime, timedelta

import requests
from loguru import logger


class DartCollector:
    """DART 전자공시 수집."""

    BASE_URL = "https://opendart.fss.or.kr/api"

    def __init__(self):
        self.api_key = os.getenv("DART_API_KEY", "")

    def collect_recent_disclosures(
        self,
        corp_code: str | None = None,
        days: int = 1,
        count: int = 20,
    ) -> list[dict]:
        """최근 공시 수집."""
        if not self.api_key:
            logger.warning("DART_API_KEY not set, skipping disclosure collection")
            return []

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        params = {
            "crtfc_key": self.api_key,
            "bgn_de": start_date,
            "end_de": end_date,
            "page_count": str(count),
            "sort": "date",
            "sort_mth": "desc",
        }

        if corp_code:
            params["corp_code"] = corp_code

        try:
            resp = requests.get(
                f"{self.BASE_URL}/list.json", params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "000":
                logger.warning(f"DART API: {data.get('message', 'unknown error')}")
                return []

            disclosures = []
            for item in data.get("list", []):
                disclosures.append({
                    "corp_name": item.get("corp_name", ""),
                    "corp_code": item.get("corp_code", ""),
                    "stock_code": item.get("stock_code", ""),
                    "report_nm": item.get("report_nm", ""),
                    "rcept_dt": item.get("rcept_dt", ""),
                    "flr_nm": item.get("flr_nm", ""),  # 공시 제출인
                })

            logger.info(f"Collected {len(disclosures)} DART disclosures")
            return disclosures

        except Exception as e:
            logger.warning(f"DART collection failed: {e}")
            return []

    def summarize_for_llm(self, disclosures: list[dict]) -> str:
        """공시를 LLM 입력용 텍스트로 변환."""
        if not disclosures:
            return ""

        lines = []
        for d in disclosures:
            stock = f"({d['stock_code']})" if d.get("stock_code") else ""
            lines.append(f"- [{d['rcept_dt']}] {d['corp_name']}{stock}: {d['report_nm']}")

        return "\n".join(lines)
