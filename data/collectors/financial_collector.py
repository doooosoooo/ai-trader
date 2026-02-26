"""재무제표 수집 — DART API 기반."""

import os
from datetime import datetime

import requests
from loguru import logger


class FinancialCollector:
    """기업 재무제표 데이터 수집."""

    BASE_URL = "https://opendart.fss.or.kr/api"

    def __init__(self):
        self.api_key = os.getenv("DART_API_KEY", "")

    def get_financial_summary(self, corp_code: str, year: str = "") -> dict:
        """재무제표 주요 항목 수집."""
        if not self.api_key:
            return {}

        if not year:
            year = str(datetime.now().year - 1)

        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bsns_year": year,
            "reprt_code": "11011",  # 사업보고서
        }

        try:
            resp = requests.get(
                f"{self.BASE_URL}/fnlttSinglAcnt.json",
                params=params, timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "000":
                return {}

            summary = {}
            for item in data.get("list", []):
                account = item.get("account_nm", "")
                value = item.get("thstrm_amount", "0")
                try:
                    value = int(value.replace(",", ""))
                except (ValueError, AttributeError):
                    value = 0

                if "매출액" in account:
                    summary["revenue"] = value
                elif "영업이익" in account and "비영업" not in account:
                    summary["operating_profit"] = value
                elif "당기순이익" in account:
                    summary["net_income"] = value
                elif "자산총계" in account:
                    summary["total_assets"] = value
                elif "부채총계" in account:
                    summary["total_liabilities"] = value

            return summary

        except Exception as e:
            logger.warning(f"Financial data collection failed: {e}")
            return {}
