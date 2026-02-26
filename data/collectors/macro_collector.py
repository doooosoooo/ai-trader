"""매크로 지표 수집 — 금리, 환율, VIX 등."""

from datetime import datetime

import requests
from loguru import logger


class MacroCollector:
    """매크로 경제 지표 수집."""

    def __init__(self, market_client=None):
        self.market = market_client
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AITrader/1.0)",
        }

    def collect_all(self) -> dict:
        """모든 매크로 지표 수집."""
        data = {}

        # 환율 (USD/KRW)
        exchange = self._get_exchange_rate()
        if exchange:
            data["usd_krw"] = exchange

        # VIX — Yahoo Finance 또는 대체 소스
        vix = self._get_vix()
        if vix is not None:
            data["vix"] = vix

        # 코스피 지수
        if self.market:
            try:
                kospi = self.market.get_kospi_index()
                data["kospi"] = kospi
            except Exception as e:
                logger.warning(f"KOSPI collection failed: {e}")

        data["collected_at"] = datetime.now().isoformat()
        return data

    def _get_exchange_rate(self) -> dict | None:
        """USD/KRW 환율 수집."""
        try:
            # 네이버 금융 환율 페이지에서 간단 수집
            url = "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_USDKRW"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                import re
                match = re.search(r'class="no_today">\s*<em[^>]*>\s*([\d,.]+)', resp.text)
                if match:
                    rate = float(match.group(1).replace(",", ""))
                    return {"rate": rate, "currency": "USD/KRW"}
        except Exception as e:
            logger.warning(f"Exchange rate collection failed: {e}")
        return None

    def _get_vix(self) -> float | None:
        """VIX (공포지수) 수집."""
        try:
            # 간단한 VIX 수집 — 여러 소스 시도
            url = "https://finance.naver.com/world/sise.naver?symbol=VIX"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                import re
                match = re.search(r'class="no_today">\s*<em[^>]*>\s*([\d,.]+)', resp.text)
                if match:
                    return float(match.group(1).replace(",", ""))
        except Exception as e:
            logger.warning(f"VIX collection failed: {e}")
        return None

    def format_for_llm(self, macro_data: dict) -> dict:
        """LLM에게 전달할 매크로 데이터 정리."""
        result = {}

        if "kospi" in macro_data:
            k = macro_data["kospi"]
            result["코스피"] = f"{k.get('index', 0):,.1f} ({k.get('change_pct', 0):+.2f}%)"

        if "usd_krw" in macro_data:
            result["환율(USD/KRW)"] = f"{macro_data['usd_krw'].get('rate', 0):,.1f}원"

        if "vix" in macro_data:
            vix = macro_data["vix"]
            level = "높음(주의)" if vix >= 30 else "보통" if vix >= 20 else "낮음"
            result["VIX"] = f"{vix:.1f} ({level})"

        return result
