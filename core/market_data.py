"""KIS OpenAPI 연동 — 인증, 시세 조회, 주문."""

import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from loguru import logger


def _safe_int(val, default=0):
    """빈 문자열/None도 안전하게 int 변환."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=0.0):
    """빈 문자열/None도 안전하게 float 변환."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class KISAuth:
    """KIS API 인증 관리 — 토큰 발급 및 갱신."""

    def __init__(self, config: dict):
        self.account_type = config.get("broker", {}).get("account_type", "virtual")
        self.account_product_code = os.getenv("KIS_ACCOUNT_PRODUCT_CODE", "01")

        # 모의/실전에 따라 별도 키·계좌 사용
        if self.account_type == "virtual":
            self.app_key = os.getenv("KIS_VIRTUAL_APP_KEY", "")
            self.app_secret = os.getenv("KIS_VIRTUAL_APP_SECRET", "")
            self.account_no = os.getenv("KIS_VIRTUAL_ACCOUNT_NO", "")
            self.base_url = config.get("broker", {}).get(
                "base_url_virtual",
                "https://openapivts.koreainvestment.com:29443",
            )
        else:
            self.app_key = os.getenv("KIS_REAL_APP_KEY", "")
            self.app_secret = os.getenv("KIS_REAL_APP_SECRET", "")
            self.account_no = os.getenv("KIS_REAL_ACCOUNT_NO", "")
            self.base_url = config.get("broker", {}).get(
                "base_url_real",
                "https://openapi.koreainvestment.com:9443",
            )

        self._access_token: str = ""
        self._token_expires: datetime = datetime.min

    @property
    def is_authenticated(self) -> bool:
        return bool(self._access_token) and datetime.now() < self._token_expires

    def get_token(self) -> str:
        if self.is_authenticated:
            return self._access_token
        return self._issue_token()

    def _issue_token(self) -> str:
        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        try:
            resp = requests.post(url, json=body, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self._access_token = data["access_token"]
            # 토큰 유효시간 — 보통 24시간, 여유 있게 23시간으로 설정
            self._token_expires = datetime.now() + timedelta(hours=23)
            logger.info("KIS API token issued successfully")
            return self._access_token
        except Exception as e:
            logger.error(f"KIS token issue failed: {e}")
            raise

    def get_hashkey(self, body: dict) -> str:
        """POST 요청용 hashkey 생성."""
        url = f"{self.base_url}/uapi/hashkey"
        headers = {
            "Content-Type": "application/json",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp.json()["HASH"]
        except Exception as e:
            logger.error(f"Hashkey generation failed: {e}")
            raise

    def build_headers(self, tr_id: str, hashkey: str = "") -> dict:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.get_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
        }
        if hashkey:
            headers["hashkey"] = hashkey
        return headers


class MarketDataClient:
    """KIS API를 통한 시장 데이터 수집."""

    def __init__(self, auth: KISAuth):
        self.auth = auth

    def _get(self, path: str, tr_id: str, params: dict) -> dict:
        url = f"{self.auth.base_url}{path}"
        headers = self.auth.build_headers(tr_id)
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") != "0":
                logger.warning(f"KIS API warning: {data.get('msg1', 'unknown')}")
            return data
        except Exception as e:
            logger.error(f"KIS API GET failed [{path}]: {e}")
            raise

    def get_current_price(self, ticker: str) -> dict:
        """현재가 조회 (국내 주식)."""
        path = "/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 주식
            "FID_INPUT_ISCD": ticker,
        }
        # 모의투자/실전 tr_id 구분
        tr_id = "FHKST01010100"
        data = self._get(path, tr_id, params)
        output = data.get("output", {})
        return {
            "ticker": ticker,
            "name": output.get("hts_kor_isnm", ""),
            "price": _safe_int(output.get("stck_prpr", 0)),
            "change_pct": _safe_float(output.get("prdy_ctrt", 0)),
            "volume": _safe_int(output.get("acml_vol", 0)),
            "high": _safe_int(output.get("stck_hgpr", 0)),
            "low": _safe_int(output.get("stck_lwpr", 0)),
            "open": _safe_int(output.get("stck_oprc", 0)),
            "prev_close": _safe_int(output.get("stck_sdpr", 0)),
            "market_cap": _safe_int(output.get("hts_avls", 0)),
            "per": _safe_float(output.get("per", 0)),
            "pbr": _safe_float(output.get("pbr", 0)),
            "timestamp": datetime.now().isoformat(),
        }

    def get_daily_ohlcv(self, ticker: str, period: str = "D", count: int = 100) -> list[dict]:
        """일봉/주봉/월봉 OHLCV 조회.

        period: D(일), W(주), M(월)
        """
        path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=count * 2)).strftime("%Y%m%d")

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": period,
            "FID_ORG_ADJ_PRC": "0",  # 수정주가
        }
        tr_id = "FHKST03010100"
        data = self._get(path, tr_id, params)

        records = []
        for item in data.get("output2", [])[:count]:
            records.append({
                "date": item.get("stck_bsop_date", ""),
                "open": _safe_int(item.get("stck_oprc", 0)),
                "high": _safe_int(item.get("stck_hgpr", 0)),
                "low": _safe_int(item.get("stck_lwpr", 0)),
                "close": _safe_int(item.get("stck_clpr", 0)),
                "volume": _safe_int(item.get("acml_vol", 0)),
                "amount": _safe_int(item.get("acml_tr_pbmn", 0)),
            })

        return sorted(records, key=lambda x: x["date"])

    def get_minute_ohlcv(self, ticker: str, interval: str = "1") -> list[dict]:
        """분봉 데이터 조회. interval: 1, 3, 5, 10, 15, 30, 60."""
        path = "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        now = datetime.now().strftime("%H%M%S")

        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_HOUR_1": now,
            "FID_PW_DATA_INCU_YN": "Y",
        }
        tr_id = "FHKST03010200"
        data = self._get(path, tr_id, params)

        records = []
        for item in data.get("output2", []):
            records.append({
                "time": item.get("stck_cntg_hour", ""),
                "open": _safe_int(item.get("stck_oprc", 0)),
                "high": _safe_int(item.get("stck_hgpr", 0)),
                "low": _safe_int(item.get("stck_lwpr", 0)),
                "close": _safe_int(item.get("stck_prpr", 0)),
                "volume": _safe_int(item.get("cntg_vol", 0)),
            })

        return records

    def get_investor_trends(self, ticker: str) -> dict:
        """투자자별 매매동향 (외국인, 기관)."""
        path = "/uapi/domestic-stock/v1/quotations/inquire-investor"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
        }
        tr_id = "FHKST01010900"
        data = self._get(path, tr_id, params)

        output = data.get("output", [])
        if not output:
            return {"foreign_net": 0, "institution_net": 0}

        # 첫 번째 항목이 당일 데이터
        today = output[0] if output else {}
        return {
            "foreign_net": _safe_int(today.get("frgn_ntby_qty", 0)),
            "institution_net": _safe_int(today.get("orgn_ntby_qty", 0)),
            "individual_net": _safe_int(today.get("prsn_ntby_qty", 0)),
        }

    def get_kospi_index(self) -> dict:
        """코스피 지수 조회."""
        path = "/uapi/domestic-stock/v1/quotations/inquire-index-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": "U",
            "FID_INPUT_ISCD": "0001",  # 코스피
        }
        tr_id = "FHPUP02100000"
        data = self._get(path, tr_id, params)
        output = data.get("output", {})
        return {
            "index": _safe_float(output.get("bstp_nmix_prpr", 0)),
            "change_pct": _safe_float(output.get("bstp_nmix_prdy_ctrt", 0)),
            "volume": _safe_int(output.get("acml_vol", 0)),
        }
