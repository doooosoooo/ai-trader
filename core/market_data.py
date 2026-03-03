"""KIS OpenAPI 연동 — 인증, 시세 조회, 주문."""

import hashlib
import json
import os
import threading
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
        self._token_lock = threading.Lock()

    @property
    def is_authenticated(self) -> bool:
        return bool(self._access_token) and datetime.now() < self._token_expires

    def get_token(self) -> str:
        with self._token_lock:
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
            rt_cd = data.get("rt_cd", "")
            if rt_cd != "0":
                error_msg = data.get("msg1", "unknown")
                logger.error(f"KIS API error [{path}] rt_cd={rt_cd}: {error_msg}")
                raise RuntimeError(f"KIS API error [{rt_cd}]: {error_msg}")
            return data
        except RuntimeError:
            raise
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

    def get_daily_ohlcv_range(
        self, ticker: str, start_date: str, end_date: str, period: str = "D",
    ) -> list[dict]:
        """날짜 범위 지정 일봉 조회 (페이지네이션 지원).

        KIS API가 1회 ~100바 제한이므로 end_date를 뒤로 이동하며 반복 호출.
        Args:
            start_date: "20240301" 형식
            end_date: "20260227" 형식
        """
        path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        tr_id = "FHKST03010100"
        all_records = []
        current_end = end_date

        for _ in range(15):  # 최대 15회 (~1500 거래일 ≈ 6년)
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": current_end,
                "FID_PERIOD_DIV_CODE": period,
                "FID_ORG_ADJ_PRC": "0",
            }
            data = self._get(path, tr_id, params)
            batch = data.get("output2", [])
            if not batch:
                break

            for item in batch:
                date = item.get("stck_bsop_date", "")
                if not date or date < start_date:
                    continue
                all_records.append({
                    "date": date,
                    "open": _safe_int(item.get("stck_oprc", 0)),
                    "high": _safe_int(item.get("stck_hgpr", 0)),
                    "low": _safe_int(item.get("stck_lwpr", 0)),
                    "close": _safe_int(item.get("stck_clpr", 0)),
                    "volume": _safe_int(item.get("acml_vol", 0)),
                    "amount": _safe_int(item.get("acml_tr_pbmn", 0)),
                })

            # 배치에서 가장 오래된 날짜 찾기
            oldest = min(item.get("stck_bsop_date", "") for item in batch)
            if oldest <= start_date:
                break

            # 다음 호출: oldest 하루 전까지
            oldest_dt = datetime.strptime(oldest, "%Y%m%d")
            current_end = (oldest_dt - timedelta(days=1)).strftime("%Y%m%d")
            time.sleep(0.5)  # API 부하 방지

        # 중복 제거 + 정렬
        seen = set()
        unique = []
        for r in all_records:
            if r["date"] not in seen:
                seen.add(r["date"])
                unique.append(r)

        return sorted(unique, key=lambda x: x["date"])

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

    def get_account_balance(self) -> dict:
        """계좌 잔고 및 보유종목 조회 (KIS API).

        Returns:
            {
                "cash": int,               # 예수금
                "total_asset": int,         # 총자산
                "positions": [              # 보유종목 리스트
                    {"ticker": str, "name": str, "quantity": int,
                     "avg_price": float, "current_price": int, "pnl": float, "pnl_pct": float},
                ]
            }
        """
        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        # 실전/모의 tr_id 구분
        if self.auth.account_type == "virtual":
            tr_id = "VTTC8434R"
        else:
            tr_id = "TTTC8434R"

        params = {
            "CANO": self.auth.account_no[:8],
            "ACNT_PRDT_CD": self.auth.account_product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        try:
            data = self._get(path, tr_id, params)
        except RuntimeError as e:
            # 장 외 시간 등에서 rt_cd != "0" 반환 시 빈 결과로 안전하게 처리
            logger.warning(f"Account balance query failed (may be outside trading hours): {e}")
            return {"cash": 0, "total_asset": 0, "positions": []}

        # 보유종목 파싱
        positions = []
        for item in data.get("output1", []):
            qty = _safe_int(item.get("hldg_qty", 0))
            if qty <= 0:
                continue
            positions.append({
                "ticker": item.get("pdno", ""),
                "name": item.get("prdt_name", ""),
                "quantity": qty,
                "avg_price": _safe_float(item.get("pchs_avg_pric", 0)),
                "current_price": _safe_int(item.get("prpr", 0)),
                "pnl": _safe_float(item.get("evlu_pfls_amt", 0)),
                "pnl_pct": _safe_float(item.get("evlu_pfls_rt", 0)),
            })

        # 계좌 요약 — 여러 필드 시도 (증권사 API 응답 형식 차이 대응)
        output2 = data.get("output2", [{}])
        summary = output2[0] if output2 else {}

        invested = sum(p["quantity"] * p["current_price"] for p in positions)

        # 총평가금액(tot_evlu_amt)이 가장 신뢰할 수 있는 기준값
        total_asset = _safe_int(summary.get("tot_evlu_amt", 0))

        # 예수금: dnca_tot_amt 사용. 0이면 total_asset에서 역산
        cash = _safe_int(summary.get("dnca_tot_amt", 0))

        if total_asset > 0 and invested > 0:
            # tot_evlu_amt(총평가금액)이 가장 정확 → cash를 항상 역산
            # dnca_tot_amt는 결제완료 예수금만 포함하여 미결제 매도대금이 빠짐
            cash = total_asset - invested
        elif total_asset == 0 and cash > 0:
            total_asset = cash + invested
        elif total_asset == 0 and cash == 0:
            # 둘 다 0이면 nass_amt(순자산) 시도
            nass = _safe_int(summary.get("nass_amt", 0))
            if nass > 0:
                total_asset = nass
                cash = nass - invested
            else:
                total_asset = invested
                cash = 0

        logger.info(f"Account balance raw: dnca_tot_amt={summary.get('dnca_tot_amt')}, "
                    f"nass_amt={summary.get('nass_amt')}, tot_evlu_amt={summary.get('tot_evlu_amt')}, "
                    f"scts_evlu_amt={summary.get('scts_evlu_amt')}, "
                    f"pchs_amt_smtl_amt={summary.get('pchs_amt_smtl_amt')}, "
                    f"evlu_amt_smtl_amt={summary.get('evlu_amt_smtl_amt')}, "
                    f"→ cash={cash}, total_asset={total_asset}, invested={invested}")

        return {
            "cash": cash,
            "total_asset": total_asset,
            "positions": positions,
        }

    def get_today_orders(self) -> list[dict]:
        """당일 주문 체결 조회."""
        from datetime import datetime
        path = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        today = datetime.now().strftime("%Y%m%d")
        tr_id = "TTTC8001R"
        if self.auth.account_type == "virtual":
            tr_id = "VTTC8001R"

        params = {
            "CANO": self.auth.account_no[:8],
            "ACNT_PRDT_CD": self.auth.account_product_code,
            "INQR_STRT_DT": today,
            "INQR_END_DT": today,
            "SLL_BUY_DVSN_CD": "00",
            "INQR_DVSN": "01",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        try:
            data = self._get(path, tr_id, params)
        except RuntimeError as e:
            logger.warning(f"Order inquiry failed: {e}")
            return []

        orders = []
        for o in data.get("output1", []):
            ord_qty = _safe_int(o.get("ord_qty", 0))
            if ord_qty <= 0:
                continue
            ccld_qty = _safe_int(o.get("tot_ccld_qty", 0))
            orders.append({
                "ticker": o.get("pdno", ""),
                "name": o.get("prdt_name", ""),
                "side": "매수" if o.get("sll_buy_dvsn_cd") == "02" else "매도",
                "ord_qty": ord_qty,
                "ccld_qty": ccld_qty,
                "ord_price": _safe_int(o.get("ord_unpr", 0)),
                "ccld_price": _safe_int(o.get("avg_prvs", 0)),
                "ccld_amount": _safe_int(o.get("tot_ccld_amt", 0)),
                "order_no": o.get("odno", ""),
                "order_time": o.get("ord_tmd", ""),
                "filled": ccld_qty >= ord_qty,
            })
        return orders

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
